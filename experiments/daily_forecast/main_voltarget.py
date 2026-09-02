"""거래 응용 시도: 변동성 예측 기반 포지션 사이징 백테스트.

확인된 우위 (D_bank 의 5봉 실현변동성 예측 R² 0.52) 를 포트폴리오 위험 배분에 사용:
매 거래일, 각 종목의 예측 변동성 σ̂ 의 역수에 비례해 비중을 배분하는 long-only 포트폴리오.

비교 대상 (동일 종목 집합, 일별 리밸런스):
  EW     동일 비중
  TRAIL  w ∝ 1/σ_trail20 (직전 20봉 실현변동성 — 관행적 기준)
  PRED   w ∝ 1/σ̂_pred   (D_bank 5봉 변동성 예측)

지표: 연환산 수익률·변동성·Sharpe, 최대낙폭, 일평균 turnover,
      비용 반영 Sharpe (turnover × 10bp 차감), 20봉 롤링 변동성의 표준편차 (위험 안정성).

주의: 예측력 → 위험 관리 유용성 검증이 목적. 수수료·체결·차입 등 실거래 요소는 단순화.
실행: python -m experiments.daily_forecast.main_voltarget
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import torch

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4, ReceptorBankForecaster
from experiments.receptor_real_data.cdm_data import LARGE_CAP, SMALL_CAP
from experiments.daily_forecast.data import build_arrays, load_stocks, make_inputs
from experiments.daily_forecast.main import DEVICE, HORIZONS, SEED, WINDOW, YEARS, train_model
from experiments.daily_forecast.main_vol import make_vol_targets

RESULTS_DIR = Path(__file__).parent / "results"

MIN_STOCKS = 30
COST_PER_TURNOVER = 0.0010  # 10bp


def predict_test_sigma(model, hocl_all, v_all, test_starts, mu, sd):
    """test 윈도우별 5봉 σ̂ (log RV 예측 → 역표준화 → exp)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_starts), 8192):
            h, v = make_inputs(hocl_all, v_all, test_starts[i:i + 8192], WINDOW)
            preds.append(model(h, v))
    y_pred = torch.cat(preds) * sd + mu                     # (B, 5) log RV
    return torch.exp(y_pred[:, -1])                          # 5봉 σ̂


def trailing_sigma(hocl_all, starts, window: int, lookback: int = 20):
    logc = hocl_all[:, 2]
    end = starts + window - 1
    idx = end.unsqueeze(-1) + torch.arange(-lookback + 1, 1, device=end.device)
    r = logc[idx] - logc[idx - 1]
    return r.std(dim=1) * math.sqrt(5.0)                     # 5봉 스케일 정합


def next_day_returns(hocl_all, starts, window: int):
    logc = hocl_all[:, 2]
    end = starts + window - 1
    return torch.exp(logc[end + 1] - logc[end]) - 1.0


def portfolio_stats(name, daily_ret, daily_turnover):
    daily_ret = torch.tensor(daily_ret)
    to = torch.tensor(daily_turnover)
    mean, std = daily_ret.mean().item(), daily_ret.std().item()
    ann_ret, ann_vol = mean * 252, std * math.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    net = daily_ret - to * COST_PER_TURNOVER
    net_sharpe = (net.mean().item() * 252) / (net.std().item() * math.sqrt(252) + 1e-12)

    cum = torch.cumprod(1.0 + daily_ret, dim=0)
    peak = torch.cummax(cum, dim=0).values
    maxdd = ((cum - peak) / peak).min().item()

    n = len(daily_ret)
    roll = torch.stack([daily_ret[i:i + 20].std() for i in range(n - 20)]) if n > 40 else daily_ret.std().unsqueeze(0)
    vol_stability = roll.std().item() * math.sqrt(252)

    return {
        "name": name, "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "maxdd": maxdd, "turnover": to.mean().item(), "net_sharpe": net_sharpe,
        "vol_stability": vol_stability,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    api = CandleDataAPI()
    stocks = load_stocks(LARGE_CAP + SMALL_CAP, api, years=YEARS)
    hocl_all, v_all, ts_all, starts, tick_ids = build_arrays(stocks, WINDOW, max(HORIZONS))
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    ts_all = ts_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}
    tick_ids = {k: v.to(DEVICE) for k, v in tick_ids.items()}

    # 변동성 예측 모델 학습 (main_vol 의 D_bank 구성 그대로)
    ys = {k: make_vol_targets(hocl_all, starts[k], WINDOW, HORIZONS) for k in starts}
    mu, sd = ys["train"].mean(dim=0, keepdim=True), ys["train"].std(dim=0, keepdim=True)
    y_n = {k: (v - mu) / sd for k, v in ys.items()}
    torch.manual_seed(SEED)
    _, _, model = train_model(
        ReceptorBankForecaster(HL_K4, "pyramid"),
        hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"],
    )

    st = starts["test"]
    sigma_pred = predict_test_sigma(model, hocl_all, v_all, st, mu.to(DEVICE), sd.to(DEVICE))
    sigma_trail = trailing_sigma(hocl_all, st, WINDOW)
    ret_next = next_day_returns(hocl_all, st, WINDOW)
    tids = tick_ids["test"]
    end_ts = ts_all[st + WINDOW - 1]

    # 날짜별 그룹 구성 (시간 순)
    uniq_ts, inv = torch.unique(end_ts, sorted=True, return_inverse=True)
    schemes = {"EW": None, "TRAIL": sigma_trail, "PRED": sigma_pred}
    rets = {k: [] for k in schemes}
    turnovers = {k: [] for k in schemes}
    prev_w = {k: {} for k in schemes}

    n_dates_used = 0
    for d in range(len(uniq_ts)):
        m = inv == d
        if int(m.sum().item()) < MIN_STOCKS:
            continue
        n_dates_used += 1
        r = ret_next[m]
        ids = tids[m].tolist()
        for name, sig in schemes.items():
            if sig is None:
                w = torch.full_like(r, 1.0 / len(r))
            else:
                inv_s = 1.0 / (sig[m] + 1e-8)
                w = inv_s / inv_s.sum()
            rets[name].append((w * r).sum().item())

            w_map = dict(zip(ids, w.tolist()))
            prev = prev_w[name]
            all_ids = set(w_map) | set(prev)
            turnovers[name].append(
                sum(abs(w_map.get(i, 0.0) - prev.get(i, 0.0)) for i in all_ids) / 2.0
            )
            prev_w[name] = w_map

    lines = [
        f"변동성 예측 기반 포지션 사이징 백테스트: {len(stocks)} 종목, test {n_dates_used} 거래일, "
        f"일별 리밸런스, long-only, 비용 {COST_PER_TURNOVER*1e4:.0f}bp/turnover",
        "",
        f"{'scheme':>6} | {'연수익':>7} | {'연변동':>7} | {'Sharpe':>6} | {'netShp':>6} | "
        f"{'MDD':>7} | {'턴오버':>6} | {'위험안정':>7}",
        "-" * 75,
    ]
    for name in schemes:
        s = portfolio_stats(name, rets[name], turnovers[name])
        lines.append(
            f"{s['name']:>6} | {s['ann_ret']:>+7.2%} | {s['ann_vol']:>7.2%} | {s['sharpe']:>6.2f} | "
            f"{s['net_sharpe']:>6.2f} | {s['maxdd']:>+7.2%} | {s['turnover']:>6.3f} | {s['vol_stability']:>7.3%}"
        )
        print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_voltarget.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
