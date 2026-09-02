"""거래 응용 2차: vol-targeting overlay 백테스트.

EW 포트폴리오의 총 노출을 목표 변동성 (연 15%) 에 맞춰 일별 스케일링:
    L_t = clip( σ_target / σ̂_p(t), 0.2, 2.0 ),  일수익률 = L_t × r_EW(t)

포트폴리오 변동성 예측 σ̂_p 두 방식 (상관 구조는 공통으로 trailing 사용, 수준만 다름):
  TRAIL  σ̂_p = 직전 20일 EW 포트폴리오 실현변동성 (지속성 가정)
  PRED   σ̂_p = TRAIL × ( 종목평균 σ̂_bank / 종목평균 σ_trail )  — bank 의 5봉 예측으로 수준 갱신

비중 배분 (1/σ) 과 달리 노출 스케일링은 예측 오차가 실현 변동성의 목표 이탈로 직결되므로
예측 정밀도가 경제적 지표 (목표 추적 오차) 로 직접 번역된다.

지표: 목표 추적 (20일 롤링 실현변동성의 |편차| 평균·표준편차), Sharpe, net Sharpe
      (|ΔL| × 10bp 비용), MDD. 부가: 날짜별 σ̂_p vs 실현 미래 5일 포트폴리오 변동성의 상관.

실행: python -m experiments.daily_forecast.main_overlay
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import torch

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4, ReceptorBankForecaster
from experiments.receptor_real_data.cdm_data import LARGE_CAP, SMALL_CAP
from experiments.daily_forecast.data import build_arrays, load_stocks
from experiments.daily_forecast.main import DEVICE, HORIZONS, SEED, WINDOW, YEARS, train_model
from experiments.daily_forecast.main_vol import make_vol_targets
from experiments.daily_forecast.main_voltarget import (
    MIN_STOCKS,
    next_day_returns,
    predict_test_sigma,
    trailing_sigma,
)

RESULTS_DIR = Path(__file__).parent / "results"

TARGET_ANN_VOL = 0.15
LEV_MIN, LEV_MAX = 0.2, 2.0
WARMUP_DAYS = 20
COST_PER_TURNOVER = 0.0010


def overlay_stats(name, rets, levs):
    rets = torch.tensor(rets)
    levs = torch.tensor(levs)
    n = len(rets)

    roll = torch.stack([rets[i:i + 20].std() for i in range(n - 20)]) * math.sqrt(252)
    track_mae = (roll - TARGET_ANN_VOL).abs().mean().item()
    track_std = roll.std().item()

    mean, std = rets.mean().item(), rets.std().item()
    ann_ret, ann_vol = mean * 252, std * math.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    dl = torch.cat([levs[:1] * 0, (levs[1:] - levs[:-1]).abs()])
    net = rets - dl * COST_PER_TURNOVER
    net_sharpe = (net.mean().item() * 252) / (net.std().item() * math.sqrt(252) + 1e-12)

    cum = torch.cumprod(1.0 + rets, dim=0)
    peak = torch.cummax(cum, dim=0).values
    maxdd = ((cum - peak) / peak).min().item()

    return (
        f"{name:>6} | {ann_vol:>7.2%} | {track_mae:>8.3%} | {track_std:>8.3%} | "
        f"{ann_ret:>+7.2%} | {sharpe:>6.2f} | {net_sharpe:>6.2f} | {maxdd:>+7.2%} | "
        f"{levs.mean().item():>5.2f}"
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    api = CandleDataAPI()
    stocks = load_stocks(LARGE_CAP + SMALL_CAP, api, years=YEARS)
    hocl_all, v_all, ts_all, starts, _ = build_arrays(stocks, WINDOW, max(HORIZONS))
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    ts_all = ts_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}

    ys = {k: make_vol_targets(hocl_all, starts[k], WINDOW, HORIZONS) for k in starts}
    mu, sd = ys["train"].mean(dim=0, keepdim=True), ys["train"].std(dim=0, keepdim=True)
    y_n = {k: (v - mu) / sd for k, v in ys.items()}
    torch.manual_seed(SEED)
    _, _, model = train_model(
        ReceptorBankForecaster(HL_K4, "pyramid"),
        hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"],
    )

    def build_day_series(split: str):
        """분할별 날짜 시계열: EW 일수익률, 종목평균 σ̂/σ_trail 비율, trailing·미래 포트폴리오 변동성."""
        st = starts[split]
        sigma_pred = predict_test_sigma(model, hocl_all, v_all, st, mu.to(DEVICE), sd.to(DEVICE))
        sigma_trail = trailing_sigma(hocl_all, st, WINDOW)
        ret_next = next_day_returns(hocl_all, st, WINDOW)
        end_ts = ts_all[st + WINDOW - 1]

        uniq_ts, inv = torch.unique(end_ts, sorted=True, return_inverse=True)
        r_list, ratio_list = [], []
        for d in range(len(uniq_ts)):
            m = inv == d
            if int(m.sum().item()) < MIN_STOCKS:
                continue
            r_list.append(ret_next[m].mean().item())
            ratio_list.append((sigma_pred[m].mean() / (sigma_trail[m].mean() + 1e-12)).item())

        n = len(r_list)
        r_ew = torch.tensor(r_list)
        ratio = torch.tensor(ratio_list)
        sigma_p_trail = torch.tensor([
            r_ew[max(0, t - WARMUP_DAYS):t].std().item() * math.sqrt(252) if t >= 5 else float("nan")
            for t in range(n)
        ])
        fwd_vol = torch.tensor([
            r_ew[t:t + 5].std().item() * math.sqrt(252) if t + 5 <= n else float("nan")
            for t in range(n)
        ])
        return r_ew, ratio, sigma_p_trail, fwd_vol

    r_ew, ratio, sigma_p_trail, fwd_vol = build_day_series("test")
    n = len(r_ew)

    # val 구간에서 두 예측기의 수준 보정 상수 산출 (log 평균 일치) — test 정보 미사용
    r_ew_v, ratio_v, trail_v, fwd_v = build_day_series("val")
    valid_v = torch.arange(WARMUP_DAYS, len(r_ew_v) - 5)
    cal_trail = math.exp((torch.log(fwd_v[valid_v]) - torch.log(trail_v[valid_v])).mean().item())
    cal_pred = math.exp(
        (torch.log(fwd_v[valid_v]) - torch.log(trail_v[valid_v] * ratio_v[valid_v])).mean().item()
    )

    # 예측 품질 부가 지표: 미래 5일 실현 포트폴리오 변동성과의 log 상관 (test)
    valid = torch.arange(WARMUP_DAYS, n - 5)
    lt = torch.log(sigma_p_trail[valid])
    lp = torch.log(sigma_p_trail[valid] * ratio[valid])
    lf = torch.log(fwd_vol[valid])
    corr_trail = torch.corrcoef(torch.stack([lt, lf]))[0, 1].item()
    corr_pred = torch.corrcoef(torch.stack([lp, lf]))[0, 1].item()

    def lev(sigma_hat: float) -> float:
        return min(max(TARGET_ANN_VOL / (sigma_hat + 1e-12), LEV_MIN), LEV_MAX)

    # overlay 실행 (warm-up 이후). *_CAL 은 val 보정 상수 적용
    schemes = {
        "BASE": lambda t: 1.0,
        "TRAIL": lambda t: lev(sigma_p_trail[t].item()),
        "PRED": lambda t: lev(sigma_p_trail[t].item() * ratio[t].item()),
        "TR_CAL": lambda t: lev(sigma_p_trail[t].item() * cal_trail),
        "PR_CAL": lambda t: lev(sigma_p_trail[t].item() * ratio[t].item() * cal_pred),
    }
    lines = [
        f"vol-targeting overlay: {len(stocks)} 종목 EW, test {n - WARMUP_DAYS} 거래일, "
        f"목표 연변동성 {TARGET_ANN_VOL:.0%}, 레버리지 [{LEV_MIN}, {LEV_MAX}], 비용 10bp×|ΔL|",
        f"포트폴리오 변동성 예측의 미래 5일 실현변동성과 log 상관: "
        f"TRAIL {corr_trail:+.3f}, PRED {corr_pred:+.3f}",
        f"val 보정 상수: TRAIL ×{cal_trail:.3f}, PRED ×{cal_pred:.3f}",
        "",
        f"{'scheme':>6} | {'연변동':>7} | {'목표MAE':>8} | {'롤링std':>8} | "
        f"{'연수익':>7} | {'Sharpe':>6} | {'netShp':>6} | {'MDD':>7} | {'평균L':>5}",
        "-" * 90,
    ]
    for name, lev_fn in schemes.items():
        levs = [lev_fn(t) for t in range(WARMUP_DAYS, n)]
        rets = [levs[i] * r_ew[WARMUP_DAYS + i].item() for i in range(len(levs))]
        lines.append(overlay_stats(name, rets, levs))
        print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_overlay.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
