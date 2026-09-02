"""숏·롱숏 거래 가능성: 횡단면 신호 기반 시장중립 전략 백테스트.

신호: 같은 날짜 종목 집합 내 미래 수익률 z-score 예측 (main_cross 와 동일 학습).
매 test 거래일, 예측 점수 상위 20% 를 동일비중 매수 (long), 하위 20% 를 동일비중 공매도 (short).
달러중립 (매수 1 : 공매도 1, 총노출 2). 일별 리밸런스, horizon-1 점수 사용.

비교: LS_linear (momentum ridge), LS_bank (receptor→DecayBank).
다리별 분해로 숏 단독 가능성도 판정:
  - long leg alpha  = 상위 바스켓 수익률 − 전체 EW
  - short leg alpha = 전체 EW − 하위 바스켓 수익률 (공매도 기여)
  - short 단독 P&L  = −하위 바스켓 수익률 (하락장 외에는 음수가 정상)

비용: 10bp × turnover. 공매도 차입비용은 미모델 — 연 0.5% (숏 노출 기준) 민감도만 병기.
실행: python -m experiments.daily_forecast.main_longshort
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import torch

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4, ReceptorBankForecaster
from experiments.receptor_real_data.cdm_data import LARGE_CAP, SMALL_CAP
from experiments.daily_forecast.data import build_arrays, load_stocks, make_inputs, make_targets
from experiments.daily_forecast.main import (
    DEVICE,
    HORIZONS,
    SEED,
    WINDOW,
    YEARS,
    linear_features,
    train_model,
)
from experiments.daily_forecast.main_vol import ridge
from experiments.daily_forecast.main_cross import (
    MIN_GROUP,
    build_groups,
    cs_normalize,
    daily_ic,
    filter_small_groups,
)
from experiments.daily_forecast.main_voltarget import next_day_returns

RESULTS_DIR = Path(__file__).parent / "results"

QUANTILE = 0.2
COST_PER_TURNOVER = 0.0010
BORROW_ANN = 0.005
SIGNAL_H = 0  # horizon-1 점수로 일별 리밸런스


def backtest_ls(scores, ret_next, tids, gid, n_dates):
    """일별 상/하위 QUANTILE 바스켓 시계열. 반환: dict of 일별 리스트."""
    out = {"long": [], "short": [], "ew": [], "turnover": []}
    prev_w = {}
    for d in range(n_dates):
        m = gid == d
        k = int(m.sum().item())
        if k < MIN_GROUP:
            continue
        s = scores[m]
        r = ret_next[m]
        ids = tids[m].tolist()
        n_leg = max(1, int(k * QUANTILE))
        order = torch.argsort(s, descending=True)
        long_idx, short_idx = order[:n_leg], order[-n_leg:]

        out["long"].append(r[long_idx].mean().item())
        out["short"].append(r[short_idx].mean().item())
        out["ew"].append(r.mean().item())

        w = {}
        for i in long_idx.tolist():
            w[ids[i]] = 1.0 / n_leg
        for i in short_idx.tolist():
            w[ids[i]] = w.get(ids[i], 0.0) - 1.0 / n_leg
        all_ids = set(w) | set(prev_w)
        out["turnover"].append(
            sum(abs(w.get(i, 0.0) - prev_w.get(i, 0.0)) for i in all_ids) / 2.0
        )
        prev_w = w
    return out


def ls_stats(name, series):
    long_r = torch.tensor(series["long"])
    short_r = torch.tensor(series["short"])
    ew_r = torch.tensor(series["ew"])
    to = torch.tensor(series["turnover"])

    ls = long_r - short_r
    ann_ret = ls.mean().item() * 252
    ann_vol = ls.std().item() * math.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    net = ls - to * COST_PER_TURNOVER
    net_ann = net.mean().item() * 252 - BORROW_ANN
    net_sharpe = net_ann / (net.std().item() * math.sqrt(252) + 1e-12)
    cum = torch.cumprod(1.0 + ls, dim=0)
    maxdd = ((cum - torch.cummax(cum, dim=0).values) / torch.cummax(cum, dim=0).values).min().item()

    long_alpha = (long_r - ew_r).mean().item() * 252
    short_alpha = (ew_r - short_r).mean().item() * 252
    short_pnl = (-short_r).mean().item() * 252

    return (
        f"{name:>9} | {ann_ret:>+7.2%} | {ann_vol:>6.2%} | {sharpe:>6.2f} | {net_sharpe:>6.2f} | "
        f"{maxdd:>+7.2%} | {to.mean().item():>5.3f} | {long_alpha:>+7.2%} | {short_alpha:>+7.2%} | {short_pnl:>+8.2%}"
    )


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

    date_ids, _ = build_groups(ts_all, starts, WINDOW)
    keep_masks = {}
    for k in starts:
        cnt = torch.bincount(date_ids[k])
        keep_masks[k] = cnt[date_ids[k]] >= MIN_GROUP
    starts = {k: starts[k][keep_masks[k]] for k in starts}
    tick_ids = {k: tick_ids[k][keep_masks[k]] for k in tick_ids}
    gids = {}
    for k in starts:
        end_ts = ts_all[starts[k] + WINDOW - 1]
        _, gids[k] = torch.unique(end_ts, sorted=True, return_inverse=True)

    y_n = {}
    for k in starts:
        y_raw = make_targets(hocl_all, starts[k], WINDOW, HORIZONS)
        y_n[k] = cs_normalize(y_raw, gids[k])

    ret_next = next_day_returns(hocl_all, starts["test"], WINDOW)
    n_dates = int(gids["test"].max().item()) + 1

    lines = [
        f"롱숏 백테스트: {len(stocks)} 종목, 상/하위 {QUANTILE:.0%} 바스켓, 달러중립, "
        f"일별 리밸런스 (horizon-1 점수), 비용 10bp×turnover + 차입 {BORROW_ANN:.1%}/년",
        f"test {n_dates} 거래일",
        "",
        f"{'model':>9} | {'연수익':>7} | {'연변동':>6} | {'Sharpe':>6} | {'netShp':>6} | "
        f"{'MDD':>7} | {'턴오버':>5} | {'롱알파':>7} | {'숏알파':>7} | {'숏단독PL':>8}",
        "-" * 105,
    ]

    # 선형 momentum ridge
    x_tr = linear_features(hocl_all, v_all, starts["train"])
    x_te = linear_features(hocl_all, v_all, starts["test"])
    pred_lin = ridge(x_tr, y_n["train"], x_te)
    ic_lin, t_lin = daily_ic(pred_lin, y_n["test"], gids["test"])
    series = backtest_ls(pred_lin[:, SIGNAL_H], ret_next, tick_ids["test"], gids["test"], n_dates)
    lines.append(ls_stats("LS_linear", series))
    print(lines[-1])

    # bank
    torch.manual_seed(SEED)
    _, hls, model = train_model(
        ReceptorBankForecaster(HL_K4, "pyramid"),
        hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"],
    )
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(starts["test"]), 8192):
            h, v = make_inputs(hocl_all, v_all, starts["test"][i:i + 8192], WINDOW)
            preds.append(model(h, v))
    pred_bank = torch.cat(preds)
    ic_bank, t_bank = daily_ic(pred_bank, y_n["test"], gids["test"])
    series = backtest_ls(pred_bank[:, SIGNAL_H], ret_next, tick_ids["test"], gids["test"], n_dates)
    lines.append(ls_stats("LS_bank", series))
    print(lines[-1])

    lines += [
        "",
        f"IC@1 (t): linear {ic_lin[0].item():+.4f} ({t_lin[0].item():+.1f}), "
        f"bank {ic_bank[0].item():+.4f} ({t_bank[0].item():+.1f})",
        f"bank learned hl: [" + ", ".join(f"{h:.1f}" for h in hls) + "]",
        "참고: 숏알파 > 0 이면 하위 바스켓이 시장을 하회 (공매도가 상대가치 기여). "
        "숏단독PL 은 하락장 외에는 음수가 정상.",
    ]
    report = "\n".join(lines)
    (RESULTS_DIR / "report_longshort.txt").write_text(report, encoding="utf-8")
    print(report.split(chr(10))[-3])
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
