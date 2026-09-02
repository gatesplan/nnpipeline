"""4h lead-lag 신호 + 트레일링 스탑 청산의 이벤트 기반 백테스트.

고정 시간 청산 (main_crypto_leadlag) 과 달리 청산을 경로 의존으로:
- 진입: 매 4h 봉, 빈 슬롯이 있으면 cross-lag ridge 예측 상위 (롱) / 하위 (숏) 종목 진입.
  슬롯 = 롱 3 + 숏 3, 종목당 비중 1/6.
- 청산: 트레일링 스탑 — 롱은 진입 후 최고가 대비 m × σ (종목별 20봉 실현 4h 변동성) 하락 시,
  숏은 최저가 대비 동일 폭 상승 시. 봉 내 고가/저가로 판정, 갭이면 시가 체결.
  안전용 최대보유 60봉 (10일).
- 격자 전체 보고 (사후 선택 방지): m ∈ {1, 2, 3} × 수수료 {taker 5bp, maker 2bp} (편도).

실행: python -m experiments.daily_forecast.main_crypto_trailstop
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import numpy as np
import torch

from candle_data_manager import CandleDataAPI

from experiments.daily_forecast.main_crypto_leadlag import (
    BARS_PER_YEAR,
    MAX_LAG,
    OWN_LAGS,
    load_4h,
)
from experiments.daily_forecast.main_vol import ridge

RESULTS_DIR = Path(__file__).parent / "results"

N_SLOT = 3
MAX_HOLD = 60
VOL_LOOKBACK = 20
STOP_MULTS = (1.0, 2.0, 3.0)
FEES = (0.0005, 0.0002)


def build_ohlc_panel(coins):
    ts_sets = [set(df["timestamp"].values.tolist()) for _, df in coins]
    common = sorted(set.intersection(*ts_sets))
    names = [n for n, _ in coins]
    arrs = {}
    for key in ("high", "low", "close", "open"):
        mat = np.zeros((len(common), len(coins)))
        for j, (_, df) in enumerate(coins):
            m = dict(zip(df["timestamp"].values.tolist(), df[key].values))
            mat[:, j] = [m[t] for t in common]
        arrs[key] = mat
    return names, np.array(common, dtype=np.int64), arrs


def cross_predictions(ret):
    """cross-lag ridge (leadlag Part B 와 동일 구성). 반환: test 구간 예측 (n_te, N) 과 오프셋."""
    T, N = ret.shape
    t0 = MAX_LAG
    X = np.concatenate(
        [np.stack([ret[t0 - l:T - l, j] for l in range(1, OWN_LAGS + 1)], axis=1) for j in range(N)],
        axis=1,
    )
    Y = ret[t0:]
    n = len(Y)
    n_tr, n_va = int(n * 0.7), int(n * 0.85)
    preds = np.zeros((n - n_va, N))
    for j in range(N):
        y = torch.tensor(Y[:, j:j + 1])
        sd = y[:n_tr].std() + 1e-12
        preds[:, j] = ridge(
            torch.tensor(X[:n_tr]), y[:n_tr] / sd, torch.tensor(X[n_va:]), alpha=10.0
        )[:, 0].numpy()
    return preds, t0 + n_va  # ret 인덱스 기준 test 시작 오프셋


def run_backtest(names, arrs, preds, test_off, stop_mult, fee):
    """이벤트 기반 백테스트. 반환: (일련 수익률, 거래 통계 dict)."""
    N = len(names)
    close, high, low, opn = arrs["close"], arrs["high"], arrs["low"], arrs["open"]
    # 종목별 4h 수익률 표준편차 (진입 시점 기준 trailing)
    logc = np.log(close)
    r_all = np.diff(logc, axis=0)

    n_bars = len(preds)
    positions = {}   # coin_idx -> dict(side, entry_px, extreme, bars, stop_dist)
    equity_ret = []
    trades = {"n": 0, "win": 0, "hold": [], "exit_stop": 0, "exit_time": 0, "pnl": []}

    for t in range(n_bars - 1):
        bar = test_off + t          # ret 인덱스: preds[t] 는 close[bar] → close[bar+1] 구간 예측
        px_idx = bar + 1            # 이번 스텝에 전개되는 봉의 OHLC 행 (진입 = 이 봉 시가)

        # 1) 청산 판정: 이번 봉 (px_idx) 의 시가/고가/저가로 스탑 체결
        bar_ret = 0.0
        to_close = []
        for j, p in positions.items():
            o, h, l, c = opn[px_idx, j], high[px_idx, j], low[px_idx, j], close[px_idx, j]
            exit_px = None
            if p["side"] > 0:
                stop = p["extreme"] * (1.0 - p["stop_dist"])
                if o <= stop:
                    exit_px = o
                elif l <= stop:
                    exit_px = stop
                p["extreme"] = max(p["extreme"], h)
            else:
                stop = p["extreme"] * (1.0 + p["stop_dist"])
                if o >= stop:
                    exit_px = o
                elif h >= stop:
                    exit_px = stop
                p["extreme"] = min(p["extreme"], l)
            p["bars"] += 1
            if exit_px is None and p["bars"] >= MAX_HOLD:
                exit_px = c
                trades["exit_time"] += 1
            elif exit_px is not None:
                trades["exit_stop"] += 1

            ref_px = p["last_px"]
            seg_px = exit_px if exit_px is not None else c
            bar_ret += p["side"] * (seg_px / ref_px - 1.0) / (2 * N_SLOT)
            p["last_px"] = seg_px
            if exit_px is not None:
                bar_ret -= fee / (2 * N_SLOT)
                pnl = p["side"] * (exit_px / p["entry_px"] - 1.0)
                trades["pnl"].append(pnl)
                trades["win"] += pnl > 0
                trades["hold"].append(p["bars"])
                to_close.append(j)
        for j in to_close:
            del positions[j]

        # 2) 진입: 빈 슬롯에 현재 예측 상/하위 종목 (미보유) 진입 — 다음 봉 시가 체결
        n_long = sum(1 for p in positions.values() if p["side"] > 0)
        n_short = len(positions) - n_long
        order = np.argsort(-preds[t])
        sig_vol = r_all[max(0, bar - VOL_LOOKBACK):bar].std(axis=0)
        for j in order:
            if n_long >= N_SLOT:
                break
            if j not in positions:
                entry = opn[px_idx, j]
                positions[j] = dict(
                    side=+1, entry_px=entry, last_px=entry, extreme=entry,
                    bars=0, stop_dist=stop_mult * max(sig_vol[j], 1e-4),
                )
                bar_ret -= fee / (2 * N_SLOT)
                trades["n"] += 1
                n_long += 1
        for j in order[::-1]:
            if n_short >= N_SLOT:
                break
            if j not in positions:
                entry = opn[px_idx, j]
                positions[j] = dict(
                    side=-1, entry_px=entry, last_px=entry, extreme=entry,
                    bars=0, stop_dist=stop_mult * max(sig_vol[j], 1e-4),
                )
                bar_ret -= fee / (2 * N_SLOT)
                trades["n"] += 1
                n_short += 1

        equity_ret.append(bar_ret)

    return np.array(equity_ret), trades


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    api = CandleDataAPI()
    coins = load_4h(api)
    names, ts, arrs = build_ohlc_panel(coins)
    ret = np.diff(np.log(arrs["close"]), axis=0)
    preds, test_off = cross_predictions(ret)

    lines = [
        f"트레일링 스탑 백테스트: {len(names)} 종목, 슬롯 롱{N_SLOT}+숏{N_SLOT}, "
        f"스탑 = m × 20봉 실현 4h 변동성, 최대보유 {MAX_HOLD}봉, test {len(preds)}봉",
        "",
        f"{'m':>4} | {'fee':>4} | {'net연수익':>9} | {'netShp':>6} | {'MDD':>7} | "
        f"{'거래수':>5} | {'승률':>5} | {'평균보유':>7} | {'스탑청산':>7}",
        "-" * 85,
    ]

    for m in STOP_MULTS:
        for fee in FEES:
            rets, tr = run_backtest(names, arrs, preds, test_off, m, fee)
            ann = rets.mean() * BARS_PER_YEAR
            vol = rets.std() * math.sqrt(BARS_PER_YEAR)
            shp = ann / vol if vol > 0 else float("nan")
            cum = np.cumprod(1.0 + rets)
            peak = np.maximum.accumulate(cum)
            mdd = ((cum - peak) / peak).min()
            wr = tr["win"] / max(len(tr["pnl"]), 1)
            stop_frac = tr["exit_stop"] / max(tr["exit_stop"] + tr["exit_time"], 1)
            lines.append(
                f"{m:>4.1f} | {fee*1e4:>3.0f}bp | {ann:>+9.1%} | {shp:>6.2f} | {mdd:>+7.1%} | "
                f"{len(tr['pnl']):>5d} | {wr:>5.1%} | {np.mean(tr['hold']):>6.1f}봉 | {stop_frac:>7.1%}"
            )
            print(lines[-1])

    lines += [
        "",
        "참고: 진입 = cross-lag 예측 상/하위 종목 (다음 봉 시가 체결), 청산 = 트레일링 스탑 "
        "(봉 내 고가/저가 판정, 갭 시 시가). 수수료는 편도, 진입·청산 각 1회.",
    ]
    report = "\n".join(lines)
    (RESULTS_DIR / "report_trailstop.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
