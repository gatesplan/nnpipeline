"""롱/숏 각각 트레일링 스탑으로 수익이 나는가 — 방향별 분리 측정.

해상도 15m / 1h / 4h, 진입 2종 × 방향 2종:
  breakout-L : 20봉 신고가 돌파 시 롱      breakout-S : 20봉 신저가 이탈 시 숏
  signal-L   : cross-lag 예측 최상위 롱     signal-S   : cross-lag 예측 최하위 숏
청산: 트레일링 스탑 m×σ (m=2, 4), 안전용 최대보유 60봉. 수수료 선물 taker 5bp 편도.
보고: net 연수익 / Sharpe / 거래수 / 승률 / 거래당 평균손익 (bp, 수수료 차감 전·후).

실행: python -m experiments.daily_forecast.main_crypto_sides
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import numpy as np
import pandas as pd
import torch

from candle_data_manager import CandleDataAPI

from experiments.daily_forecast.main_crypto_leadlag import load_4h
from experiments.daily_forecast.main_crypto_trailstop import build_ohlc_panel
from experiments.daily_forecast.main_crypto_multires import (
    RES_SPECS,
    load_resampled,
    panel_from_dfs,
)
from experiments.daily_forecast.main_vol import ridge

RESULTS_DIR = Path(__file__).parent / "results"

STOP_MULTS = (2.0, 4.0)
FEE = 0.0005
MAX_HOLD = 60
VOL_LOOKBACK = 20
DONCHIAN_K = 20


def crosslag_preds(ret):
    """전 구간 cross-lag ridge 예측 (train 70% 적합, 이후 구간 예측). 반환 (T-6, N), 오프셋 6."""
    T, N = ret.shape
    t0 = 6
    X = np.concatenate(
        [np.stack([ret[t0 - l:T - l, j] for l in range(1, 7)], axis=1) for j in range(N)], axis=1
    )
    Y = ret[t0:]
    n_tr = int(len(Y) * 0.7)
    preds = np.zeros((len(Y), N))
    for j in range(N):
        y = torch.tensor(Y[:, j:j + 1])
        sd = y[:n_tr].std() + 1e-12
        preds[:, j] = ridge(
            torch.tensor(X[:n_tr]), y[:n_tr] / sd, torch.tensor(X), alpha=10.0
        )[:, 0].numpy()
    return preds, t0, n_tr + t0


def run_side(arrs, entries, side, m, bars_per_year, eval_start):
    """방향 단일 이벤트 백테스트. entries[t, j] = True 면 t 종가 시점 진입 신호.

    side: +1 롱 / -1 숏. 동시 보유 무제한 (신호가 뜬 종목마다 독립 1단위).
    반환: 통계 dict.
    """
    opn, high, low, close = arrs["open"], arrs["high"], arrs["low"], arrs["close"]
    T, N = close.shape
    r = np.diff(np.log(close), axis=0)
    sig_roll = pd.DataFrame(r).rolling(VOL_LOOKBACK).std().values

    port = np.zeros(T - 1)
    gross_pnls, net_pnls, holds = [], [], []
    open_count = np.zeros(T - 1)

    for j in range(N):
        pos = 0
        entry = extreme = last = stop_d = 0.0
        bars_held = 0
        for t in range(max(VOL_LOOKBACK + 1, eval_start), T - 1):
            o, h, l, c = opn[t + 1, j], high[t + 1, j], low[t + 1, j], close[t + 1, j]
            if pos != 0:
                exit_px = None
                if side > 0:
                    stop = extreme * (1.0 - stop_d)
                    if o <= stop:
                        exit_px = o
                    elif l <= stop:
                        exit_px = stop
                    extreme = max(extreme, h)
                else:
                    stop = extreme * (1.0 + stop_d)
                    if o >= stop:
                        exit_px = o
                    elif h >= stop:
                        exit_px = stop
                    extreme = min(extreme, l)
                bars_held += 1
                if exit_px is None and bars_held >= MAX_HOLD:
                    exit_px = c
                seg = exit_px if exit_px is not None else c
                port[t] += side * (seg / last - 1.0)
                open_count[t] += 1
                last = seg
                if exit_px is not None:
                    port[t] -= FEE
                    g = side * (exit_px / entry - 1.0)
                    gross_pnls.append(g)
                    net_pnls.append(g - 2 * FEE)
                    holds.append(bars_held)
                    pos = 0
            if pos == 0 and entries[t, j]:
                sigma = sig_roll[t - 1, j]
                if not np.isfinite(sigma):
                    continue
                pos, entry, last, extreme = side, o, o, o
                stop_d, bars_held = m * max(sigma, 1e-4), 0
                port[t] -= FEE

    avg_open = max(open_count.mean(), 1e-9)
    port_scaled = port / max(avg_open, 1.0)     # 평균 동시 포지션 수로 정규화한 1단위 기준
    ann = port_scaled.mean() * bars_per_year
    vol = port_scaled.std() * math.sqrt(bars_per_year) + 1e-12
    return {
        "ann": ann, "sharpe": ann / vol,
        "n": len(net_pnls),
        "win": np.mean([p > 0 for p in net_pnls]) if net_pnls else 0.0,
        "gross_bp": np.mean(gross_pnls) * 1e4 if gross_pnls else 0.0,
        "net_bp": np.mean(net_pnls) * 1e4 if net_pnls else 0.0,
        "hold": np.mean(holds) if holds else 0.0,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    api = CandleDataAPI()

    panels = {}
    resampled = load_resampled(api)
    for tf in RES_SPECS:
        names, arrs = panel_from_dfs(resampled[tf])
        panels[tf] = (names, arrs, RES_SPECS[tf][1])
    n4, _, arrs4 = build_ohlc_panel(load_4h(api))
    panels["4h"] = (n4, arrs4, 6 * 365)

    lines = [
        f"방향별 트레일링 스탑: 진입 2종 (20봉 돌파 / cross-lag 극단), 스탑 m×σ, "
        f"최대보유 {MAX_HOLD}봉, 수수료 {FEE*1e4:.0f}bp 편도. 평가 = 각 패널 후반 30% (신호 학습 구간 제외)",
        "",
        f"{'해상도':>4} | {'진입':>10} | {'방향':>2} | {'m':>3} | {'net연수익':>9} | {'Sharpe':>6} | "
        f"{'거래수':>5} | {'승률':>5} | {'거래당 총/순 (bp)':>18} | {'평균보유':>6}",
        "-" * 100,
    ]

    for tf in ("15m", "1h", "4h"):
        names, arrs, bpy = panels[tf]
        close = arrs["close"]
        T, N = close.shape
        ret = np.diff(np.log(close), axis=0)

        # 진입 신호 행렬 — 채널은 직전 봉까지 (현재 봉 포함 시 돌파가 거의 발생 불가)
        hh = pd.DataFrame(arrs["high"]).rolling(DONCHIAN_K).max().shift(1).values
        ll = pd.DataFrame(arrs["low"]).rolling(DONCHIAN_K).min().shift(1).values
        with np.errstate(invalid="ignore"):
            brk_l = close >= hh
            brk_s = close <= ll
        brk_l[:DONCHIAN_K + 1] = False
        brk_s[:DONCHIAN_K + 1] = False

        preds, t0, eval_start = crosslag_preds(ret)
        sig_l = np.zeros((T, N), dtype=bool)
        sig_s = np.zeros((T, N), dtype=bool)
        rows = np.arange(len(preds))
        sig_l[t0 + rows, np.argmax(preds, axis=1)] = True
        sig_s[t0 + rows, np.argmin(preds, axis=1)] = True

        eval_from = int(T * 0.7)
        for ent_name, ent_l, ent_s in (("breakout", brk_l, brk_s), ("signal", sig_l, sig_s)):
            for side, ent in ((+1, ent_l), (-1, ent_s)):
                for m in STOP_MULTS:
                    s = run_side(arrs, ent, side, m, bpy, eval_from)
                    lines.append(
                        f"{tf:>4} | {ent_name:>10} | {'롱' if side > 0 else '숏':>2} | {m:>3.0f} | "
                        f"{s['ann']:>+9.1%} | {s['sharpe']:>6.2f} | {s['n']:>5d} | {s['win']:>5.1%} | "
                        f"{s['gross_bp']:>+8.1f}/{s['net_bp']:>+8.1f} | {s['hold']:>5.1f}봉"
                    )
                    print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_sides.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
