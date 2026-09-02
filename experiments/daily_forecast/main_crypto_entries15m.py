"""15m 진입 전략 전수 조사: 다양한 진입 × 롱/숏 × 트레일링 스탑.

종목: BTC/ETH/SOL/XRP/DOGE (1m 리샘플 15m, ~2020-09~2026-03, 전 기간 평가 — 규칙 기반이라 적합 없음).
청산: 트레일링 스탑 m×σ (m=2, 4), 최대보유 60봉, 수수료 선물 taker 5bp 편도.

진입 8계열 (롱 조건 / 숏 조건):
  don96    96봉(1일) 신고가 돌파 / 신저가 이탈          — 추세
  don288   288봉(3일) 신고가 돌파 / 신저가 이탈         — 추세 (장기)
  emax     EMA20 이 EMA96 상향 교차 / 하향 교차          — 추세 전환
  rsi14    RSI14 < 30 / > 70                             — 역추세
  boll20   종가 < SMA20-2σ / > SMA20+2σ                  — 역추세
  dip1h    직전 1h(4봉) 수익률 < -2σ / > +2σ             — 급락 반등 / 급등 페이드
  mom1d    직전 96봉 수익률 > +1σ / < -1σ                — 시계열 모멘텀
  volspk   거래량 > 3×평균 이고 양봉 / 음봉              — 거래량 급증 추종

실행: python -m experiments.daily_forecast.main_crypto_entries15m
"""

from pathlib import Path

from loguru import logger

logger.remove()

import numpy as np
import pandas as pd

from candle_data_manager import CandleDataAPI

from experiments.daily_forecast.main_crypto_multires import (
    RES_SPECS,
    load_resampled,
    panel_from_dfs,
)
from experiments.daily_forecast.main_crypto_sides import run_side
from experiments.daily_forecast.report_notify import notify

RESULTS_DIR = Path(__file__).parent / "results"

STOP_MULTS = (2.0, 4.0)
BARS_PER_YEAR = RES_SPECS["15m"][1]
WARMUP = 600


def rsi(close_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    diff = close_df.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def build_entries(arrs, vol_mat):
    """전략별 (롱 진입, 숏 진입) bool (T, N) 행렬."""
    close = pd.DataFrame(arrs["close"])
    high = pd.DataFrame(arrs["high"])
    low = pd.DataFrame(arrs["low"])
    vol = pd.DataFrame(vol_mat)
    logc = np.log(close)
    r1 = logc.diff()

    out = {}

    for k, name in ((96, "don96"), (288, "don288")):
        hh = high.rolling(k).max().shift(1)
        ll = low.rolling(k).min().shift(1)
        out[name] = ((close >= hh).values, (close <= ll).values)

    e_fast = close.ewm(span=20, adjust=False).mean()
    e_slow = close.ewm(span=96, adjust=False).mean()
    above = e_fast > e_slow
    out["emax"] = (
        (above & ~above.shift(1).fillna(False)).values,
        (~above & above.shift(1).fillna(True)).values,
    )

    rs = rsi(close)
    out["rsi14"] = ((rs < 30).values, (rs > 70).values)

    ma = close.rolling(20).mean()
    sd = close.rolling(20).std()
    out["boll20"] = ((close < ma - 2 * sd).values, (close > ma + 2 * sd).values)

    r4 = logc.diff(4)
    s4 = r4.rolling(96).std()
    out["dip1h"] = ((r4 < -2 * s4).values, (r4 > 2 * s4).values)

    r96 = logc.diff(96)
    s96 = r96.rolling(480).std()
    out["mom1d"] = ((r96 > s96).values, (r96 < -s96).values)

    vma = vol.rolling(96).mean()
    spike = vol > 3 * vma
    out["volspk"] = ((spike & (r1 > 0)).values, (spike & (r1 < 0)).values)

    for name, (l, s) in out.items():
        l[:WARMUP] = False
        s[:WARMUP] = False
        out[name] = (np.nan_to_num(l).astype(bool), np.nan_to_num(s).astype(bool))
    return out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    api = CandleDataAPI()
    resampled = load_resampled(api)
    names, arrs = panel_from_dfs(resampled["15m"])
    vol_mat = np.zeros_like(arrs["close"])
    ts_sets = [set(df["timestamp"].tolist()) for df in resampled["15m"].values()]
    common = sorted(set.intersection(*ts_sets))
    for j, nm in enumerate(names):
        m = dict(zip(resampled["15m"][nm]["timestamp"].tolist(), resampled["15m"][nm]["volume"].tolist()))
        vol_mat[:, j] = [m[t] for t in common]

    entries = build_entries(arrs, vol_mat)

    lines = [
        f"15m 진입 전략 전수 조사: {len(names)} 종목 ({', '.join(names)}), "
        f"{len(arrs['close'])}봉 전 기간, 스탑 m×σ, 수수료 5bp 편도",
        "",
        f"{'전략':>7} | {'방향':>2} | {'m':>3} | {'net연수익':>9} | {'Sharpe':>6} | "
        f"{'거래수':>6} | {'승률':>5} | {'거래당 총/순 (bp)':>18} | {'평균보유':>6}",
        "-" * 95,
    ]

    for name, (ent_l, ent_s) in entries.items():
        for side, ent in ((+1, ent_l), (-1, ent_s)):
            for m in STOP_MULTS:
                s = run_side(arrs, ent, side, m, BARS_PER_YEAR, WARMUP)
                lines.append(
                    f"{name:>7} | {'롱' if side > 0 else '숏':>2} | {m:>3.0f} | "
                    f"{s['ann']:>+9.1%} | {s['sharpe']:>6.2f} | {s['n']:>6d} | {s['win']:>5.1%} | "
                    f"{s['gross_bp']:>+8.1f}/{s['net_bp']:>+8.1f} | {s['hold']:>5.1f}봉"
                )
                print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_entries15m.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")
    notify("15m 진입 전략 전수 조사 계산 완료 — 분석 보고 이어집니다.", prefix="[nnpipeline]")


if __name__ == "__main__":
    main()
