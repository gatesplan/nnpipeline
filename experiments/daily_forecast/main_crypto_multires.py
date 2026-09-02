"""다중 해상도 (15m / 1h / 4h) 크립토 특성 조사.

데이터: 15m·1h 는 1m 봉 (BTC/ETH/SOL/XRP/DOGE) 리샘플, 4h 는 기존 19종목.
리샘플 결과는 .cache/crypto_resample/ 에 캐싱.

Part A. lead-lag 지도 (해상도별, 전·후반 재현성 필터)
Part B. cross-lag 신호 특성: 다음 봉 상/하위 1종목 롱숏의 총/순 성과 (수수료 5bp·2bp)
Part C. 추세 진입 (Donchian K봉 돌파) + 트레일링 스탑 (m×σ) — 스탑 청산의 본령 검증.
        종목별 시스템, 포트폴리오 = 종목 동일비중. 격자 (K × m × fee) 전체 보고.

실행: python -m experiments.daily_forecast.main_crypto_multires
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import numpy as np
import pandas as pd

from candle_data_manager import CandleDataAPI

from experiments.daily_forecast.main_crypto_leadlag import load_4h
from experiments.daily_forecast.main_crypto_trailstop import build_ohlc_panel
from experiments.daily_forecast.report_notify import notify

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(".cache/crypto_resample")

COINS_1M = ("BTC", "ETH", "SOL", "XRP", "DOGE")
RES_SPECS = {"15m": ("15min", 4 * 24 * 365), "1h": ("1h", 24 * 365)}
MAX_LAG = 8
DONCHIAN_KS = (20, 55)
STOP_MULTS = (2.0, 3.0, 4.0)
FEES = (0.0005, 0.0002)
VOL_LOOKBACK = 20


def load_resampled(api):
    """1m → 15m/1h 리샘플 (캐시 사용). 반환: {tf: {coin: DataFrame(ts index, OHLCV)}}."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {tf: {} for tf in RES_SPECS}
    missing = [
        c for c in COINS_1M
        if not all((CACHE_DIR / f"{c}_{tf}.parquet").exists() for tf in RES_SPECS)
    ]
    raw = {}
    if missing:
        markets = api.load(
            archetype="CRYPTO", exchange="BINANCE", tradetype="FUTURES",
            quote="USDT", timeframe="1m",
        )
        for m in markets:
            if m.symbol.base in missing:
                raw[m.symbol.base] = m.candles

    for coin in COINS_1M:
        for tf, (rule, _) in RES_SPECS.items():
            cache = CACHE_DIR / f"{coin}_{tf}.parquet"
            if cache.exists():
                out[tf][coin] = pd.read_parquet(cache)
                continue
            df = raw[coin][["timestamp", "open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            rs = df.resample(rule).agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
            ).dropna()
            rs = rs[rs["volume"] > 0]
            rs["timestamp"] = rs.index.view("int64") // 10 ** 9
            out[tf][coin] = rs.reset_index(drop=True)
            rs.reset_index(drop=True).to_parquet(cache)
    return out


def panel_from_dfs(dfs):
    """공통 timestamp OHLC 패널. 반환: (names, arrs dict of (T,N))."""
    names = list(dfs.keys())
    ts_sets = [set(df["timestamp"].tolist()) for df in dfs.values()]
    common = sorted(set.intersection(*ts_sets))
    arrs = {}
    for key in ("open", "high", "low", "close"):
        mat = np.zeros((len(common), len(names)))
        for j, nm in enumerate(names):
            m = dict(zip(dfs[nm]["timestamp"].tolist(), dfs[nm][key].tolist()))
            mat[:, j] = [m[t] for t in common]
        arrs[key] = mat
    return names, arrs


def leadlag_summary(names, ret, max_lag=MAX_LAG):
    T, N = ret.shape
    half = T // 2
    thr = 2.0 / math.sqrt(half)
    found = []
    for lag in range(1, max_lag + 1):
        x, y = ret[:-lag], ret[lag:]
        cs = []
        for seg in ((slice(None, half)), (slice(half, None))):
            xs = (x[seg] - x[seg].mean(0)) / (x[seg].std(0) + 1e-12)
            ys = (y[seg] - y[seg].mean(0)) / (y[seg].std(0) + 1e-12)
            cs.append(xs.T @ ys / len(xs))
        c1, c2 = cs
        for i in range(N):
            for j in range(N):
                if i != j and abs(c1[i, j]) > thr and abs(c2[i, j]) > thr and c1[i, j] * c2[i, j] > 0:
                    found.append((lag, names[i], names[j], (c1[i, j] + c2[i, j]) / 2))
    found.sort(key=lambda x: -abs(x[3]))
    lines = [f"  재현 쌍 {len(found)}개 (기준 |corr|>{thr:.4f} 양쪽 반)"]
    for lag, ni, nj, c in found[:8]:
        lines.append(f"    lag {lag}: {ni} -> {nj}  {c:+.4f}")
    return lines


def crosslag_ls(names, ret, bars_per_year):
    """cross-lag ridge, 상/하위 1종목 롱숏 (1봉 보유)."""
    import torch
    from experiments.daily_forecast.main_vol import ridge

    T, N = ret.shape
    t0 = 6
    X = np.concatenate(
        [np.stack([ret[t0 - l:T - l, j] for l in range(1, 7)], axis=1) for j in range(N)], axis=1
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
    Yt = Y[n_va:]
    prev = np.zeros(N)
    ls, tos = [], []
    for t in range(len(Yt)):
        w = np.zeros(N)
        w[np.argmax(preds[t])] = 1.0
        w[np.argmin(preds[t])] = -1.0
        ls.append(float(w @ Yt[t]))
        tos.append(float(np.abs(w - prev).sum() / 2))
        prev = w
    ls, tos = np.array(ls), np.array(tos)
    g = ls.mean() * bars_per_year
    gv = ls.std() * math.sqrt(bars_per_year) + 1e-12
    row = f"  총 {g:+8.1%}/년 Shp {g/gv:5.2f} 턴오버 {tos.mean():.2f}/봉"
    for fee in FEES:
        net = ls - tos * fee
        na = net.mean() * bars_per_year
        row += f" | net({fee*1e4:.0f}bp) {na:+8.1%} Shp {na/(net.std()*math.sqrt(bars_per_year)+1e-12):5.2f}"
    return [row]


def donchian_trail(names, arrs, bars_per_year, k, m, fee):
    """종목별 Donchian K봉 돌파 진입 + m×σ 트레일링 스탑. 포트폴리오 동일비중."""
    opn, high, low, close = arrs["open"], arrs["high"], arrs["low"], arrs["close"]
    T, N = close.shape
    logc = np.log(close)
    r = np.diff(logc, axis=0)

    # rolling 지표 사전 계산 (루프 내 O(k) 슬라이스 제거)
    hh_roll = pd.DataFrame(high).rolling(k).max().values
    ll_roll = pd.DataFrame(low).rolling(k).min().values
    sig_roll = pd.DataFrame(r).rolling(VOL_LOOKBACK).std().values   # r 인덱스 기준

    port = np.zeros(T - 1)
    n_trades, n_wins, holds = 0, 0, []
    for j in range(N):
        pos, entry, extreme, stop_d, bars_held, last = 0, 0.0, 0.0, 0.0, 0, 0.0
        pnl_j = np.zeros(T - 1)
        for t in range(k + VOL_LOOKBACK, T - 1):
            o, h, l, c = opn[t + 1, j], high[t + 1, j], low[t + 1, j], close[t + 1, j]
            if pos != 0:
                exit_px = None
                if pos > 0:
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
                seg = exit_px if exit_px is not None else c
                pnl_j[t] += pos * (seg / last - 1.0)
                last = seg
                bars_held += 1
                if exit_px is not None:
                    pnl_j[t] -= fee
                    n_trades += 1
                    n_wins += pos * (exit_px / entry - 1.0) > 0
                    holds.append(bars_held)
                    pos = 0
            if pos == 0:
                sig = 0
                if close[t, j] >= hh_roll[t, j]:
                    sig = +1
                elif close[t, j] <= ll_roll[t, j]:
                    sig = -1
                if sig != 0:
                    sigma = sig_roll[t - 1, j]
                    pos, entry, last, extreme = sig, o, o, o
                    stop_d, bars_held = m * max(sigma, 1e-4), 0
                    pnl_j[t] -= fee
        port += pnl_j / N

    ann = port.mean() * bars_per_year
    vol = port.std() * math.sqrt(bars_per_year) + 1e-12
    cum = np.cumprod(1.0 + port)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min()
    return (
        f"  K={k:>2} m={m:.0f} fee={fee*1e4:.0f}bp: net {ann:+8.1%}/년 Shp {ann/vol:5.2f} "
        f"MDD {mdd:+6.1%} 거래 {n_trades} 승률 {n_wins/max(n_trades,1):.0%} "
        f"평균보유 {np.mean(holds) if holds else 0:.0f}봉"
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    api = CandleDataAPI()

    panels = {}
    resampled = load_resampled(api)
    for tf in RES_SPECS:
        names, arrs = panel_from_dfs(resampled[tf])
        panels[tf] = (names, arrs, RES_SPECS[tf][1])
    coins4h = load_4h(api)
    n4, ts4, arrs4 = build_ohlc_panel(coins4h)
    panels["4h"] = (n4, arrs4, 6 * 365)

    lines = []
    for tf in ("15m", "1h", "4h"):
        names, arrs, bpy = panels[tf]
        ret = np.diff(np.log(arrs["close"]), axis=0)
        lines += [
            "",
            f"===== {tf} ({len(names)}종목, {len(ret)}봉) =====",
            "-- Part A. lead-lag --",
        ]
        lines += leadlag_summary(names, ret)
        lines += ["-- Part B. cross-lag 상/하위 1종목 롱숏 --"]
        lines += crosslag_ls(names, ret, bpy)
        lines += ["-- Part C. Donchian 돌파 + 트레일링 스탑 --"]
        for k in DONCHIAN_KS:
            for m in STOP_MULTS:
                for fee in FEES:
                    lines.append(donchian_trail(names, arrs, bpy, k, m, fee))
        print("\n".join(lines))

    report = "\n".join(lines)
    (RESULTS_DIR / "report_multires.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")
    notify("다중 해상도 조사 계산 완료 — 분석 보고 이어집니다.", prefix="[nnpipeline]")


if __name__ == "__main__":
    main()
