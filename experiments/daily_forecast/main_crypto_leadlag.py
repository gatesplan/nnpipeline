"""Binance 4h 봉 특성 탐색: 종목 간 선행-후행 (lead-lag) 구조 + 4h 변동성 예측.

Part A. lead-lag 상관 지도
  전 종목 공통 구간에서 corr(r_i[t-lag], r_j[t]), lag 1~6 (4h~24h).
  전·후반 분할 재현성 필터 (양쪽 모두 |corr| > 2/sqrt(T/2), 같은 부호) 통과 쌍만 보고.

Part B. 예측 가능성: own-lag vs cross-lag
  각 코인의 다음 4h 수익률을 (a) 자기 과거 6개 lag (b) 전 코인 과거 6개 lag 로 ridge 예측.
  (b) - (a) 가 양수면 타 종목 정보가 전파된다는 뜻. 상위/하위 3종목 롱숏 (5bp) 도 평가.

Part C. 4h 변동성 예측 (표본 확대 검증)
  일봉 크립토에서 bank 가 선형에 소폭 뒤졌던 것이 표본 (3.5만) 문제인지, 4h (약 20만 윈도우) 로 재검.

실행: python -m experiments.daily_forecast.main_crypto_leadlag
"""

from pathlib import Path

from loguru import logger

logger.remove()

import math

import numpy as np
import torch

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4, ReceptorBankForecaster
from experiments.daily_forecast.data import build_arrays
from experiments.daily_forecast.main import DEVICE, SEED, WINDOW, r2_per_horizon, train_model
from experiments.daily_forecast.main_vol import make_vol_targets, ridge, vol_features
from experiments.daily_forecast.main import HORIZONS

RESULTS_DIR = Path(__file__).parent / "results"

MAX_LAG = 6
OWN_LAGS = 6
FEE = 0.0005
BARS_PER_YEAR = 6 * 365


def load_4h(api, min_candles: int = 3000):
    markets = api.load(
        archetype="CRYPTO", exchange="BINANCE", tradetype="FUTURES",
        quote="USDT", timeframe="4h",
    )
    out = []
    for m in markets:
        df = m.candles
        df = df[["timestamp", "high", "open", "close", "low", "volume"]].copy()
        df = df[df["volume"] > 0]
        if len(df) < min_candles:
            continue
        out.append((m.symbol.base, df))
    return out


def build_panel(coins):
    """공통 timestamp 구간의 log 종가 행렬. 반환: (names, ts (T,), logc (T, N))."""
    ts_sets = [set(df["timestamp"].values.tolist()) for _, df in coins]
    common = sorted(set.intersection(*ts_sets))
    names = [n for n, _ in coins]
    logc = np.zeros((len(common), len(coins)), dtype=np.float64)
    for j, (_, df) in enumerate(coins):
        m = dict(zip(df["timestamp"].values.tolist(), np.log(df["close"].values)))
        logc[:, j] = [m[t] for t in common]
    return names, np.array(common, dtype=np.int64), logc


def leadlag_map(names, ret):
    """전·후반 재현되는 lead-lag 쌍. ret (T, N). 반환: 보고 문자열 리스트."""
    T, N = ret.shape
    half = T // 2
    thr = 2.0 / math.sqrt(half)
    out = []
    found = []
    for lag in range(1, MAX_LAG + 1):
        a_lead = ret[:-lag]        # r_i[t-lag]
        a_fol = ret[lag:]          # r_j[t]
        for h, (x, y) in enumerate([(a_lead[:half], a_fol[:half]), (a_lead[half:], a_fol[half:])]):
            xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
            ys = (y - y.mean(0)) / (y.std(0) + 1e-12)
            c = xs.T @ ys / len(xs)                          # (N, N): i 선행 → j 추종
            if h == 0:
                c1 = c
            else:
                c2 = c
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if abs(c1[i, j]) > thr and abs(c2[i, j]) > thr and c1[i, j] * c2[i, j] > 0:
                    found.append((lag, names[i], names[j], (c1[i, j] + c2[i, j]) / 2))
    found.sort(key=lambda x: -abs(x[3]))
    out.append(f"재현성 필터 (전·후반 각 |corr|>{thr:.3f}, 동일 부호) 통과: {len(found)}쌍")
    for lag, ni, nj, c in found[:15]:
        out.append(f"  lag {lag} ({lag*4}h): {ni} -> {nj}  corr {c:+.4f}")
    # 시장 리더십 요약: lag-1 에서 행 (선행자) 평균
    x = ret[:-1]
    y = ret[1:]
    xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
    ys = (y - y.mean(0)) / (y.std(0) + 1e-12)
    c = xs.T @ ys / len(xs)
    np.fill_diagonal(c, np.nan)
    lead_score = np.nanmean(c, axis=1)
    order = np.argsort(-lead_score)
    out.append("lag-1 선행 점수 (자신 제외 전 종목에 대한 평균 상관) 상위:")
    out.append("  " + ", ".join(f"{names[i]} {lead_score[i]:+.4f}" for i in order[:6]))
    out.append("하위 (후행 성향):")
    out.append("  " + ", ".join(f"{names[i]} {lead_score[i]:+.4f}" for i in order[-3:]))
    return out


def predictability(names, ret):
    """own-lag vs cross-lag ridge 의 OOS R² + 상/하위 3종목 롱숏."""
    T, N = ret.shape
    t0 = MAX_LAG
    X_own = {j: np.stack([ret[t0 - l:T - l, j] for l in range(1, OWN_LAGS + 1)], axis=1) for j in range(N)}
    X_cross = np.concatenate(
        [np.stack([ret[t0 - l:T - l, j] for l in range(1, OWN_LAGS + 1)], axis=1) for j in range(N)],
        axis=1,
    )                                                        # (T-t0, N*6)
    Y = ret[t0:]                                             # (T-t0, N)
    n = len(Y)
    n_tr, n_va = int(n * 0.7), int(n * 0.85)

    rows = []
    r2_own_all, r2_cross_all = [], []
    preds_test = np.zeros((n - n_va, N))
    for j in range(N):
        y = torch.tensor(Y[:, j:j + 1])
        sd = y[:n_tr].std() + 1e-12
        y = y / sd
        xo = torch.tensor(X_own[j])
        xc = torch.tensor(X_cross)
        po = ridge(xo[:n_tr], y[:n_tr], xo[n_va:], alpha=10.0)
        pc = ridge(xc[:n_tr], y[:n_tr], xc[n_va:], alpha=10.0)
        r2o = r2_per_horizon(po, y[n_va:])[0].item()
        r2c = r2_per_horizon(pc, y[n_va:])[0].item()
        r2_own_all.append(r2o)
        r2_cross_all.append(r2c)
        preds_test[:, j] = pc[:, 0].numpy()
        rows.append((names[j], r2o, r2c))

    out = [f"{'coin':>6} | {'own R2':>8} | {'cross R2':>8} | 개선"]
    for nm, a, b in sorted(rows, key=lambda r: -(r[2] - r[1])):
        out.append(f"{nm:>6} | {a:>+8.4f} | {b:>+8.4f} | {'+' if b > a else ''}")
    n_better = sum(b > a for _, a, b in rows)
    out.append(
        f"평균: own {np.mean(r2_own_all):+.4f}, cross {np.mean(r2_cross_all):+.4f} "
        f"(cross 우위 {n_better}/{N} 종목)"
    )

    # 상/하위 3종목 롱숏 (test 구간, 매 4h 리밸런스)
    Yt = Y[n_va:]
    prev = None
    ls, tos = [], []
    for t in range(len(Yt)):
        order = np.argsort(-preds_test[t])
        w = np.zeros(N)
        w[order[:3]] = 1 / 3
        w[order[-3:]] = -1 / 3
        ls.append(float(w @ Yt[t]))
        tos.append(0.0 if prev is None else float(np.abs(w - prev).sum() / 2))
        prev = w
    ls, tos = np.array(ls), np.array(tos)
    ann_ret = ls.mean() * BARS_PER_YEAR
    ann_vol = ls.std() * math.sqrt(BARS_PER_YEAR)
    net = ls - tos * FEE
    out.append(
        f"cross 신호 상/하위 3종목 롱숏: 총 {ann_ret:+.1%}/년, Sharpe {ann_ret/ann_vol:.2f}, "
        f"턴오버 {tos.mean():.3f}/봉, net {net.mean()*BARS_PER_YEAR:+.1%}/년 "
        f"(netShp {net.mean()*BARS_PER_YEAR/(net.std()*math.sqrt(BARS_PER_YEAR)+1e-12):.2f})"
    )
    return out


def vol_4h(coins):
    """4h 변동성 예측: A/B/D 비교 (표본 확대 검증)."""
    stocks_fmt = []
    for name, df in coins:
        ts = df["timestamp"].values.astype(np.int64)
        log_hocl = np.log(
            np.stack([df["high"].values, df["open"].values, df["close"].values, df["low"].values], axis=1)
        ).astype(np.float32)
        log_v = np.log(df["volume"].values).astype(np.float32)
        stocks_fmt.append((name, ts, log_hocl, log_v))

    hocl_all, v_all, _, starts, _ = build_arrays(stocks_fmt, WINDOW, max(HORIZONS))
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}

    ys = {k: make_vol_targets(hocl_all, starts[k], WINDOW, HORIZONS) for k in starts}
    mu, sd = ys["train"].mean(dim=0, keepdim=True), ys["train"].std(dim=0, keepdim=True)
    y_n = {k: (v - mu) / sd for k, v in ys.items()}

    out = [f"windows: " + ", ".join(f"{k}={len(starts[k])}" for k in starts)]
    out.append(
        "A_zero    | " + " | ".join(
            f"{v:+.4f}" for v in r2_per_horizon(torch.zeros_like(y_n['test']), y_n['test']).tolist()
        )
    )
    xv_tr = vol_features(hocl_all, v_all, starts["train"], WINDOW)
    xv_te = vol_features(hocl_all, v_all, starts["test"], WINDOW)
    out.append(
        "B_linear  | " + " | ".join(
            f"{v:+.4f}" for v in r2_per_horizon(ridge(xv_tr, y_n['train'], xv_te), y_n['test']).tolist()
        )
    )
    torch.manual_seed(SEED)
    r2, hls, _ = train_model(
        ReceptorBankForecaster(HL_K4, "pyramid"),
        hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"],
    )
    out.append(
        "D_bank    | " + " | ".join(f"{v:+.4f}" for v in r2.tolist())
        + " | [" + ", ".join(f"{h:.1f}" for h in hls) + "]"
    )
    return out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    api = CandleDataAPI()
    coins = load_4h(api)
    names, ts, logc = build_panel(coins)
    ret = np.diff(logc, axis=0)
    from datetime import datetime, timezone
    t0 = datetime.fromtimestamp(int(ts[0]), tz=timezone.utc).date()
    t1 = datetime.fromtimestamp(int(ts[-1]), tz=timezone.utc).date()

    lines = [
        f"Binance 4h lead-lag 탐색: {len(names)} 종목, 공통 구간 {t0} ~ {t1}, {len(ret)} 봉",
        "",
        "== Part A. lead-lag 상관 지도 (lag 1~6 = 4h~24h) ==",
    ]
    lines += leadlag_map(names, ret)
    print("\n".join(lines))

    lines += ["", "== Part B. 예측 가능성: own-lag vs cross-lag ridge (다음 4h 수익률) =="]
    lines += predictability(names, ret)
    print("\n".join(lines[-8:]))

    lines += ["", "== Part C. 4h 변동성 예측 (1~5봉 = 4h~20h, A/B/D) =="]
    lines += vol_4h(coins)
    print("\n".join(lines[-4:]))

    report = "\n".join(lines)
    (RESULTS_DIR / "report_crypto_4h.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
