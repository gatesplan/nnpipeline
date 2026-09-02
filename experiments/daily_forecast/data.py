"""일봉 다종목 로딩 + 시간순 분할 + GPU 배치 구성.

- candle_data_manager 에서 종목별 일봉을 timestamp 포함으로 로드
- 전 종목 공통의 달력 기준 시간순 분할 (train 70% / val 15% / test 15%)
- 분할 경계를 걸치는 윈도우는 어느 쪽에도 넣지 않음 (purge)
- 배치는 시작 인덱스에서 GPU 상에서 즉석 구성 (윈도우 겹침 때문에 사전 전개하지 않음)
- 정규화: HOCL 은 윈도우 마지막 종가 차감, V 는 윈도우 내 z-score (receptor 규약)
"""

import numpy as np
import torch

from experiments.receptor_real_data.cdm_data import (
    filter_recent_years,
    load_market_df,
)


def load_stocks(tickers, api, years: int = 10, min_candles: int = 600):
    """반환: list[(ticker, ts (T,), log_hocl (T,4), log_v (T,))]. 실패 종목은 건너뜀."""
    results = []
    seen = set()
    for ticker in tickers:
        if ticker in seen:
            continue
        seen.add(ticker)
        df = load_market_df(api, ticker)
        if df is None:
            continue
        df = filter_recent_years(df, years=years)
        if df is None or len(df) < min_candles:
            continue
        ts = df["timestamp"].values.astype(np.int64)
        log_hocl = np.log(
            np.stack(
                [df["High"].values, df["Open"].values, df["Close"].values, df["Low"].values],
                axis=1,
            )
        )
        log_v = np.log(df["Volume"].values)
        results.append((ticker, ts, log_hocl.astype(np.float32), log_v.astype(np.float32)))
    return results


def build_arrays(stocks, window: int, max_horizon: int,
                 train_frac: float = 0.70, val_frac: float = 0.15,
                 cutoff_ts=None):
    """전 종목 연결 배열 + 분할별 윈도우 시작 인덱스.

    cutoff_ts=(train_end, val_end[, test_end]) 를 주면 절대 timestamp 로 분할
    (walk-forward 용). test_end 이후 윈도우는 어느 분할에도 넣지 않음.
    없으면 전체 달력 구간의 비율 (train_frac / val_frac) 로 분할.

    반환: (hocl_all (Ttot,4), v_all (Ttot,), ts_all (Ttot,) int64,
           starts: dict[str, LongTensor],
           tick_ids: dict[str, LongTensor] — 시작 인덱스별 종목 번호)
    """
    test_end = None
    if cutoff_ts is not None:
        train_end, val_end = cutoff_ts[0], cutoff_ts[1]
        if len(cutoff_ts) > 2:
            test_end = cutoff_ts[2]
    else:
        t_min = min(int(ts[0]) for _, ts, _, _ in stocks)
        t_max = max(int(ts[-1]) for _, ts, _, _ in stocks)
        span = t_max - t_min
        train_end = t_min + train_frac * span
        val_end = t_min + (train_frac + val_frac) * span

    hocl_parts, v_parts, ts_parts = [], [], []
    starts = {"train": [], "val": [], "test": []}
    tick_ids = {"train": [], "val": [], "test": []}
    offset = 0
    for tid, (_, ts, log_hocl, log_v) in enumerate(stocks):
        T = len(ts)
        hocl_parts.append(log_hocl)
        v_parts.append(log_v)
        ts_parts.append(ts)
        n_valid = T - window - max_horizon + 1
        for i in range(n_valid):
            ts_first = ts[i]
            ts_last = ts[i + window + max_horizon - 1]
            if ts_last < train_end:
                key = "train"
            elif ts_first >= train_end and ts_last < val_end:
                key = "val"
            elif ts_first >= val_end and (test_end is None or ts_last < test_end):
                key = "test"
            else:
                continue  # 경계 걸침 또는 test_end 이후 — purge/제외
            starts[key].append(offset + i)
            tick_ids[key].append(tid)
        offset += T

    hocl_all = torch.from_numpy(np.concatenate(hocl_parts, axis=0))
    v_all = torch.from_numpy(np.concatenate(v_parts, axis=0))
    ts_all = torch.from_numpy(np.concatenate(ts_parts, axis=0))
    starts = {k: torch.tensor(v, dtype=torch.long) for k, v in starts.items()}
    tick_ids = {k: torch.tensor(v, dtype=torch.long) for k, v in tick_ids.items()}
    return hocl_all, v_all, ts_all, starts, tick_ids


def make_inputs(hocl_all, v_all, batch_starts, window: int):
    """시작 인덱스 배치 → 정규화된 (hocl (B,W,4), v (B,W,1))."""
    idx = batch_starts.unsqueeze(-1) + torch.arange(window, device=batch_starts.device)
    hocl = hocl_all[idx]                                    # (B, W, 4)
    v = v_all[idx]                                          # (B, W)
    anchor = hocl[:, -1:, 2:3]                              # 윈도우 마지막 종가
    hocl = hocl - anchor
    v = (v - v.mean(dim=1, keepdim=True)) / (v.std(dim=1, keepdim=True) + 1e-8)
    return hocl, v.unsqueeze(-1)


def make_targets(hocl_all, batch_starts, window: int, horizons):
    """미래 k 봉 누적 log 수익률. 반환 (B, len(horizons))."""
    logc = hocl_all[:, 2]
    end = batch_starts + window - 1
    return torch.stack([logc[end + k] - logc[end] for k in horizons], dim=-1)
