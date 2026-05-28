"""yfinance 데이터 + per-window 정규화.

전처리: 전체 데이터 log 변환만.
학습 시: 윈도우 마다 ref close 차감 (HOCL), rolling z-score (V).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import torch

CACHE_DIR = Path(".cache/yfinance")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(ticker: str, period: str = "5y", force: bool = False) -> pd.DataFrame:
    """yfinance에서 OHLCV 다운로드. .cache/yfinance/에 csv 캐싱."""
    cache_file = CACHE_DIR / f"{ticker}_{period}.csv"
    if cache_file.exists() and not force:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, auto_adjust=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df.dropna()
        # Volume = 0인 행 제거 (log 적용 위해)
        df = df[df["Volume"] > 0]
        df.to_csv(cache_file)
    return df


def to_log_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """OHLCV → (log_hocl: (T, 4), log_v: (T,)).

    HOCL 채널 순서: [H, O, C, L].
    """
    log_h = np.log(df["High"].values)
    log_o = np.log(df["Open"].values)
    log_c = np.log(df["Close"].values)
    log_l = np.log(df["Low"].values)
    log_v = np.log(df["Volume"].values)
    log_hocl = np.stack([log_h, log_o, log_c, log_l], axis=1)  # (T, 4)
    return log_hocl, log_v


@dataclass
class WindowDataset:
    """Sliding window dataset. 각 윈도우는 길이 N의 (HOCL_norm, V_norm) 쌍.

    forecast 모드에서는 t+1 캔들도 함께 반환 (target).
    """

    log_hocl: np.ndarray   # (T, 4)
    log_v: np.ndarray      # (T,)
    window: int            # N
    forecast_step: int = 0  # 0=autoencoder (target=current window last), 1=다음 캔들 예측

    def __len__(self) -> int:
        # forecast: 마지막 윈도우는 t+forecast_step까지 필요
        return len(self.log_hocl) - self.window - self.forecast_step + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 윈도우 슬라이스
        win_hocl = self.log_hocl[idx : idx + self.window]      # (N, 4)
        win_v = self.log_v[idx : idx + self.window]            # (N,)

        # HOCL 정규화: 윈도우 마지막 close (idx 2 of HOCL [H,O,C,L]) 차감
        ref_close = win_hocl[-1, 2]
        hocl_norm = win_hocl - ref_close                       # (N, 4)

        # V 정규화: rolling z-score (윈도우 자체 통계 사용)
        v_mean = win_v.mean()
        v_std = win_v.std() + 1e-8
        v_norm = (win_v - v_mean) / v_std                      # (N,)
        v_norm = v_norm[:, None]                                # (N, 1)

        # Target — forecast 모드면 t+forecast_step 캔들
        if self.forecast_step > 0:
            tgt_idx = idx + self.window + self.forecast_step - 1
            tgt_hocl = self.log_hocl[tgt_idx] - ref_close          # (4,) 같은 ref로 정규화
            tgt_v = (self.log_v[tgt_idx] - v_mean) / v_std         # scalar
            tgt_hoclv = np.concatenate([tgt_hocl, [tgt_v]])        # (5,)
            tgt = torch.from_numpy(tgt_hoclv).float()
        else:
            # autoencoder: 입력 자체가 target
            tgt = torch.from_numpy(
                np.concatenate([hocl_norm, v_norm], axis=1)
            ).float()  # (N, 5)

        hocl_tensor = torch.from_numpy(hocl_norm).float()         # (N, 4)
        v_tensor = torch.from_numpy(v_norm).float()                # (N, 1)
        ref = torch.tensor([ref_close, v_mean, v_std]).float()    # 정규화 메타
        return hocl_tensor, v_tensor, tgt, ref


def split_train_val_test(
    n_total: int, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[range, range, range]:
    """시계열 분할 — 시간 순서 유지."""
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))
    return range(0, train_end), range(train_end, val_end), range(val_end, n_total)
