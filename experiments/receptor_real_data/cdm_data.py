"""candle_data_manager 기반 multi-stock 데이터 로딩.

대형주 100 (curated) + 잡주 100 (curated + sampling).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from candle_data_manager import CandleDataAPI

from .data import WindowDataset, split_train_val_test


# 대형주 100 (NASDAQ + 일부 NYSE 명단, 잘 알려진 mega/large cap)
LARGE_CAP = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    # Large tech / growth
    "ADBE", "CRM", "AMD", "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU", "INTU",
    "NOW", "PANW", "SNPS", "CDNS", "MRVL", "KLAC", "LRCX", "ASML", "ADI", "MCHP",
    # Finance / consumer
    "COST", "PEP", "SBUX", "BKNG", "AMGN", "GILD", "VRTX", "REGN", "ISRG", "MDLZ",
    # Established mid-large
    "NFLX", "TMUS", "CMCSA", "HON", "ADP", "PYPL", "FISV", "AMAT", "MNST", "KLAC",
    # Pharma / healthcare large
    "BIIB", "ILMN", "MRNA", "DXCM", "IDXX", "ZTS", "PFE", "JNJ", "UNH", "LLY",
    # Industrial / consumer
    "UPS", "FDX", "DLTR", "ULTA", "EXC", "AEP", "XEL", "PCAR", "ROST", "CTAS",
    # Semi / other tech
    "WDAY", "FTNT", "MSTR", "TEAM", "DDOG", "ZS", "CRWD", "OKTA", "MDB", "NET",
    # Add'l well-known
    "ABNB", "UBER", "LYFT", "DASH", "ROKU", "SHOP", "SQ", "COIN", "PINS", "SNAP",
    # Established mid
    "EBAY", "EA", "ATVI", "TTD", "MELI", "PDD", "JD", "BABA", "NTES", "LULU",
    "EXPE", "MAR", "ORLY", "WBA", "WBD", "PARA", "FOX", "NWS", "DIS", "NKE",
]

# 잡주 100 (high-vol, small/mid-cap, speculative, biotech, meme 등)
SMALL_CAP = [
    # Meme / retail favorites
    "MVIS", "AMC", "GME", "BBBYQ", "CLOV", "WISH", "WKHS", "RIDE", "NKLA", "HYZN",
    # SPAC / EV / clean energy speculative
    "MARA", "RIOT", "HUT", "BTBT", "BITF", "CIFR", "CLSK", "ARQQ", "QUBT", "RGTI",
    # Cannabis
    "TLRY", "CGC", "ACB", "SNDL", "HEXO", "CRON", "OGI", "VFF", "GTBIF", "TCNNF",
    # Biotech / pharma small-cap
    "OCGN", "INO", "VXRT", "ATOS", "SAVA", "ANVS", "ACET", "TGTX", "PRPL", "INFI",
    # Fintech / crypto exposure
    "SOFI", "AFRM", "UPST", "OPEN", "HOOD", "MARA", "BKKT", "EQOS", "OSTK", "FUBO",
    # Penny / micro speculation
    "IDEX", "SOS", "EBON", "EBET", "BTCS", "GREE", "ANY", "EHTH", "CYCC", "ELYM",
    # Electric / energy speculative
    "PLUG", "FCEL", "BLNK", "CHPT", "EVGO", "QS", "FFIE", "MULN", "GOEV", "ARVL",
    # Small biotech
    "AMRN", "JAGX", "ENZN", "RIBT", "EYEN", "CYRX", "DRRX", "AVCT", "HCWB", "GANX",
    # Other speculative
    "BBIG", "PROG", "ATER", "GREE", "SPRT", "IRNT", "OPAD", "MMAT", "OPK", "CCJ",
    # Add more variety
    "TIRX", "HOFV", "MULN", "CXAI", "GFAI", "SNTI", "ENVB", "ABVC", "VVOS", "GRRR",
]


def load_market_df(api: CandleDataAPI, ticker: str) -> pd.DataFrame | None:
    """단일 종목 데이터 로드. 실패 시 None."""
    try:
        markets = api.load(
            archetype="STOCK", exchange="NASDAQ", tradetype="SPOT",
            base=ticker, quote="USD", timeframe="1d",
        )
        if not markets:
            return None
        df = markets[0].candles
        # 컬럼 정리
        df = df[["timestamp", "high", "open", "close", "low", "volume"]].copy()
        df.columns = ["timestamp", "High", "Open", "Close", "Low", "Volume"]
        df = df[df["Volume"] > 0].copy()
        return df
    except Exception:
        return None


def filter_recent_years(df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """timestamp 기준 최근 N년만."""
    if df is None or len(df) == 0:
        return df
    latest_ts = df["timestamp"].max()
    cutoff = latest_ts - years * 365 * 86400
    return df[df["timestamp"] >= cutoff].copy()


def df_to_log_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """OHLCV → (log_hocl: (T, 4), log_v: (T,)). 채널 순서 [H, O, C, L]."""
    log_h = np.log(df["High"].values)
    log_o = np.log(df["Open"].values)
    log_c = np.log(df["Close"].values)
    log_l = np.log(df["Low"].values)
    log_v = np.log(df["Volume"].values)
    log_hocl = np.stack([log_h, log_o, log_c, log_l], axis=1)
    return log_hocl, log_v


def load_multi_stock(
    tickers: List[str],
    api: CandleDataAPI,
    years: int = 5,
    min_candles: int = 200,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """다중 종목 로드. 충분한 데이터(min_candles)를 가진 종목만 반환."""
    results = []
    for ticker in tickers:
        df = load_market_df(api, ticker)
        if df is None:
            continue
        df = filter_recent_years(df, years=years)
        if len(df) < min_candles:
            continue
        log_hocl, log_v = df_to_log_arrays(df)
        results.append((ticker, log_hocl, log_v))
    return results


def build_multi_stock_dataset(
    stocks: List[Tuple[str, np.ndarray, np.ndarray]],
    window: int = 60,
    forecast_step: int = 1,
    split: str = "train",
) -> ConcatDataset:
    """종목별 WindowDataset을 시간 분할 후 ConcatDataset으로 합침."""
    subsets = []
    for ticker, log_hocl, log_v in stocks:
        full_ds = WindowDataset(log_hocl, log_v, window=window, forecast_step=forecast_step)
        train_idx, val_idx, test_idx = split_train_val_test(len(full_ds))
        if split == "train":
            idx = list(train_idx)
        elif split == "val":
            idx = list(val_idx)
        elif split == "test":
            idx = list(test_idx)
        else:
            raise ValueError(split)
        from torch.utils.data import Subset
        subsets.append(Subset(full_ds, idx))
    return ConcatDataset(subsets)
