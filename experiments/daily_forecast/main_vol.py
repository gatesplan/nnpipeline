"""일봉 변동성 예측 비교: 120봉 → 다음 1~5봉 log 실현변동성.

방향 수익률 (main.py) 과 달리 변동성은 실데이터에서 예측 가능성이 실증된 타깃.
타깃: y_k = log( sqrt( Σ_{i=1..k} r_{t+i}² ) + eps ), r 은 일별 log 수익률.

비교 모델 (main.py 와 동일 분할·프로토콜, seed 1개):
  A zero    항상 0 예측 (표준화 후) — 기준점
  B linear  과거 실현변동성 (5/20/60/119봉) + 수익률 lag + 거래량 → ridge (HAR 회귀에 해당)
  C flatten receptor → flatten → Pyramid
  D bank    receptor → DecayBank(K=4) → Pyramid
  E dual    D + robust_clip=3, robust_dual

실행: python -m experiments.daily_forecast.main_vol
"""

from pathlib import Path

from loguru import logger

logger.remove()

import torch
from torch import nn

from candle_data_manager import CandleDataAPI

from experiments.decay_bank_forecast.main_decompose import HL_K4, ReceptorBankForecaster
from experiments.receptor_real_data.cdm_data import LARGE_CAP, SMALL_CAP
from experiments.daily_forecast.data import build_arrays, load_stocks, make_inputs
from experiments.daily_forecast.main import (
    DEVICE,
    EPOCHS,
    HORIZONS,
    SEED,
    WINDOW,
    YEARS,
    FlattenForecaster,
    r2_per_horizon,
    train_model,
)

RESULTS_DIR = Path(__file__).parent / "results"

EPS = 1e-12
RV_LOOKBACKS = (5, 20, 60, 119)
RET_LAGS = (1, 5, 20)


def make_vol_targets(hocl_all, batch_starts, window: int, horizons):
    """미래 k 봉 log 실현변동성. 반환 (B, len(horizons))."""
    logc = hocl_all[:, 2]
    end = batch_starts + window - 1
    idx = end.unsqueeze(-1) + torch.arange(1, max(horizons) + 1, device=end.device)
    r = logc[idx] - logc[idx - 1]                            # (B, max_h) 미래 일별 수익률
    cum_var = torch.cumsum(r ** 2, dim=1)
    y = 0.5 * torch.log(cum_var + EPS)                       # log sqrt(Σ r²)
    return torch.stack([y[:, k - 1] for k in horizons], dim=-1)


def vol_features(hocl_all, v_all, batch_starts, window: int):
    """ridge baseline 특징: 과거 log 실현변동성 (여러 구간) + 수익률 lag + 거래량 z."""
    logc = hocl_all[:, 2]
    end = batch_starts + window - 1
    feats = []
    for lb in RV_LOOKBACKS:
        idx = end.unsqueeze(-1) + torch.arange(-lb + 1, 1, device=end.device)
        r = logc[idx] - logc[idx - 1]
        feats.append(0.5 * torch.log((r ** 2).sum(dim=1) + EPS))
    for lag in RET_LAGS:
        feats.append(logc[end] - logc[end - lag])
    _, v = make_inputs(hocl_all, v_all, batch_starts, window)
    feats.append(v[:, -1, 0])
    return torch.stack(feats, dim=-1)


def ridge(x_tr, y_tr, x_te, alpha: float = 1.0):
    x_tr, x_te = x_tr.double(), x_te.double()
    mu, sd = x_tr.mean(dim=0), x_tr.std(dim=0) + 1e-12
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    ones_tr = torch.ones(len(x_tr), 1, device=x_tr.device, dtype=x_tr.dtype)
    ones_te = torch.ones(len(x_te), 1, device=x_te.device, dtype=x_te.dtype)
    x_tr = torch.cat([x_tr, ones_tr], dim=-1)
    x_te = torch.cat([x_te, ones_te], dim=-1)
    gram = x_tr.T @ x_tr + alpha * torch.eye(x_tr.shape[1], device=x_tr.device, dtype=x_tr.dtype)
    w = torch.linalg.solve(gram, x_tr.T @ y_tr.double())
    return (x_te @ w).float()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    api = CandleDataAPI()
    stocks = load_stocks(LARGE_CAP + SMALL_CAP, api, years=YEARS)
    hocl_all, v_all, _, starts, _ = build_arrays(stocks, WINDOW, max(HORIZONS))
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}

    ys = {k: make_vol_targets(hocl_all, starts[k], WINDOW, HORIZONS) for k in starts}
    mu, sd = ys["train"].mean(dim=0, keepdim=True), ys["train"].std(dim=0, keepdim=True)
    y_n = {k: (v - mu) / sd for k, v in ys.items()}

    lines = [
        f"일봉 변동성 예측: {len(stocks)} 종목, window={WINDOW}, horizons={HORIZONS}, "
        f"epochs={EPOCHS}, device={DEVICE}",
        f"windows: train={len(starts['train'])}, val={len(starts['val'])}, test={len(starts['test'])}",
        "",
        f"{'model':>9} | " + " | ".join(f"R2@{k}" for k in HORIZONS) + " | learned hl",
        "-" * 95,
    ]

    def fmt(name, r2, hls=None):
        row = f"{name:>9} | " + " | ".join(f"{v:+.4f}" for v in r2.tolist())
        if hls:
            row += " | [" + ", ".join(f"{h:.1f}" for h in hls) + "]"
        return row

    r2_zero = r2_per_horizon(torch.zeros_like(y_n["test"]), y_n["test"])
    lines.append(fmt("A_zero", r2_zero))
    print(lines[-1])

    x_tr = vol_features(hocl_all, v_all, starts["train"], WINDOW)
    x_te = vol_features(hocl_all, v_all, starts["test"], WINDOW)
    r2_lin = r2_per_horizon(ridge(x_tr, y_n["train"], x_te), y_n["test"])
    lines.append(fmt("B_linear", r2_lin))
    print(lines[-1])

    configs = [
        ("C_flatten", lambda: FlattenForecaster()),
        ("D_bank", lambda: ReceptorBankForecaster(HL_K4, "pyramid")),
        ("E_dual", lambda: ReceptorBankForecaster(HL_K4, "pyramid", robust_clip=3.0, robust_dual=True)),
    ]
    e_model = None
    for name, factory in configs:
        torch.manual_seed(SEED)
        r2, hls, trained = train_model(
            factory(), hocl_all, v_all, starts, y_n["train"], y_n["val"], y_n["test"]
        )
        lines.append(fmt(name, r2, hls))
        print(lines[-1])
        if name == "E_dual":
            e_model = trained

    if e_model is not None:
        w = e_model.head[0].weight.detach()
        half = w.shape[1] // 2
        lines.append("")
        lines.append(
            f"E_dual head 입력 가중치 norm — 미적용 블록: {w[:, :half].norm().item():.3f}, "
            f"적용 블록: {w[:, half:].norm().item():.3f}"
        )

    report = "\n".join(lines)
    (RESULTS_DIR / "report_vol.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
