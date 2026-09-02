"""일봉 예측 비교 실험: 120봉 → 다음 1~5봉 누적수익률 (미국 주식 ~200종목, 10년).

비교 모델 (동일 데이터·동일 분할, seed 1개):
  A zero    항상 0 예측 — R² 기준점
  B linear  과거 수익률·변동성·거래량 특징 → ridge 회귀 (닫힌 형태)
  C flatten receptor → flatten(120×3) → Pyramid — 기존 방식 대조군
  D bank    receptor → DecayBank(K=4) → Pyramid — 본 파이프라인
  E dual    D + robust_clip=3, robust_dual — 이중 상태 변형

평가: test 구간 (최근 15%) out-of-sample R², horizon 별.
실행: python -m experiments.daily_forecast.main
"""

from pathlib import Path

from loguru import logger

logger.remove()  # candle_data_manager 로그 억제

import torch
from torch import nn

from candle_data_manager import CandleDataAPI

from nnpipeline import OHLCVReceptor, Pyramid
from experiments.decay_bank_forecast.main_decompose import (
    HL_K4,
    ReceptorBankForecaster,
)
from experiments.receptor_real_data.cdm_data import LARGE_CAP, SMALL_CAP
from experiments.daily_forecast.data import (
    build_arrays,
    load_stocks,
    make_inputs,
    make_targets,
)

RESULTS_DIR = Path(__file__).parent / "results"

WINDOW = 120
HORIZONS = (1, 2, 3, 4, 5)
YEARS = 10
BATCH = 4096
EPOCHS = 40
LR = 5e-3
LR_LAMBDA = 2e-2
SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RETURN_LAGS = (1, 2, 4, 8, 16, 32, 64, 119)


class FlattenForecaster(nn.Module):
    """기존 방식: receptor 임베딩을 시간축 그대로 펼쳐 MLP 에 입력."""

    def __init__(self):
        super().__init__()
        self.receptor = OHLCVReceptor()
        self.head = Pyramid(WINDOW * 3, len(HORIZONS), depth=3, interlayer=[nn.LeakyReLU()])

    def forward(self, hocl, v):
        z = self.receptor(hocl, v)
        return self.head(z.flatten(start_dim=1))


def r2_per_horizon(pred, target):
    mse = ((pred - target) ** 2).mean(dim=0)
    return 1.0 - mse / target.var(dim=0, unbiased=False)


def eval_chunked(model, hocl_all, v_all, starts, y_n):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(starts), 8192):
            h, v = make_inputs(hocl_all, v_all, starts[i:i + 8192], WINDOW)
            preds.append(model(h, v))
    return r2_per_horizon(torch.cat(preds), y_n)


def train_model(model, hocl_all, v_all, starts, y_tr_n, y_va_n, y_te_n):
    model = model.to(DEVICE)
    bank = getattr(model, "bank", None)
    if bank is not None:
        lam = [bank.lambda_logit]
        others = [p for p in model.parameters() if p is not bank.lambda_logit]
        opt = torch.optim.Adam([{"params": others, "lr": LR}, {"params": lam, "lr": LR_LAMBDA}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n_tr = len(starts["train"])
    best_loss, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH):
            sel = perm[i:i + BATCH]
            h, v = make_inputs(hocl_all, v_all, starts["train"][sel], WINDOW)
            opt.zero_grad()
            loss = loss_fn(model(h, v), y_tr_n[sel])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, len(starts["val"]), 8192):
                h, v = make_inputs(hocl_all, v_all, starts["val"][i:i + 8192], WINDOW)
                val_preds.append(model(h, v))
            val_loss = loss_fn(torch.cat(val_preds), y_va_n).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    r2 = eval_chunked(model, hocl_all, v_all, starts["test"], y_te_n)
    hls = bank.half_lives.detach().tolist() if bank is not None else None
    return r2, hls, model


def linear_features(hocl_all, v_all, batch_starts):
    """ridge baseline 특징: 과거 누적수익률 (여러 lag) + 20봉 실현변동성 + 최근 거래량 z."""
    logc = hocl_all[:, 2]
    end = batch_starts + WINDOW - 1
    feats = [logc[end] - logc[end - lag] for lag in RETURN_LAGS]

    idx20 = end.unsqueeze(-1) + torch.arange(-19, 1, device=end.device)
    ret20 = logc[idx20] - logc[idx20 - 1]
    feats.append(ret20.std(dim=1))

    _, v = make_inputs(hocl_all, v_all, batch_starts, WINDOW)
    feats.append(v[:, -1, 0])
    return torch.stack(feats, dim=-1)


def ridge_baseline(hocl_all, v_all, starts, y_tr_n, y_te_n, alpha: float = 1.0):
    x_tr = linear_features(hocl_all, v_all, starts["train"]).double()
    x_te = linear_features(hocl_all, v_all, starts["test"]).double()
    mu, sd = x_tr.mean(dim=0), x_tr.std(dim=0) + 1e-12
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    x_tr = torch.cat([x_tr, torch.ones(len(x_tr), 1, device=x_tr.device, dtype=x_tr.dtype)], dim=-1)
    x_te = torch.cat([x_te, torch.ones(len(x_te), 1, device=x_te.device, dtype=x_te.dtype)], dim=-1)
    gram = x_tr.T @ x_tr + alpha * torch.eye(x_tr.shape[1], device=x_tr.device, dtype=x_tr.dtype)
    w = torch.linalg.solve(gram, x_tr.T @ y_tr_n.double())
    return r2_per_horizon((x_te @ w).float(), y_te_n)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    api = CandleDataAPI()
    stocks = load_stocks(LARGE_CAP + SMALL_CAP, api, years=YEARS)
    hocl_all, v_all, _, starts, _ = build_arrays(stocks, WINDOW, max(HORIZONS))
    hocl_all, v_all = hocl_all.to(DEVICE), v_all.to(DEVICE)
    starts = {k: v.to(DEVICE) for k, v in starts.items()}

    ys = {k: make_targets(hocl_all, starts[k], WINDOW, HORIZONS) for k in starts}
    mu, sd = ys["train"].mean(dim=0, keepdim=True), ys["train"].std(dim=0, keepdim=True)
    y_n = {k: (v - mu) / sd for k, v in ys.items()}

    lines = [
        f"일봉 예측 비교: {len(stocks)} 종목, window={WINDOW}, horizons={HORIZONS}, "
        f"epochs={EPOCHS}, device={DEVICE}",
        f"windows: train={len(starts['train'])}, val={len(starts['val'])}, test={len(starts['test'])}",
        "",
        f"{'model':>9} | " + " | ".join(f"R2@{k}" for k in HORIZONS) + " | learned hl",
        "-" * 90,
    ]

    def fmt(name, r2, hls=None):
        row = f"{name:>9} | " + " | ".join(f"{v:+.4f}" for v in r2.tolist())
        if hls:
            row += " | [" + ", ".join(f"{h:.1f}" for h in hls) + "]"
        return row

    # A: zero
    r2_zero = r2_per_horizon(torch.zeros_like(y_n["test"]), y_n["test"])
    lines.append(fmt("A_zero", r2_zero))
    print(lines[-1])

    # B: ridge
    r2_lin = ridge_baseline(hocl_all, v_all, starts, y_n["train"], y_n["test"])
    lines.append(fmt("B_linear", r2_lin))
    print(lines[-1])

    # C / D / E
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

    # E 의 dual 블록 사용도: head 첫 Linear 의 블록별 입력 가중치 norm
    if e_model is not None:
        w = e_model.head[0].weight.detach()                  # (out, 42)
        half = w.shape[1] // 2
        raw_n, rob_n = w[:, :half].norm().item(), w[:, half:].norm().item()
        lines.append("")
        lines.append(
            f"E_dual head 입력 가중치 norm — 미적용 블록: {raw_n:.3f}, 적용 블록: {rob_n:.3f}"
        )

    report = "\n".join(lines)
    (RESULTS_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
