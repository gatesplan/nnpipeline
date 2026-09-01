"""Receptor → DecayBank → Pyramid head 로 120 봉에서 다음 1~5 봉 누적수익률 예측.

목적은 예측 성능 자체보다 학습 중 λ (반감기) 거동 관찰:
- 데이터에 심은 진짜 반감기 (4, 16, 64) 쪽으로 이동하는가
- 스케일들이 collapse 하는가, 극단 (0/∞) 으로 포화하는가
- oracle R² (도달 가능 상한) 대비 얼마나 뽑는가

실행: python -m experiments.decay_bank_forecast.main
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from nnpipeline.prototype.decay_bank import DecayBank
from nnpipeline.prototype.ohlcv_receptor import OHLCVReceptor
from nnpipeline.prototype.pyramid import Pyramid
from experiments.decay_bank_forecast.synthesize import (
    TRUE_HALF_LIVES,
    make_forecast_dataset,
)

RESULTS_DIR = Path(__file__).parent / "results"

WINDOW = 120
HORIZONS = (1, 2, 3, 4, 5)
INIT_HALF_LIVES = (2.0, 8.0, 32.0, 96.0)
N_TRAIN = 8192
N_VAL = 2048
BATCH = 256
EPOCHS = 150
LR = 5e-3
LR_LAMBDA = 2e-2   # λ logit 은 별도 (높은) lr — gradient 스케일이 작아 묻히는 것 방지
SEED = 7


class ForecastModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.receptor = OHLCVReceptor()
        self.bank = DecayBank(half_lives=INIT_HALF_LIVES, learnable=True)
        feat_dim = self.bank.out_scales * 3
        self.head = Pyramid(feat_dim, len(HORIZONS), depth=3, interlayer=[nn.LeakyReLU()])

    def forward(self, hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        emb = self.receptor(hocl, v)              # (B, n, 3)
        feats = self.bank(emb).flatten(start_dim=-2)  # (B, out_scales*3)
        return self.head(feats)


def r2_per_horizon(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean(dim=0)
    var = target.var(dim=0, unbiased=False)
    return 1.0 - mse / var


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    hocl_tr, v_tr, y_tr, _ = make_forecast_dataset(N_TRAIN, WINDOW, HORIZONS, seed=SEED)
    hocl_va, v_va, y_va, oracle_va = make_forecast_dataset(N_VAL, WINDOW, HORIZONS, seed=SEED + 999)

    # 타깃 표준화 (train 통계) — MSE 스케일 안정화. R² 는 표준화 불변이 아니므로 동일 기준 적용
    mu, sd = y_tr.mean(dim=0, keepdim=True), y_tr.std(dim=0, keepdim=True)
    y_tr_n = (y_tr - mu) / sd
    y_va_n = (y_va - mu) / sd
    oracle_va_n = (oracle_va - mu) / sd

    model = ForecastModel()
    lambda_params = [model.bank.lambda_logit]
    other_params = [p for p in model.parameters() if p is not model.bank.lambda_logit]
    opt = torch.optim.Adam(
        [{"params": other_params, "lr": LR}, {"params": lambda_params, "lr": LR_LAMBDA}]
    )
    loss_fn = nn.MSELoss()

    hl_history = [model.bank.half_lives.detach().clone()]
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(hocl_tr[idx], v_tr[idx]), y_tr_n[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(hocl_va, v_va), y_va_n).item()
        val_losses.append(val_loss)
        hl_history.append(model.bank.half_lives.detach().clone())

    model.eval()
    with torch.no_grad():
        pred_va = model(hocl_va, v_va)
    r2_model = r2_per_horizon(pred_va, y_va_n)
    r2_oracle = r2_per_horizon(oracle_va_n, y_va_n)

    hl = torch.stack(hl_history)  # (EPOCHS+1, K)

    # 리포트
    lines = [
        f"DecayBank forecast: window={WINDOW}, horizons={HORIZONS}, "
        f"init hl={INIT_HALF_LIVES}, true hl={TRUE_HALF_LIVES}",
        f"final val loss={val_losses[-1]:.4f}",
        "",
        f"{'horizon':>8} | {'model R2':>9} | {'oracle R2':>9}",
        "-" * 34,
    ]
    for j, k in enumerate(HORIZONS):
        lines.append(f"{k:>8} | {r2_model[j].item():>9.4f} | {r2_oracle[j].item():>9.4f}")
    lines += [
        "",
        f"half-lives init : {[f'{h:.1f}' for h in hl[0].tolist()]}",
        f"half-lives final: {[f'{h:.1f}' for h in hl[-1].tolist()]}",
    ]
    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "report.txt").write_text(report, encoding="utf-8")

    # λ 거동 플롯
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(hl.shape[1]):
        ax.plot(hl[:, k], label=f"scale {k} (init {INIT_HALF_LIVES[k]:.0f})")
    for tau in TRUE_HALF_LIVES:
        ax.axhline(tau, color="gray", ls="--", lw=0.8)
        ax.text(EPOCHS, tau, f" true {tau:.0f}", va="center", color="gray", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("half-life (candles, log scale)")
    ax.set_title("Learned half-life trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "half_life_trajectories.png", dpi=120)
    plt.close(fig)

    # 손실 곡선
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(val_losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MSE (standardized)")
    ax.set_title("Validation loss")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "val_loss.png", dpi=120)
    plt.close(fig)

    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
