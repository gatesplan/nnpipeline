"""oracle 대비 손실 30% 의 출처 분해 + λ 개수 (K) 효과.

같은 데이터·같은 학습 프로토콜로 사다리식 구성을 비교해 각 단계의 손실을 분리:

  true_linear     진짜 잠재 상태 (3) + linear head        — ceiling 재현 sanity check
  delta_K4_lin    원시 증분 → bank(K=4) + linear head     — 필터링 손실만 (EMA vs Kalman)
  delta_K8_lin    원시 증분 → bank(K=8) + linear head     — K 를 늘리면 필터링 손실이 닫히는가
  recep_K4_lin    캔들 → receptor → bank(K=4) + linear    — + receptor 손실
  recep_K4_pyr    캔들 → receptor → bank(K=4) + Pyramid   — 기존 구성 (head 효과 비교)
  recep_K8_lin    캔들 → receptor → bank(K=8) + linear
  recep_K8_pyr    캔들 → receptor → bank(K=8) + Pyramid

해석:
  oracle - delta_K8       : K 로도 안 닫히는 잔여 필터링 손실
  delta_K4 - delta_K8     : K 부족분 (λ 개수의 가치)
  delta_Kx - recep_Kx_lin : receptor 병목 손실 (캔들화 + 3-dim 압축 + wick/V 잡음 채널)
  _lin vs _pyr            : head 용량 손실

실행: python -m experiments.decay_bank_forecast.main_decompose
"""

from pathlib import Path

import torch
from torch import nn

from nnpipeline.prototype.decay_bank import DecayBank
from nnpipeline.prototype.ohlcv_receptor import OHLCVReceptor
from nnpipeline.prototype.pyramid import Pyramid
from experiments.decay_bank_forecast.main import r2_per_horizon
from experiments.decay_bank_forecast.synthesize import make_forecast_dataset

RESULTS_DIR = Path(__file__).parent / "results"

WINDOW = 120
HORIZONS = (1, 2, 3, 4, 5)
HL_K4 = (2.0, 8.0, 32.0, 96.0)
HL_K8 = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 120.0)
N_TRAIN = 8192
N_VAL = 2048
BATCH = 256
EPOCHS = 100
LR = 5e-3
LR_LAMBDA = 2e-2
SEED = 7


class BankForecaster(nn.Module):
    """임베딩 시퀀스 (..., n, d) → bank → head. 입력이 이미 시퀀스 형태인 경로용."""

    def __init__(self, half_lives: tuple, in_dim: int, head: str):
        super().__init__()
        self.bank = DecayBank(half_lives=half_lives, learnable=True)
        feat_dim = self.bank.out_scales * in_dim
        self.head = _make_head(head, feat_dim)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.head(self.bank(e).flatten(start_dim=-2))


class ReceptorBankForecaster(nn.Module):

    def __init__(
        self, half_lives: tuple, head: str,
        robust_clip: float = None, robust_dual: bool = False,
    ):
        super().__init__()
        self.receptor = OHLCVReceptor()
        self.bank = DecayBank(
            half_lives=half_lives, learnable=True,
            robust_clip=robust_clip, robust_dual=robust_dual,
        )
        feat_dim = self.bank.out_scales * 3
        self.head = _make_head(head, feat_dim)

    def forward(self, hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        emb = self.receptor(hocl, v)
        return self.head(self.bank(emb).flatten(start_dim=-2))


def _make_head(kind: str, feat_dim: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(feat_dim, len(HORIZONS))
    return Pyramid(feat_dim, len(HORIZONS), depth=3, interlayer=[nn.LeakyReLU()])


def train_and_eval(model, inputs_tr, inputs_va, y_tr_n, y_va_n) -> tuple:
    """공통 학습 프로토콜. 반환: (val R² per horizon, 학습된 half-lives or None)."""
    bank = getattr(model, "bank", None)
    if bank is not None:
        lam = [bank.lambda_logit]
        others = [p for p in model.parameters() if p is not bank.lambda_logit]
        opt = torch.optim.Adam(
            [{"params": others, "lr": LR}, {"params": lam, "lr": LR_LAMBDA}]
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # receptor 의 BatchNorm running stats 가 활성값 통계 변화를 따라가지 못해 eval 성능이
    # epoch 간 크게 요동칠 수 있음 → 매 epoch val 을 재고 최고 시점 가중치를 복원.
    n_train = y_tr_n.shape[0]
    best_loss, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(*(x[idx] for x in inputs_tr)), y_tr_n[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(*inputs_va), y_va_n).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(*inputs_va)
    r2 = r2_per_horizon(pred, y_va_n)
    hls = bank.half_lives.detach().tolist() if bank is not None else None
    return r2, hls


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    hocl_tr, v_tr, y_tr, _, delta_tr, states_tr = make_forecast_dataset(
        N_TRAIN, WINDOW, HORIZONS, seed=SEED, extras=True
    )
    hocl_va, v_va, y_va, oracle_va, delta_va, states_va = make_forecast_dataset(
        N_VAL, WINDOW, HORIZONS, seed=SEED + 999, extras=True
    )

    mu, sd = y_tr.mean(dim=0, keepdim=True), y_tr.std(dim=0, keepdim=True)
    y_tr_n, y_va_n = (y_tr - mu) / sd, (y_va - mu) / sd
    r2_oracle = r2_per_horizon((oracle_va - mu) / sd, y_va_n)

    # 입력 표준화 — 잠재 상태·증분의 원 스케일 (~2e-3) 그대로면 head 가중치가 수백 규모여야
    # 해서 SGD 가 도달 불능. 스케일만 맞추고 정보는 불변.
    d_scale = delta_tr.std()
    d_tr = (delta_tr / d_scale).unsqueeze(-1)                 # (B, n, 1)
    d_va = (delta_va / d_scale).unsqueeze(-1)
    s_scale = states_tr.std()
    states_tr_n, states_va_n = states_tr / s_scale, states_va / s_scale

    # true_linear 는 SGD 대신 닫힌 형태 최소제곱 — 최적화 요인 없는 순수 ceiling 검증
    X = torch.cat([states_tr_n, torch.ones(states_tr_n.shape[0], 1)], dim=-1)
    W = torch.linalg.lstsq(X, y_tr_n).solution                # (4, K_h)
    X_va = torch.cat([states_va_n, torch.ones(states_va_n.shape[0], 1)], dim=-1)
    r2_true = r2_per_horizon(X_va @ W, y_va_n)

    configs = [
        ("delta_K4_lin", lambda: BankForecaster(HL_K4, 1, "linear"), (d_tr,), (d_va,)),
        ("delta_K8_lin", lambda: BankForecaster(HL_K8, 1, "linear"), (d_tr,), (d_va,)),
        ("recep_K4_lin", lambda: ReceptorBankForecaster(HL_K4, "linear"), (hocl_tr, v_tr), (hocl_va, v_va)),
        ("recep_K4_pyr", lambda: ReceptorBankForecaster(HL_K4, "pyramid"), (hocl_tr, v_tr), (hocl_va, v_va)),
        ("recep_K8_lin", lambda: ReceptorBankForecaster(HL_K8, "linear"), (hocl_tr, v_tr), (hocl_va, v_va)),
        ("recep_K8_pyr", lambda: ReceptorBankForecaster(HL_K8, "pyramid"), (hocl_tr, v_tr), (hocl_va, v_va)),
    ]

    lines = [
        f"손실 분해: window={WINDOW}, horizons={HORIZONS}, epochs={EPOCHS}",
        f"oracle R2: {[f'{r:.3f}' for r in r2_oracle.tolist()]}",
        "",
        f"{'config':>13} | {'R2@1':>6} | {'R2@5':>6} | {'capture@5':>9} | learned half-lives",
        "-" * 80,
        f"{'true_lstsq':>13} | {r2_true[0].item():>6.3f} | {r2_true[-1].item():>6.3f} | "
        f"{r2_true[-1].item() / r2_oracle[-1].item():>9.1%} | -",
    ]
    print(lines[-1])

    for name, factory, inp_tr, inp_va in configs:
        torch.manual_seed(SEED)
        model = factory()
        r2, hls = train_and_eval(model, inp_tr, inp_va, y_tr_n, y_va_n)
        cap = r2[-1].item() / r2_oracle[-1].item()
        hl_str = "[" + ", ".join(f"{h:.1f}" for h in hls) + "]" if hls else "-"
        lines.append(
            f"{name:>13} | {r2[0].item():>6.3f} | {r2[-1].item():>6.3f} | {cap:>9.1%} | {hl_str}"
        )
        print(lines[-1])

    report = "\n".join(lines)
    (RESULTS_DIR / "report_decompose.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
