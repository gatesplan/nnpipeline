"""DecayBank 구조 검증 — 학습 없는 감쇠 상태만으로 모멘텀 형상 3 부류가 분리되는가.

가설: h_fast - h_slow (인접 스케일 차이) 가 "움직임이 약해지고 있는가" 를 구조적으로
표현한다. 총 변위가 동일한 weakening / steady / accelerating 증분 시퀀스를
**미학습** DecayBank 에 통과시켜:

1. diff 좌표 (h_f-h_m, h_m-h_s) 산점도에서 3 부류가 시각적으로 분리되는지
2. 선형 probe 정확도 — bank 특징 (5-dim) vs 원시 시퀀스 (64-dim) vs 마지막 증분 (1-dim)
3. 부류별 평균 상태 궤적 (return_sequence) 이 시간에 따라 어떻게 갈라지는지

를 확인한다. 결과는 results/ 에 저장 (gitignore 대상).

실행: python -m experiments.decay_bank_verification.main
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from nnpipeline.prototype.decay_bank import DecayBank
from experiments.decay_bank_verification.synthesize import CLASSES, make_dataset

RESULTS_DIR = Path(__file__).parent / "results"

N = 64
N_SAMPLES = 300
NOISE_LEVELS = (0.0, 0.5, 1.0, 2.0)
HALF_LIVES = (2.0, 8.0, 32.0)
SEED = 42
CLASS_COLORS = {"weakening": "#d1495b", "steady": "#8d99ae", "accelerating": "#2e6f95"}


def linear_probe_accuracy(feats: torch.Tensor, y: torch.Tensor, seed: int = 0) -> float:
    """표준화 후 선형 (softmax) probe 를 70/30 분할로 학습, 테스트 정확도 반환."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(feats.shape[0], generator=gen)
    feats, y = feats[perm], y[perm]

    n_train = int(feats.shape[0] * 0.7)
    mu = feats[:n_train].mean(dim=0, keepdim=True)
    sd = feats[:n_train].std(dim=0, keepdim=True).clamp_min(1e-8)
    feats = (feats - mu) / sd

    torch.manual_seed(seed)
    probe = nn.Linear(feats.shape[1], len(CLASSES))
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    x_tr, y_tr = feats[:n_train], y[:n_train]
    for _ in range(300):
        opt.zero_grad()
        loss_fn(probe(x_tr), y_tr).backward()
        opt.step()

    with torch.no_grad():
        pred = probe(feats[n_train:]).argmax(dim=-1)
    return (pred == y[n_train:]).float().mean().item()


def plot_diff_scatter(feats: torch.Tensor, y: torch.Tensor, noise: float, path: Path):
    """diff 좌표 (h_f-h_m, h_m-h_s) 산점도. feats 는 (B, 5) — [h_f, h_m, h_s, d_fm, d_ms]."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, kind in enumerate(CLASSES):
        m = y == i
        ax.scatter(
            feats[m, 3], feats[m, 4],
            s=8, alpha=0.5, color=CLASS_COLORS[kind], label=kind,
        )
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("h_fast − h_mid")
    ax.set_ylabel("h_mid − h_slow")
    ax.set_title(f"Diff-space separation (noise={noise})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_state_trajectories(bank: DecayBank, x: torch.Tensor, y: torch.Tensor, path: Path):
    """부류별 평균 상태 궤적: 스케일별 h_k[t] 와 인접 diff[t]."""
    with torch.no_grad():
        seq = bank(x, return_sequence=True)  # (B, n, 5, 1)
    seq = seq[..., 0]  # (B, n, 5)

    labels = ["h_fast (hl=2)", "h_mid (hl=8)", "h_slow (hl=32)", "h_f − h_m", "h_m − h_s"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5), sharex=True)
    for s, (ax, lab) in enumerate(zip(axes, labels)):
        for i, kind in enumerate(CLASSES):
            mean_traj = seq[y == i, :, s].mean(dim=0)
            ax.plot(mean_traj, color=CLASS_COLORS[kind], label=kind)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(lab)
        ax.set_xlabel("t")
    axes[0].legend()
    fig.suptitle("Class-mean state trajectories (untrained bank)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    bank = DecayBank(half_lives=HALF_LIVES, learnable=False)

    lines = [
        f"DecayBank 구조 검증: half_lives={HALF_LIVES}, n={N}, {N_SAMPLES} samples/class",
        f"{'noise':>6} | {'bank(5d)':>9} | {'raw(64d)':>9} | {'last(1d)':>9}",
        "-" * 46,
    ]

    for noise in NOISE_LEVELS:
        x, y = make_dataset(n=N, n_samples=N_SAMPLES, noise=noise, seed=SEED)
        with torch.no_grad():
            feats = bank(x)[..., 0]  # (B, 5, 1) → (B, 5)

        acc_bank = linear_probe_accuracy(feats, y)
        acc_raw = linear_probe_accuracy(x[..., 0], y)
        acc_last = linear_probe_accuracy(x[:, -1, :], y)
        lines.append(
            f"{noise:>6.1f} | {acc_bank:>9.3f} | {acc_raw:>9.3f} | {acc_last:>9.3f}"
        )

        plot_diff_scatter(feats, y, noise, RESULTS_DIR / f"diff_scatter_noise{noise}.png")
        if noise == NOISE_LEVELS[1]:
            plot_state_trajectories(bank, x, y, RESULTS_DIR / "state_trajectories.png")

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
