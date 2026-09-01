"""DecayBank + OHLCVReceptor 결합 검증.

1 차원 증분이 아니라 실제 파이프라인 형태 — 합성 OHLCV 캔들 → (미학습) OHLCVReceptor
per-candle 임베딩 (n, 3) → (미학습) DecayBank — 로 모멘텀 형상 3 부류가 분리되는지 확인.

receptor 는 랜덤 초기화 3 개 seed 로 반복하여 초기화 운에 의존하지 않는지 본다.
비교 baseline: 원시 정규화 HOCL+V flatten (n*5 차원), 마지막 캔들 (5 차원).

실행: python -m experiments.decay_bank_verification.main_receptor
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from nnpipeline.prototype.decay_bank import DecayBank
from nnpipeline.prototype.ohlcv_receptor import OHLCVReceptor
from experiments.decay_bank_verification.main import (
    CLASS_COLORS,
    HALF_LIVES,
    N,
    N_SAMPLES,
    NOISE_LEVELS,
    RESULTS_DIR,
    SEED,
    linear_probe_accuracy,
)
from experiments.decay_bank_verification.synthesize import CLASSES, make_ohlcv_dataset

RECEPTOR_SEEDS = (0, 1, 2)


def bank_features(
    receptor: OHLCVReceptor, bank: DecayBank, hocl: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """(미학습) receptor → bank 특징 추출. 반환 (B, out_scales * 3)."""
    receptor.eval()  # BatchNorm 을 running stats (초기값 0/1) 로 고정 — 결정적 추출
    with torch.no_grad():
        emb = receptor(hocl, v)          # (B, n, 3)
        feats = bank(emb)                # (B, out_scales, 3)
    return feats.flatten(start_dim=-2)   # (B, out_scales*3)


def plot_diff_pca(feats: torch.Tensor, y: torch.Tensor, bank: DecayBank, noise: float, path: Path):
    """diff 스케일 성분 (K-1 개 × 3 채널) 을 PCA 2 차원으로 투영한 산점도."""
    k = bank.n_scales
    diffs = feats.view(feats.shape[0], bank.out_scales, 3)[:, k:, :].flatten(start_dim=1)
    centered = diffs - diffs.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, kind in enumerate(CLASSES):
        m = y == i
        ax.scatter(proj[m, 0], proj[m, 1], s=8, alpha=0.5, color=CLASS_COLORS[kind], label=kind)
    ax.set_xlabel("diff PC1")
    ax.set_ylabel("diff PC2")
    ax.set_title(f"Receptor+Bank diff-space PCA (noise={noise})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    bank = DecayBank(half_lives=HALF_LIVES, learnable=False)
    feat_dim = bank.out_scales * 3

    lines = [
        f"Receptor+DecayBank 결합 검증: half_lives={HALF_LIVES}, n={N}, "
        f"{N_SAMPLES} samples/class, receptor seeds={RECEPTOR_SEEDS}",
        f"{'noise':>6} | {'bank(' + str(feat_dim) + 'd)':>10} | {'raw(' + str(N * 5) + 'd)':>10} | {'last(5d)':>9}",
        "-" * 48,
    ]

    for noise in NOISE_LEVELS:
        hocl, v, y = make_ohlcv_dataset(n=N, n_samples=N_SAMPLES, noise=noise, seed=SEED)

        accs = []
        for rs in RECEPTOR_SEEDS:
            torch.manual_seed(rs)
            receptor = OHLCVReceptor()
            feats = bank_features(receptor, bank, hocl, v)
            accs.append(linear_probe_accuracy(feats, y))
            if rs == RECEPTOR_SEEDS[0]:
                plot_diff_pca(feats, y, bank, noise, RESULTS_DIR / f"receptor_diff_pca_noise{noise}.png")
        acc_bank = sum(accs) / len(accs)

        raw = torch.cat([hocl, v], dim=-1).flatten(start_dim=1)   # (B, n*5)
        acc_raw = linear_probe_accuracy(raw, y)
        acc_last = linear_probe_accuracy(torch.cat([hocl, v], dim=-1)[:, -1, :], y)

        lines.append(f"{noise:>6.1f} | {acc_bank:>10.3f} | {acc_raw:>10.3f} | {acc_last:>9.3f}")

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "report_receptor.txt").write_text(report, encoding="utf-8")
    print(f"\nsaved -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
