"""Canonical 패턴 산점도 시각화."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # Headless
import matplotlib.pyplot as plt
import numpy as np
import torch

from .receptor import CandleAutoencoder
from .synthesize import normalize_hocl, sample_canonical_candles


def canonical_scatter(
    model: CandleAutoencoder,
    device: torch.device,
    out_path: str | Path,
    per_pattern: int = 100,
    seed: int = 21,
) -> None:
    hocl, _, labels = sample_canonical_candles(
        per_pattern, seed=seed, device=str(device)
    )
    model.eval()
    with torch.no_grad():
        z = model.encode(normalize_hocl(hocl).to(device)).cpu().numpy()

    label_arr = np.array(labels)
    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab10", len(unique))

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(unique):
        mask = label_arr == name
        ax.scatter(z[mask, 0], z[mask, 1], color=cmap(i), label=name, alpha=0.6, s=20)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("out_1 (upper-aspect 의도)")
    ax.set_ylabel("out_2 (lower-aspect 의도)")
    ax.set_title("Canonical pattern scatter — 학습된 receptor의 2D 임베딩")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.set_title("Autoencoder reconstruction loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def jacobian_heatmap(jac: dict, out_path: str | Path) -> None:
    rows = list(jac.keys())
    cols = list(jac[rows[0]].keys())
    mat = np.array([[jac[r][c] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    color="white" if mat[i,j] < mat.max()*0.5 else "black", fontsize=9)
    ax.set_title("Jacobian |∂out_i/∂x_j| 평균")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
