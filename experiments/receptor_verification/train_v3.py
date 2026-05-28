"""v3 학습: autoencoder + auxiliary upper/lower loss.

연구계획 §7.1 fallback 옵션 구현.

Aux loss:
    L_aux = λ · [(out_1 - upper_target)² + (out_2 - lower_target)²]

upper_target, lower_target은 정규화된 HOCL에서 derive:
    upper_target = H_norm - max(O_norm, C_norm)   (upper wick 길이, 정규화 단위)
    lower_target = min(O_norm, C_norm) - L_norm   (lower wick 길이, 정규화 단위)

이 값을 직접 출력하지 않더라도, 강한 상관을 가지면 분리 성공.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .receptor import CandleAutoencoder
from .synthesize import normalize_hocl, sample_random_candles


@dataclass
class TrainConfigV3:
    receptor_hidden: int = 16
    decoder_hidden: int = 32
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    seed: int = 1
    train_n: int = 50_000
    val_n: int = 10_000
    device: str = "cuda"
    aux_lambda: float = 0.05
    # aux target scaling: 정규화된 wick 값에 곱해서 receptor 출력 스케일에 맞춤
    aux_target_scale: float = 1.0


@dataclass
class TrainResultV3:
    config: TrainConfigV3
    model: CandleAutoencoder
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_aux_losses: List[float] = field(default_factory=list)
    val_aux_losses: List[float] = field(default_factory=list)
    final_train_loss: float = float("nan")
    final_val_loss: float = float("nan")


def derive_wick_targets(hocl_norm: torch.Tensor) -> torch.Tensor:
    """정규화된 HOCL에서 (upper_target, lower_target) 추출.

    Returns
    -------
    (..., 2) tensor: [upper_wick_norm, lower_wick_norm]
    """
    H = hocl_norm[..., 0]
    O = hocl_norm[..., 1]
    C = hocl_norm[..., 2]
    L = hocl_norm[..., 3]
    upper = H - torch.maximum(O, C)
    lower = torch.minimum(O, C) - L
    return torch.stack([upper, lower], dim=-1)


def build_loaders(cfg: TrainConfigV3):
    train_hocl, _ = sample_random_candles(cfg.train_n, seed=cfg.seed)
    val_hocl, _ = sample_random_candles(cfg.val_n, seed=cfg.seed + 1000)

    train_norm = normalize_hocl(train_hocl)
    val_norm = normalize_hocl(val_hocl)
    train_targets = derive_wick_targets(train_norm) * cfg.aux_target_scale
    val_targets = derive_wick_targets(val_norm) * cfg.aux_target_scale

    train_ds = TensorDataset(train_norm, train_targets)
    val_ds = TensorDataset(val_norm, val_targets)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def train_v3(cfg: TrainConfigV3) -> TrainResultV3:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(cfg)
    model = CandleAutoencoder(
        receptor_hidden=cfg.receptor_hidden, decoder_hidden=cfg.decoder_hidden
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    result = TrainResultV3(config=cfg, model=model)

    for epoch in range(cfg.max_epochs):
        model.train()
        tr_recon, tr_aux = [], []
        for batch_norm, batch_target in train_loader:
            batch_norm = batch_norm.to(device)
            batch_target = batch_target.to(device)
            recon, z = model(batch_norm)
            recon_loss = mse(recon, batch_norm)
            aux_loss = mse(z, batch_target)
            loss = recon_loss + cfg.aux_lambda * aux_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_recon.append(recon_loss.item())
            tr_aux.append(aux_loss.item())
        result.train_losses.append(sum(tr_recon) / len(tr_recon))
        result.train_aux_losses.append(sum(tr_aux) / len(tr_aux))

        model.eval()
        va_recon, va_aux = [], []
        with torch.no_grad():
            for batch_norm, batch_target in val_loader:
                batch_norm = batch_norm.to(device)
                batch_target = batch_target.to(device)
                recon, z = model(batch_norm)
                va_recon.append(mse(recon, batch_norm).item())
                va_aux.append(mse(z, batch_target).item())
        result.val_losses.append(sum(va_recon) / len(va_recon))
        result.val_aux_losses.append(sum(va_aux) / len(va_aux))

    result.final_train_loss = result.train_losses[-1]
    result.final_val_loss = result.val_losses[-1]
    return result


if __name__ == "__main__":
    cfg = TrainConfigV3()
    r = train_v3(cfg)
    print(f"final recon train={r.final_train_loss:.6f} val={r.final_val_loss:.6f} "
          f"aux train={r.train_aux_losses[-1]:.6f} val={r.val_aux_losses[-1]:.6f}")
