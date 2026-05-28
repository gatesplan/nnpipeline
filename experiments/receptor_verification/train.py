"""Autoencoder 학습 루프.

연구계획 §3, §8.2를 따른다.
- MAX_EPOCHS=100, early stopping OFF
- Adam 10^-3 baseline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .receptor import CandleAutoencoder
from .synthesize import (
    CandleFactors,
    normalize_hocl,
    sample_random_candles,
)


@dataclass
class TrainConfig:
    receptor_hidden: int = 16
    decoder_hidden: int = 32
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    seed: int = 1
    train_n: int = 50_000
    val_n: int = 10_000
    device: str = "cuda"


@dataclass
class TrainResult:
    config: TrainConfig
    model: CandleAutoencoder
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    final_train_loss: float = float("nan")
    final_val_loss: float = float("nan")


def build_loaders(
    train_n: int, val_n: int, batch_size: int, seed: int, device: str
) -> tuple[DataLoader, DataLoader, CandleFactors, CandleFactors]:
    train_hocl, train_factors = sample_random_candles(train_n, seed=seed)
    val_hocl, val_factors = sample_random_candles(val_n, seed=seed + 1000)

    train_norm = normalize_hocl(train_hocl)
    val_norm = normalize_hocl(val_hocl)

    train_ds = TensorDataset(train_norm)
    val_ds = TensorDataset(val_norm)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_factors, val_factors


def train(cfg: TrainConfig) -> TrainResult:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, _ = build_loaders(
        cfg.train_n, cfg.val_n, cfg.batch_size, cfg.seed, str(device)
    )

    model = CandleAutoencoder(
        receptor_hidden=cfg.receptor_hidden, decoder_hidden=cfg.decoder_hidden
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    result = TrainResult(config=cfg, model=model)

    for epoch in range(cfg.max_epochs):
        # Train
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)
        result.train_losses.append(train_loss)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                val_losses.append(loss_fn(recon, batch).item())
        val_loss = sum(val_losses) / len(val_losses)
        result.val_losses.append(val_loss)

    result.final_train_loss = result.train_losses[-1]
    result.final_val_loss = result.val_losses[-1]
    return result


if __name__ == "__main__":
    cfg = TrainConfig()
    res = train(cfg)
    print(f"final train={res.final_train_loss:.6f} val={res.final_val_loss:.6f}")
