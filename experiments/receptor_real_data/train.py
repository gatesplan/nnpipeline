"""학습 루프 — autoencoder, forecast 두 종류."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import WindowDataset, split_train_val_test
from .models import CandleAutoencoder, CandleForecaster


@dataclass
class TrainConfig:
    ticker: str
    paradigm: str           # "autoencoder" or "forecast"
    comb_activation: str = "leaky_relu"   # "leaky_relu" or "relu"
    window: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    volume_loss_weight: float = 0.01  # forecast 모드에서 V loss 가중치 (HOCL 4ch 합 대비)
    seed: int = 1
    device: str = "cuda"


@dataclass
class TrainResult:
    config: TrainConfig
    model: nn.Module
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    per_channel_train_losses: List[dict] = field(default_factory=list)
    per_channel_val_losses: List[dict] = field(default_factory=list)
    final_train_loss: float = float("nan")
    final_val_loss: float = float("nan")
    duration_sec: float = 0.0


def _make_loaders(
    log_hocl, log_v, window: int, forecast_step: int, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    full_ds = WindowDataset(log_hocl, log_v, window=window, forecast_step=forecast_step)
    train_idx, val_idx, test_idx = split_train_val_test(len(full_ds))
    train_loader = DataLoader(Subset(full_ds, list(train_idx)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, list(val_idx)),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(full_ds, list(test_idx)),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _per_channel_mse(pred: torch.Tensor, target: torch.Tensor) -> dict:
    # pred, target shape (..., 5)
    names = ["H", "O", "C", "L", "V"]
    diff_sq = (pred - target) ** 2
    # leading dim 평균
    per_ch = diff_sq.reshape(-1, 5).mean(0)
    return {n: float(per_ch[i].item()) for i, n in enumerate(names)}


def train_autoencoder(cfg: TrainConfig, log_hocl, log_v) -> TrainResult:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = _make_loaders(
        log_hocl, log_v, cfg.window, forecast_step=0, batch_size=cfg.batch_size
    )

    model = CandleAutoencoder().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    result = TrainResult(config=cfg, model=model)
    t0 = time.time()

    for epoch in range(cfg.max_epochs):
        # Train
        model.train()
        tr_losses, tr_per_ch = [], []
        for hocl, v, tgt, _ref in train_loader:
            hocl, v, tgt = hocl.to(device), v.to(device), tgt.to(device)
            recon, _ = model(hocl, v)
            loss = mse(recon, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())
            tr_per_ch.append(_per_channel_mse(recon, tgt))
        result.train_losses.append(sum(tr_losses) / len(tr_losses))
        # avg per channel
        avg_ch = {k: sum(d[k] for d in tr_per_ch) / len(tr_per_ch) for k in tr_per_ch[0]}
        result.per_channel_train_losses.append(avg_ch)

        # Val
        model.eval()
        va_losses, va_per_ch = [], []
        with torch.no_grad():
            for hocl, v, tgt, _ref in val_loader:
                hocl, v, tgt = hocl.to(device), v.to(device), tgt.to(device)
                recon, _ = model(hocl, v)
                va_losses.append(mse(recon, tgt).item())
                va_per_ch.append(_per_channel_mse(recon, tgt))
        result.val_losses.append(sum(va_losses) / len(va_losses))
        avg_ch_v = {k: sum(d[k] for d in va_per_ch) / len(va_per_ch) for k in va_per_ch[0]}
        result.per_channel_val_losses.append(avg_ch_v)

    result.duration_sec = time.time() - t0
    result.final_train_loss = result.train_losses[-1]
    result.final_val_loss = result.val_losses[-1]
    return result


def train_forecast(cfg: TrainConfig, log_hocl, log_v) -> TrainResult:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = _make_loaders(
        log_hocl, log_v, cfg.window, forecast_step=1, batch_size=cfg.batch_size
    )

    model = CandleForecaster(window=cfg.window).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse_none = nn.MSELoss(reduction='none')  # 채널별로 계산용
    lam_v = cfg.volume_loss_weight

    result = TrainResult(config=cfg, model=model)
    t0 = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        tr_losses, tr_per_ch = [], []
        for hocl, v, tgt, _ref in train_loader:
            hocl, v, tgt = hocl.to(device), v.to(device), tgt.to(device)
            pred, _ = model(hocl, v)  # (B, 5)
            err_sq = mse_none(pred, tgt)  # (B, 5)
            # HOCL 4 채널 합 + V 채널 * lam_v
            loss = err_sq[:, :4].mean() + lam_v * err_sq[:, 4].mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_losses.append(loss.item())
            tr_per_ch.append(_per_channel_mse(pred, tgt))
        result.train_losses.append(sum(tr_losses) / len(tr_losses))
        avg_ch = {k: sum(d[k] for d in tr_per_ch) / len(tr_per_ch) for k in tr_per_ch[0]}
        result.per_channel_train_losses.append(avg_ch)

        model.eval()
        va_losses, va_per_ch = [], []
        with torch.no_grad():
            for hocl, v, tgt, _ref in val_loader:
                hocl, v, tgt = hocl.to(device), v.to(device), tgt.to(device)
                pred, _ = model(hocl, v)
                err_sq = mse_none(pred, tgt)
                va_losses.append((err_sq[:, :4].mean() + lam_v * err_sq[:, 4].mean()).item())
                va_per_ch.append(_per_channel_mse(pred, tgt))
        result.val_losses.append(sum(va_losses) / len(va_losses))
        avg_ch_v = {k: sum(d[k] for d in va_per_ch) / len(va_per_ch) for k in va_per_ch[0]}
        result.per_channel_val_losses.append(avg_ch_v)

    result.duration_sec = time.time() - t0
    result.final_train_loss = result.train_losses[-1]
    result.final_val_loss = result.val_losses[-1]
    return result
