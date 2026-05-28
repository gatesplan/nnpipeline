"""Multi-stock FC 학습. 표준 TB 로깅.

기록:
- train/val loss (total)
- val per-channel MSE (H, O, C, L, V)
- val vs persistence baseline (per-channel ratio)
- receptor 출력 통계 (out_1, out_2, out_v: mean, std)
- 주요 disentanglement R² (각 출력의 top factor R²)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .cdm_data import build_multi_stock_dataset
from .evaluate import FACTOR_NAMES, derive_factors
from .models import CandleForecaster


@dataclass
class MultiTrainConfig:
    run_name: str
    window: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    volume_loss_weight: float = 0.01
    seed: int = 1
    device: str = "cuda"
    tb_log_dir: str = "runs"


def _per_channel_mse(pred: torch.Tensor, target: torch.Tensor):
    return ((pred - target) ** 2).reshape(-1, 5).mean(0)


def _baseline_persistence_mse(loader, device):
    H_e, O_e, C_e, L_e, V_e = [], [], [], [], []
    for hocl, v, tgt, _ in loader:
        hocl, v, tgt = hocl.to(device), v.to(device), tgt.to(device)
        pred = torch.cat([hocl[:, -1, :], v[:, -1, :]], dim=-1)
        err = (pred - tgt) ** 2
        H_e.append(err[:, 0]); O_e.append(err[:, 1]); C_e.append(err[:, 2])
        L_e.append(err[:, 3]); V_e.append(err[:, 4])
    return {
        "H": torch.cat(H_e).mean().item(), "O": torch.cat(O_e).mean().item(),
        "C": torch.cat(C_e).mean().item(), "L": torch.cat(L_e).mean().item(),
        "V": torch.cat(V_e).mean().item(),
    }


def train_multi_stock(cfg: MultiTrainConfig, stocks: List[Tuple[str, np.ndarray, np.ndarray]]):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_ds = build_multi_stock_dataset(stocks, window=cfg.window, forecast_step=1, split="train")
    val_ds = build_multi_stock_dataset(stocks, window=cfg.window, forecast_step=1, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"[train_multi] train: {len(train_ds)} windows, val: {len(val_ds)} windows")

    model = CandleForecaster(window=cfg.window).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse_none = nn.MSELoss(reduction="none")
    lam_v = cfg.volume_loss_weight

    log_dir = Path(cfg.tb_log_dir) / cfg.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"[train_multi] TensorBoard: {log_dir}")

    print("[train_multi] persistence baseline 계산 중...")
    baseline = _baseline_persistence_mse(val_loader, device)
    for ch, val in baseline.items():
        writer.add_scalar(f"baseline/{ch}", val, 0)

    t0 = time.time()

    for epoch in range(cfg.max_epochs):
        # Train
        model.train()
        epoch_train = []
        for hocl, v, tgt, _ in train_loader:
            hocl, v, tgt = hocl.to(device), v.to(device), tgt.to(device)
            pred, _ = model(hocl, v)
            err_sq = mse_none(pred, tgt)
            loss = err_sq[:, :4].mean() + lam_v * err_sq[:, 4].mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_train.append(loss.item())
        train_loss = float(np.mean(epoch_train))
        writer.add_scalar("loss/train", train_loss, epoch)

        # Val
        model.eval()
        epoch_val, epoch_per_ch, z_collect, f_collect = [], [], [], []
        with torch.no_grad():
            for hocl, v, tgt, _ in val_loader:
                hocl, v, tgt = hocl.to(device), v.to(device), tgt.to(device)
                pred, z = model(hocl, v)
                err_sq = mse_none(pred, tgt)
                vloss = err_sq[:, :4].mean() + lam_v * err_sq[:, 4].mean()
                epoch_val.append(vloss.item())
                epoch_per_ch.append(_per_channel_mse(pred, tgt).detach().cpu().numpy())
                z_collect.append(z.detach().cpu())
                f_collect.append(derive_factors(hocl, v).detach().cpu())

        val_loss = float(np.mean(epoch_val))
        per_ch = np.mean(epoch_per_ch, axis=0)
        writer.add_scalar("loss/val", val_loss, epoch)
        for j, ch in enumerate(["H", "O", "C", "L", "V"]):
            writer.add_scalar(f"val_per_channel/{ch}", float(per_ch[j]), epoch)
            ratio = per_ch[j] / (baseline[ch] + 1e-12)
            writer.add_scalar(f"vs_baseline/{ch}", float(ratio), epoch)

        # Receptor 출력 통계 + R²
        z_flat = torch.cat(z_collect, dim=0).reshape(-1, 3).numpy()
        f_flat = torch.cat(f_collect, dim=0).reshape(-1, 5).numpy()
        for i, oname in enumerate(["out_1", "out_2", "out_v"]):
            v_col = z_flat[:, i]
            writer.add_scalar(f"receptor/{oname}_std", float(v_col.std()), epoch)
            # Top R²
            r2_max = 0.0
            for k in range(5):
                if v_col.std() < 1e-9:
                    continue
                corr = np.corrcoef(v_col, f_flat[:, k])[0, 1]
                r2 = float(corr ** 2) if not np.isnan(corr) else 0.0
                r2_max = max(r2_max, r2)
            writer.add_scalar(f"r2_top/{oname}", r2_max, epoch)

        if epoch % 10 == 0 or epoch == cfg.max_epochs - 1:
            print(f"[ep {epoch:3d}] train={train_loss:.5f} val={val_loss:.5f} "
                  f"per_ch C={per_ch[2]:.4f} L={per_ch[3]:.4f} V={per_ch[4]:.3f}")

    writer.close()
    duration = time.time() - t0
    print(f"[train_multi] 학습 완료. {duration:.1f}s 소요")
    return model, log_dir, duration
