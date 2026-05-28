"""Disentanglement 평가.

실데이터 ground truth factor가 없으므로 derived factor와 receptor 출력의 관계 측정.

Derived factors (정규화 공간에서 계산):
- upper_wick = H_norm - max(O_norm, C_norm)
- lower_wick = min(O_norm, C_norm) - L_norm
- body_signed = C_norm - O_norm   (방향 + 크기)
- range = H_norm - L_norm
- volume_z = V_norm (그대로)

평가 지표:
- Linear probing R² (out_i 단독 → factor)
- Jacobian |∂out_i / ∂x_j|
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import WindowDataset, split_train_val_test


FACTOR_NAMES = ["upper_wick", "lower_wick", "body_signed", "range", "volume_z"]


def derive_factors(hocl_norm: torch.Tensor, v_norm: torch.Tensor) -> torch.Tensor:
    """(.., N, 4), (.., N, 1) → (.., N, 5) factor matrix."""
    H = hocl_norm[..., 0]
    O = hocl_norm[..., 1]
    C = hocl_norm[..., 2]
    L = hocl_norm[..., 3]
    upper_wick = H - torch.maximum(O, C)
    lower_wick = torch.minimum(O, C) - L
    body_signed = C - O
    range_ = H - L
    volume_z = v_norm[..., 0]
    return torch.stack([upper_wick, lower_wick, body_signed, range_, volume_z], dim=-1)


def collect_receptor_outputs(
    model, log_hocl, log_v, window: int, device: torch.device,
    batch_size: int = 64, subset: str = "test"
):
    """학습된 모델의 receptor만 통과시켜 (z, factor) 수집.

    subset: "train", "val", "test", "all"
    """
    full_ds = WindowDataset(log_hocl, log_v, window=window, forecast_step=0)
    train_idx, val_idx, test_idx = split_train_val_test(len(full_ds))
    if subset == "train":
        ds = Subset(full_ds, list(train_idx))
    elif subset == "val":
        ds = Subset(full_ds, list(val_idx))
    elif subset == "test":
        ds = Subset(full_ds, list(test_idx))
    else:
        ds = full_ds

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    receptor = model.receptor

    zs, factors = [], []
    receptor.eval()
    with torch.no_grad():
        for hocl, v, _tgt, _ref in loader:
            hocl = hocl.to(device)
            v = v.to(device)
            z = receptor(hocl, v)  # (B, N, 3)
            f = derive_factors(hocl, v)  # (B, N, 5)
            zs.append(z.cpu().numpy())
            factors.append(f.cpu().numpy())

    z_all = np.concatenate(zs, axis=0)  # (M, N, 3)
    f_all = np.concatenate(factors, axis=0)  # (M, N, 5)
    # Flatten batch와 sequence
    z_flat = z_all.reshape(-1, 3)
    f_flat = f_all.reshape(-1, 5)
    return z_flat, f_flat


def linear_probing(z: np.ndarray, factors: np.ndarray) -> Dict[str, Dict[str, float]]:
    """각 (out_i, factor) 쌍에 대해 단변량 선형 회귀 R²."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    n_out = z.shape[1]  # 3
    n_fac = factors.shape[1]  # 5
    result = {}
    for i in range(n_out):
        row = {}
        for k in range(n_fac):
            reg = LinearRegression().fit(z[:, i:i+1], factors[:, k])
            pred = reg.predict(z[:, i:i+1])
            row[FACTOR_NAMES[k]] = float(r2_score(factors[:, k], pred))
        result[f"out_{i+1}"] = row
    return result


def jacobian_analysis(
    model, log_hocl, log_v, window: int, device: torch.device,
    n_samples: int = 500, seed: int = 99
):
    """receptor에 대해 |∂out_i / ∂x_j|, x = (H_norm, O_norm, C_norm, L_norm, V_norm) 측정."""
    full_ds = WindowDataset(log_hocl, log_v, window=window, forecast_step=0)
    train_idx, _, test_idx = split_train_val_test(len(full_ds))

    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(list(test_idx), size=min(n_samples, len(test_idx)), replace=False)

    receptor = model.receptor
    receptor.eval()

    INPUT_NAMES = ["H", "O", "C", "L", "V"]
    # 윈도우 마지막 캔들에 대해서만 Jacobian 측정
    sum_jac = np.zeros((3, 5))
    count = 0
    for idx in sample_idx:
        hocl, v, _tgt, _ref = full_ds[idx]
        # 마지막 캔들만 (1, 1, 4), (1, 1, 1)
        hocl_last = hocl[-1:].unsqueeze(0).to(device).requires_grad_(True)
        v_last = v[-1:].unsqueeze(0).to(device).requires_grad_(True)

        z = receptor(hocl_last, v_last)  # (1, 1, 3)
        z = z[0, 0]  # (3,)

        # 각 출력에 대해 ∂z[i]/∂(hocl_last, v_last)
        for i in range(3):
            grads = torch.autograd.grad(z[i], [hocl_last, v_last], retain_graph=(i < 2))
            g_hocl = grads[0].abs().squeeze().detach().cpu().numpy()  # (4,)
            g_v = grads[1].abs().squeeze().detach().cpu().numpy()      # scalar
            sum_jac[i, :4] += g_hocl
            sum_jac[i, 4] += float(g_v)
        count += 1

    jac_mean = sum_jac / count
    out = {}
    for i in range(3):
        out[f"out_{i+1}"] = {INPUT_NAMES[j]: float(jac_mean[i, j]) for j in range(5)}
    return out


def evaluate_all(model, log_hocl, log_v, window: int, device: torch.device) -> Dict:
    z, f = collect_receptor_outputs(model, log_hocl, log_v, window, device, subset="test")
    lp = linear_probing(z, f)
    jac = jacobian_analysis(model, log_hocl, log_v, window, device)

    # Correlation matrix
    z_centered = z - z.mean(0, keepdims=True)
    f_centered = f - f.mean(0, keepdims=True)
    z_std = z.std(0, keepdims=True) + 1e-8
    f_std = f.std(0, keepdims=True) + 1e-8
    corr = (z_centered.T @ f_centered) / (z.shape[0] * z_std.T @ f_std)
    corr_dict = {}
    for i in range(3):
        corr_dict[f"out_{i+1}"] = {
            FACTOR_NAMES[k]: float(corr[i, k]) for k in range(5)
        }

    return {
        "linear_probing_r2": lp,
        "jacobian_abs_mean": jac,
        "correlation": corr_dict,
        "n_test_samples": z.shape[0],
    }


def verdict(eval_res: Dict) -> Dict:
    """간단 판정: 각 out_i가 어떤 factor와 가장 강한 상관 가지는지."""
    lp = eval_res["linear_probing_r2"]
    summary = {}
    for out_name, r2_row in lp.items():
        sorted_factors = sorted(r2_row.items(), key=lambda x: -x[1])
        summary[out_name] = {
            "top_factor": sorted_factors[0][0],
            "top_r2": sorted_factors[0][1],
            "second_factor": sorted_factors[1][0],
            "second_r2": sorted_factors[1][1],
            "gap": sorted_factors[0][1] - sorted_factors[1][1],
        }
    return summary
