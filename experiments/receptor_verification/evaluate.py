"""다중지표 평가.

연구계획 §5에 정의된 평가 지표를 모두 구현한다.

표준 disentanglement:
- MIG (Mutual Information Gap)
- DCI (Disentanglement, Completeness, Informativeness)
- SAP (Separated Attribute Predictability)
- Linear probing R²
- FactorVAE score

도메인 특화:
- Jacobian analysis
- Causal intervention
- SeparationScore

표준 지표의 MI/regression은 sklearn으로 구현한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .receptor import CandleAutoencoder
from .synthesize import (
    CandleFactors,
    factors_to_hocl,
    normalize_hocl,
    sample_canonical_candles,
    sample_random_candles,
)


FACTOR_NAMES = ["center", "magnitude", "direction", "upper_wick", "lower_wick"]


# ---------- 유틸 ----------

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def encode_dataset(
    model: CandleAutoencoder, hocl: torch.Tensor, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model.encode(normalize_hocl(hocl).to(device))
    return _to_numpy(z)


def factors_matrix(factors: CandleFactors) -> np.ndarray:
    """(N, 5) factor matrix. order = FACTOR_NAMES."""
    cols = [_to_numpy(getattr(factors, name)) for name in FACTOR_NAMES]
    return np.stack(cols, axis=1)


# ---------- MIG ----------

def mig_score(z: np.ndarray, factors: np.ndarray, n_bins: int = 20) -> Dict[str, float]:
    """Mutual Information Gap.

    각 factor에 대해, z 차원과의 MI를 모두 계산. 최대 MI에서 두 번째 MI를 빼고
    factor의 entropy로 정규화. 평균 = MIG.

    sklearn의 mutual_info_regression이 continuous → continuous MI를 추정.
    """
    n_factors = factors.shape[1]
    n_latent = z.shape[1]
    mig_per_factor = []
    detail = {}

    for k in range(n_factors):
        f = factors[:, k]
        mis = []
        for j in range(n_latent):
            mi = mutual_info_regression(z[:, j:j+1], f, random_state=0)[0]
            mis.append(max(mi, 0.0))
        mis_sorted = sorted(mis, reverse=True)
        # factor entropy estimation (continuous → discretize)
        f_disc = np.digitize(f, np.linspace(f.min(), f.max(), n_bins))
        _, counts = np.unique(f_disc, return_counts=True)
        p = counts / counts.sum()
        h = -(p * np.log(p + 1e-12)).sum()
        gap = (mis_sorted[0] - mis_sorted[1]) / max(h, 1e-6)
        mig_per_factor.append(gap)
        detail[FACTOR_NAMES[k]] = float(gap)

    return {"mig_mean": float(np.mean(mig_per_factor)), "per_factor": detail}


# ---------- DCI ----------

def dci_score(z: np.ndarray, factors: np.ndarray) -> Dict[str, float]:
    """DCI: Disentanglement, Completeness, Informativeness.

    각 factor를 GradientBoosting으로 회귀, feature_importances_를 행으로 쌓아
    importance matrix R ∈ R^{D_latent x D_factor}를 만든다.
    """
    n_latent = z.shape[1]
    n_factors = factors.shape[1]
    R = np.zeros((n_latent, n_factors))
    informativeness = np.zeros(n_factors)

    for k in range(n_factors):
        f = factors[:, k]
        model = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, random_state=0
        )
        model.fit(z, f)
        R[:, k] = model.feature_importances_
        informativeness[k] = r2_score(f, model.predict(z))

    # 정규화
    R_d = R / (R.sum(axis=1, keepdims=True) + 1e-12)   # row normalize for disentanglement
    R_c = R / (R.sum(axis=0, keepdims=True) + 1e-12)   # col normalize for completeness

    # Disentanglement per latent
    def entropy(p):
        p = np.asarray(p)
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    H_max_d = np.log(n_factors)
    H_max_c = np.log(n_latent)
    disent_per = np.array([
        1 - entropy(R_d[i, :]) / max(H_max_d, 1e-12) for i in range(n_latent)
    ])
    comple_per = np.array([
        1 - entropy(R_c[:, k]) / max(H_max_c, 1e-12) for k in range(n_factors)
    ])

    # Weighted average (by importance)
    weight_d = R.sum(axis=1) / (R.sum() + 1e-12)
    weight_c = R.sum(axis=0) / (R.sum() + 1e-12)

    return {
        "disentanglement": float(np.sum(disent_per * weight_d)),
        "completeness": float(np.sum(comple_per * weight_c)),
        "informativeness_mean_r2": float(np.mean(informativeness)),
        "per_factor_r2": {
            FACTOR_NAMES[k]: float(informativeness[k]) for k in range(n_factors)
        },
    }


# ---------- SAP ----------

def sap_score(z: np.ndarray, factors: np.ndarray) -> Dict[str, float]:
    """SAP: 각 factor에 대해 모든 잠재 차원의 예측력(R²) 계산.
    상위 2개의 차이를 평균.
    """
    n_factors = factors.shape[1]
    n_latent = z.shape[1]
    gaps = []
    detail = {}

    for k in range(n_factors):
        f = factors[:, k]
        scores = []
        for j in range(n_latent):
            reg = LinearRegression().fit(z[:, j:j+1], f)
            scores.append(r2_score(f, reg.predict(z[:, j:j+1])))
        scores_sorted = sorted(scores, reverse=True)
        gap = scores_sorted[0] - scores_sorted[1]
        gaps.append(gap)
        detail[FACTOR_NAMES[k]] = float(gap)

    return {"sap_mean": float(np.mean(gaps)), "per_factor": detail}


# ---------- Linear probing ----------

def linear_probing(z: np.ndarray, factors: np.ndarray) -> Dict[str, Dict[str, float]]:
    """latent 각 차원 단독으로 factor 회귀, R² 표."""
    n_latent = z.shape[1]
    n_factors = factors.shape[1]
    out: Dict[str, Dict[str, float]] = {}
    for j in range(n_latent):
        row = {}
        for k in range(n_factors):
            reg = LinearRegression().fit(z[:, j:j+1], factors[:, k])
            row[FACTOR_NAMES[k]] = float(r2_score(factors[:, k], reg.predict(z[:, j:j+1])))
        out[f"out_{j+1}"] = row
    return out


# ---------- FactorVAE score ----------

def factor_vae_score(
    model: CandleAutoencoder,
    device: torch.device,
    n_votes: int = 200,
    samples_per_vote: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """FactorVAE-style score.

    각 vote마다:
        1. 한 factor를 임의로 고정
        2. 그 factor를 고정한 상태로 다른 factor들은 자유 sampling
        3. 잠재 차원의 분산이 가장 작은 차원이 그 factor와 연관됐다고 추정
        4. (factor, argmin 차원) majority vote.

    정확도 = 정답률 (factor가 자기 차원에 모이면 1에 가까움).
    """
    rng = np.random.RandomState(seed)
    fixed_factor_choices = rng.randint(0, len(FACTOR_NAMES), n_votes)
    votes = np.zeros((len(FACTOR_NAMES), 2))  # 2 = num latents

    model.eval()
    with torch.no_grad():
        for v in range(n_votes):
            k = fixed_factor_choices[v]
            # 모든 factor 무작위 sampling
            sub_seed = int(rng.randint(0, 10_000_000))
            hocl, factors = sample_random_candles(
                samples_per_vote, seed=sub_seed, device=str(device)
            )
            # k번째 factor만 한 값으로 고정 → 다시 hocl 합성
            f_dict = factors.as_dict()
            fixed_value = f_dict[FACTOR_NAMES[k]][0]  # 첫 sample 값으로 고정
            f_dict[FACTOR_NAMES[k]] = torch.full_like(
                f_dict[FACTOR_NAMES[k]], float(fixed_value)
            )
            new_factors = CandleFactors(**f_dict)
            hocl_fixed = factors_to_hocl(new_factors).to(device)
            z = model.encode(normalize_hocl(hocl_fixed))
            # 표준화 후 분산
            z_std = (z - z.mean(0, keepdim=True)) / (
                z.std(0, keepdim=True) + 1e-6
            )
            var_per_dim = z_std.var(0)
            argmin = int(var_per_dim.argmin().item())
            votes[k, argmin] += 1

    # 각 factor에 대해 majority dim 정답률
    accuracies = []
    for k in range(len(FACTOR_NAMES)):
        if votes[k].sum() == 0:
            accuracies.append(0.0)
        else:
            accuracies.append(float(votes[k].max() / votes[k].sum()))

    return {
        "factorvae_mean_accuracy": float(np.mean(accuracies)),
        "per_factor_accuracy": {
            FACTOR_NAMES[k]: accuracies[k] for k in range(len(FACTOR_NAMES))
        },
        "votes_matrix": votes.tolist(),
    }


# ---------- Jacobian analysis ----------

def jacobian_analysis(
    model: CandleAutoencoder,
    device: torch.device,
    n_samples: int = 1000,
    seed: int = 99,
) -> Dict[str, float]:
    """test 캔들에 대해 ∂out_i/∂x_j (x ∈ {H, O, C, L}) 평균 절대값."""
    hocl, _ = sample_random_candles(n_samples, seed=seed, device=str(device))
    hocl = hocl.detach()
    norm_input = normalize_hocl(hocl).requires_grad_(True)

    model.eval()
    z = model.encode(norm_input)  # (N, 2)

    # 각 출력에 대해 Jacobian 계산
    # grad_out_i / grad_input
    grads = {}
    for i in range(2):
        ones = torch.ones(n_samples, device=device)
        g = torch.autograd.grad(
            z[:, i], norm_input, grad_outputs=ones, retain_graph=(i == 0)
        )[0]  # (N, 4)
        grads[f"out_{i+1}"] = g.abs().mean(0).detach().cpu().numpy()

    INPUT_NAMES = ["H", "O", "C", "L"]
    jac_dict = {
        f"out_{i+1}": {
            INPUT_NAMES[j]: float(grads[f"out_{i+1}"][j]) for j in range(4)
        }
        for i in range(2)
    }

    sep_score = _separation_score(jac_dict)
    return {"jacobian_abs_mean": jac_dict, "separation_score": sep_score}


def _separation_score(jac: Dict[str, Dict[str, float]]) -> float:
    j1H = jac["out_1"]["H"]; j1L = jac["out_1"]["L"]
    j2H = jac["out_2"]["H"]; j2L = jac["out_2"]["L"]
    eps = 1e-9
    s1 = (abs(j1H) - abs(j1L)) / (abs(j1H) + abs(j1L) + eps)
    s2 = (abs(j2L) - abs(j2H)) / (abs(j2L) + abs(j2H) + eps)
    return float(s1 + s2)


# ---------- Causal intervention ----------

def causal_intervention(
    model: CandleAutoencoder,
    device: torch.device,
    delta: float = 0.5,
    n_samples: int = 1000,
    seed: int = 77,
) -> Dict[str, Dict[str, float]]:
    """베이스라인 캔들에서 H 또는 L만 delta만큼 이동 → out 변화 측정."""
    hocl, _ = sample_random_candles(n_samples, seed=seed, device=str(device))
    model.eval()

    base_norm = normalize_hocl(hocl).to(device)
    with torch.no_grad():
        z0 = model.encode(base_norm)

    # H 증가
    hocl_h = hocl.clone()
    hocl_h[:, 0] = hocl_h[:, 0] + delta
    with torch.no_grad():
        z_h = model.encode(normalize_hocl(hocl_h).to(device))
    dh = (z_h - z0).abs().mean(0).cpu().numpy()

    # L 감소 (제약 유지)
    hocl_l = hocl.clone()
    hocl_l[:, 3] = hocl_l[:, 3] - delta
    with torch.no_grad():
        z_l = model.encode(normalize_hocl(hocl_l).to(device))
    dl = (z_l - z0).abs().mean(0).cpu().numpy()

    return {
        "delta_out_for_H_perturbation": {"out_1": float(dh[0]), "out_2": float(dh[1])},
        "delta_out_for_L_perturbation": {"out_1": float(dl[0]), "out_2": float(dl[1])},
    }


# ---------- Reconstruction quality ----------

def single_dim_reconstruction(
    model: CandleAutoencoder,
    device: torch.device,
    n_samples: int = 5000,
    seed: int = 55,
) -> Dict[str, Dict[str, float]]:
    """out_1만 사용한 디코딩 vs out_2만 사용한 디코딩.

    구현: 나머지 차원을 0으로 mask 후 decoder 통과. 채널별 MSE 측정.
    """
    hocl, _ = sample_random_candles(n_samples, seed=seed, device=str(device))
    norm = normalize_hocl(hocl).to(device)

    model.eval()
    with torch.no_grad():
        z = model.encode(norm)

        # out_1만
        z_1 = z.clone()
        z_1[:, 1] = 0.0
        recon_1 = model.decode(z_1)

        # out_2만
        z_2 = z.clone()
        z_2[:, 0] = 0.0
        recon_2 = model.decode(z_2)

        # 양쪽 다
        recon_full = model.decode(z)

    target = norm
    INPUT_NAMES = ["H", "O", "C", "L"]
    def per_ch_mse(recon):
        err = (recon - target) ** 2
        return {INPUT_NAMES[j]: float(err[:, j].mean().item()) for j in range(4)}

    return {
        "full_decoder_mse_per_channel": per_ch_mse(recon_full),
        "out1_only_mse_per_channel": per_ch_mse(recon_1),
        "out2_only_mse_per_channel": per_ch_mse(recon_2),
    }


# ---------- 전체 평가 묶음 ----------

@dataclass
class EvaluationResult:
    mig: Dict
    dci: Dict
    sap: Dict
    linear_probing: Dict
    factorvae: Dict
    jacobian: Dict
    causal: Dict
    reconstruction: Dict


def evaluate_all(
    model: CandleAutoencoder,
    device: torch.device,
    n_eval: int = 10_000,
    seed: int = 123,
) -> EvaluationResult:
    hocl, factors = sample_random_candles(n_eval, seed=seed)
    z_np = encode_dataset(model, hocl, device)
    factor_mat = factors_matrix(factors)

    return EvaluationResult(
        mig=mig_score(z_np, factor_mat),
        dci=dci_score(z_np, factor_mat),
        sap=sap_score(z_np, factor_mat),
        linear_probing=linear_probing(z_np, factor_mat),
        factorvae=factor_vae_score(model, device),
        jacobian=jacobian_analysis(model, device),
        causal=causal_intervention(model, device),
        reconstruction=single_dim_reconstruction(model, device),
    )


# ---------- 판정 ----------

def verdict(eval_res: EvaluationResult) -> Dict[str, object]:
    """연구계획 §6의 결정 기준 적용."""
    conditions = {}

    # 1. MIG >= 0.3
    conditions["mig>=0.3"] = eval_res.mig["mig_mean"] >= 0.3

    # 2. DCI Disentanglement >= 0.5
    conditions["dci_disent>=0.5"] = eval_res.dci["disentanglement"] >= 0.5

    # 3. Linear probing 비대각/대각 비율
    lp = eval_res.linear_probing
    # 대각 = out_1↔upper_wick, out_2↔lower_wick
    diag = [lp["out_1"]["upper_wick"], lp["out_2"]["lower_wick"]]
    off_diag = [lp["out_1"]["lower_wick"], lp["out_2"]["upper_wick"]]
    if min(diag) > 1e-6:
        ratio = max(off_diag) / min(diag)
    else:
        ratio = float("inf")
    conditions["linear_probing_off/diag<=0.3"] = ratio <= 0.3

    # 4. SeparationScore >= 1.0
    conditions["separation>=1.0"] = eval_res.jacobian["separation_score"] >= 1.0

    n_pass = sum(conditions.values())

    if n_pass >= 3:
        outcome = "분리 성공"
    elif (
        eval_res.mig["mig_mean"] < 0.1
        or ratio > 1.5
    ):
        outcome = "분리 실패"
    else:
        outcome = "모호"

    return {
        "conditions": conditions,
        "n_pass": int(n_pass),
        "linear_probing_diag": diag,
        "linear_probing_off_diag": off_diag,
        "outcome": outcome,
    }
