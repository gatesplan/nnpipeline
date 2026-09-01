"""다중 시간스케일 잠재 구조를 심은 합성 수익률 시계열.

log 증분을 반감기 τ_j 가 알려진 잠재 AR(1) 성분들의 합 + 백색잡음으로 생성:

    s_j[t] = φ_j s_j[t-1] + η_j,   φ_j = 2^(-1/τ_j),  stationary std = comp_std
    δ_t    = Σ_j s_j[t] + ε_t

미래 k 봉 누적수익률의 조건부 기대값은 잠재 상태의 선형 결합 (닫힌 형태) 으로 계산
가능하므로, 도달 가능한 예측력 상한 (oracle R²) 과 "진짜 반감기" (τ_j) 를 둘 다 안다.
DecayBank 의 학습된 반감기가 τ_j 쪽으로 이동하는지 검증하는 것이 목적.
"""

import torch

from experiments.decay_bank_verification.synthesize import candles_from_increments

TRUE_HALF_LIVES = (4.0, 16.0, 64.0)
COMP_STD = 0.002      # 각 잠재 성분의 정상 표준편차 (log 수익률 단위)
NOISE_STD = 0.004     # 예측 불가능한 백색잡음 표준편차


def make_series(
    n_samples: int,
    n_total: int,
    seed: int = 0,
    taus: tuple = TRUE_HALF_LIVES,
    comp_std: float = COMP_STD,
    noise_std: float = NOISE_STD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """반환: (delta, states) — delta (B, n_total), states (B, n_total, J)."""
    gen = torch.Generator().manual_seed(seed)
    phis = torch.tensor([2.0 ** (-1.0 / t) for t in taus])            # (J,)
    innov_std = comp_std * torch.sqrt(1.0 - phis ** 2)                # 정상성 유지

    s = torch.randn(n_samples, len(taus), generator=gen) * comp_std   # 정상 분포 초기화
    states = [s]
    for _ in range(n_total - 1):
        eta = torch.randn(n_samples, len(taus), generator=gen) * innov_std
        s = phis * s + eta
        states.append(s)
    states = torch.stack(states, dim=1)                               # (B, n_total, J)

    eps = torch.randn(n_samples, n_total, generator=gen) * noise_std
    delta = states.sum(dim=-1) + eps
    return delta, states


def oracle_predictions(
    states_at_n: torch.Tensor,
    horizons: tuple,
    taus: tuple = TRUE_HALF_LIVES,
) -> torch.Tensor:
    """생성 과정을 아는 관측자의 최적 예측 E[Σ_{i=1..k} δ_{n+i} | s[n]].

    Σ_{i=1..k} φ^i = φ(1-φ^k)/(1-φ). 반환 (B, len(horizons)).
    """
    phis = torch.tensor([2.0 ** (-1.0 / t) for t in taus])            # (J,)
    preds = []
    for k in horizons:
        coef = phis * (1.0 - phis ** k) / (1.0 - phis)                # (J,)
        preds.append((states_at_n * coef).sum(dim=-1))
    return torch.stack(preds, dim=-1)


def make_forecast_dataset(
    n_samples: int,
    window: int = 120,
    horizons: tuple = (1, 2, 3, 4, 5),
    seed: int = 0,
    extras: bool = False,
):
    """반환: (hocl, v, targets, oracle) — extras=True 면 (…, delta_window, states_at_n) 추가.

    - hocl (B, window, 4), v (B, window, 1): 정규화된 캔들 (candles_from_increments 규약)
    - targets (B, len(horizons)): 미래 k 봉 누적 log 수익률
    - oracle (B, len(horizons)): 잠재 상태 기반 최적 예측 (상한 측정용)
    - delta_window (B, window): 원시 log 증분 (receptor 우회 경로용)
    - states_at_n (B, J): 윈도우 마지막 시점의 진짜 잠재 상태 (ceiling 검증용)
    """
    n_total = window + max(horizons)
    delta, states = make_series(n_samples, n_total, seed=seed)

    hocl, v = candles_from_increments(
        delta[:, :window], mean_step=delta.abs().mean().item(), seed=seed
    )

    future = delta[:, window:]                                        # (B, max_h)
    cum = torch.cumsum(future, dim=-1)
    targets = torch.stack([cum[:, k - 1] for k in horizons], dim=-1)  # (B, K_h)

    states_at_n = states[:, window - 1, :]
    oracle = oracle_predictions(states_at_n, horizons)
    if extras:
        return hocl, v, targets, oracle, delta[:, :window], states_at_n
    return hocl, v, targets, oracle
