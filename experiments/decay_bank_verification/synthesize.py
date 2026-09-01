"""합성 시계열 생성 — 모멘텀 형상 3 부류.

세 부류 모두 총 변위 (증분 합) 를 1 로 정규화하여, 부류 구분이 총수익률 크기가 아니라
**누적의 형상** (약해짐 / 일정 / 가속) 에서만 나오도록 강제한다.

- weakening:    δ_t ∝ r^t, r < 1 — 증분이 점점 약해지는 상승
- steady:       δ_t ∝ 1 — 일정한 상승
- accelerating: δ_t ∝ r^t, r > 1 — 증분이 점점 강해지는 상승
"""

import torch

CLASSES = ("weakening", "steady", "accelerating")

# 부류별 증분 성장률
_RATES = {
    "weakening": 0.94,
    "steady": 1.0,
    "accelerating": 1.06,
}


def make_increments(
    kind: str,
    n: int = 64,
    n_samples: int = 300,
    noise: float = 0.1,
    seed: int = 0,
) -> torch.Tensor:
    """증분 시퀀스 생성. 반환 shape (n_samples, n, 1).

    noise 는 평균 증분 크기 대비 가우시안 잡음 표준편차 비율.
    """
    if kind not in _RATES:
        raise ValueError(f"kind 는 {CLASSES} 중 하나여야 합니다. 받은 값: {kind!r}")

    r = _RATES[kind]
    base = torch.tensor([r ** t for t in range(n)], dtype=torch.float32)
    base = base / base.sum()  # 총 변위 1 로 정규화 — 형상만 남김

    gen = torch.Generator().manual_seed(seed)
    mean_step = 1.0 / n
    eps = torch.randn(n_samples, n, generator=gen) * (noise * mean_step)
    x = base.unsqueeze(0) + eps
    return x.unsqueeze(-1)


def make_dataset(
    n: int = 64,
    n_samples: int = 300,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3 부류 합본. 반환: (X, y) — X (3*n_samples, n, 1), y (3*n_samples,) 부류 인덱스."""
    xs, ys = [], []
    for i, kind in enumerate(CLASSES):
        xs.append(make_increments(kind, n=n, n_samples=n_samples, noise=noise, seed=seed + i))
        ys.append(torch.full((n_samples,), i, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def candles_from_increments(
    delta: torch.Tensor,
    mean_step: float,
    seed: int = 0,
    wick_scale: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """log 증분 시퀀스 (B, n) 를 OHLCV 캔들로 전개. receptor 입력 규약에 맞춰 정규화까지 수행.

    - log 가격 경로: logP_t = logP_{t-1} + δ_t
    - O = 직전 close, C = 현재 close, H/L = body ± 랜덤 wick (스케일: wick_scale × mean_step)
    - HOCL 정규화: 윈도우 마지막 close 차감 (log(x / C_n))
    - V: |증분| 에 비례 + lognormal 잡음 → log + 윈도우 내 z-norm

    반환: (hocl, v) — hocl (B, n, 4) [H,O,C,L], v (B, n, 1)
    """
    n_samples, n = delta.shape
    log_c = torch.cumsum(delta, dim=-1)                      # (B, n)
    log_o = log_c - delta                                    # 직전 close
    body_hi = torch.maximum(log_o, log_c)
    body_lo = torch.minimum(log_o, log_c)

    gen = torch.Generator().manual_seed(seed + 1000)
    wick_up = torch.randn(n_samples, n, generator=gen).abs() * (wick_scale * mean_step)
    wick_dn = torch.randn(n_samples, n, generator=gen).abs() * (wick_scale * mean_step)
    log_h = body_hi + wick_up
    log_l = body_lo - wick_dn

    # 정규화: 윈도우 마지막 close 기준 차감 (마지막 캔들 C = 0 anchor)
    anchor = log_c[:, -1:]
    hocl = torch.stack(
        [log_h - anchor, log_o - anchor, log_c - anchor, log_l - anchor], dim=-1
    )

    # 거래량: |증분| 비례 + lognormal 잡음 → log + z-norm
    vol_noise = torch.randn(n_samples, n, generator=gen) * 0.3
    log_v = torch.log(0.5 + delta.abs() / mean_step) + vol_noise
    log_v = (log_v - log_v.mean(dim=-1, keepdim=True)) / log_v.std(dim=-1, keepdim=True).clamp_min(1e-8)

    return hocl, log_v.unsqueeze(-1)


def make_candles(
    kind: str,
    n: int = 64,
    n_samples: int = 300,
    noise: float = 0.1,
    seed: int = 0,
    total_log_move: float = 0.3,
    wick_scale: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """모멘텀 형상 부류의 증분을 캔들로 전개. candles_from_increments 참조."""
    inc = make_increments(kind, n=n, n_samples=n_samples, noise=noise, seed=seed)
    delta = inc[..., 0] * total_log_move
    return candles_from_increments(
        delta, mean_step=total_log_move / n, seed=seed, wick_scale=wick_scale
    )


def make_ohlcv_dataset(
    n: int = 64,
    n_samples: int = 300,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3 부류 OHLCV 합본. 반환: (hocl, v, y)."""
    hs, vs, ys = [], [], []
    for i, kind in enumerate(CLASSES):
        h, v = make_candles(kind, n=n, n_samples=n_samples, noise=noise, seed=seed + i)
        hs.append(h)
        vs.append(v)
        ys.append(torch.full((n_samples,), i, dtype=torch.long))
    return torch.cat(hs), torch.cat(vs), torch.cat(ys)
