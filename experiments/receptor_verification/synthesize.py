"""합성 캔들 생성기.

연구계획 §4에 정의된 factor of variation을 기반으로 캔들을 생성한다.

Factors:
- body_center (c): 가격 레벨 (절대값 의미는 없음, 정규화 가정)
- body_magnitude (m): |C - O|
- body_direction (d): +1 (bull) or -1 (bear)
- upper_wick (u): H - max(O, C)
- lower_wick (l): min(O, C) - L

매핑:
- O = c - d * m / 2
- C = c + d * m / 2
- H = max(O, C) + u
- L = min(O, C) - l
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class CandleFactors:
    """Ground truth factor 텐서 묶음. shape: (N,) 모두 동일."""

    center: torch.Tensor       # c
    magnitude: torch.Tensor    # m
    direction: torch.Tensor    # d in {-1, +1}
    upper_wick: torch.Tensor   # u
    lower_wick: torch.Tensor   # l

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "center": self.center,
            "magnitude": self.magnitude,
            "direction": self.direction,
            "upper_wick": self.upper_wick,
            "lower_wick": self.lower_wick,
        }


def factors_to_hocl(factors: CandleFactors) -> torch.Tensor:
    """factor → HOCL (N, 4) 텐서. 채널 순서: H, O, C, L."""
    c, m, d, u, l = (
        factors.center,
        factors.magnitude,
        factors.direction,
        factors.upper_wick,
        factors.lower_wick,
    )
    O = c - d * m / 2
    C = c + d * m / 2
    upper_body = torch.maximum(O, C)
    lower_body = torch.minimum(O, C)
    H = upper_body + u
    L = lower_body - l
    return torch.stack([H, O, C, L], dim=-1)


def sample_random_candles(
    n: int,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, CandleFactors]:
    """무작위 캔들 n개 생성. 연구계획 §4.1 분포를 따른다."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    center = torch.rand(n, generator=g) * 100.0          # U(0, 100)
    magnitude = torch.rand(n, generator=g) * 5.0         # U(0, 5)
    direction_bits = torch.randint(0, 2, (n,), generator=g)
    direction = (direction_bits * 2 - 1).float()         # {-1, +1}
    upper_wick = torch.rand(n, generator=g) * 5.0        # U(0, 5)
    lower_wick = torch.rand(n, generator=g) * 5.0        # U(0, 5)

    factors = CandleFactors(
        center=center.to(device),
        magnitude=magnitude.to(device),
        direction=direction.to(device),
        upper_wick=upper_wick.to(device),
        lower_wick=lower_wick.to(device),
    )
    hocl = factors_to_hocl(factors)
    return hocl, factors


CANONICAL_DEFS = {
    # 각 패턴: (m, |d| (fixed), u, l)
    # direction은 패턴마다 fix 또는 random
    "Marubozu_Bull":    dict(m=4.0, d=+1, u=0.0, l=0.0),
    "Marubozu_Bear":    dict(m=4.0, d=-1, u=0.0, l=0.0),
    "Hammer":           dict(m=1.0, d=+1, u=0.0, l=4.0),
    "Inverted_Hammer":  dict(m=1.0, d=+1, u=4.0, l=0.0),
    "Shooting_Star":    dict(m=1.0, d=-1, u=4.0, l=0.0),
    "Doji":             dict(m=0.05, d=+1, u=1.0, l=1.0),
    "Long_Legged_Doji": dict(m=0.05, d=+1, u=3.0, l=3.0),
    "Spinning_Top":     dict(m=1.0, d=+1, u=2.0, l=2.0),
}


def sample_canonical_candles(
    per_pattern: int = 100,
    *,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, CandleFactors, list[str]]:
    """캐노니컬 패턴 8종 × per_pattern. center만 무작위.

    Returns
    -------
    hocl : (8*per_pattern, 4)
    factors : CandleFactors (동일 shape)
    labels : list[str] of length 8*per_pattern, 패턴 이름
    """
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)

    centers, mags, dirs, ups, lows, labels = [], [], [], [], [], []
    for name, spec in CANONICAL_DEFS.items():
        c = torch.rand(per_pattern, generator=g) * 100.0
        m = torch.full((per_pattern,), float(spec["m"]))
        d = torch.full((per_pattern,), float(spec["d"]))
        u = torch.full((per_pattern,), float(spec["u"]))
        l = torch.full((per_pattern,), float(spec["l"]))
        centers.append(c)
        mags.append(m)
        dirs.append(d)
        ups.append(u)
        lows.append(l)
        labels.extend([name] * per_pattern)

    factors = CandleFactors(
        center=torch.cat(centers).to(device),
        magnitude=torch.cat(mags).to(device),
        direction=torch.cat(dirs).to(device),
        upper_wick=torch.cat(ups).to(device),
        lower_wick=torch.cat(lows).to(device),
    )
    hocl = factors_to_hocl(factors)
    return hocl, factors, labels


def normalize_hocl(hocl: torch.Tensor) -> torch.Tensor:
    """Instance-wise 정규화 v2. 절대 가격 레벨 제거하되 H/L 변동성 보존.

    각 instance에 대해:
        c = (O + C) / 2                  (body center)
        s = max(H - L, eps)              (instance scale)
        out = (hocl - c) / s

    v1과 차이: center를 range center (H+L)/2 대신 body center (O+C)/2로.
    이러면 H_norm, L_norm이 instance마다 다른 값 (각각 upper/lower wick + body half)
    → receptor가 H, L 입력에서 의미 있는 신호를 받음.

    범위 (대략): H_norm ∈ [0, +1], L_norm ∈ [-1, 0], O_norm, C_norm ∈ [-0.5, +0.5]
    """
    H, O, C, L = hocl[..., 0], hocl[..., 1], hocl[..., 2], hocl[..., 3]
    center = (O + C) / 2
    scale = (H - L).clamp_min(1e-6)
    norm = (hocl - center.unsqueeze(-1)) / scale.unsqueeze(-1)
    return norm


def denormalize_hocl(
    norm: torch.Tensor, center: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """normalize_hocl의 역. center/scale 별도 보관해야 사용 가능."""
    return norm * scale.unsqueeze(-1) + center.unsqueeze(-1)
