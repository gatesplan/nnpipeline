"""검증용 wrapper. 핵심 receptor는 nnpipeline 라이브러리 사용.

OHLCVReceptor를 autoencoder 형태로 감싸 disentanglement 검증에 사용.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from nnpipeline import OHLCVReceptor


class CandleDecoder(nn.Module):
    """(..., 3) → (..., 5). receptor 출력을 HOCLV로 복원."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CandleAutoencoder(nn.Module):
    """OHLCVReceptor + CandleDecoder. autoencoder 검증용 wrapper."""

    def __init__(
        self,
        receptor_hidden: int = 2,
        receptor_side_dim: int = 2,
        receptor_hidden_v: int = 4,
        decoder_hidden: int = 32,
    ):
        super().__init__()
        self.receptor = OHLCVReceptor(
            hidden=receptor_hidden,
            side_dim=receptor_side_dim,
            hidden_v=receptor_hidden_v,
        )
        self.decoder = CandleDecoder(hidden=decoder_hidden)

    def encode(self, hocl_norm: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.receptor(hocl_norm, v)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, hocl_norm: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(hocl_norm, v)
        recon = self.decode(z)
        return recon, z
