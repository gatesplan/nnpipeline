"""Autoencoder / Forecast wrapper. OHLCVReceptor 라이브러리 사용."""
from __future__ import annotations

import torch
from torch import nn

from nnpipeline import OHLCVReceptor


class CandleAutoencoder(nn.Module):
    """OHLCVReceptor + decoder. 윈도우 전체 재구성."""

    def __init__(self, decoder_hidden: int = 32):
        super().__init__()
        self.receptor = OHLCVReceptor()
        self.decoder = nn.Sequential(
            nn.Linear(3, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 5),
        )

    def encode(self, hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.receptor(hocl, v)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, hocl: torch.Tensor, v: torch.Tensor):
        z = self.encode(hocl, v)
        recon = self.decode(z)
        return recon, z


class CandleForecaster(nn.Module):
    """OHLCVReceptor (per-candle) + 간단 head로 다음 캔들 예측."""

    def __init__(self, window: int, head_hidden: int = 32):
        super().__init__()
        self.window = window
        self.receptor = OHLCVReceptor()
        self.head = nn.Sequential(
            nn.Linear(window * 3, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 5),
        )

    def forward(self, hocl: torch.Tensor, v: torch.Tensor):
        z = self.receptor(hocl, v)
        z_flat = z.reshape(z.shape[0], -1)
        pred = self.head(z_flat)
        return pred, z
