"""Interchangeable model architectures with the same contract.

Every model takes a (batch, 132, 30) window and returns a (batch, 1) fall probability, so the API,
the browser demo (ONNX) and the promotion pipeline do not depend on which one is the champion.
"""

from __future__ import annotations

import torch
from torch import nn

from fall_detection.model import FallCNN1D


class GRUNet(nn.Module):
    """One recurrent layer reading the window frame by frame; the last state is classified."""

    def __init__(self, input_channels: int, seq_length: int, hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x.transpose(1, 2))  # (batch, frames, hidden)
        return torch.sigmoid(self.fc(out[:, -1]))


class TCN(nn.Module):
    """Dilated temporal convolutions (receptive field covers the whole window), mean pooling."""

    def __init__(self, input_channels: int, seq_length: int, width: int = 32):
        super().__init__()
        layers, channels = [], input_channels
        for dilation in (1, 2, 4):
            layers += [
                nn.Conv1d(channels, width, 3, padding=dilation, dilation=dilation),
                nn.ReLU(),
            ]
            channels = width
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(self.body(x).mean(-1)))


class TinyTransformer(nn.Module):
    """Two self-attention layers over the frames, learned positions, mean pooling."""

    def __init__(self, input_channels: int, seq_length: int, d_model: int = 32, layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(input_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.transpose(1, 2)) + self.pos[:, : x.shape[2]]
        return torch.sigmoid(self.fc(self.encoder(h).mean(1)))


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "cnn1d": FallCNN1D,
    "gru": GRUNet,
    "tcn": TCN,
    "transformer": TinyTransformer,
}
DEFAULT_ARCH = "cnn1d"


def build_model(arch: str = DEFAULT_ARCH, input_channels: int = 132, seq_length: int = 30):
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}, choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](input_channels=input_channels, seq_length=seq_length)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
