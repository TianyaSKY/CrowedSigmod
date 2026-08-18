"""步长 4 的非负密度头。"""

from __future__ import annotations

import torch
from torch import nn

from .yolo_encoder import ConvBNAct


class DensityHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_channels: int = 64, bottleneck_channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, 3),
            ConvBNAct(hidden_channels, bottleneck_channels, 3),
        )
        self.output = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)
        self.activation = nn.Softplus()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.output(self.features(features)))
