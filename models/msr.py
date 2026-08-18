"""多尺度残差（MSR）密度细化模块。"""

from __future__ import annotations

import torch
from torch import nn

from .yolo_encoder import ConvBNAct


class MSRBlock(nn.Module):
    """并行空洞卷积，随后进行残差投影。"""

    def __init__(
        self,
        channels: int = 128,
        branch_channels: int = 48,
        dilations: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must contain positive integers")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, 3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.SiLU(inplace=True),
                )
                for dilation in dilations
            ]
        )
        self.project = ConvBNAct(branch_channels * len(dilations), channels, 1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([branch(features) for branch in self.branches], dim=1)
        return self.activation(self.project(merged) + features)


class MSRRefinement(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 3, dilations: tuple[int, ...] = (1, 2, 3)) -> None:
        super().__init__()
        if blocks < 0:
            raise ValueError("blocks must be non-negative")
        self.blocks = nn.Sequential(*(MSRBlock(channels, dilations=dilations) for _ in range(blocks)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.blocks(features)
