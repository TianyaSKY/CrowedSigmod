"""多尺度残差（MSR）密度细化模块。"""

from __future__ import annotations

import torch
from torch import nn

from .yolo_encoder import ConvBNAct


class MSRBlock(nn.Module):
    """多尺度残差块：不同空洞率的并行 3×3 卷积捕获多档感受野，
    拼接后经 1×1 投影压回原通道数并与输入残差相加。"""

    def __init__(
        self,
        channels: int = 128,
        branch_channels: int = 48,
        dilations: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()
        # 空洞率必须为正；每个分支对应一档等效感受野（rate 1/2/3 ≈ 3×3/5×5/7×7），
        # 并行覆盖从单人到密集人群的尺度范围
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must contain positive integers")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    # padding=dilation 与空洞卷积配套，保证各分支输出分辨率不变
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
        # 各分支输出沿通道拼接（宽度 = branch_channels × 分支数），
        # 由 1×1 投影降回 channels 后与输入恒等相加，激活置于相加之后
        merged = torch.cat([branch(features) for branch in self.branches], dim=1)
        return self.activation(self.project(merged) + features)


class MSRRefinement(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 3, dilations: tuple[int, ...] = (1, 2, 3)) -> None:
        super().__init__()
        # 允许 0 块：此时模块退化为恒等，方便消融时整体旁路 MSR 细化
        if blocks < 0:
            raise ValueError("blocks must be non-negative")
        self.blocks = nn.Sequential(*(MSRBlock(channels, dilations=dilations) for _ in range(blocks)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.blocks(features)
