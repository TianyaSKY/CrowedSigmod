"""步长 4 的非负密度头：Softplus 激活保证每个像素的密度值 ≥ 0，
从而可安全地对整张密度图求和得到人数。"""

from __future__ import annotations

import torch
from torch import nn

from .yolo_encoder import ConvBNAct


class DensityHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_channels: int = 64, bottleneck_channels: int = 32) -> None:
        super().__init__()
        # 与概率头同构的轻量两段压缩，输出单通道密度 logits
        self.features = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, 3),
            ConvBNAct(hidden_channels, bottleneck_channels, 3),
        )
        self.output = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)
        # 用 Softplus 而非 ReLU：同样保证非负，但处处光滑可导，
        # 0 附近梯度不截断，密度像素的数值稳定性更好
        self.activation = nn.Softplus()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 输出为连续非负密度；上层 count = ΣD 是实数期望值而非整数，
        # 刻意不做取整，以保留亚像素级的人数估计精度
        return self.activation(self.output(self.features(features)))
