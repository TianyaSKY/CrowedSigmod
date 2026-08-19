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
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化输出卷积层权重与偏置。
        
        权重使用小高斯分布 (std=0.01) 初始化以保证反向传播时密度头及上游长残差梯度即刻可导；
        偏置初始化为 -5.0，使得初始 Softplus(-5.0) ≈ 0.0067。
        在 160×160 网格上，初始全图积分预测约为 160×160×0.0067 ≈ 171 人，
        避免未显式初始化时 Softplus(0) ≈ 0.693 导致初始预测高达 17740 人的异常量级。
        """
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, -5.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 输出为连续非负密度；上层 count = ΣD 是实数期望值而非整数，
        # 刻意不做取整，以保留亚像素级的人数估计精度
        return self.activation(self.output(self.features(features)))
