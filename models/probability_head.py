"""轻量级人群概率头：两段卷积压缩特征，sigmoid 输出逐像素人群存在概率。"""

from __future__ import annotations

import torch
from torch import nn

from .yolo_encoder import ConvBNAct


class ProbabilityHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_channels: int = 64, bottleneck_channels: int = 32) -> None:
        super().__init__()
        # 两段 3×3 卷积把特征逐级压缩到最窄的瓶颈宽度（padding 保持分辨率不变），
        # 在保留局部上下文的同时控制头部参数规模
        self.features = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, 3),
            ConvBNAct(hidden_channels, bottleneck_channels, 3),
        )
        # 1×1 卷积作逐像素分类器，输出单通道 logits
        self.logit = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # sigmoid 把 logits 压到 [0,1]，得到逐像素存在概率，
        # 供空间注意力作为外部先验图使用
        return torch.sigmoid(self.logit(self.features(features)))
