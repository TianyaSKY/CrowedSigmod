"""ECA 通道注意力与概率引导的空间注意力。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ECAChannelAttention(nn.Module):
    """无两层压缩瓶颈的高效通道注意力（Efficient Channel Attention）。"""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.channels = channels

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.pool(features).squeeze(-1).transpose(1, 2)
        weights = self.conv(weights).transpose(1, 2).unsqueeze(-1).sigmoid()
        return features * weights, weights


class ProbabilityGuidedAttention(nn.Module):
    """将 ECA 与空间统计量及外部概率图融合。"""

    def __init__(self, channels: int = 128, spatial_kernel: int = 7) -> None:
        super().__init__()
        if spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be odd")
        self.channel = ECAChannelAttention(channels)
        self.spatial = nn.Conv2d(3, 1, spatial_kernel, padding=spatial_kernel // 2, bias=True)
        # 初始化为恒等变换：第一阶段中，预训练/冻结的特征
        # 不会被随机的空间门控相乘。
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channel_features, _ = self.channel(features)
        if probability.shape[-2:] != channel_features.shape[-2:]:
            probability = F.interpolate(probability, size=channel_features.shape[-2:], mode="bilinear", align_corners=False)
        average = channel_features.mean(dim=1, keepdim=True)
        maximum = channel_features.amax(dim=1, keepdim=True)
        spatial = torch.sigmoid(self.spatial(torch.cat((probability, average, maximum), dim=1)))
        attended = channel_features * (1.0 + self.alpha * spatial)
        return attended, spatial
