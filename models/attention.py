"""ECA 通道注意力与概率引导的空间注意力。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ECAChannelAttention(nn.Module):
    """高效通道注意力（Efficient Channel Attention）：用一维卷积替代 SE 的
    两层 MLP 瓶颈，避免降维重建带来的信息损失，参数仅一个 1D 卷积核。"""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        # 要求奇数核：padding = kernel_size // 2 恰好使卷积不改变通道序列长度
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd")
        # 全局平均池化得到通道描述子后，用 1D 卷积建模相邻通道间的依赖
        # （核宽 k 即感受野覆盖 k 个通道），相比 SE 无降维、参数更少
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.channels = channels

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [B,C,1,1] → [B,C] → [B,1,C]：把通道维摆成序列送入 Conv1d，
        # 输出还原形状后经 sigmoid 得到 [0,1] 的逐通道权重
        weights = self.pool(features).squeeze(-1).transpose(1, 2)
        weights = self.conv(weights).transpose(1, 2).unsqueeze(-1).sigmoid()
        # 乘法门控缩放特征；同时返回权重图，便于可视化与消融分析
        return features * weights, weights


class ProbabilityGuidedAttention(nn.Module):
    """将 ECA 与空间统计量及外部概率图融合。"""

    def __init__(self, channels: int = 128, spatial_kernel: int = 7) -> None:
        super().__init__()
        if spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be odd")
        self.channel = ECAChannelAttention(channels)
        # 空间门控：输入是「概率图 + 通道均值 + 通道最大值」三通道拼接，
        # 用大核卷积融合局部空间邻域后输出单通道门控
        self.spatial = nn.Conv2d(3, 1, spatial_kernel, padding=spatial_kernel // 2, bias=True)
        # 初始化为恒等变换：第一阶段中，预训练/冻结的特征
        # 不会被随机的空间门控相乘。
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channel_features, _ = self.channel(features)
        # 概率图可能来自不同分辨率网格（如消融时用零占位图），显式上采样对齐
        if probability.shape[-2:] != channel_features.shape[-2:]:
            probability = F.interpolate(probability, size=channel_features.shape[-2:], mode="bilinear", align_corners=False)
        # 空间统计量：均值编码全局上下文，最大值突出强响应位置，
        # 与外部概率先验拼接成三通道输入
        average = channel_features.mean(dim=1, keepdim=True)
        maximum = channel_features.amax(dim=1, keepdim=True)
        spatial = torch.sigmoid(self.spatial(torch.cat((probability, average, maximum), dim=1)))
        # 乘法门控围绕恒等展开：alpha=0 时输出即 channel_features，
        # 训练初期注意力不影响梯度，强度由 alpha 的梯度逐步学出
        attended = channel_features * (1.0 + self.alpha * spatial)
        return attended, spatial
