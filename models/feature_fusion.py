"""P2/P3/P4 投影与步长 4 融合。"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .yolo_encoder import ConvBNAct, ResidualC2f


class MultiScaleFusion(nn.Module):
    """将所有 YOLO 尺度投影到统一网格，并在 P2 分辨率下进行融合。"""

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        *,
        projection_channels: int = 64,
        fusion_channels: int = 128,
    ) -> None:
        super().__init__()
        # 强制按 (P2, P3, P4) 顺序传入通道数，索引即尺度契约
        if len(in_channels) != 3:
            raise ValueError("in_channels must contain P2, P3 and P4 channels")
        # 三个 1×1 投影把各尺度通道统一到 projection_channels：
        # 先降通道再上采样，可显著减少后续双线性插值的计算量
        self.p2_projection = ConvBNAct(in_channels[0], projection_channels, 1)
        self.p3_projection = ConvBNAct(in_channels[1], projection_channels, 1)
        self.p4_projection = ConvBNAct(in_channels[2], projection_channels, 1)
        # 拼接后先用 1×1 把 3×projection_channels 压到 fusion_channels 降维，
        # 再由残差块做跨尺度非线性融合
        self.fuse = nn.Sequential(
            ConvBNAct(projection_channels * 3, fusion_channels, 1),
            ResidualC2f(fusion_channels),
        )

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> torch.Tensor:
        p2 = self.p2_projection(p2)
        # 以 P2（stride 4）为基准分辨率：P3/P4 分别是 stride 8/16，需上采样对齐
        target_size = p2.shape[-2:]
        # bilinear + align_corners=False 与检测/分割主流的像素网格对齐方式一致
        p3 = F.interpolate(self.p3_projection(p3), size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.p4_projection(p4), size=target_size, mode="bilinear", align_corners=False)
        # 通道维拼接三尺度特征，交由 fuse 降维并融合
        return self.fuse(torch.cat((p2, p3, p4), dim=1))
