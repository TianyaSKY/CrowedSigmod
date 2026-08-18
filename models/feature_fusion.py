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
        if len(in_channels) != 3:
            raise ValueError("in_channels must contain P2, P3 and P4 channels")
        self.p2_projection = ConvBNAct(in_channels[0], projection_channels, 1)
        self.p3_projection = ConvBNAct(in_channels[1], projection_channels, 1)
        self.p4_projection = ConvBNAct(in_channels[2], projection_channels, 1)
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
        target_size = p2.shape[-2:]
        p3 = F.interpolate(self.p3_projection(p3), size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.p4_projection(p4), size=target_size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((p2, p3, p4), dim=1))
