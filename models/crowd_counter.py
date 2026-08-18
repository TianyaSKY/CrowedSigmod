"""End-to-end YOLO-PGMD crowd counter."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .attention import ProbabilityGuidedAttention
from .density_head import DensityHead
from .feature_fusion import MultiScaleFusion
from .msr import MSRRefinement
from .probability_head import ProbabilityHead
from .yolo_encoder import YOLOBackbone


class CrowdCounter(nn.Module):
    """Predict probability, attention, density and count from a fixed crop.

    The public forward contract is intentionally identical for training and
    inference.  Count is always the sum of the non-negative density map; there
    is no fully connected count regressor.
    """

    def __init__(
        self,
        *,
        backbone_name: str = "yolo11n.yaml",
        pretrained: str | Path | None = None,
        use_ultralytics: bool = True,
        fusion_channels: int = 128,
        projection_channels: int = 64,
        msr_blocks: int = 3,
        msr_dilations: tuple[int, ...] = (1, 2, 3),
        use_probability: bool = True,
        use_attention: bool = True,
        use_msr: bool = True,
    ) -> None:
        super().__init__()
        self.output_stride = 4
        self.backbone = YOLOBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            use_ultralytics=use_ultralytics,
        )
        self.fusion = MultiScaleFusion(
            self.backbone.feature_channels,
            projection_channels=projection_channels,
            fusion_channels=fusion_channels,
        )
        self.probability_head = ProbabilityHead(fusion_channels)
        self.attention = ProbabilityGuidedAttention(fusion_channels)
        self.refinement = MSRRefinement(fusion_channels, blocks=msr_blocks, dilations=msr_dilations)
        self.density_head = DensityHead(fusion_channels)
        self.use_probability = bool(use_probability)
        self.use_attention = bool(use_attention)
        self.use_msr = bool(use_msr)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")
        p2, p3, p4 = self.backbone(images)
        features = self.fusion(p2, p3, p4)
        if self.use_probability:
            probability = self.probability_head(features)
        else:
            probability = torch.zeros(
                (features.shape[0], 1, features.shape[2], features.shape[3]),
                dtype=features.dtype,
                device=features.device,
            )
        if self.use_attention:
            attended, attention = self.attention(features, probability)
        else:
            attended = features
            attention = torch.ones_like(probability)
        if self.use_msr:
            refined = self.refinement(attended)
        else:
            refined = attended
        density = self.density_head(refined)
        count = density.flatten(1).sum(dim=1)
        return {
            "probability": probability,
            "attention": attention,
            "density": density,
            "count": count,
        }

    def inference_count(self, images: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that preserves the same forward path."""

        return self(images)["count"]
