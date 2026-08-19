"""端到端 YOLO-PGMD 人群计数器。"""

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
    """从固定裁剪区域预测概率、注意力、密度与人数。

    公共 forward 契约在训练与推理阶段刻意保持一致：同一份代码、同一份
    计算图，无需切换 train/eval 分支。人数始终等于非负密度图之和；
    不存在全连接的人数回归器，计数能力完全来自逐像素回归。
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
        # 整条流水线的工作分辨率：密度图相对原图下采样 4 倍（P2 网格）
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
        # 可学习长残差缩放系数：从 0 初始化，初始阶段严格等价于原结构，
        # 网络自适应学习需要直接透传多少 YOLO 融合特征给密度头，防止深层精化过程中的特征塌缩。
        self.density_residual_alpha = nn.Parameter(torch.zeros(1))
        self.use_probability = bool(use_probability)
        self.use_attention = bool(use_attention)
        self.use_msr = bool(use_msr)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # 输入契约 [B, 3, H, W]：在入口处校验，避免形状错误在深层网络中
        # 以难读的维度不匹配错误形式暴露
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")
        p2, p3, p4 = self.backbone(images)
        features = self.fusion(p2, p3, p4)
        if self.use_probability:
            probability = self.probability_head(features)
        else:
            # 消融开关：零图占位保持后续注意力分支的输入契约不变，
            # 全零概率使空间注意力退化为纯特征统计（均值/最大值）
            probability = torch.zeros(
                (features.shape[0], 1, features.shape[2], features.shape[3]),
                dtype=features.dtype,
                device=features.device,
            )
        if self.use_attention:
            attended, attention = self.attention(features, probability)
        else:
            # 关闭注意力时直接透传特征，并用全 1 图占位，
            # 保持输出字典的键与形状在所有消融配置下一致
            attended = features
            attention = torch.ones_like(probability)
        if self.use_msr:
            refined = self.refinement(attended)
        else:
            # 关闭 MSR 时跳过细化，验证多尺度残差模块的独立贡献
            refined = attended

        # 长残差跳跃连接：融合特征直接跨越注意力与 MSR 精化块，向密度头补充底层空间细节
        refined = refined + self.density_residual_alpha * features

        density = self.density_head(refined)
        # 人数 = 密度图全部像素求和：密度由 Softplus 保证非负，
        # 求和结果稳定且保留小数精度；无 FC 回归头（见类 docstring）
        count = density.flatten(1).sum(dim=1)
        return {
            "probability": probability,
            "attention": attention,
            "density": density,
            "count": count,
        }

    def inference_count(self, images: torch.Tensor) -> torch.Tensor:
        """保持相同前向路径的便捷封装。"""

        # 与训练共用 self.forward，保证推理人数与训练时学到的统计一致
        return self(images)["count"]
