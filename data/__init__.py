"""数据加载与标签生成工具。

对外公开的模块级接口：
- CrowdDataset / crowd_collate：训练与评估的数据入口及批次整理；
- TargetConfig 与 generate_* / validate_density_conservation：标签图
  生成与守恒校验；
- crop_points：半开区间点裁剪，评估分块与数据增广中复用。
"""

from .crowd_dataset import CrowdDataset, CrowdRecord, crowd_collate
from .target_generator import (
    TargetConfig,
    crop_points,
    generate_density_target,
    generate_probability_target,
    generate_targets,
    validate_density_conservation,
)

__all__ = [
    "CrowdDataset",
    "CrowdRecord",
    "crowd_collate",
    "TargetConfig",
    "crop_points",
    "generate_density_target",
    "generate_probability_target",
    "generate_targets",
    "validate_density_conservation",
]
