"""人群计数器模型组件：对外统一暴露骨干、融合、注意力、细化与计数 API，
供训练脚本与推理脚本从 models 顶层导入。"""

from .attention import ECAChannelAttention, ProbabilityGuidedAttention
from .crowd_counter import CrowdCounter
from .density_head import DensityHead
from .feature_fusion import MultiScaleFusion
from .msr import MSRBlock, MSRRefinement
from .probability_head import ProbabilityHead
from .yolo_encoder import YOLOBackbone

# 显式列出公共 API：既约束 `from models import *` 的导出面，
# 也避免把各模块内部实现细节（如各类辅助模块）泄漏给调用方
__all__ = [
    "CrowdCounter",
    "DensityHead",
    "ECAChannelAttention",
    "MSRBlock",
    "MSRRefinement",
    "MultiScaleFusion",
    "ProbabilityGuidedAttention",
    "ProbabilityHead",
    "YOLOBackbone",
]
