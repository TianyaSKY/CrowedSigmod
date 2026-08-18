"""训练、冻结与验证辅助工具。"""

# trainer 依赖 freeze_scheduler 完成阶段切换与优化器分组，
# schedules 提供学习率调度，evaluator 提供整图验证指标；
# 以下按模块聚合对外导出，供训练脚本统一从 engine 包导入。
from .evaluator import counting_metrics, evaluate_tiled
from .freeze_scheduler import FreezeScheduler, TrainingPhase, build_optimizer
from .schedules import apply_warmup_cosine, warmup_cosine_factor
from .trainer import CrowdTrainer

# __all__ 固定公共 API：星号导入只暴露这些符号，屏蔽各模块内部实现细节。
__all__ = [
    "CrowdTrainer",
    "FreezeScheduler",
    "TrainingPhase",
    "apply_warmup_cosine",
    "build_optimizer",
    "counting_metrics",
    "evaluate_tiled",
    "warmup_cosine_factor",
]
