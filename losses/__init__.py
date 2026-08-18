"""人群计数损失。"""

# 对外只暴露三个符号：CrowdLoss（训练主损失）、LossWeights（可覆盖的权重配置）、
# global_count_loss（实验用的独立全局人数损失）；实现细节（如 F、nn）不导出。
from .crowd_loss import CrowdLoss, LossWeights, global_count_loss

__all__ = ["CrowdLoss", "LossWeights", "global_count_loss"]
