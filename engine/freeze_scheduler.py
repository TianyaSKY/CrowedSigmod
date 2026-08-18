"""骨干网络冻结调度与优化器参数分组。"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import torch
from torch import nn


class TrainingPhase(str, Enum):
    FROZEN = "frozen"
    PARTIAL = "partial"
    FULL = "full"


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    # 冻结时同步切换 eval：除 BN 统计外，也避免 dropout 等随机层
    # 在特征提取阶段引入不确定性，保证冻结行为与推理一致。
    for parameter in module.parameters():
        parameter.requires_grad = trainable
    if not trainable:
        module.eval()


def _set_batch_norm_eval(module: nn.Module) -> None:
    # 冻结阶段的 batch 通常很小（只有头部在训练），若 BN 继续按当前
    # batch 更新运行统计，会以小样本估计污染预训练统计量，故强制 eval。
    # affine 参数（weight/bias）一并固定：归一化统计已冻结，再更新它们
    # 会使尺度/偏移与统计量不一致。
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            child.weight.requires_grad = False
            child.bias.requires_grad = False


class FreezeScheduler:
    """应用冻结 → 部分解冻 → 完全解冻的过渡。"""

    def __init__(self, freeze_epochs: int = 10, partial_unfreeze_epoch: int = 10, full_unfreeze_epoch: int = 30) -> None:
        if not (0 <= freeze_epochs <= partial_unfreeze_epoch <= full_unfreeze_epoch):
            raise ValueError("freeze_epochs <= partial_unfreeze_epoch <= full_unfreeze_epoch is required")
        self.freeze_epochs = int(freeze_epochs)
        self.partial_unfreeze_epoch = int(partial_unfreeze_epoch)
        self.full_unfreeze_epoch = int(full_unfreeze_epoch)
        self.phase = TrainingPhase.FROZEN

    def phase_for_epoch(self, epoch: int) -> TrainingPhase:
        # FROZEN：骨干完全冻结，仅头部用预训练特征快速收敛；
        # PARTIAL：只解冻高层参数，低层通用特征仍冻结，微调代价最小；
        # FULL：全部参数放开，进入整体微调。
        if epoch < self.partial_unfreeze_epoch:
            return TrainingPhase.FROZEN
        if epoch < self.full_unfreeze_epoch:
            return TrainingPhase.PARTIAL
        return TrainingPhase.FULL

    @staticmethod
    def _backbone_parts(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        if not hasattr(model, "backbone"):
            raise AttributeError("model must expose a backbone module")
        backbone = model.backbone
        if hasattr(backbone, "backbone_stage_parameters"):
            return backbone.backbone_stage_parameters()
        # 兜底启发式：没有按 stage 分组的接口时，按参数声明顺序对半
        # 划分，近似区分低层/高层，保证 PARTIAL 阶段仍只解冻后半段。
        parameters = list(backbone.parameters())
        midpoint = max(1, len(parameters) // 2)
        return parameters[:midpoint], parameters[midpoint:]

    def apply(self, model: nn.Module, epoch: int) -> TrainingPhase:
        phase = self.phase_for_epoch(epoch)
        self.phase = phase
        model.train()
        backbone = model.backbone
        low, high = self._backbone_parts(model)
        for parameter in low + high:
            parameter.requires_grad = False
        if phase is TrainingPhase.PARTIAL:
            # 只解冻高层：高层特征与计数任务语义更相关，低层保留预训练
            # 的通用边缘/纹理特征；高层学习率由 build_optimizer 中的
            # backbone_high_multiplier（0.1）单独压低。
            for parameter in high:
                parameter.requires_grad = True
        elif phase is TrainingPhase.FULL:
            for parameter in low + high:
                parameter.requires_grad = True

        # 所有新建的头部在每一阶段都保持可训练。
        for name, module in model.named_children():
            if name != "backbone":
                _set_module_trainable(module, True)
        # 冻结的骨干网络不得更新运行中的 BN 统计量。在 model.train() 之后
        # 应用此设置，可在每个阶段切换时保持该不变式成立。
        self.enforce_batch_norm_state(model)
        return phase

    def enforce_batch_norm_state(self, model: nn.Module) -> None:
        """在外部调用 ``model.train()`` 后，重新对冻结的骨干网络 BN 应用 eval 模式。"""

        # FULL 阶段骨干 BN 恢复正常训练，统计量随微调同步更新；
        # 其余阶段必须保持 eval，防止小 batch 污染预训练统计。
        if self.phase is not TrainingPhase.FULL:
            _set_batch_norm_eval(model.backbone)

    def state_dict(self) -> dict[str, int | str]:
        return {
            "freeze_epochs": self.freeze_epochs,
            "partial_unfreeze_epoch": self.partial_unfreeze_epoch,
            "full_unfreeze_epoch": self.full_unfreeze_epoch,
            "phase": self.phase.value,
        }


def _unique_parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    # 按对象身份去重并剔除冻结参数：同一 Parameter 可能同时落在多个
    # 集合（如 head 与骨干划分重叠），交给优化器前必须唯一，否则
    # AdamW 会对同一权重重复更新。
    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for parameter in parameters:
        if id(parameter) not in seen and parameter.requires_grad:
            result.append(parameter)
            seen.add(id(parameter))
    return result


def build_optimizer(
    model: nn.Module,
    *,
    base_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    backbone_high_multiplier: float = 0.1,
    backbone_low_multiplier: float = 0.03,
) -> torch.optim.Optimizer:
    """为头部、颈部与骨干网络各阶段创建 AdamW 参数组。"""

    low, high = model.backbone.backbone_stage_parameters()
    low_set = {id(parameter) for parameter in low}
    high_set = {id(parameter) for parameter in high}
    backbone_ids = low_set | high_set
    # 头部用基础学习率快速拟合；骨干是预训练权重，只需小步微调，且低层
    # 比高层更低（0.03 vs 0.1），分组便于各阶段独立控制学习率乘数。
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]
    groups = []
    if head_parameters:
        groups.append({"params": _unique_parameters(head_parameters), "lr": base_lr})
    high_parameters = _unique_parameters(parameter for parameter in high if parameter.requires_grad)
    if high_parameters:
        groups.append({"params": high_parameters, "lr": base_lr * backbone_high_multiplier})
    low_parameters = _unique_parameters(parameter for parameter in low if parameter.requires_grad)
    if low_parameters:
        groups.append({"params": low_parameters, "lr": base_lr * backbone_low_multiplier})
    if not groups:
        raise ValueError("no trainable parameters available for optimizer")
    optimizer = torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)
    # 记录各组自身的初始学习率：warmup/cosine 调度以 initial_lr 为基准
    # 缩放，使头部/骨干各组的学习率比例在训练全程保持不变。阶段切换时
    # 新增解冻的参数此前被 requires_grad 过滤在组外，需重建优化器
    # 才能获得参数组与 AdamW 状态。
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
