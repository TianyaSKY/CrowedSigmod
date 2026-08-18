"""Backbone freeze schedule and optimizer parameter groups."""

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
    for parameter in module.parameters():
        parameter.requires_grad = trainable
    if not trainable:
        module.eval()


def _set_batch_norm_eval(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            child.weight.requires_grad = False
            child.bias.requires_grad = False


class FreezeScheduler:
    """Apply frozen → partially unfrozen → fully unfrozen transitions."""

    def __init__(self, freeze_epochs: int = 10, partial_unfreeze_epoch: int = 10, full_unfreeze_epoch: int = 30) -> None:
        if not (0 <= freeze_epochs <= partial_unfreeze_epoch <= full_unfreeze_epoch):
            raise ValueError("freeze_epochs <= partial_unfreeze_epoch <= full_unfreeze_epoch is required")
        self.freeze_epochs = int(freeze_epochs)
        self.partial_unfreeze_epoch = int(partial_unfreeze_epoch)
        self.full_unfreeze_epoch = int(full_unfreeze_epoch)
        self.phase = TrainingPhase.FROZEN

    def phase_for_epoch(self, epoch: int) -> TrainingPhase:
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
            for parameter in high:
                parameter.requires_grad = True
        elif phase is TrainingPhase.FULL:
            for parameter in low + high:
                parameter.requires_grad = True

        # All newly initialized heads remain trainable in every phase.
        for name, module in model.named_children():
            if name != "backbone":
                _set_module_trainable(module, True)
        # A frozen backbone must not update running BN statistics.  Applying this
        # after model.train() keeps the invariant true at every phase transition.
        self.enforce_batch_norm_state(model)
        return phase

    def enforce_batch_norm_state(self, model: nn.Module) -> None:
        """Re-apply frozen-backbone BN eval after an outer ``model.train()``."""

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
    """Create AdamW groups for heads, neck, and backbone stages."""

    low, high = model.backbone.backbone_stage_parameters()
    low_set = {id(parameter) for parameter in low}
    high_set = {id(parameter) for parameter in high}
    backbone_ids = low_set | high_set
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
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
