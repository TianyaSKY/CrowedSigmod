"""Warmup plus cosine learning-rate scheduling."""

from __future__ import annotations

import math

import torch


def warmup_cosine_factor(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Return a scalar LR multiplier for the start of ``epoch``."""

    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if warmup_epochs < 0 or warmup_epochs > total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs]")
    if warmup_epochs and epoch < warmup_epochs:
        return max((epoch + 1) / warmup_epochs, 1e-6)
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(max(epoch - warmup_epochs, 0) / cosine_epochs, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 0,
) -> list[float]:
    """Apply the same schedule to all parameter groups, preserving LR ratios."""

    factor = warmup_cosine_factor(epoch, total_epochs, warmup_epochs)
    learning_rates: list[float] = []
    for group in optimizer.param_groups:
        base_lr = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = base_lr * factor
        learning_rates.append(group["lr"])
    return learning_rates
