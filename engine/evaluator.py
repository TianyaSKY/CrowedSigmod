"""Validation metrics for full-image crowd counting."""

from __future__ import annotations

from typing import Iterable

import torch

from inference.tiled_inference import DensityTiler


def counting_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = predicted.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    if predicted.numel() != target.numel() or not predicted.numel():
        raise ValueError("predicted and target counts must be non-empty and equally sized")
    error = predicted - target
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "nae": float((error.abs() / target.clamp_min(1.0)).mean()),
    }


def evaluate_tiled(
    model: torch.nn.Module,
    samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    tiler: DensityTiler | None = None,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Evaluate deterministic full images; each sample is ``(image, count_gt)``."""

    tiler = tiler or DensityTiler()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for image, count_gt in samples:
        result = tiler(model, image, device=device)
        predictions.append(result.count.cpu())
        targets.append(torch.as_tensor(count_gt).reshape(1).cpu())
    return counting_metrics(torch.cat(predictions), torch.cat(targets))
