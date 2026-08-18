"""Probability, density and count-consistency losses for YOLO-PGMD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    probability: float = 1.0
    density: float = 1.0
    count: float = 0.5
    local: float = 0.25
    local_grid: int = 4


class CrowdLoss(nn.Module):
    """Compute the weighted four-term objective.

    ``forward`` returns a scalar tensor, matching normal PyTorch criteria.
    ``compute`` additionally exposes named live tensors for logging.
    """

    def __init__(
        self,
        *,
        probability_weight: float = 1.0,
        density_weight: float = 1.0,
        count_weight: float = 0.5,
        local_weight: float = 0.25,
        local_grid: int = 4,
        dice_weight: float = 0.2,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if local_grid <= 0:
            raise ValueError("local_grid must be positive")
        self.weights = LossWeights(
            probability=probability_weight,
            density=density_weight,
            count=count_weight,
            local=local_weight,
            local_grid=local_grid,
        )
        self.dice_weight = float(dice_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)

    @staticmethod
    def _as_count(count: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        count = torch.as_tensor(count, device=device, dtype=dtype).reshape(-1)
        if count.numel() != batch_size:
            raise ValueError(f"count target has {count.numel()} values for batch size {batch_size}")
        return count

    def _local_region_sums(self, density: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = density.shape
        grid = self.weights.local_grid
        if height % grid or width % grid:
            raise ValueError(f"density shape {(height, width)} is not divisible by local_grid={grid}")
        return density.reshape(batch, channels, grid, height // grid, grid, width // grid).sum(dim=(1, 3, 5))

    def compute(self, outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        probability = outputs["probability"]
        density = outputs["density"]
        probability_gt = targets["probability_gt"].to(device=probability.device, dtype=probability.dtype)
        density_gt = targets["density_gt"].to(device=density.device, dtype=density.dtype)
        if probability.shape != probability_gt.shape:
            raise ValueError(f"probability shape {tuple(probability.shape)} != target {tuple(probability_gt.shape)}")
        if density.shape != density_gt.shape:
            raise ValueError(f"density shape {tuple(density.shape)} != target {tuple(density_gt.shape)}")
        count_gt = self._as_count(targets["count_gt"], density.shape[0], density.device, density.dtype)

        probability_clamped = probability.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(probability_clamped, probability_gt)
        intersection = (probability * probability_gt).flatten(1).sum(dim=1)
        denominator = probability.flatten(1).sum(dim=1) + probability_gt.flatten(1).sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        probability_loss = bce + self.dice_weight * dice

        density_loss = F.smooth_l1_loss(density, density_gt, beta=self.smooth_l1_beta)
        predicted_count = density.flatten(1).sum(dim=1)
        count_loss = ((predicted_count - count_gt).abs() / (count_gt + 1.0)).mean()

        predicted_local = self._local_region_sums(density)
        target_local = self._local_region_sums(density_gt)
        local_loss = ((predicted_local - target_local).abs() / (target_local + 1.0)).mean()

        total = (
            self.weights.probability * probability_loss
            + self.weights.density * density_loss
            + self.weights.count * count_loss
            + self.weights.local * local_loss
        )
        return {
            "total": total,
            "loss": total,
            "probability": probability_loss,
            "density": density_loss,
            "count": count_loss,
            "local": local_loss,
            "predicted_count": predicted_count.detach(),
        }

    def forward(self, outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.compute(outputs, targets)["total"]


def global_count_loss(predicted_density: torch.Tensor, count_gt: torch.Tensor) -> torch.Tensor:
    """Standalone normalized global count loss for experiments."""

    predicted_count = predicted_density.flatten(1).sum(dim=1)
    count_gt = count_gt.to(device=predicted_density.device, dtype=predicted_density.dtype).reshape(-1)
    return ((predicted_count - count_gt).abs() / (count_gt + 1.0)).mean()
