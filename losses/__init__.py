"""Crowd counting losses."""

from .crowd_loss import CrowdLoss, LossWeights, global_count_loss

__all__ = ["CrowdLoss", "LossWeights", "global_count_loss"]
