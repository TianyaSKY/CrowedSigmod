"""Minimal training loop built around the shared model/loss interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .freeze_scheduler import FreezeScheduler, build_optimizer
from .schedules import apply_warmup_cosine


class CrowdTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
        base_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_scheduler: FreezeScheduler | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.freeze_scheduler = freeze_scheduler or FreezeScheduler()
        self.freeze_scheduler.apply(self.model, 0)
        self.optimizer = build_optimizer(
            self.model,
            base_lr=base_lr,
            weight_decay=weight_decay,
        )
        self.grad_clip_norm = grad_clip_norm
        self.start_epoch = 0

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = dict(batch)
        for key in ("image", "probability_gt", "density_gt", "count_gt"):
            if key in moved:
                moved[key] = moved[key].to(self.device, non_blocking=True)
        return moved

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        self.freeze_scheduler.enforce_batch_norm_state(self.model)
        running: dict[str, float] = {}
        steps = 0
        for batch in loader:
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch["image"])
            details = self.criterion.compute(outputs, batch)  # type: ignore[attr-defined]
            details["total"].backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            for key, value in details.items():
                if value.ndim == 0 and torch.isfinite(value):
                    running[key] = running.get(key, 0.0) + float(value.detach())
            steps += 1
        if steps == 0:
            raise ValueError("training loader is empty")
        return {key: value / steps for key, value in running.items()}

    def fit(
        self,
        loader: DataLoader,
        *,
        epochs: int,
        checkpoint_dir: str | Path | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        warmup_epochs: int = 0,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(self.start_epoch, int(epochs)):
            phase = self.freeze_scheduler.apply(self.model, epoch)
            # Newly enabled backbone parameters need a group. Rebuilding is
            # explicit and keeps LR multipliers correct at phase boundaries.
            if epoch == self.start_epoch or phase.value != getattr(self, "_last_phase", None):
                self.optimizer = build_optimizer(self.model, base_lr=self.optimizer.defaults["lr"])
            self._last_phase = phase.value
            if scheduler is not None:
                scheduler.step()
            else:
                apply_warmup_cosine(self.optimizer, epoch=epoch, total_epochs=epochs, warmup_epochs=warmup_epochs)
            metrics = self.train_epoch(loader)
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            if checkpoint_dir is not None:
                self.save_checkpoint(Path(checkpoint_dir) / f"epoch_{epoch:03d}.pt", epoch, metrics)
        return history

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": int(epoch),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics or {},
                "freeze_scheduler": self.freeze_scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        return checkpoint
