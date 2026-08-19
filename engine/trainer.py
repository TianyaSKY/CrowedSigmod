"""围绕共享的模型/损失接口构建的最小训练循环。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        # non_blocking=True 在 CUDA 上发起异步拷贝，可与后续前向计算
        # 重叠；键可能缺失（训练与验证的标注组合不同），缺失时跳过即可。
        for key in ("image", "probability_gt", "density_gt", "count_gt"):
            if key in moved:
                moved[key] = moved[key].to(self.device, non_blocking=True)
        return moved

    def train_epoch(
        self,
        loader: DataLoader,
        *,
        epoch: int | None = None,
        total_epochs: int | None = None,
        show_pbar: bool = True,
    ) -> dict[str, float]:
        self.model.train()
        # model.train() 会递归把含冻结骨干在内的所有模块切回训练模式，
        # 因此必须紧接着把骨干 BN 重新压回 eval——顺序不能颠倒，否则
        # 冻结阶段的运行统计会在当前 epoch 被小 batch 更新。
        self.freeze_scheduler.enforce_batch_norm_state(self.model)
        running: dict[str, float] = {}
        steps = 0

        desc = (
            f"Epoch [{epoch + 1:03d}/{total_epochs:03d}]"
            if epoch is not None and total_epochs is not None
            else "Training"
        )
        pbar = (
            tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
            if show_pbar
            else loader
        )

        for batch in pbar:
            batch = self._move_batch(batch)
            # set_to_none=True 把梯度置为 None 而非零张量：省去逐元素
            # 清零的写操作，释放旧梯度内存，反向传播时会按需重新分配。
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch["image"])
            details = self.criterion.compute(outputs, batch)  # type: ignore[attr-defined]
            details["total"].backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            for key, value in details.items():
                # 只累计标量且有限的项（跳过 NaN/Inf 与向量项），
                # 最后按步数取平均，得到每 epoch 的损失/指标汇总。
                if value.ndim == 0 and torch.isfinite(value):
                    running[key] = running.get(key, 0.0) + float(value.detach())
            steps += 1

            if show_pbar and hasattr(pbar, "set_postfix"):
                postfix = {"loss": f"{float(details['total'].detach()):.4f}"}
                if "count" in details:
                    postfix["count_loss"] = f"{float(details['count'].detach()):.4f}"
                pbar.set_postfix(postfix)

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
        writer: SummaryWriter | None = None,
        show_pbar: bool = True,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for epoch in range(self.start_epoch, int(epochs)):
            phase = self.freeze_scheduler.apply(self.model, epoch)
            # 新启用的骨干网络参数需要新的参数组。显式重建可确保
            # 在阶段切换边界处学习率乘数保持正确。
            if epoch == self.start_epoch or phase.value != getattr(self, "_last_phase", None):
                self.optimizer = build_optimizer(self.model, base_lr=self.optimizer.defaults["lr"])
            self._last_phase = phase.value
            if scheduler is not None:
                scheduler.step()
            else:
                # 未传入外部 scheduler 时使用内置 warmup+cosine；
                # 二选一执行，避免对学习率做双重缩放。
                apply_warmup_cosine(self.optimizer, epoch=epoch, total_epochs=epochs, warmup_epochs=warmup_epochs)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch + 1:03d}/{int(epochs):03d}] start | Phase: {phase.value} | LR: {current_lr:.6e}"
            )

            metrics = self.train_epoch(
                loader,
                epoch=epoch,
                total_epochs=int(epochs),
                show_pbar=show_pbar,
            )
            metrics["epoch"] = float(epoch)
            history.append(metrics)

            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k != "epoch")
            logger.info(f"Epoch [{epoch + 1:03d}/{int(epochs):03d}] result | {metric_str}")

            if writer is not None:
                for key, val in metrics.items():
                    if key != "epoch":
                        writer.add_scalar(f"train/{key}", val, epoch + 1)
                writer.add_scalar("train/lr", current_lr, epoch + 1)
                writer.flush()

            if checkpoint_dir is not None:
                ckpt_dir = Path(checkpoint_dir)
                ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
                self.save_checkpoint(ckpt_path, epoch, metrics)
                logger.info(f"Saved checkpoint -> {ckpt_path}")
                metrics_json = ckpt_dir / "metrics.json"
                metrics_json.write_text(json.dumps(history, indent=2, ensure_ascii=False))
        return history

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 快照内容：模型权重、优化器状态（Adam 的动量/方差，续训必需）、
        # 冻结调度器状态（恢复后阶段与解冻边界一致）、当前 epoch 与指标。
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
            # 恢复优化器状态以无缝续训；旧版 checkpoint 可能没有该键，判空兼容。
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        # 从断点后的下一个 epoch 继续训练，配合 fit 的 range(start_epoch, epochs)。
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        return checkpoint
