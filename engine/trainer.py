"""围绕共享的模型/损失接口构建的最小训练循环与验证/保存机制。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from inference.tiled_inference import DensityTiler
from .evaluator import counting_metrics, evaluate_tiled
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
        backbone_high_multiplier: float = 0.02,
        backbone_low_multiplier: float = 0.005,
        freeze_scheduler: FreezeScheduler | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.freeze_scheduler = freeze_scheduler or FreezeScheduler()
        self.freeze_scheduler.apply(self.model, 0)
        self.weight_decay = float(weight_decay)
        self.backbone_high_multiplier = float(backbone_high_multiplier)
        self.backbone_low_multiplier = float(backbone_low_multiplier)
        self.optimizer = build_optimizer(
            self.model,
            base_lr=base_lr,
            weight_decay=weight_decay,
            backbone_high_multiplier=self.backbone_high_multiplier,
            backbone_low_multiplier=self.backbone_low_multiplier,
        )
        self.grad_clip_norm = grad_clip_norm
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.best_epoch = -1


    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = dict(batch)
        # non_blocking=True 在 CUDA 上发起异步拷贝，可与后续前向计算
        # 重叠；键可能缺失（训练与验证的标注组合不同），缺失时跳过即可。
        for key in ("image", "probability_gt", "density_gt", "count_gt"):
            if key in moved and isinstance(moved[key], torch.Tensor):
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
            f"Epoch [{epoch + 1:03d}/{total_epochs:03d}] Train"
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
                if isinstance(value, torch.Tensor) and value.ndim == 0 and torch.isfinite(value):
                    running[key] = running.get(key, 0.0) + float(value.detach())
                elif isinstance(value, (int, float)):
                    running[key] = running.get(key, 0.0) + float(value)
            steps += 1

            if show_pbar and hasattr(pbar, "set_postfix"):
                postfix = {"loss": f"{float(details['total'].detach()):.4f}"}
                if "mae" in details:
                    postfix["mae"] = f"{float(details['mae'].detach()):.2f}"
                elif "count" in details:
                    postfix["count_loss"] = f"{float(details['count'].detach()):.4f}"
                pbar.set_postfix(postfix)

        if steps == 0:
            raise ValueError("training loader is empty")
        return {key: value / steps for key, value in running.items()}

    def evaluate_dataset(
        self,
        dataset: Any,
        *,
        tiler: DensityTiler | None = None,
        show_pbar: bool = True,
    ) -> dict[str, float]:
        """使用整图瓦片平铺（Tiled Inference）执行确定性验证。"""
        self.model.eval()

        def _samples() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for index in range(len(dataset)):
                item = (
                    dataset.full_image(index)
                    if hasattr(dataset, "full_image")
                    else (dataset[index]["image"], dataset[index]["count_gt"])
                )
                if isinstance(item, dict):
                    yield item["image"], item["count_gt"]
                else:
                    yield item[0], item[1]

        return evaluate_tiled(
            self.model,
            _samples(),
            tiler=tiler or DensityTiler(),
            device=self.device,
            total_samples=len(dataset),
            show_pbar=show_pbar,
        )

    def evaluate_loader(
        self,
        loader: DataLoader,
        *,
        show_pbar: bool = True,
    ) -> dict[str, float]:
        """在常规 DataLoader 批次上执行验证评估。"""
        self.model.eval()
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        val_losses: dict[str, float] = {}
        steps = 0

        iterator = (
            tqdm(loader, desc="Validating", dynamic_ncols=True, leave=False)
            if show_pbar
            else loader
        )

        with torch.no_grad():
            for batch in iterator:
                batch = self._move_batch(batch)
                outputs = self.model(batch["image"])
                details = self.criterion.compute(outputs, batch)  # type: ignore[attr-defined]
                predictions.append(outputs["count"].detach().cpu())
                targets.append(torch.as_tensor(batch["count_gt"]).detach().cpu().reshape(-1))
                for key, value in details.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 0 and torch.isfinite(value):
                        val_losses[key] = val_losses.get(key, 0.0) + float(value.detach())
                steps += 1

        if not predictions:
            raise ValueError("validation loader is empty")

        metrics = counting_metrics(torch.cat(predictions), torch.cat(targets))
        if steps > 0:
            for key, value in val_losses.items():
                metrics[f"loss_{key}"] = value / steps
        return metrics

    def fit(
        self,
        loader: DataLoader,
        *,
        epochs: int,
        checkpoint_dir: str | Path | None = None,
        val_dataset: Any = None,
        val_loader: DataLoader | None = None,
        val_interval: int = 1,
        tiler: DensityTiler | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        warmup_epochs: int = 0,
        writer: SummaryWriter | None = None,
        show_pbar: bool = True,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for epoch in range(self.start_epoch, int(epochs)):
            phase = self.freeze_scheduler.apply(self.model, epoch)
            # 新启用的骨干网络参数需要新的参数组。显式重建可确保
            # 在阶段切换边界处学习率乘数保持正确。
            if epoch == self.start_epoch or phase.value != getattr(self, "_last_phase", None):
                self.optimizer = build_optimizer(
                    self.model,
                    base_lr=self.optimizer.defaults["lr"],
                    weight_decay=self.weight_decay,
                    backbone_high_multiplier=self.backbone_high_multiplier,
                    backbone_low_multiplier=self.backbone_low_multiplier,
                )
            self._last_phase = phase.value
            if scheduler is not None:
                scheduler.step()
            else:
                # 未传入外部 scheduler 时使用内置 warmup+cosine；
                # 二选一执行，避免对学习率做双重缩放。
                apply_warmup_cosine(self.optimizer, epoch=epoch, total_epochs=epochs, warmup_epochs=warmup_epochs)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch + 1:03d}/{int(epochs):03d}] Start | Phase: {phase.value} | LR: {current_lr:.6e}"
            )

            train_metrics = self.train_epoch(
                loader,
                epoch=epoch,
                total_epochs=int(epochs),
                show_pbar=show_pbar,
            )

            # 格式化训练日志
            train_log_parts = [
                f"Total: {train_metrics.get('total', 0.0):.4f}",
                f"Prob: {train_metrics.get('probability', 0.0):.4f}",
                f"Dens: {train_metrics.get('density', 0.0):.4f}",
                f"Count: {train_metrics.get('count', 0.0):.4f}",
                f"Local: {train_metrics.get('local', 0.0):.4f}",
            ]
            if "mae" in train_metrics:
                train_log_parts.append(f"MAE: {train_metrics['mae']:.2f}")
            train_log_str = " | ".join(train_log_parts)

            val_metrics: dict[str, float] = {}
            do_val = (val_dataset is not None or val_loader is not None) and (
                (epoch + 1) % max(1, val_interval) == 0 or epoch + 1 == int(epochs)
            )

            if do_val:
                if val_dataset is not None:
                    val_metrics = self.evaluate_dataset(val_dataset, tiler=tiler, show_pbar=show_pbar)
                elif val_loader is not None:
                    val_metrics = self.evaluate_loader(val_loader, show_pbar=show_pbar)

                val_mae = val_metrics.get("mae", float("inf"))
                is_best = val_mae < self.best_metric
                if is_best:
                    self.best_metric = val_mae
                    self.best_epoch = epoch

                val_log_str = (
                    f"MAE: {val_metrics.get('mae', 0.0):.2f} | "
                    f"RMSE: {val_metrics.get('rmse', 0.0):.2f} | "
                    f"NAE: {val_metrics.get('nae', 0.0):.4f} | "
                    f"Best MAE: {self.best_metric:.2f} (Ep {self.best_epoch + 1})"
                )
                logger.info(
                    f"Epoch [{epoch + 1:03d}/{int(epochs):03d}] Result -> Train [{train_log_str}] | Val [{val_log_str}]"
                )
            else:
                # 无验证集时以训练指标为准跟踪最优模型
                current_score = train_metrics.get("mae", train_metrics.get("total", float("inf")))
                is_best = current_score < self.best_metric
                if is_best:
                    self.best_metric = current_score
                    self.best_epoch = epoch
                logger.info(
                    f"Epoch [{epoch + 1:03d}/{int(epochs):03d}] Result -> Train [{train_log_str}]"
                )

            # 写入 TensorBoard
            if writer is not None:
                # 1. 训练阶段各项指标
                for key, val in train_metrics.items():
                    writer.add_scalar(f"train/{key}", val, epoch + 1)
                writer.add_scalar("train/lr", current_lr, epoch + 1)

                # 2. 验证阶段各项指标
                if val_metrics:
                    for key, val in val_metrics.items():
                        writer.add_scalar(f"val/{key}", val, epoch + 1)
                    writer.add_scalar("val/best_mae", self.best_metric, epoch + 1)

                    # 3. 定期向 TensorBoard 记录可视化抽样图像
                    try:
                        from utils.visualization import create_composite_figure, figure_to_tensor, save_figure
                        if val_dataset is not None and len(val_dataset) > 0:
                            sample_item = val_dataset.full_image(0)
                            sample_img = sample_item["image"]
                            sample_gt = float(sample_item["count_gt"].item())
                            active_tiler = tiler or DensityTiler()
                            sample_res = active_tiler(self.model, sample_img, device=self.device)
                            fig = create_composite_figure(
                                sample_img,
                                sample_res.density,
                                pred_count=float(sample_res.count.item()),
                                gt_count=sample_gt,
                                title=f"Epoch {epoch + 1} Val Sample | GT: {sample_gt:.1f} | Pred: {float(sample_res.count.item()):.1f}",
                            )
                            vis_tensor = figure_to_tensor(fig)
                            writer.add_image("val/visual_predictions", vis_tensor, epoch + 1)
                            import matplotlib.pyplot as plt
                            plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Failed to log visual sample to TensorBoard: {e}")

                writer.flush()

            # 记录历史
            epoch_record = {
                "epoch": epoch,
                "lr": current_lr,
                "train": train_metrics,
                "val": val_metrics,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
            }
            history.append(epoch_record)

            # 保存 checkpoint：只保留 last.pt 与 best.pt
            if checkpoint_dir is not None:
                ckpt_dir = Path(checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                # 始终保存/覆盖 last.pt
                last_path = ckpt_dir / "last.pt"
                self.save_checkpoint(last_path, epoch, epoch_record)

                # 达到更优指标时保存/覆盖 best.pt
                if is_best:
                    best_path = ckpt_dir / "best.pt"
                    self.save_checkpoint(best_path, epoch, epoch_record)
                    logger.success(
                        f"★ New best checkpoint (Metric: {self.best_metric:.4f}) saved -> {best_path}"
                    )

                metrics_json = ckpt_dir / "metrics.json"
                metrics_json.write_text(json.dumps(history, indent=2, ensure_ascii=False))
        return history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 快照内容：模型权重、优化器状态、最佳指标信息、冻结调度器状态
        torch.save(
            {
                "epoch": int(epoch),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics or {},
                "best_metric": float(self.best_metric),
                "best_epoch": int(self.best_epoch),
                "freeze_scheduler": self.freeze_scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        saved_epoch = int(checkpoint.get("epoch", -1))
        if "freeze_scheduler" in checkpoint and hasattr(self.freeze_scheduler, "load_state_dict"):
            self.freeze_scheduler.load_state_dict(checkpoint["freeze_scheduler"])
        elif saved_epoch >= 0:
            self.freeze_scheduler.apply(self.model, saved_epoch)

        if "optimizer" in checkpoint:
            # 恢复与检查点所在阶段相匹配的参数组结构，再加载优化器状态
            self.optimizer = build_optimizer(
                self.model,
                base_lr=self.optimizer.defaults.get("lr", 1e-3),
                weight_decay=self.weight_decay,
                backbone_high_multiplier=self.backbone_high_multiplier,
                backbone_low_multiplier=self.backbone_low_multiplier,
            )
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "best_metric" in checkpoint:
            self.best_metric = float(checkpoint["best_metric"])
        if "best_epoch" in checkpoint:
            self.best_epoch = int(checkpoint["best_epoch"])
        self.start_epoch = saved_epoch + 1
        return checkpoint

