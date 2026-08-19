"""全数据集一键训练与跨数据集验证基准评测脚本 (Train & Validate on All Datasets)。

支持三种工作模式：
1. sequential (默认)：按序对每个数据集独立训练、验证、测试，并生成统一的跨数据集 Benchmark 对比大表与可视化柱状图。
2. joint：将所有数据集的训练集联合混合训练单一通用模型，并在所有独立数据集的测试集上评测泛化指标。
3. eval_only：批量加载各数据集的最佳权重（或单一通用权重），对所有数据集执行瓦片推理测试并汇总评估报告。

示例用法：
  # 对所有数据集依次训练与测试（使用默认配置）
  python train_all.py --device cuda

  # 快速对指定数据集训练 20 轮并输出报告
  python train_all.py --datasets ucf_qnrf shanghaitech_AB --epochs 20 --output runs/exp_all

  # 联合所有数据集混合训练
  python train_all.py --mode joint --epochs 50 --device cuda

  # 对所有数据集执行现有模型评测
  python train_all.py --mode eval_only --checkpoint runs/all_datasets/ucf_qnrf/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import matplotlib
matplotlib.use("Agg")  # 保证无 GUI 环境安全绘图
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.crowd_dataset import CrowdDataset, crowd_collate
from engine.evaluator import evaluate_tiled
from engine.freeze_scheduler import FreezeScheduler
from engine.trainer import CrowdTrainer
from inference.tiled_inference import DensityTiler
from losses.crowd_loss import CrowdLoss
from models.crowd_counter import CrowdCounter
from utils.visualization import create_composite_figure, plot_count_scatter, save_figure


@dataclass
class DatasetSpec:
    """数据集描述元数据。"""
    name: str
    root: Path
    train_split: str
    val_split: str | None = None
    test_split: str | None = None
    fold: int | None = None
    description: str = ""


@dataclass
class DatasetRunResult:
    """单个数据集的运行结果汇总。"""
    name: str
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    best_epoch: int = -1
    best_val_mae: float | None = None
    best_val_rmse: float | None = None
    best_val_nae: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_nae: float | None = None
    duration_seconds: float = 0.0
    checkpoint_path: str = ""
    error_message: str = ""


def setup_logger(log_file: Path | None = None) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO",
            rotation="20 MB",
            encoding="utf-8",
        )


def discover_available_datasets(
    datasets_arg: Sequence[str] | str = "all",
    base_datasets_dir: Path = Path("datasets"),
    base_data_dir: Path = Path("data"),
    ucf_cc50_folds: str = "0",
) -> list[DatasetSpec]:
    """自动发现并整理待训练/评测的数据集列表。"""
    specs: list[DatasetSpec] = []
    
    # 统一转换参数列表
    if isinstance(datasets_arg, str):
        if datasets_arg.strip().lower() == "all":
            target_names = ["ucf_qnrf", "shanghaitech_AB", "jhu_crowd", "ucf_cc50"]
        else:
            target_names = [name.strip() for name in datasets_arg.split() if name.strip()]
    else:
        target_names = list(datasets_arg)
        if len(target_names) == 1 and target_names[0].strip().lower() == "all":
            target_names = ["ucf_qnrf", "shanghaitech_AB", "jhu_crowd", "ucf_cc50"]

    for name in target_names:
        # 1. 检查是否为显式路径
        explicit_path = Path(name)
        if explicit_path.exists() and explicit_path.is_dir():
            spec = _create_spec_from_path(explicit_path)
            if spec:
                specs.append(spec)
            continue

        # 2. 检查 datasets/ 下的预置标准数据集
        candidate_datasets = base_datasets_dir / name
        if candidate_datasets.exists():
            if "cc50" in name.lower():
                # 处理 UCF-CC-50 交叉验证折
                folds_to_run = _parse_folds(ucf_cc50_folds)
                for f in folds_to_run:
                    specs.append(
                        DatasetSpec(
                            name=f"ucf_cc50_fold{f}",
                            root=candidate_datasets,
                            train_split=f"fold{f}_train",
                            val_split=f"fold{f}_val",
                            test_split=f"fold{f}_test",
                            fold=f,
                            description=f"UCF-CC-50 Cross-Validation Fold {f}",
                        )
                    )
            else:
                val_dir = candidate_datasets / "images" / "val"
                test_dir = candidate_datasets / "images" / "test"
                specs.append(
                    DatasetSpec(
                        name=name,
                        root=candidate_datasets,
                        train_split="train",
                        val_split="val" if (val_dir.exists() and any(val_dir.glob("*"))) else None,
                        test_split="test" if (test_dir.exists() and any(test_dir.glob("*"))) else None,
                        description=f"Standard Dataset: {name}",
                    )
                )
            continue

        # 3. 检查 data/ 原始数据目录
        candidate_data = base_data_dir / name
        if candidate_data.exists():
            spec = _create_spec_from_path(candidate_data)
            if spec:
                specs.append(spec)
            continue

        logger.warning(f"Dataset '{name}' not found under '{base_datasets_dir}' or '{base_data_dir}'. Skipping.")

    return specs


def _parse_folds(folds_arg: str) -> list[int]:
    if str(folds_arg).strip().lower() == "all":
        return list(range(5))
    result = []
    for item in str(folds_arg).replace(";", ",").split(","):
        item = item.strip()
        if item.isdigit():
            result.append(int(item))
    return result or [0]


def _create_spec_from_path(path: Path) -> DatasetSpec | None:
    """根据目录实际结构推断 split 构成。"""
    train_candidates = ["train", "Train", "train_data", "images/train"]
    val_candidates = ["val", "Val", "val_data", "images/val"]
    test_candidates = ["test", "Test", "test_data", "images/test"]

    train_split = None
    for cand in train_candidates:
        if (path / cand).exists() or (path / cand.split("/")[0]).exists():
            train_split = cand.split("/")[-1]
            break
    if train_split is None:
        train_split = "train"

    val_split = None
    for cand in val_candidates:
        if (path / cand).exists() or (path / cand.split("/")[0]).exists():
            val_split = cand.split("/")[-1]
            break

    test_split = None
    for cand in test_candidates:
        if (path / cand).exists() or (path / cand.split("/")[0]).exists():
            test_split = cand.split("/")[-1]
            break

    return DatasetSpec(
        name=path.name,
        root=path,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        description=f"Custom Dataset from {path}",
    )


def create_model_and_trainer(
    config: dict[str, Any],
    device: str,
    epochs: int,
    checkpoint_path: Path | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> tuple[CrowdCounter, CrowdLoss, FreezeScheduler, CrowdTrainer]:
    """根据配置创建模型、损失函数与训练器。"""
    model_cfg = config["model"]
    train_cfg = config["training"]

    model = CrowdCounter(
        backbone_name=model_cfg.get("backbone", "yolo11n.yaml"),
        pretrained=model_cfg.get("pretrained"),
        use_ultralytics=bool(model_cfg.get("use_ultralytics", True)),
        fusion_channels=int(model_cfg.get("fusion_channels", 128)),
        projection_channels=int(model_cfg.get("projection_channels", 64)),
        msr_blocks=int(model_cfg.get("msr_blocks", 3)),
        msr_dilations=tuple(model_cfg.get("dilations", [1, 2, 3])),
    )

    if checkpoint_path is not None and checkpoint_path.exists():
        logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    criterion = CrowdLoss(
        probability_weight=float(config["loss"]["probability"]),
        density_weight=float(config["loss"]["density"]),
        count_weight=float(config["loss"]["count"]),
        local_weight=float(config["loss"]["local"]),
        local_grid=int(config["loss"].get("local_grid", 4)),
        dice_weight=float(config["loss"].get("dice", 0.2)),
    )

    freeze = FreezeScheduler(
        freeze_epochs=int(train_cfg["freeze_epochs"]),
        partial_unfreeze_epoch=int(train_cfg["partial_unfreeze_epoch"]),
        full_unfreeze_epoch=int(train_cfg["full_unfreeze_epoch"]),
    )

    trainer = CrowdTrainer(
        model,
        criterion,
        device=device,
        base_lr=float(learning_rate if learning_rate is not None else train_cfg["learning_rate"]),
        weight_decay=float(weight_decay if weight_decay is not None else train_cfg["weight_decay"]),
        backbone_high_multiplier=float(train_cfg.get("backbone_high_multiplier", 0.02)),
        backbone_low_multiplier=float(train_cfg.get("backbone_low_multiplier", 0.005)),
        freeze_scheduler=freeze,
    )

    return model, criterion, freeze, trainer


def evaluate_dataset_split(
    model: CrowdCounter,
    dataset: CrowdDataset,
    tiler: DensityTiler,
    device: str,
    out_dir: Path,
    split_name: str = "test",
    save_scatter: bool = True,
    num_vis: int = 6,
    colormap: str = "jet",
) -> dict[str, float]:
    """对指定数据集 split 进行确定性整图滑窗测试并导出散点图与定性可视化。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluating split '{split_name}' ({len(dataset)} images) -> {out_dir}")

    def _samples() -> Iterator[dict[str, Any]]:
        for i in range(len(dataset)):
            yield dataset.full_image(i)

    metrics, detailed_records = evaluate_tiled(
        model,
        _samples(),
        tiler=tiler,
        device=device,
        total_samples=len(dataset),
        return_details=True,
    )  # type: ignore[misc]

    metric_str = " | ".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items())
    logger.info(f"Split [{split_name}] Evaluation Finished -> {metric_str}")

    # 保存指标 JSON
    metrics_file = out_dir / f"{split_name}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 保存散点图
    if save_scatter and detailed_records:
        gt_counts = [r["target_count"] for r in detailed_records]
        pred_counts = [r["pred_count"] for r in detailed_records]
        scatter_fig = plot_count_scatter(
            gt_counts,
            pred_counts,
            metrics=metrics,
            title=f"Dataset {out_dir.name} [{split_name}]: GT vs. Pred Count",
        )
        scatter_path = out_dir / f"{split_name}_scatter.png"
        save_figure(scatter_fig, scatter_path)

    # 导出定性分析图 (难例与优秀样本)
    if num_vis > 0 and detailed_records:
        vis_dir = out_dir / f"{split_name}_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        sorted_records = sorted(detailed_records, key=lambda r: abs(r["error"]), reverse=True)
        half = max(1, num_vis // 2)
        selected = []
        for rank, r in enumerate(sorted_records[:half]):
            selected.append((f"worst_{rank + 1:02d}", r))
        for rank, r in enumerate(sorted_records[-half:]):
            selected.append((f"best_{rank + 1:02d}", r))

        for tag, record in selected:
            img_id = (record["image_id"] or "sample").replace("/", "_").replace("\\", "_")
            pred_c = record["pred_count"]
            tgt_c = record["target_count"]
            fig = create_composite_figure(
                image=record["image"],
                pred_density=record["density"],
                pred_count=pred_c,
                gt_count=tgt_c,
                title=f"[{tag.upper()}] {img_id} | GT: {tgt_c:.1f} | Pred: {pred_c:.1f} (Err: {abs(pred_c - tgt_c):.1f})",
                colormap=colormap,
            )
            save_figure(fig, vis_dir / f"{tag}_{img_id}_gt{tgt_c:.0f}_pred{pred_c:.0f}.jpg")

    return metrics


def train_single_dataset(
    spec: DatasetSpec,
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> DatasetRunResult:
    """训练并评估单个数据集（顺序基准模式的核心单元）。"""
    start_time = time.time()
    ds_output_dir = output_dir / spec.name
    ds_output_dir.mkdir(parents=True, exist_ok=True)

    result = DatasetRunResult(
        name=spec.name,
        status="RUNNING",
        checkpoint_path=str(ds_output_dir / "best.pt"),
    )

    target_cfg = config["targets"]
    train_cfg = config["training"]
    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg["batch_size"])
    workers = int(args.workers if args.workers is not None else train_cfg.get("workers", 4))

    # 1. 检查是否跳过已完成项
    if args.skip_completed and (ds_output_dir / "test_metrics.json").exists():
        logger.info(f"[{spec.name}] Found existing test_metrics.json. Skipping (--skip-completed enabled).")
        try:
            test_m = json.loads((ds_output_dir / "test_metrics.json").read_text())
            val_m = json.loads((ds_output_dir / "eval_metrics.json").read_text()) if (ds_output_dir / "eval_metrics.json").exists() else {}
            result.status = "SKIPPED"
            result.test_mae = test_m.get("mae")
            result.test_rmse = test_m.get("rmse")
            result.test_nae = test_m.get("nae")
            result.best_val_mae = val_m.get("mae")
            result.best_val_rmse = val_m.get("rmse")
            result.best_val_nae = val_m.get("nae")
            return result
        except Exception:
            pass

    logger.info("=" * 70)
    logger.info(f"Starting Training Dataset: [{spec.name}] ({spec.description})")
    logger.info(f"Root: {spec.root} | Train Split: {spec.train_split} | Val: {spec.val_split} | Test: {spec.test_split}")
    logger.info(f"Output Dir: {ds_output_dir}")
    logger.info("=" * 70)

    try:
        # 2. 构建训练数据集
        train_dataset = CrowdDataset(
            spec.root,
            split=spec.train_split,
            crop_size=int(config["image_size"]),
            output_stride=int(config["output_stride"]),
            probability_sigma=float(target_cfg["probability_sigma"]),
            density_sigma=float(target_cfg["density_sigma"]),
            adaptive_density=bool(target_cfg.get("adaptive_density", False)),
        )
        result.train_samples = len(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=crowd_collate,
            pin_memory=args.device.startswith("cuda"),
        )

        # 3. 构建验证数据集
        val_dataset = None
        if spec.val_split:
            try:
                val_dataset = CrowdDataset(
                    spec.root,
                    split=spec.val_split,
                    crop_size=int(config["image_size"]),
                    output_stride=int(config["output_stride"]),
                    dynamic_crop=False,
                    augment=False,
                )
                result.val_samples = len(val_dataset)
            except Exception as e:
                logger.warning(f"[{spec.name}] Validation split '{spec.val_split}' load failed: {e}")

        # 4. 构建测试数据集
        test_dataset = None
        if spec.test_split:
            try:
                test_dataset = CrowdDataset(
                    spec.root,
                    split=spec.test_split,
                    crop_size=int(config["image_size"]),
                    output_stride=int(config["output_stride"]),
                    dynamic_crop=False,
                    augment=False,
                )
                result.test_samples = len(test_dataset)
            except Exception as e:
                logger.warning(f"[{spec.name}] Test split '{spec.test_split}' load failed: {e}")

        # 5. 初始化模型与训练器
        model, criterion, freeze, trainer = create_model_and_trainer(
            config,
            device=args.device,
            epochs=epochs,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            learning_rate=args.lr,
        )

        tiler = DensityTiler(
            tile_size=int(config.get("inference", {}).get("tile_size", 640)),
            tile_stride=int(config.get("inference", {}).get("tile_stride", 512)),
            output_stride=int(config.get("output_stride", 4)),
        )

        tb_writer = SummaryWriter(log_dir=str(ds_output_dir / "tensorboard"))

        # 6. 执行训练循环
        trainer.fit(
            train_loader,
            epochs=epochs,
            checkpoint_dir=ds_output_dir,
            val_dataset=val_dataset,
            val_interval=int(args.val_interval if args.val_interval is not None else train_cfg.get("val_interval", 1)),
            tiler=tiler,
            warmup_epochs=min(int(train_cfg.get("warmup_epochs", 5)), epochs),
            writer=tb_writer,
        )
        tb_writer.close()

        result.best_epoch = trainer.best_epoch + 1
        result.best_val_mae = trainer.best_metric if trainer.best_metric != float("inf") else None

        # 7. 训练后测试与全套评估
        best_pt_path = ds_output_dir / "best.pt"
        eval_weights = best_pt_path if best_pt_path.exists() else (ds_output_dir / "last.pt")

        if eval_weights.exists():
            logger.info(f"[{spec.name}] Loading best checkpoint for post-training testing: {eval_weights}")
            ckpt = torch.load(eval_weights, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.to(args.device)

            # 评估验证集 (若存在)
            if val_dataset is not None:
                val_metrics = evaluate_dataset_split(
                    model, val_dataset, tiler, args.device, ds_output_dir,
                    split_name="val", save_scatter=args.save_scatter, num_vis=args.num_vis,
                )
                result.best_val_mae = val_metrics.get("mae")
                result.best_val_rmse = val_metrics.get("rmse")
                result.best_val_nae = val_metrics.get("nae")

            # 评估测试集 (若存在)
            if test_dataset is not None and args.test_after_train:
                test_metrics = evaluate_dataset_split(
                    model, test_dataset, tiler, args.device, ds_output_dir,
                    split_name="test", save_scatter=args.save_scatter, num_vis=args.num_vis,
                )
                result.test_mae = test_metrics.get("mae")
                result.test_rmse = test_metrics.get("rmse")
                result.test_nae = test_metrics.get("nae")

        result.status = "SUCCESS"

    except Exception as e:
        logger.error(f"[{spec.name}] Training failed with error: {e}")
        logger.error(traceback.format_exc())
        result.status = "FAILED"
        result.error_message = str(e)

    result.duration_seconds = time.time() - start_time
    logger.info(f"[{spec.name}] Completed in {result.duration_seconds / 60:.2f} minutes | Status: {result.status}")
    return result


def run_sequential_mode(
    specs: list[DatasetSpec],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[DatasetRunResult]:
    """模式 1：对所有数据集逐一独立训练与评估。"""
    logger.info(f"=== Starting Sequential Multi-Dataset Training ({len(specs)} datasets) ===")
    results: list[DatasetRunResult] = []

    for idx, spec in enumerate(specs, start=1):
        logger.info(f"\n>>> [{idx}/{len(specs)}] Processing dataset: {spec.name} <<<")
        res = train_single_dataset(spec, config, args, output_dir)
        results.append(res)

    return results


def run_joint_mode(
    specs: list[DatasetSpec],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[DatasetRunResult]:
    """模式 2：将所有数据集联合为一个大训练集进行联合训练，并在各个独立测试集上评测。"""
    joint_output_dir = output_dir / "joint_model"
    joint_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== Starting Joint Multi-Dataset Training ({len(specs)} datasets combined) ===")

    target_cfg = config["targets"]
    train_cfg = config["training"]
    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg["batch_size"])
    workers = int(args.workers if args.workers is not None else train_cfg.get("workers", 4))

    # 1. 组合所有训练集
    train_datasets = []
    val_datasets_map: dict[str, CrowdDataset] = {}
    test_datasets_map: dict[str, CrowdDataset] = {}

    for spec in specs:
        try:
            ds = CrowdDataset(
                spec.root,
                split=spec.train_split,
                crop_size=int(config["image_size"]),
                output_stride=int(config["output_stride"]),
                probability_sigma=float(target_cfg["probability_sigma"]),
                density_sigma=float(target_cfg["density_sigma"]),
                adaptive_density=bool(target_cfg.get("adaptive_density", False)),
            )
            train_datasets.append(ds)
            logger.info(f"Added training dataset: {spec.name} ({len(ds)} images)")
        except Exception as e:
            logger.error(f"Failed to load training split for {spec.name}: {e}")

        if spec.val_split:
            try:
                val_datasets_map[spec.name] = CrowdDataset(
                    spec.root,
                    split=spec.val_split,
                    crop_size=int(config["image_size"]),
                    output_stride=int(config["output_stride"]),
                    dynamic_crop=False,
                    augment=False,
                )
            except Exception:
                pass

        if spec.test_split:
            try:
                test_datasets_map[spec.name] = CrowdDataset(
                    spec.root,
                    split=spec.test_split,
                    crop_size=int(config["image_size"]),
                    output_stride=int(config["output_stride"]),
                    dynamic_crop=False,
                    augment=False,
                )
            except Exception:
                pass

    if not train_datasets:
        raise ValueError("No valid training datasets found to combine.")

    combined_train_dataset = ConcatDataset(train_datasets)
    logger.info(f"Combined Joint Dataset size: {len(combined_train_dataset)} images from {len(train_datasets)} datasets")

    combined_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=crowd_collate,
        pin_memory=args.device.startswith("cuda"),
    )

    # 2. 构建模型与训练器
    model, criterion, freeze, trainer = create_model_and_trainer(
        config,
        device=args.device,
        epochs=epochs,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        learning_rate=args.lr,
    )

    tiler = DensityTiler(
        tile_size=int(config.get("inference", {}).get("tile_size", 640)),
        tile_stride=int(config.get("inference", {}).get("tile_stride", 512)),
        output_stride=int(config.get("output_stride", 4)),
    )

    tb_writer = SummaryWriter(log_dir=str(joint_output_dir / "tensorboard"))

    # 首选主验证集
    primary_val_ds = next(iter(val_datasets_map.values())) if val_datasets_map else None

    start_time = time.time()
    trainer.fit(
        combined_loader,
        epochs=epochs,
        checkpoint_dir=joint_output_dir,
        val_dataset=primary_val_ds,
        val_interval=int(args.val_interval if args.val_interval is not None else train_cfg.get("val_interval", 1)),
        tiler=tiler,
        warmup_epochs=min(int(train_cfg.get("warmup_epochs", 5)), epochs),
        writer=tb_writer,
    )
    tb_writer.close()
    duration = time.time() - start_time

    # 3. 加载最佳权重并在每个独立数据集上评测
    best_pt = joint_output_dir / "best.pt"
    eval_weights = best_pt if best_pt.exists() else (joint_output_dir / "last.pt")
    if eval_weights.exists():
        ckpt = torch.load(eval_weights, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(args.device)

    results: list[DatasetRunResult] = []
    for spec in specs:
        res = DatasetRunResult(
            name=spec.name,
            status="SUCCESS",
            checkpoint_path=str(eval_weights),
            duration_seconds=duration / len(specs),
        )
        ds_eval_dir = joint_output_dir / "eval_per_dataset" / spec.name
        ds_eval_dir.mkdir(parents=True, exist_ok=True)

        if spec.name in val_datasets_map:
            val_m = evaluate_dataset_split(
                model, val_datasets_map[spec.name], tiler, args.device, ds_eval_dir,
                split_name="val", save_scatter=args.save_scatter, num_vis=args.num_vis,
            )
            res.val_samples = len(val_datasets_map[spec.name])
            res.best_val_mae = val_m.get("mae")
            res.best_val_rmse = val_m.get("rmse")
            res.best_val_nae = val_m.get("nae")

        if spec.name in test_datasets_map:
            test_m = evaluate_dataset_split(
                model, test_datasets_map[spec.name], tiler, args.device, ds_eval_dir,
                split_name="test", save_scatter=args.save_scatter, num_vis=args.num_vis,
            )
            res.test_samples = len(test_datasets_map[spec.name])
            res.test_mae = test_m.get("mae")
            res.test_rmse = test_m.get("rmse")
            res.test_nae = test_m.get("nae")

        results.append(res)

    return results


def run_eval_only_mode(
    specs: list[DatasetSpec],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[DatasetRunResult]:
    """模式 3：仅批量评估已训练好的检查点。"""
    logger.info(f"=== Starting Multi-Dataset Batch Evaluation ({len(specs)} datasets) ===")
    results: list[DatasetRunResult] = []

    model_cfg = config["model"]
    model = CrowdCounter(
        backbone_name=model_cfg.get("backbone", "yolo11n.yaml"),
        use_ultralytics=bool(model_cfg.get("use_ultralytics", True)),
        fusion_channels=int(model_cfg.get("fusion_channels", 128)),
        projection_channels=int(model_cfg.get("projection_channels", 64)),
        msr_blocks=int(model_cfg.get("msr_blocks", 3)),
        msr_dilations=tuple(model_cfg.get("dilations", [1, 2, 3])),
    ).to(args.device)

    tiler = DensityTiler(
        tile_size=int(config.get("inference", {}).get("tile_size", 640)),
        tile_stride=int(config.get("inference", {}).get("tile_stride", 512)),
        output_stride=int(config.get("output_stride", 4)),
    )

    for spec in specs:
        start_t = time.time()
        ds_out_dir = output_dir / spec.name
        ds_out_dir.mkdir(parents=True, exist_ok=True)

        res = DatasetRunResult(name=spec.name, status="RUNNING")

        candidate_ckpts = []
        if args.checkpoint:
            candidate_ckpts.append(Path(args.checkpoint))
        candidate_ckpts.extend([
            ds_out_dir / "best.pt",
            ds_out_dir / "last.pt",
            Path("runs/crowd/best.pt"),
            Path("runs/crowd/last.pt"),
        ])

        chosen_ckpt = None
        for ck in candidate_ckpts:
            if ck.exists():
                chosen_ckpt = ck
                break

        if chosen_ckpt is None:
            logger.warning(f"[{spec.name}] No checkpoint found in candidate paths. Skipping.")
            res.status = "FAILED"
            res.error_message = "No checkpoint found"
            results.append(res)
            continue

        res.checkpoint_path = str(chosen_ckpt)
        logger.info(f"[{spec.name}] Evaluating with checkpoint: {chosen_ckpt}")
        ckpt_data = torch.load(chosen_ckpt, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt_data["model"] if "model" in ckpt_data else ckpt_data)

        # 评估验证集
        if spec.val_split:
            try:
                val_ds = CrowdDataset(spec.root, split=spec.val_split, crop_size=int(config["image_size"]), output_stride=int(config["output_stride"]), dynamic_crop=False, augment=False)
                res.val_samples = len(val_ds)
                val_m = evaluate_dataset_split(model, val_ds, tiler, args.device, ds_out_dir, split_name="val", save_scatter=args.save_scatter, num_vis=args.num_vis)
                res.best_val_mae = val_m.get("mae")
                res.best_val_rmse = val_m.get("rmse")
                res.best_val_nae = val_m.get("nae")
            except Exception as e:
                logger.warning(f"[{spec.name}] Val eval failed: {e}")

        # 评估测试集
        if spec.test_split:
            try:
                test_ds = CrowdDataset(spec.root, split=spec.test_split, crop_size=int(config["image_size"]), output_stride=int(config["output_stride"]), dynamic_crop=False, augment=False)
                res.test_samples = len(test_ds)
                test_m = evaluate_dataset_split(model, test_ds, tiler, args.device, ds_out_dir, split_name="test", save_scatter=args.save_scatter, num_vis=args.num_vis)
                res.test_mae = test_m.get("mae")
                res.test_rmse = test_m.get("rmse")
                res.test_nae = test_m.get("nae")
            except Exception as e:
                logger.warning(f"[{spec.name}] Test eval failed: {e}")

        res.status = "SUCCESS"
        res.duration_seconds = time.time() - start_t
        results.append(res)

    return results


def generate_benchmark_summary(
    results: list[DatasetRunResult],
    output_dir: Path,
    mode: str,
    args: argparse.Namespace,
) -> None:
    """生成漂亮的跨数据集基准评测报告（Terminal表格、Markdown报告、JSON、CSV与可视化柱状图）。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 打印控制台表格
    headers = ["Dataset", "Train", "Val", "Test", "Best Ep", "Val MAE", "Val RMSE", "Test MAE", "Test RMSE", "Test NAE", "Status", "Time"]
    rows = []

    valid_test_maes = []
    valid_test_rmses = []
    valid_val_maes = []

    for r in results:
        v_mae_str = f"{r.best_val_mae:.2f}" if r.best_val_mae is not None else "-"
        v_rmse_str = f"{r.best_val_rmse:.2f}" if r.best_val_rmse is not None else "-"
        t_mae_str = f"{r.test_mae:.2f}" if r.test_mae is not None else "-"
        t_rmse_str = f"{r.test_rmse:.2f}" if r.test_rmse is not None else "-"
        t_nae_str = f"{r.test_nae:.4f}" if r.test_nae is not None else "-"
        time_str = f"{r.duration_seconds / 60:.1f}m" if r.duration_seconds > 0 else "-"
        ep_str = str(r.best_epoch) if r.best_epoch > 0 else "-"

        if r.test_mae is not None:
            valid_test_maes.append(r.test_mae)
        if r.test_rmse is not None:
            valid_test_rmses.append(r.test_rmse)
        if r.best_val_mae is not None:
            valid_val_maes.append(r.best_val_mae)

        rows.append([
            r.name,
            str(r.train_samples),
            str(r.val_samples),
            str(r.test_samples),
            ep_str,
            v_mae_str,
            v_rmse_str,
            t_mae_str,
            t_rmse_str,
            t_nae_str,
            r.status,
            time_str,
        ])

    # 计算宏平均 (Macro-Average)
    if valid_test_maes or valid_val_maes:
        avg_v_mae = f"{np.mean(valid_val_maes):.2f}" if valid_val_maes else "-"
        avg_t_mae = f"{np.mean(valid_test_maes):.2f}" if valid_test_maes else "-"
        avg_t_rmse = f"{np.mean(valid_test_rmses):.2f}" if valid_test_rmses else "-"
        rows.append([
            "★ AVERAGE",
            "-",
            "-",
            "-",
            "-",
            avg_v_mae,
            "-",
            avg_t_mae,
            avg_t_rmse,
            "-",
            "-",
            "-",
        ])

    try:
        from tabulate import tabulate
        table_str = tabulate(rows, headers=headers, tablefmt="fancy_grid")
    except ImportError:
        table_str = "\n".join(["\t".join(headers)] + ["\t".join(r) for r in rows])

    logger.info("\n" + "=" * 80 + "\nBENCHMARK SUMMARY RESULTS:\n" + table_str + "\n" + "=" * 80)

    # 2. 保存 JSON 与 CSV
    json_path = output_dir / "summary_metrics.json"
    results_dict = [asdict(r) for r in results]
    json_path.write_text(json.dumps({"mode": mode, "results": results_dict}, indent=2, ensure_ascii=False))

    csv_path = output_dir / "summary_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # 3. 生成 Markdown 总结报告
    md_content = [
        f"# YOLO-PGMD 跨数据集人群计数基准评测报告 (Benchmark Report)\n",
        f"- **运行模式**: `{mode}`",
        f"- **运行设备**: `{args.device}`",
        f"- **生成时间**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **配置文件**: `{args.config}`\n",
        "## 1. 跨数据集总体指标汇总\n",
    ]
    
    md_header = "| " + " | ".join(headers) + " |"
    md_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    md_rows = ["| " + " | ".join(r) + " |" for r in rows]
    md_content.extend([md_header, md_sep] + md_rows + ["\n"])

    md_content.append("## 2. 逐数据集详情与产物链接\n")
    for r in results:
        md_content.append(f"### 数据集: `{r.name}`")
        md_content.append(f"- **状态**: {r.status}")
        md_content.append(f"- **样本规模**: Train={r.train_samples}, Val={r.val_samples}, Test={r.test_samples}")
        if r.best_val_mae is not None:
            md_content.append(f"- **Validation 指标**: MAE = `{r.best_val_mae:.2f}`, RMSE = `{r.best_val_rmse:.2f}`, NAE = `{r.best_val_nae:.4f}`")
        if r.test_mae is not None:
            md_content.append(f"- **Test 指标**: MAE = `{r.test_mae:.2f}`, RMSE = `{r.test_rmse:.2f}`, NAE = `{r.test_nae:.4f}`")
        if r.best_epoch > 0:
            md_content.append(f"- **最佳 Epoch**: {r.best_epoch}")
        if r.checkpoint_path:
            md_content.append(f"- **模型权重**: `{r.checkpoint_path}`")
        if r.error_message:
            md_content.append(f"- **错误信息**: `{r.error_message}`")
        md_content.append("")

    report_path = output_dir / "summary_report.md"
    report_path.write_text("\n".join(md_content), encoding="utf-8")
    logger.success(f"Benchmark summary report saved to: {report_path}")

    # 4. 生成对比柱状图
    try:
        plot_benchmark_bar_chart(results, output_dir / "benchmark_comparison.png")
    except Exception as e:
        logger.warning(f"Failed to generate benchmark bar chart: {e}")


def plot_benchmark_bar_chart(results: list[DatasetRunResult], save_path: Path) -> None:
    """绘制跨数据集 MAE / RMSE 对比柱状图。"""
    valid_results = [r for r in results if r.status == "SUCCESS" and (r.test_mae is not None or r.best_val_mae is not None)]
    if not valid_results:
        return

    names = [r.name for r in valid_results]
    maes = [r.test_mae if r.test_mae is not None else r.best_val_mae for r in valid_results]
    rmses = [r.test_rmse if r.test_rmse is not None else r.best_val_rmse for r in valid_results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5), dpi=200)
    rects1 = ax.bar(x - width / 2, maes, width, label="MAE", color="#1f77b4", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width / 2, rmses, width, label="RMSE", color="#ff7f0e", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Error (Counts)")
    ax.set_title("Crowd Counting Cross-Dataset Benchmark Performance (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        if h is not None and not np.isnan(h):
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        if h is not None and not np.isnan(h):
            ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_figure(fig, save_path)
    logger.success(f"Benchmark comparison plot saved to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/crowd.yaml"), help="Path to base YAML config")
    parser.add_argument("--mode", choices=["sequential", "joint", "eval_only"], default="sequential", help="Workflow mode: sequential (bench each), joint (combine all), eval_only (eval checkpoints)")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Datasets to run: 'all', or list of names e.g. 'ucf_qnrf shanghaitech_AB'")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training/eval")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for training")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output", type=Path, default=Path("runs/all_datasets"), help="Root directory for multi-dataset outputs")
    parser.add_argument("--val-interval", type=int, default=None, help="Validation interval epochs")
    parser.add_argument("--test-after-train", action="store_true", default=True, help="Run test evaluation on best checkpoint after training")
    parser.add_argument("--no-test-after-train", dest="test_after_train", action="store_false", help="Skip post-training test set evaluation")
    parser.add_argument("--num-vis", type=int, default=6, help="Number of qualitative visual samples to save per dataset")
    parser.add_argument("--save-scatter", action="store_true", default=True, help="Save GT vs Pred scatter plots")
    parser.add_argument("--no-scatter", dest="save_scatter", action="store_false", help="Disable scatter plots")
    parser.add_argument("--ucf-cc50-folds", default="0", help="Folds for UCF-CC-50: '0' (default), 'all' (5 folds CV), or '0,1,2'")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path to resume or for eval_only mode")
    parser.add_argument("--skip-completed", action="store_true", default=False, help="Skip datasets that already have completed evaluation metrics")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader num_workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "train_all.log"
    setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("YOLO-PGMD Multi-Dataset Training & Benchmarking System")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config} | Mode: {args.mode} | Device: {args.device} | Output: {output_dir}")

    config = yaml.safe_load(args.config.read_text())

    # 1. 发现可用数据集
    specs = discover_available_datasets(
        datasets_arg=args.datasets,
        ucf_cc50_folds=args.ucf_cc50_folds,
    )
    if not specs:
        logger.error("No valid datasets discovered! Please check --datasets argument or dataset directory structure.")
        sys.exit(1)

    logger.info(f"Discovered {len(specs)} dataset specification(s):")
    for s in specs:
        logger.info(f"  - [{s.name}] root: {s.root}, train: {s.train_split}, val: {s.val_split}, test: {s.test_split}")

    # 2. 根据模式执行
    if args.mode == "sequential":
        results = run_sequential_mode(specs, config, args, output_dir)
    elif args.mode == "joint":
        results = run_joint_mode(specs, config, args, output_dir)
    elif args.mode == "eval_only":
        results = run_eval_only_mode(specs, config, args, output_dir)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # 3. 汇总报告与可视化大表
    generate_benchmark_summary(results, output_dir, mode=args.mode, args=args)
    logger.success("All multi-dataset tasks completed successfully!")


if __name__ == "__main__":
    main()
