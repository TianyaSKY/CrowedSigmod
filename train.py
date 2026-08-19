"""根据 YAML 配置训练 YOLO-PGMD。

示例：``python train.py --config configs/crowd.yaml --device cuda``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.crowd_dataset import CrowdDataset, crowd_collate
from engine.freeze_scheduler import FreezeScheduler
from engine.trainer import CrowdTrainer
from losses.crowd_loss import CrowdLoss
from models.crowd_counter import CrowdCounter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/crowd.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/crowd"))
    return parser.parse_args()


def setup_logger(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    log_file = output_dir / "train.log"
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="20 MB",
        encoding="utf-8",
    )


def main() -> None:
    # 入口约定：命令行只覆盖少量运行参数（config/device/epochs/output），
    # 模型、损失与训练超参全部从 YAML 读取，保证实验配置可复现。
    args = parse_args()
    output_dir = Path(args.output)
    setup_logger(output_dir)

    logger.info(f"Loading config from {args.config}")
    config = yaml.safe_load(args.config.read_text())
    data_cfg = config["data"]
    model_cfg = config["model"]
    target_cfg = config["targets"]
    train_cfg = config["training"]

    root = Path(data_cfg["root"])
    dataset = CrowdDataset(
        root,
        data_cfg.get("train_split", "train"),
        crop_size=int(config["image_size"]),
        output_stride=int(config["output_stride"]),
        probability_sigma=float(target_cfg["probability_sigma"]),
        density_sigma=float(target_cfg["density_sigma"]),
        adaptive_density=bool(target_cfg.get("adaptive_density", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("workers", 0)),
        collate_fn=crowd_collate,
        # GPU 训练时启用 pin_memory，加速主机到设备的批量拷贝。
        pin_memory=args.device.startswith("cuda"),
    )
    logger.info(f"Dataset initialized from {root} (samples: {len(dataset)}, batch_size: {train_cfg['batch_size']})")

    model = CrowdCounter(
        backbone_name=model_cfg.get("backbone", "yolo11n.yaml"),
        pretrained=model_cfg.get("pretrained"),
        use_ultralytics=bool(model_cfg.get("use_ultralytics", True)),
        fusion_channels=int(model_cfg.get("fusion_channels", 128)),
        projection_channels=int(model_cfg.get("projection_channels", 64)),
        msr_blocks=int(model_cfg.get("msr_blocks", 3)),
        msr_dilations=tuple(model_cfg.get("dilations", [1, 2, 3])),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created (total params: {total_params:,}, trainable: {trainable_params:,})"
    )

    criterion = CrowdLoss(
        probability_weight=float(config["loss"]["probability"]),
        density_weight=float(config["loss"]["density"]),
        count_weight=float(config["loss"]["count"]),
        local_weight=float(config["loss"]["local"]),
        local_grid=int(config["loss"].get("local_grid", 4)),
        dice_weight=float(config["loss"].get("dice", 0.2)),
    )
    # 分阶段解冻：先冻结主干稳定早期训练，再逐步解冻以微调全网络，
    # 避免随机初始化阶段主干梯度不稳定。
    freeze = FreezeScheduler(
        freeze_epochs=int(train_cfg["freeze_epochs"]),
        partial_unfreeze_epoch=int(train_cfg["partial_unfreeze_epoch"]),
        full_unfreeze_epoch=int(train_cfg["full_unfreeze_epoch"]),
    )
    trainer = CrowdTrainer(
        model,
        criterion,
        device=args.device,
        base_lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        freeze_scheduler=freeze,
    )
    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])

    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard summary writer active at {tb_dir}")
    logger.info(f"Starting training on device '{args.device}' for {epochs} epochs")

    try:
        trainer.fit(
            loader,
            epochs=epochs,
            checkpoint_dir=output_dir,
            # warmup 取 min(warmup_epochs, epochs)，防止命令行缩短训练轮数后预热越界。
            warmup_epochs=min(int(train_cfg.get("warmup_epochs", 0)), epochs),
            writer=writer,
        )
        logger.success("Training completed successfully!")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
