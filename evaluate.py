import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import torch
import yaml
from loguru import logger

from data.crowd_dataset import CrowdDataset
from engine.evaluator import evaluate_tiled
from inference.tiled_inference import DensityTiler
from models.crowd_counter import CrowdCounter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/crowd.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Directory to save eval logs and metrics")
    args = parser.parse_args()

    out_dir = args.output or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_log_file = out_dir / "eval.log"

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    logger.add(
        str(eval_log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="10 MB",
        encoding="utf-8",
    )

    config = yaml.safe_load(args.config.read_text())
    model_cfg = config["model"]
    model = CrowdCounter(
        backbone_name=model_cfg.get("backbone", "yolo11n.yaml"),
        use_ultralytics=bool(model_cfg.get("use_ultralytics", True)),
        fusion_channels=int(model_cfg.get("fusion_channels", 128)),
        projection_channels=int(model_cfg.get("projection_channels", 64)),
        msr_blocks=int(model_cfg.get("msr_blocks", 3)),
        msr_dilations=tuple(model_cfg.get("dilations", [1, 2, 3])),
    ).to(args.device)
    # weights_only=False 兼容旧版 torch.save 保存的完整 checkpoint 结构。
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # 关闭动态裁剪与增强，保证同一检查点每次评估得到完全相同的结果。
    dataset = CrowdDataset(
        config["data"]["root"],
        config["data"].get("val_split", "val"),
        crop_size=int(config["image_size"]),
        output_stride=int(config["output_stride"]),
        dynamic_crop=False,
        augment=False,
    )
    logger.info(
        f"Evaluating checkpoint '{args.checkpoint}' on split '{config['data'].get('val_split', 'val')}' ({len(dataset)} images)"
    )

    # 训练时模型只见过固定尺寸裁剪，任意分辨率的整图必须切瓦片推理；
    # 固定分块 + 密度拼接后求和即得整图人数，且每次评估结果完全一致。
    # 这也与训练分布保持一致，避免直接缩放整图带来的计数偏差。
    def _samples() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for index in range(len(dataset)):
            item = dataset.full_image(index)
            yield item["image"], item["count_gt"]

    metrics = evaluate_tiled(
        model,
        _samples(),
        tiler=DensityTiler(
            tile_size=int(config["inference"]["tile_size"]),
            tile_stride=int(config["inference"]["tile_stride"]),
            output_stride=int(config["output_stride"]),
        ),
        device=args.device,
        total_samples=len(dataset),
    )
    metric_str = " | ".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items())
    logger.success(f"Evaluation finished -> {metric_str}")

    metrics_out = out_dir / "eval_metrics.json"
    metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info(f"Saved evaluation metrics to {metrics_out}")
    print(metrics)


if __name__ == "__main__":
    main()
