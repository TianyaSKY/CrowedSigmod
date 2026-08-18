"""在确定性的整图分块上评估检查点。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import torch
import yaml

from data.crowd_dataset import CrowdDataset
from engine.evaluator import evaluate_tiled
from inference.tiled_inference import DensityTiler
from models.crowd_counter import CrowdCounter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/crowd.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
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
    )
    print(metrics)


if __name__ == "__main__":
    main()
