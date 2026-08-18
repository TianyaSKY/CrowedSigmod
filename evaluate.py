"""Evaluate a checkpoint on deterministic full-image tiles."""

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
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    dataset = CrowdDataset(
        config["data"]["root"],
        config["data"].get("val_split", "val"),
        crop_size=int(config["image_size"]),
        output_stride=int(config["output_stride"]),
        dynamic_crop=False,
        augment=False,
    )
    # Deterministic full-image validation: fixed tiling + density stitching,
    # identical for every evaluation run.
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
