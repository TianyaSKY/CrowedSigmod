"""全数据集批量质检与真实密度图可视化脚本。

支持遍历 datasets/ 或 data/ 下的所有数据集与划分（train/val/test/folds），
批量生成包含 [Full Image+Points | Full Density GT | Full Probability GT | Density Overlay] 的 4 面板高质检图。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import yaml
from loguru import logger

from data.crowd_dataset import CrowdDataset
from tools.visualize_dataset import create_dataset_inspection_figure, create_full_image_inspection_figure
from utils.visualization import save_figure


def get_all_dataset_targets(base_dir: Path) -> list[tuple[str, Path, str]]:
    """发现 base_dir 下的所有数据集及其可用的划分 (split)。"""
    targets: list[tuple[str, Path, str]] = []
    
    if not base_dir.exists():
        return targets

    for ds_dir in sorted(base_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue
        
        # 1. 检查标准 datasets/ 结构: images/{train, val, test, fold...}
        img_dir = ds_dir / "images"
        if img_dir.exists() and img_dir.is_dir():
            for split_dir in sorted(img_dir.iterdir()):
                if split_dir.is_dir() and any(split_dir.glob("*")):
                    targets.append((ds_dir.name, ds_dir, split_dir.name))
            continue

        # 2. 检查 data/ 结构: part_A_final, UCF-QNRF_ECCV18, jhu_crowd_v2.0, UCF_CC_50
        candidates = ["train", "val", "test", "Train", "Test", "train_data", "test_data"]
        found_any = False
        for c in candidates:
            if (ds_dir / c).exists() and any((ds_dir / c).glob("*")):
                targets.append((ds_dir.name, ds_dir, c))
                found_any = True
        
        if not found_any and any(p.suffix.lower() in {".jpg", ".png", ".jpeg"} for p in ds_dir.glob("*")):
            targets.append((ds_dir.name, ds_dir, "."))

    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["datasets", "data", "both"],
        default="datasets",
        help="Source directory to visualize (default: datasets)",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=5,
        help="Number of samples to visualize per dataset split (default: 5)",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        default=False,
        help="Export all samples without capping at --samples-per-split",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "patch"],
        default="full",
        help="Visualization mode: full (whole image) or patch (cropped window)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/all_datasets_vis"),
        help="Directory to save inspection visualizations",
    )
    parser.add_argument("--crop-size", type=int, default=640, help="Crop window size for patch mode")
    parser.add_argument("--output-stride", type=int, default=4, help="Density stride reduction")
    parser.add_argument("--colormap", default="jet", help="Colormap for density maps")
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    sources = []
    if args.source in ("datasets", "both"):
        sources.append(project_root / "datasets")
    if args.source in ("data", "both"):
        sources.append(project_root / "data")

    all_targets: list[tuple[str, Path, str]] = []
    for s in sources:
        all_targets.extend(get_all_dataset_targets(s))

    if not all_targets:
        logger.error(f"No valid dataset splits found in {[str(s) for s in sources]}")
        sys.exit(1)

    logger.info(f"Discovered {len(all_targets)} dataset split targets across {len(sources)} source directories.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_images_saved = 0
    start_time = time.time()

    for ds_name, ds_root, split in all_targets:
        try:
            ds = CrowdDataset(
                ds_root,
                split,
                crop_size=args.crop_size,
                output_stride=args.output_stride,
                dynamic_crop=False,
                augment=False,
            )
        except Exception as e:
            logger.warning(f"Skipping {ds_name} ({split}) due to error loading: {e}")
            continue

        num_samples = len(ds) if args.all_samples else min(args.samples_per_split, len(ds))
        if num_samples <= 0:
            continue

        target_out_dir = args.output_dir / ds_name / split
        target_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Visualizing [{ds_name} / {split}] ({num_samples}/{len(ds)} samples, mode={args.mode}) -> {target_out_dir}")

        for idx in range(num_samples):
            try:
                if args.mode == "full":
                    full_item = ds.full_image(idx)
                    image_id = str(full_item.get("image_id", f"idx_{idx:04d}")).replace("/", "_").replace("\\", "_")
                    count = float(full_item.get("count_gt", 0.0).item() if isinstance(full_item.get("count_gt"), torch.Tensor) else full_item.get("count_gt", 0.0))
                    fig = create_full_image_inspection_figure(
                        image=full_item["image"],
                        points=full_item.get("points"),
                        count=count,
                        image_id=image_id,
                        output_stride=args.output_stride,
                        colormap=args.colormap,
                    )
                else:
                    sample = ds[idx]
                    image_id = str(sample.get("image_id", f"idx_{idx:04d}")).replace("/", "_").replace("\\", "_")
                    count = float(sample.get("count_gt", 0.0))
                    fig = create_dataset_inspection_figure(sample, colormap=args.colormap)

                out_file = target_out_dir / f"{idx + 1:03d}_{image_id}_cnt{int(round(count))}.jpg"
                save_figure(fig, out_file)
                total_images_saved += 1
            except Exception as e:
                logger.error(f"Failed to visualize sample {idx} of {ds_name} ({split}): {e}")

    duration = time.time() - start_time
    logger.success(
        f"Finished batch visualization! Total {total_images_saved} inspection figures saved to {args.output_dir} in {duration:.1f}s"
    )


if __name__ == "__main__":
    main()
