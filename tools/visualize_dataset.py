"""数据集与标签生成质检可视化脚本。

用于在训练前检查点标注解析、高斯密度图积分守恒性、概率图似然度以及动态裁剪采样的视觉质量。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image

from data.crowd_dataset import CrowdDataset
from utils.visualization import (
    create_composite_figure,
    density_to_heatmap,
    draw_points_on_image,
    overlay_density_on_image,
    save_figure,
    tensor_to_numpy_image,
)


def create_dataset_inspection_figure(
    sample: dict[str, object],
    colormap: str = "jet",
) -> plt.Figure:
    """构建数据集单样本的 4 面板质检图 (Image+Points / Density GT / Prob GT / Overlay)。"""
    image_tensor = sample["image"]
    points = sample.get("points", None)
    density_gt = sample["density_gt"]
    prob_gt = sample.get("probability_gt", None)
    count_gt = float(sample["count_gt"])
    crop_info = sample.get("crop_info", {})
    image_id = str(sample.get("image_id", "sample"))

    if isinstance(density_gt, torch.Tensor):
        dens_sum = float(density_gt.sum().item())
    else:
        dens_sum = float(np.sum(density_gt))

    img_np = tensor_to_numpy_image(image_tensor)
    marked_img = draw_points_on_image(img_np, points, radius=3) if points is not None else img_np
    density_heat = density_to_heatmap(density_gt, colormap=colormap)
    prob_heat = density_to_heatmap(prob_gt, colormap="magma") if prob_gt is not None else None
    overlay = overlay_density_on_image(img_np, density_gt, alpha=0.55, colormap=colormap)

    cols = 4 if prob_heat is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4.2, 4.2))

    # 1. 裁剪原图 + 点标注
    axes[0].imshow(marked_img)
    crop_mode = crop_info.get("mode", "fixed") if isinstance(crop_info, dict) else "fixed"
    axes[0].set_title(f"Patch + Points (N={int(round(count_gt))})\nMode: {crop_mode}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # 2. 真实密度图 GT
    axes[1].imshow(density_heat)
    diff = abs(dens_sum - count_gt)
    conserved_str = "Conserved" if diff < 1e-3 else f"Drift: {diff:.4f}"
    axes[1].set_title(f"Density GT (Sum: {dens_sum:.2f})\n{conserved_str}", fontsize=10, fontweight="bold", color="green")
    axes[1].axis("off")

    idx = 2
    # 3. 概率图 GT
    if prob_heat is not None:
        axes[idx].imshow(prob_heat)
        axes[idx].set_title("Probability GT (Likelihood)\nKernel: Sigmoid/Max", fontsize=10, fontweight="bold", color="purple")
        axes[idx].axis("off")
        idx += 1

    # 4. 密度图与原图叠加
    axes[idx].imshow(overlay)
    axes[idx].set_title(f"Density Overlay\nID: {image_id}", fontsize=10, fontweight="bold")
    axes[idx].axis("off")

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/crowd.yaml"), help="Path to config file")
    parser.add_argument("--data-root", type=Path, default=None, help="Override root directory of dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, val, test)")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of samples to visualize")
    parser.add_argument("--crop-size", type=int, default=None, help="Override crop size")
    parser.add_argument("--dynamic-crop", action="store_true", default=None, help="Enable dynamic multi-mode cropping")
    parser.add_argument("--no-dynamic-crop", dest="dynamic_crop", action="store_false", help="Disable dynamic cropping")
    parser.add_argument("--augment", action="store_true", default=False, help="Enable data augmentations")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dataset_vis"), help="Directory to save visual images")
    parser.add_argument("--colormap", default="jet", help="Colormap for density maps")
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    config: dict[str, object] = {}
    if args.config.exists():
        config = yaml.safe_load(args.config.read_text())

    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    data_root = args.data_root or Path(str(data_cfg.get("root", "data/UCF-QNRF_ECCV18")))
    crop_size = int(args.crop_size or config.get("image_size", 640))
    output_stride = int(config.get("output_stride", 4))

    if not data_root.exists():
        logger.error(f"Dataset root '{data_root}' not found. Please verify the dataset path.")
        sys.exit(1)

    dataset = CrowdDataset(
        data_root,
        args.split,
        crop_size=crop_size,
        output_stride=output_stride,
        dynamic_crop=args.dynamic_crop,
        augment=args.augment,
    )
    logger.info(f"Loaded dataset '{data_root}' split '{args.split}' with {len(dataset)} images.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = min(args.num_samples, len(dataset))

    for i in range(num_samples):
        sample = dataset[i]
        image_id = str(sample.get("image_id", f"sample_{i}")).replace("/", "_").replace("\\", "_")
        fig = create_dataset_inspection_figure(sample, colormap=args.colormap)
        save_path = args.output_dir / f"{args.split}_{i + 1:03d}_{image_id}.jpg"
        save_figure(fig, save_path)
        logger.info(f"[{i + 1}/{num_samples}] Saved inspection figure -> {save_path}")

    logger.success(f"Dataset inspection complete. {num_samples} figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
