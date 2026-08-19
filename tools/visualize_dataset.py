"""数据集与标签生成质检可视化工具。

支持整图（Full Image）与裁剪块（Patch）两种模式：
- 默认整图模式：直观展示完整图像及其真实人头点、全图高斯密度热力图、概率图与叠加图；
- 裁剪模式：可检验动态采样窗口与标签切片。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
from data.target_generator import generate_density_target, generate_probability_target
from utils.visualization import (
    create_composite_figure,
    density_to_heatmap,
    draw_points_on_image,
    overlay_density_on_image,
    save_figure,
    tensor_to_numpy_image,
)


def create_full_image_inspection_figure(
    image: torch.Tensor | np.ndarray | Image.Image,
    points: torch.Tensor | np.ndarray,
    count: float,
    image_id: str = "sample",
    output_stride: int = 4,
    colormap: str = "jet",
    density_sigma: float = 2.0,
) -> plt.Figure:
    """构建整图全景 4 面板质检图 (Full Image+Points / Full Density GT / Full Prob GT / Full Overlay)。"""
    if isinstance(image, torch.Tensor):
        img_h, img_w = image.shape[-2:]
    elif isinstance(image, Image.Image):
        img_w, img_h = image.size
    else:
        img_h, img_w = image.shape[:2]

    out_h = max(1, img_h // output_stride)
    out_w = max(1, img_w // output_stride)

    pts_t = torch.as_tensor(points, dtype=torch.float32) if points is not None else torch.empty((0, 2), dtype=torch.float32)
    if pts_t.ndim == 2 and pts_t.shape[0] > 0 and pts_t.max() <= 1.000001:
        pts_abs = pts_t.clone()
        pts_abs[:, 0] *= max(img_w - 1, 1)
        pts_abs[:, 1] *= max(img_h - 1, 1)
    else:
        pts_abs = pts_t

    density = generate_density_target(pts_abs, output_size=(out_h, out_w), output_stride=output_stride, sigma=density_sigma)
    prob = generate_probability_target(pts_abs, output_size=(out_h, out_w), output_stride=output_stride, sigma=density_sigma)

    img_np = tensor_to_numpy_image(image)
    radius = max(2, int(round(min(img_w, img_h) / 300.0)))
    marked = draw_points_on_image(img_np, pts_abs, radius=radius)
    dens_heat = density_to_heatmap(density, colormap=colormap)
    prob_heat = density_to_heatmap(prob, colormap="magma")
    overlay = overlay_density_on_image(img_np, density, alpha=0.55, colormap=colormap)

    aspect_ratio = img_w / max(img_h, 1)
    panel_w = max(3.5, min(5.5, 3.8 * aspect_ratio))
    panel_h = 4.0
    fig, axes = plt.subplots(1, 4, figsize=(panel_w * 4, panel_h))

    axes[0].imshow(marked)
    axes[0].set_title(f"Full Image + Points (N={int(round(count))})", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(dens_heat)
    axes[1].set_title(f"Full Density GT (Sum: {density.sum().item():.1f})", fontsize=10, fontweight="bold", color="green")
    axes[1].axis("off")

    axes[2].imshow(prob_heat)
    axes[2].set_title("Full Probability GT", fontsize=10, fontweight="bold", color="purple")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title(f"Density Overlay\nID: {image_id}", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    return fig


def create_dataset_inspection_figure(
    sample: dict[str, object],
    colormap: str = "jet",
) -> plt.Figure:
    """构建数据集单裁剪样本的 4 面板质检图。"""
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

    axes[0].imshow(marked_img)
    crop_mode = crop_info.get("mode", "fixed") if isinstance(crop_info, dict) else "fixed"
    axes[0].set_title(f"Patch + Points (N={int(round(count_gt))})\nMode: {crop_mode}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(density_heat)
    diff = abs(dens_sum - count_gt)
    conserved_str = "Conserved" if diff < 1e-3 else f"Drift: {diff:.4f}"
    axes[1].set_title(f"Density GT (Sum: {dens_sum:.2f})\n{conserved_str}", fontsize=10, fontweight="bold", color="green")
    axes[1].axis("off")

    idx = 2
    if prob_heat is not None:
        axes[idx].imshow(prob_heat)
        axes[idx].set_title("Probability GT (Likelihood)\nKernel: Sigmoid/Max", fontsize=10, fontweight="bold", color="purple")
        axes[idx].axis("off")
        idx += 1

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
    parser.add_argument("--mode", choices=["full", "patch"], default="full", help="Inspection mode: full image or crop patch")
    parser.add_argument("--crop-size", type=int, default=None, help="Override crop size")
    parser.add_argument("--dynamic-crop", action="store_true", default=None, help="Enable dynamic multi-mode cropping")
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
    logger.info(f"Loaded dataset '{data_root}' split '{args.split}' with {len(dataset)} images (mode={args.mode}).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = min(args.num_samples, len(dataset))

    for i in range(num_samples):
        if args.mode == "full":
            item = dataset.full_image(i)
            image_id = str(item.get("image_id", f"sample_{i}")).replace("/", "_").replace("\\", "_")
            count = float(item["count_gt"].item())
            fig = create_full_image_inspection_figure(
                image=item["image"],
                points=item.get("points"),
                count=count,
                image_id=image_id,
                output_stride=output_stride,
                colormap=args.colormap,
            )
        else:
            sample = dataset[i]
            image_id = str(sample.get("image_id", f"sample_{i}")).replace("/", "_").replace("\\", "_")
            fig = create_dataset_inspection_figure(sample, colormap=args.colormap)

        save_path = args.output_dir / f"{args.split}_{i + 1:03d}_{image_id}.jpg"
        save_figure(fig, save_path)
        logger.info(f"[{i + 1}/{num_samples}] Saved inspection figure -> {save_path}")

    logger.success(f"Dataset inspection complete. {num_samples} figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
