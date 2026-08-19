"""人群计数与密度图可视化工具模块。

提供密度热力图生成、原图热力图叠加融合、点标注绘制、多面板对比图生成以及散点回归图等功能。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def tensor_to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """将张量或数组格式的图像统一转换为 uint8 RGB NumPy 数组 (H, W, 3)。"""
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.ndim == 3 and img.shape[0] in (1, 3):  # (C, H, W)
            img = img.permute(1, 2, 0)
        img_np = img.numpy()
    else:
        img_np = np.asarray(image)

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)

    if np.issubdtype(img_np.dtype, np.floating):
        if img_np.max() <= 1.05 and img_np.min() >= -0.05:
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    elif img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return img_np


def density_to_heatmap(
    density: torch.Tensor | np.ndarray,
    colormap: str = "jet",
    normalize: bool = True,
) -> np.ndarray:
    """将二维密度图转换为 RGB 格式的伪彩色热力图 (H, W, 3)，取值范围 [0, 255] uint8。

    Args:
        density: 2D 密度矩阵 [H, W] 或张量 [1, 1, H, W]
        colormap: 颜色映射表名称（'jet', 'viridis', 'magma', 'plasma' 等）
        normalize: 是否按最大值归一化到 [0, 1] 区间
    """
    if isinstance(density, torch.Tensor):
        dens = density.detach().cpu().float()
        while dens.ndim > 2:
            dens = dens.squeeze(0)
        dens_np = dens.numpy()
    else:
        dens_np = np.asarray(density, dtype=np.float32)
        while dens_np.ndim > 2:
            dens_np = dens_np.squeeze(0)

    # 去除负值保证非负性
    dens_np = np.maximum(dens_np, 0.0)

    if normalize:
        max_val = float(np.max(dens_np)) if dens_np.size > 0 else 0.0
        if max_val > 1e-8:
            norm_dens = dens_np / max_val
        else:
            norm_dens = np.zeros_like(dens_np)
    else:
        norm_dens = np.clip(dens_np, 0.0, 1.0)

    cmap = plt.get_cmap(colormap)
    rgba = cmap(norm_dens)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def overlay_density_on_image(
    image: torch.Tensor | np.ndarray | Image.Image,
    density: torch.Tensor | np.ndarray,
    alpha: float = 0.55,
    colormap: str = "jet",
) -> np.ndarray:
    """将密度热力图按指定透明度叠加到原图上。

    自动将密度图双线性插值到原图大小，输出 uint8 RGB (H, W, 3) 数组。
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = tensor_to_numpy_image(image)

    target_h, target_w = img_np.shape[:2]

    # 将密度图缩放到原图空间大小
    if isinstance(density, np.ndarray):
        density_t = torch.from_numpy(density).float()
    else:
        density_t = density.detach().cpu().float()

    while density_t.ndim < 4:
        density_t = density_t.unsqueeze(0)

    if density_t.shape[-2:] != (target_h, target_w):
        resized_density = F.interpolate(
            density_t,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    else:
        resized_density = density_t.squeeze()

    heatmap = density_to_heatmap(resized_density, colormap=colormap, normalize=True)

    # 混合原图与热力图
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = ((1.0 - alpha) * img_np.astype(np.float32) + alpha * heatmap.astype(np.float32)).astype(np.uint8)
    return blended


def draw_points_on_image(
    image: torch.Tensor | np.ndarray | Image.Image,
    points: torch.Tensor | np.ndarray | Sequence[Sequence[float]],
    radius: int = 3,
    color: tuple[int, int, int] = (255, 30, 30),
    outline: tuple[int, int, int] | None = (255, 255, 255),
) -> np.ndarray:
    """在图像上绘制点标注（人头中心散点），返回 uint8 RGB 数组。"""
    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB").copy()
    else:
        img_np = tensor_to_numpy_image(image)
        pil_img = Image.fromarray(img_np)

    if isinstance(points, torch.Tensor):
        pts = points.detach().cpu().numpy()
    else:
        pts = np.asarray(points)

    if pts.size > 0 and pts.ndim == 2 and pts.shape[1] >= 2:
        draw = ImageDraw.Draw(pil_img)
        r = max(1, int(radius))
        for x, y in pts:
            draw.ellipse(
                [(x - r, y - r), (x + r, y + r)],
                fill=color,
                outline=outline,
            )

    return np.array(pil_img)


def create_composite_figure(
    image: torch.Tensor | np.ndarray | Image.Image,
    pred_density: torch.Tensor | np.ndarray,
    *,
    gt_density: torch.Tensor | np.ndarray | None = None,
    points: torch.Tensor | np.ndarray | None = None,
    pred_count: float | None = None,
    gt_count: float | None = None,
    title: str | None = None,
    colormap: str = "jet",
    alpha: float = 0.55,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """构建多面板对比图（Original + Points / GT Density / Pred Density / Overlay）。

    返回 matplotlib.figure.Figure 对象。
    """
    img_np = tensor_to_numpy_image(image)
    if pred_count is None and pred_density is not None:
        if isinstance(pred_density, torch.Tensor):
            pred_count = float(pred_density.sum().item())
        else:
            pred_count = float(np.sum(pred_density))

    if gt_count is None and gt_density is not None:
        if isinstance(gt_density, torch.Tensor):
            gt_count = float(gt_density.sum().item())
        else:
            gt_count = float(np.sum(gt_density))
    elif gt_count is None and points is not None:
        gt_count = float(len(points))

    # 绘制带点标注的原图
    if points is not None and len(points) > 0:
        marked_img = draw_points_on_image(img_np, points, radius=3)
    else:
        marked_img = img_np

    pred_heatmap = density_to_heatmap(pred_density, colormap=colormap)
    overlay = overlay_density_on_image(img_np, pred_density, alpha=alpha, colormap=colormap)

    has_gt = gt_density is not None
    cols = 4 if has_gt else 3
    fig_w = (cols * 4.2) if figsize is None else figsize[0]
    fig_h = 4.0 if figsize is None else figsize[1]

    fig, axes = plt.subplots(1, cols, figsize=(fig_w, fig_h), squeeze=False)
    ax_list = axes[0]

    # 面板 1: 原图（带点）
    ax_list[0].imshow(marked_img)
    gt_str = f" (GT: {gt_count:.1f})" if gt_count is not None else ""
    ax_list[0].set_title(f"Input Image{gt_str}", fontsize=11, fontweight="bold")
    ax_list[0].axis("off")

    idx = 1
    # 面板 2 (可选): GT 密度图
    if has_gt:
        assert gt_density is not None
        gt_heat = density_to_heatmap(gt_density, colormap=colormap)
        ax_list[idx].imshow(gt_heat)
        gt_title = f"GT Density (Count: {gt_count:.1f})" if gt_count is not None else "GT Density"
        ax_list[idx].set_title(gt_title, fontsize=11, fontweight="bold", color="green")
        ax_list[idx].axis("off")
        idx += 1

    # 面板 3: 预测密度图
    ax_list[idx].imshow(pred_heatmap)
    pred_title = f"Pred Density (Count: {pred_count:.1f})" if pred_count is not None else "Pred Density"
    ax_list[idx].set_title(pred_title, fontsize=11, fontweight="bold", color="crimson")
    ax_list[idx].axis("off")
    idx += 1

    # 面板 4: 叠加图
    ax_list[idx].imshow(overlay)
    if gt_count is not None and pred_count is not None:
        err = abs(pred_count - gt_count)
        overlay_title = f"Overlay (Err: {err:.1f})"
    else:
        overlay_title = "Overlay"
    ax_list[idx].set_title(overlay_title, fontsize=11, fontweight="bold")
    ax_list[idx].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout()
    return fig


def figure_to_image(fig: plt.Figure) -> Image.Image:
    """将 Matplotlib Figure 对象转换为 PIL Image。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def figure_to_tensor(fig: plt.Figure) -> torch.Tensor:
    """将 Matplotlib Figure 转换为 TensorBoard 所需的 [3, H, W] 浮点张量 [0, 1]。"""
    img = figure_to_image(fig)
    raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return raw.reshape(img.height, img.width, 3).permute(2, 0, 1).float().div_(255.0)


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """保存 Matplotlib 图像至指定文件路径并安全释放资源。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    if close:
        plt.close(fig)


def plot_count_scatter(
    gt_counts: Sequence[float] | torch.Tensor | np.ndarray,
    pred_counts: Sequence[float] | torch.Tensor | np.ndarray,
    metrics: dict[str, float] | None = None,
    title: str = "Ground Truth vs. Predicted Count",
    figsize: tuple[float, float] = (6.5, 6.0),
) -> plt.Figure:
    """绘制真实人数 vs 预测人数的回归散点图与理想对角线。"""
    if isinstance(gt_counts, torch.Tensor):
        gts = gt_counts.detach().cpu().numpy().reshape(-1)
    else:
        gts = np.asarray(gt_counts, dtype=np.float32).reshape(-1)

    if isinstance(pred_counts, torch.Tensor):
        preds = pred_counts.detach().cpu().numpy().reshape(-1)
    else:
        preds = np.asarray(pred_counts, dtype=np.float32).reshape(-1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(gts, preds, alpha=0.6, edgecolors="none", c="#1f77b4", s=30, label="Samples")

    min_val = min(float(np.min(gts)), float(np.min(preds)), 0.0)
    max_val = max(float(np.max(gts)), float(np.max(preds)), 1.0)
    margin = (max_val - min_val) * 0.05
    line_min = max(0.0, min_val - margin)
    line_max = max_val + margin

    # 绘制理想参考线 y = x
    ax.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=1.8, label="Ideal (y = x)")

    ax.set_xlabel("Ground Truth Count", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Count", fontsize=11, fontweight="bold")
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.grid(True, linestyle=":", alpha=0.6)

    # 指标说明文本框
    if metrics:
        metrics_text = "\n".join(f"{k.upper()}: {v:.3f}" for k, v in metrics.items())
        ax.text(
            0.05,
            0.92,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.85)
    plt.tight_layout()
    return fig
