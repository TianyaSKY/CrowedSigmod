from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from utils.visualization import (
    density_to_heatmap,
    draw_points_on_image,
    figure_to_image,
    figure_to_tensor,
    overlay_density_on_image,
    create_composite_figure,
    plot_count_scatter,
    save_figure,
    tensor_to_numpy_image,
)


def test_tensor_to_numpy_image() -> None:
    t = torch.rand(3, 100, 100)
    arr = tensor_to_numpy_image(t)
    assert arr.shape == (100, 100, 3)
    assert arr.dtype == np.uint8

    t4 = torch.rand(1, 3, 50, 60)
    arr4 = tensor_to_numpy_image(t4)
    assert arr4.shape == (50, 60, 3)
    assert arr4.dtype == np.uint8


def test_density_to_heatmap() -> None:
    density = torch.zeros(40, 40)
    density[10:20, 10:20] = 0.5
    heatmap = density_to_heatmap(density, colormap="jet")
    assert heatmap.shape == (40, 40, 3)
    assert heatmap.dtype == np.uint8


def test_overlay_density_on_image() -> None:
    img = torch.zeros(3, 100, 100)
    density = torch.ones(25, 25)  # 步长缩减后的密度图
    overlay = overlay_density_on_image(img, density, alpha=0.5)
    assert overlay.shape == (100, 100, 3)
    assert overlay.dtype == np.uint8


def test_draw_points_on_image() -> None:
    img = torch.zeros(3, 100, 100)
    points = torch.tensor([[20.0, 30.0], [50.0, 60.0]])
    marked = draw_points_on_image(img, points, radius=2)
    assert marked.shape == (100, 100, 3)
    assert marked.dtype == np.uint8


def test_create_composite_figure_and_save(tmp_path: Path) -> None:
    img = torch.rand(3, 120, 120)
    pred_density = torch.rand(30, 30)
    gt_density = torch.rand(30, 30)
    points = torch.tensor([[10.0, 10.0], [40.0, 50.0]])

    fig = create_composite_figure(
        image=img,
        pred_density=pred_density,
        gt_density=gt_density,
        points=points,
        pred_count=12.5,
        gt_count=10.0,
        title="Test Composite",
    )
    img_pil = figure_to_image(fig)
    assert isinstance(img_pil, Image.Image)

    tensor = figure_to_tensor(fig)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 3

    save_path = tmp_path / "composite.png"
    save_figure(fig, save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_count_scatter(tmp_path: Path) -> None:
    gts = [10.0, 20.0, 30.0, 45.0]
    preds = [11.0, 19.5, 32.0, 41.0]
    metrics = {"mae": 1.5, "rmse": 2.1, "nae": 0.08}
    fig = plot_count_scatter(gts, preds, metrics=metrics, title="Eval Scatter")
    save_path = tmp_path / "scatter.png"
    save_figure(fig, save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_create_dataset_inspection_figure(tmp_path: Path) -> None:
    from tools.visualize_dataset import create_dataset_inspection_figure

    sample = {
        "image": torch.rand(3, 160, 160),
        "points": torch.tensor([[30.0, 40.0], [80.0, 100.0]]),
        "density_gt": torch.rand(1, 40, 40),
        "probability_gt": torch.rand(1, 40, 40),
        "count_gt": 2.0,
        "crop_info": {"mode": "head-centered"},
        "image_id": "test_sample_01",
    }
    fig = create_dataset_inspection_figure(sample)
    save_path = tmp_path / "dataset_sample.jpg"
    save_figure(fig, save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0

