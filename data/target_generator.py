"""点坐标到概率图与点坐标到密度图的标签生成。

密度标签定义在输出网格上，每个可见点恰好贡献一个单位的质量，
包括高斯核被裁剪边界截断的点。概率标签刻意不使用这种归一化。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class TargetConfig:
    """步长缩减的人群标签配置。"""

    output_size: int | tuple[int, int] = 160
    output_stride: int = 4
    probability_sigma: float = 2.0
    density_sigma: float = 2.0
    probability_radius: int | None = None
    density_radius: int | None = None
    adaptive_density: bool = False
    adaptive_beta: float = 0.3
    adaptive_knn: int = 3

    def resolved_size(self) -> tuple[int, int]:
        if isinstance(self.output_size, int):
            return self.output_size, self.output_size
        if len(self.output_size) != 2:
            raise ValueError("output_size must be an int or (height, width)")
        return int(self.output_size[0]), int(self.output_size[1])


def _as_points(points: torch.Tensor | Sequence[Sequence[float]] | Iterable[Sequence[float]]) -> torch.Tensor:
    """将类点输入转换为有限的 ``[N, 2]`` 浮点张量。"""

    result = torch.as_tensor(points, dtype=torch.float32)
    if result.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"points must have shape [N, 2], got {tuple(result.shape)}")
    if not torch.isfinite(result).all():
        raise ValueError("points contain NaN or infinite coordinates")
    return result


def crop_points(
    points: torch.Tensor | Sequence[Sequence[float]],
    crop_origin: tuple[float, float],
    crop_size: tuple[int, int] | int,
) -> torch.Tensor:
    """按半开区间裁剪规则选取点，并平移为裁剪坐标。

    一个点属于 ``[x0, x0 + width) x [y0, y0 + height)``。半开
    区间可防止位于相邻瓦片边界上的点被重复计数。
    """

    points_t = _as_points(points)
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("crop dimensions must be positive")
    x0, y0 = float(crop_origin[0]), float(crop_origin[1])
    # 半开区间 [x0, x0+w) × [y0, y0+h)：落在相邻瓦片共享边界上的点
    # 恰好只属于一个瓦片，分块评估时不会被重复计数。
    mask = (
        (points_t[:, 0] >= x0)
        & (points_t[:, 0] < x0 + crop_w)
        & (points_t[:, 1] >= y0)
        & (points_t[:, 1] < y0 + crop_h)
    )
    translated = points_t[mask].clone()
    if translated.numel():
        translated[:, 0] -= x0
        translated[:, 1] -= y0
    return translated


def _draw_gaussian(
    target: torch.Tensor,
    center_xy: tuple[float, float],
    sigma: float,
    *,
    normalize: bool,
    radius: int | None = None,
) -> None:
    """将单个高斯核原地绘制进 ``target``。

    ``target`` 按 ``[height, width]`` 索引。对于密度标签，可见部分在
    裁剪后会被归一化，这正是人群计数所需的守恒
    不变量。
    """

    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")
    height, width = target.shape[-2:]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    # 默认截断半径取 ceil(3σ)：3σ 窗口已覆盖高斯核绝大部分质量，
    # ceil 保证尾部像素不被截断；中心取整到最近像素完成栅格化。
    radius = int(radius if radius is not None else ceil(3.0 * sigma))
    x_center = int(round(cx))
    y_center = int(round(cy))
    x_start = max(0, x_center - radius)
    x_end = min(width - 1, x_center + radius)
    y_start = max(0, y_center - radius)
    y_end = min(height - 1, y_center + radius)
    if x_start > x_end or y_start > y_end:
        return

    ys = torch.arange(y_start, y_end + 1, dtype=target.dtype, device=target.device)
    xs = torch.arange(x_start, x_end + 1, dtype=target.dtype, device=target.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gaussian = torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2.0 * sigma * sigma))
    # 密度分支：被图像边界截断的高斯核按可见部分重新归一化，
    # 每个点恰好贡献 1 个单位质量，这是 Σ密度图 = 人数守恒的保证。
    if normalize:
        mass = gaussian.sum()
        if mass <= 0:
            return
        gaussian = gaussian / mass
        target[y_start : y_end + 1, x_start : x_end + 1] += gaussian
    else:
        # 概率分支：取逐点最大值而非累加。重叠头部在概率图中不会
        # 被重复计数超过 1，从而保持似然值落在 [0, 1] 的语义。
        target[y_start : y_end + 1, x_start : x_end + 1] = torch.maximum(
            target[y_start : y_end + 1, x_start : x_end + 1], gaussian
        )


def _adaptive_sigmas(points: torch.Tensor, base_sigma: float, beta: float, knn: int) -> torch.Tensor:
    """返回可选的按点自适应 sigma（以输出单元为单位）。"""

    if len(points) <= 1:
        return torch.full((len(points),), base_sigma, dtype=points.dtype)
    # 几何自适应核：以 k 近邻平均间距估计局部人群密度，密集处自动
    # 收窄 σ。对角线置 inf 排除自距离；clamp 把 σ 限制在
    # [0.5σ, 4σ]，防止孤立点或极端密集处产生过宽/过窄的病态核。
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    k = max(1, min(int(knn), len(points) - 1))
    nearest = distances.topk(k=k, largest=False, dim=1).values.mean(dim=1)
    return torch.clamp(beta * nearest, min=base_sigma * 0.5, max=base_sigma * 4.0)


def generate_probability_target(
    points: torch.Tensor | Sequence[Sequence[float]],
    *,
    output_size: int | tuple[int, int] = 160,
    output_stride: int = 4,
    sigma: float = 2.0,
    radius: int | None = None,
) -> torch.Tensor:
    """生成非守恒的头部似然图，取值在 ``[0, 1]``。"""

    points_t = _as_points(points)
    if isinstance(output_size, int):
        height, width = output_size, output_size
    else:
        height, width = int(output_size[0]), int(output_size[1])
    if height <= 0 or width <= 0 or output_stride <= 0:
        raise ValueError("output_size and output_stride must be positive")
    target = torch.zeros((height, width), dtype=torch.float32)
    for x, y in points_t:
        _draw_gaussian(
            target,
            (float(x) / output_stride, float(y) / output_stride),
            sigma,
            normalize=False,
            radius=radius,
        )
    return target.clamp_(0.0, 1.0)


def generate_density_target(
    points: torch.Tensor | Sequence[Sequence[float]],
    *,
    output_size: int | tuple[int, int] = 160,
    output_stride: int = 4,
    sigma: float = 2.0,
    radius: int | None = None,
    adaptive: bool = False,
    adaptive_beta: float = 0.3,
    adaptive_knn: int = 3,
) -> torch.Tensor:
    """生成积分等于点数的密度图。

    每个点独立栅格化，其被裁剪的高斯核在累加前重新归一化。
    因此相邻点可能重叠，但任何点都不会因重叠或地图边界
    而损失或增加质量。
    """

    points_t = _as_points(points)
    if isinstance(output_size, int):
        height, width = output_size, output_size
    else:
        height, width = int(output_size[0]), int(output_size[1])
    if height <= 0 or width <= 0 or output_stride <= 0:
        raise ValueError("output_size and output_stride must be positive")
    target = torch.zeros((height, width), dtype=torch.float32)
    if adaptive and len(points_t):
        # σ 需在输出网格单元上度量：先把像素坐标按 stride 缩放，
        # 近邻间距才与高斯绘制时使用的输出单元一致。
        output_points = points_t / float(output_stride)
        sigmas = _adaptive_sigmas(output_points, sigma, adaptive_beta, adaptive_knn)
    else:
        sigmas = torch.full((len(points_t),), float(sigma), dtype=torch.float32)
    for (x, y), point_sigma in zip(points_t, sigmas):
        _draw_gaussian(
            target,
            (float(x) / output_stride, float(y) / output_stride),
            float(point_sigma),
            normalize=True,
            radius=radius,
        )
    return target


def generate_targets(points: torch.Tensor | Sequence[Sequence[float]], config: TargetConfig) -> dict[str, torch.Tensor | float]:
    """生成两张标签图以及权威的点数。"""

    output_size = config.resolved_size()
    probability = generate_probability_target(
        points,
        output_size=output_size,
        output_stride=config.output_stride,
        sigma=config.probability_sigma,
        radius=config.probability_radius,
    )
    density = generate_density_target(
        points,
        output_size=output_size,
        output_stride=config.output_stride,
        sigma=config.density_sigma,
        radius=config.density_radius,
        adaptive=config.adaptive_density,
        adaptive_beta=config.adaptive_beta,
        adaptive_knn=config.adaptive_knn,
    )
    # 权威点数直接取自点标注行数，作为密度/概率图之外的回归目标，
    # 供 count 损失以及评估时的 MAE/MSE 使用。
    count = float(_as_points(points).shape[0])
    return {"probability": probability, "density": density, "count": count}


def validate_density_conservation(
    density: torch.Tensor,
    count: torch.Tensor | float | Sequence[float],
    tolerance: float = 1e-4,
) -> None:
    """若各密度图之和与其点数不一致，则抛出 ``AssertionError``。"""

    density_t = torch.as_tensor(density)
    count_t = torch.as_tensor(count, dtype=density_t.dtype, device=density_t.device)
    if density_t.ndim == 2:
        sums = density_t.sum().reshape(1)
    elif density_t.ndim >= 3:
        sums = density_t.reshape(density_t.shape[0], -1).sum(dim=1)
    else:
        raise ValueError("density must have at least two dimensions")
    count_t = count_t.reshape(-1)
    if sums.numel() != count_t.numel():
        raise ValueError(f"density batch ({sums.numel()}) and count batch ({count_t.numel()}) differ")
    # 浮点累加与重归一化的舍入误差通常在 1e-4 量级，该容差只用于
    # 排除结构性错误（如点坐标未平移、越界点被静默丢弃）。
    error = (sums - count_t).abs().max().item() if sums.numel() else 0.0
    assert error <= tolerance, f"density/count conservation failed: max error={error:.6g}"
