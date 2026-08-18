"""面向全分辨率人群图像、考虑重叠的分块推理。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TiledPrediction:
    density: torch.Tensor
    count: torch.Tensor
    weight: torch.Tensor


def _tile_starts(length: int, tile_size: int, stride: int, alignment: int = 1) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        # 保持最后一个起点与模型输出网格对齐；随后的填充
        # 覆盖剩余像素，不留下零权重单元。
        aligned_last = max(0, last - (last % max(alignment, 1)))
        if aligned_last not in starts:
            starts.append(aligned_last)
    return sorted(set(starts))


def cosine_blend_window(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """正的二维余弦窗；与 Hann 窗不同，它不会留下零覆盖区域。"""

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    wy = 0.5 - 0.5 * torch.cos(torch.pi * (y + 0.5) / max(height, 1))
    wx = 0.5 - 0.5 * torch.cos(torch.pi * (x + 0.5) / max(width, 1))
    return (wy[:, None] * wx[None, :]).clamp_min(1e-3)


def predict_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    tile_size: int = 640,
    tile_stride: int = 512,
    output_stride: int = 4,
    device: str | torch.device | None = None,
) -> TiledPrediction:
    """在重叠的分块上运行模型，并在输出分辨率下融合密度。

    拼接的是密度而非每块的人数。每个分块的密度被放置在按步长缩小后的
    起点处，并与正余弦掩码相乘累加，再除以累加得到的掩码。这样可以避免
    重叠区域被重复计数。
    """

    if tile_size <= 0 or tile_stride <= 0 or output_stride <= 0:
        raise ValueError("tile_size, tile_stride and output_stride must be positive")
    if tile_size % output_stride:
        raise ValueError("tile_size must be divisible by output_stride")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("image must have shape [3,H,W] or [1,3,H,W]")
    run_device = torch.device(device) if device is not None else next(model.parameters(), torch.empty(0)).device
    image = image.to(run_device)
    _, _, height, width = image.shape
    output_height = (height + output_stride - 1) // output_stride
    output_width = (width + output_stride - 1) // output_stride
    tile_output_size = tile_size // output_stride
    density_sum = torch.zeros((1, 1, output_height, output_width), device=run_device, dtype=image.dtype)
    weight_sum = torch.zeros_like(density_sum)
    window = cosine_blend_window(tile_output_size, tile_output_size, device=run_device, dtype=image.dtype)[None, None]

    was_training = model.training
    model.eval()
    with torch.no_grad():
        tile_step = output_stride * max(1, tile_stride // output_stride)
        for y0 in _tile_starts(height, tile_size, tile_step, alignment=output_stride):
            for x0 in _tile_starts(width, tile_size, tile_step, alignment=output_stride):
                y1, x1 = min(y0 + tile_size, height), min(x0 + tile_size, width)
                tile = image[:, :, y0:y1, x0:x1]
                pad_h, pad_w = tile_size - tile.shape[-2], tile_size - tile.shape[-1]
                if pad_h or pad_w:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h))
                outputs = model(tile)
                density = outputs["density"]
                if density.shape[-2:] != (tile_output_size, tile_output_size):
                    density = F.interpolate(density, size=(tile_output_size, tile_output_size), mode="bilinear", align_corners=False)
                oy, ox = y0 // output_stride, x0 // output_stride
                valid_h = min(tile_output_size, output_height - oy)
                valid_w = min(tile_output_size, output_width - ox)
                if valid_h <= 0 or valid_w <= 0:
                    continue
                local_window = window[:, :, :valid_h, :valid_w]
                density_sum[:, :, oy : oy + valid_h, ox : ox + valid_w] += density[:, :, :valid_h, :valid_w] * local_window
                weight_sum[:, :, oy : oy + valid_h, ox : ox + valid_w] += local_window
    if was_training:
        model.train()
    density = density_sum / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).eps)
    return TiledPrediction(density=density, count=density.flatten(1).sum(dim=1), weight=weight_sum)


class DensityTiler:
    """可调用包装器，保留分块设置以供验证/推理使用。"""

    def __init__(self, tile_size: int = 640, tile_stride: int = 512, output_stride: int = 4) -> None:
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.output_stride = output_stride

    def __call__(self, model: torch.nn.Module, image: torch.Tensor, **kwargs: object) -> TiledPrediction:
        return predict_tiled(
            model,
            image,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
            output_stride=self.output_stride,
            **kwargs,
        )
