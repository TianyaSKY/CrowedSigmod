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
    # 起点按 stride 均匀铺开，且上界为 length - tile_size，保证每个
    # tile 都不越出图像；stride 已由调用方对齐到 output_stride 的整数倍。
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

    # 用 (i + 0.5) 采样使窗在两端取非零值（Hann 窗端点为 0），
    # clamp_min(1e-3) 兜底保证任意位置的权重和恒大于零：归一化不会
    # 除零，也不会在 tile 接缝处出现权重为零的暗线。
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
    batch_size: int = 8,
    use_amp: bool = True,
    device: str | torch.device | None = None,
) -> TiledPrediction:
    """在重叠的分块上运行模型，并在输出分辨率下融合密度。

    拼接的是密度而非每块的人数。每个分块的密度被放置在按步长缩小后的
    起点处，并与正余弦掩码相乘累加，再除以累加得到的掩码。这样可以避免
    重叠区域被重复计数。支持分批并行推理(batch_size)与混合精度(use_amp)大幅提升大图推理效率。
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
    # 设备未显式指定时从模型参数推断，避免输入与模型所在设备不一致。
    image = image.to(run_device)
    _, _, height, width = image.shape
    # 输出网格按天花板取整（ceil），保证整图都映射进输出坐标。
    output_height = (height + output_stride - 1) // output_stride
    output_width = (width + output_stride - 1) // output_stride
    tile_output_size = tile_size // output_stride
    density_sum = torch.zeros((1, 1, output_height, output_width), device=run_device, dtype=torch.float32)
    weight_sum = torch.zeros_like(density_sum)
    window = cosine_blend_window(tile_output_size, tile_output_size, device=run_device, dtype=torch.float32)[None, None]

    was_training = model.training
    model.eval()

    tile_step = output_stride * max(1, tile_stride // output_stride)
    y_starts = _tile_starts(height, tile_size, tile_step, alignment=output_stride)
    x_starts = _tile_starts(width, tile_size, tile_step, alignment=output_stride)
    coords = [(y0, x0) for y0 in y_starts for x0 in x_starts]

    with torch.inference_mode():
        step_bs = max(1, batch_size)
        for i in range(0, len(coords), step_bs):
            batch_coords = coords[i : i + step_bs]
            tiles = []
            for y0, x0 in batch_coords:
                y1, x1 = min(y0 + tile_size, height), min(x0 + tile_size, width)
                tile = image[:, :, y0:y1, x0:x1]
                pad_h, pad_w = tile_size - tile.shape[-2], tile_size - tile.shape[-1]
                if pad_h or pad_w:
                    # 起点上界已保证 tile 不越出图像，正常不会走到这里；
                    # 兜底在右/下补零，维持固定输入尺寸供模型前向。
                    tile = F.pad(tile, (0, pad_w, 0, pad_h))
                tiles.append(tile)

            batch_input = torch.cat(tiles, dim=0)
            with torch.amp.autocast(device_type=run_device.type, enabled=(use_amp and run_device.type == "cuda")):
                outputs = model(batch_input)
                densities = outputs["density"].float()

            if densities.shape[-2:] != (tile_output_size, tile_output_size):
                # 模型输出可能与 tile_size // output_stride 不一致
                # （如骨干自带更大下采样），统一双线性插值到该网格，
                # 保证所有 tile 可在同一输出坐标系下累加。
                densities = F.interpolate(
                    densities, size=(tile_output_size, tile_output_size), mode="bilinear", align_corners=False
                )

            for b, (y0, x0) in enumerate(batch_coords):
                dens = densities[b : b + 1]
                oy, ox = y0 // output_stride, x0 // output_stride
                valid_h = min(tile_output_size, output_height - oy)
                valid_w = min(tile_output_size, output_width - ox)
                if valid_h <= 0 or valid_w <= 0:
                    continue
                local_window = window[:, :, :valid_h, :valid_w]
                # 累加的是密度场（每像素人数）乘以窗权重的加权和，而非
                # 各 tile 的总人数——拼接计数会把重叠区域重复统计；
                # 除以累加权重得到无偏的加权平均密度。
                density_sum[:, :, oy : oy + valid_h, ox : ox + valid_w] += dens[:, :, :valid_h, :valid_w] * local_window
                weight_sum[:, :, oy : oy + valid_h, ox : ox + valid_w] += local_window

    if was_training:
        model.train()
    # 每个输出像素可能被多个 tile 覆盖，除以权重和即按窗权重取平均；
    # clamp eps 防止理论上的零覆盖位置除零。
    density = density_sum / weight_sum.clamp_min(torch.finfo(weight_sum.dtype).eps)
    # 恢复调用前的训练状态：本函数可能在训练循环的评估段被调用，
    # 静默把模型留在 eval 模式会破坏后续训练语义。
    return TiledPrediction(density=density, count=density.flatten(1).sum(dim=1), weight=weight_sum)


class DensityTiler:
    """可调用包装器，保留分块设置以供验证/推理使用。"""

    def __init__(
        self,
        tile_size: int = 640,
        tile_stride: int = 512,
        output_stride: int = 4,
        batch_size: int = 8,
        use_amp: bool = True,
    ) -> None:
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.output_stride = output_stride
        self.batch_size = batch_size
        self.use_amp = use_amp

    def __call__(self, model: torch.nn.Module, image: torch.Tensor, **kwargs: object) -> TiledPrediction:
        call_kwargs: dict[str, object] = {
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "output_stride": self.output_stride,
            "batch_size": self.batch_size,
            "use_amp": self.use_amp,
        }
        call_kwargs.update(kwargs)
        return predict_tiled(
            model,
            image,
            **call_kwargs,  # type: ignore[arg-type]
        )
