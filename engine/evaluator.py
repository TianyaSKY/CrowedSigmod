"""整图人群计数的验证指标。"""

from __future__ import annotations

from typing import Iterable

import torch

from inference.tiled_inference import DensityTiler


def counting_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    # 展平为向量：同一套指标函数可统一处理单图标量或批量预测。
    predicted = predicted.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    if predicted.numel() != target.numel() or not predicted.numel():
        raise ValueError("predicted and target counts must be non-empty and equally sized")
    error = predicted - target
    # MAE：平均绝对误差，单位即人数；RMSE：均方根误差，对离群大误差更
    # 敏感；NAE：误差按目标人数归一（clamp_min(1.0) 防除零，同时避免
    # 目标为 0 或 1 的样本放大误差占比）。
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "nae": float((error.abs() / target.clamp_min(1.0)).mean()),
    }


def evaluate_tiled(
    model: torch.nn.Module,
    samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    tiler: DensityTiler | None = None,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """评估确定性的整图；每个样本为 ``(image, count_gt)``。"""

    tiler = tiler or DensityTiler()
    # 在整图上推理（非随机裁剪/滑动平均），结果确定可复现；每个样本的
    # 真值是整图人数标量，reshape(1) 保持列向量以便 concat 后统一计算。
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for image, count_gt in samples:
        result = tiler(model, image, device=device)
        predictions.append(result.count.cpu())
        targets.append(torch.as_tensor(count_gt).reshape(1).cpu())
    return counting_metrics(torch.cat(predictions), torch.cat(targets))
