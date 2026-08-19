"""整图人群计数的验证指标。"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from tqdm import tqdm

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
    samples: Iterable[tuple[torch.Tensor, torch.Tensor] | dict[str, Any]],
    *,
    tiler: DensityTiler | None = None,
    device: str | torch.device | None = None,
    total_samples: int | None = None,
    show_pbar: bool = True,
    return_details: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, Any]]]:
    """评估确定性的整图；每个样本为 ``(image, count_gt)`` 或 dict 结构。"""

    tiler = tiler or DensityTiler()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    detailed_records: list[dict[str, Any]] = []

    iterator = (
        tqdm(samples, total=total_samples, desc="Evaluating", dynamic_ncols=True, leave=False)
        if show_pbar
        else samples
    )

    for item in iterator:
        if isinstance(item, dict):
            image = item["image"]
            count_gt = item["count_gt"]
            image_id = item.get("image_id", "")
        else:
            image, count_gt = item[0], item[1]
            image_id = ""

        result = tiler(model, image, device=device)
        pred_cnt = result.count.cpu()
        tgt_cnt = torch.as_tensor(count_gt).reshape(1).cpu()

        predictions.append(pred_cnt)
        targets.append(tgt_cnt)

        if return_details:
            detailed_records.append(
                {
                    "image": image.cpu(),
                    "pred_count": float(pred_cnt.item()),
                    "target_count": float(tgt_cnt.item()),
                    "error": float((pred_cnt - tgt_cnt).item()),
                    "density": result.density.cpu(),
                    "image_id": image_id,
                }
            )

    metrics = counting_metrics(torch.cat(predictions), torch.cat(targets))
    if return_details:
        return metrics, detailed_records
    return metrics

