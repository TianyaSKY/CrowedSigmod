"""用于 YOLO-PGMD 的概率、密度与人数一致性损失。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


# 四项损失的权重与局部划分粒度集中在此；frozen=True 防止训练流程中意外改写。
@dataclass(frozen=True)
class LossWeights:
    probability: float = 1.0
    density: float = 1.0
    count: float = 0.5
    local: float = 0.25
    local_grid: int = 4


class CrowdLoss(nn.Module):
    """计算加权四项目标损失。

    ``forward`` 返回标量张量，与常规 PyTorch 准则一致。
    ``compute`` 额外暴露命名实时张量，便于日志记录。

    设计要点：概率分支用 BCE+Dice 缓解背景主导的不平衡；密度分支用 SmoothL1
    兼顾峰值误差与平坦区稳定性；count/local 两项以相对误差归一，分别约束全局
    人数与空间分布。
    """

    def __init__(
        self,
        *,
        probability_weight: float = 1.0,
        density_weight: float = 1.0,
        count_weight: float = 0.5,
        local_weight: float = 0.25,
        local_grid: int = 4,
        dice_weight: float = 0.2,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        if local_grid <= 0:
            raise ValueError("local_grid must be positive")
        self.weights = LossWeights(
            probability=probability_weight,
            density=density_weight,
            count=count_weight,
            local=local_weight,
            local_grid=local_grid,
        )
        self.dice_weight = float(dice_weight)
        # beta=1.0：像素误差 <1 时用二次项（零附近平滑，梯度不会剧烈抖动），
        # 误差更大时退化为 L1 —— 密度峰值的离群误差不会被平方放大。
        self.smooth_l1_beta = float(smooth_l1_beta)

    @staticmethod
    def _as_count(count: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 目标可能以标量、[1] 或 [B] 任意形状传入，统一展平成 [B]；
        # 元素数必须与 batch 一致，防止 count 与 batch 错配后被静默广播。
        count = torch.as_tensor(count, device=device, dtype=dtype).reshape(-1)
        if count.numel() != batch_size:
            raise ValueError(f"count target has {count.numel()} values for batch size {batch_size}")
        return count

    def _local_region_sums(self, density: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = density.shape
        grid = self.weights.local_grid
        # 空间尺寸必须能被 grid 整除：这是 reshape 拆块的前提，提前报出明确错误，
        # 比让 reshape 抛晦涩的形状异常更易排查。
        if height % grid or width % grid:
            raise ValueError(f"density shape {(height, width)} is not divisible by local_grid={grid}")
        # 一次 view 把 H、W 各拆成 (grid, H//grid) 两维，再对通道与块内两个轴求和，
        # 得到 (B, grid, grid) 的"局部密度积分"——纯张量操作、无 Python 循环。
        return density.reshape(batch, channels, grid, height // grid, grid, width // grid).sum(dim=(1, 3, 5))

    def compute(self, outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        probability = outputs["probability"]
        density = outputs["density"]
        # 标签通常来自 CPU 数据管线，显式转到模型输出的设备与精度，避免跨设备/
        # 跨 dtype 计算错误（count 目标在 _as_count 中做同样的转换）。
        probability_gt = targets["probability_gt"].to(device=probability.device, dtype=probability.dtype)
        density_gt = targets["density_gt"].to(device=density.device, dtype=density.dtype)
        # 形状前置校验：任何不一致都应立刻报错，而不是被后续广播悄悄掩盖。
        if probability.shape != probability_gt.shape:
            raise ValueError(f"probability shape {tuple(probability.shape)} != target {tuple(probability_gt.shape)}")
        if density.shape != density_gt.shape:
            raise ValueError(f"density shape {tuple(density.shape)} != target {tuple(density_gt.shape)}")
        count_gt = self._as_count(targets["count_gt"], density.shape[0], density.device, density.dtype)

        # BCE 的 log 输入必须严格落在 (0,1)：clamp 到 [1e-6, 1-1e-6] 既保证
        # 数值有限，又保留接近饱和处仍有梯度（不直接截断为 0/1）。
        probability_clamped = probability.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(probability_clamped, probability_gt)
        intersection = (probability * probability_gt).flatten(1).sum(dim=1)
        denominator = probability.flatten(1).sum(dim=1) + probability_gt.flatten(1).sum(dim=1)
        # 人头像素稀疏、背景占比极大，单独 BCE 会被背景主导；Dice 按预测与真值
        # 区域的交叠度量损失，缓解类别不平衡。分子分母的 1e-6 防止空图除零；
        # Dice 刻意使用未 clamp 的概率（保留原始值域，不引入额外饱和）。
        dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        probability_loss = bce + self.dice_weight * dice

        # 空间密度误差归一化：将整图空间像素绝对误差求和后再按 (count_gt + 1) 进行样本级归一化，
        # 避免像素级均值导致 loss 长期处于 0.000x 量级、对零预测塌缩缺乏足够梯度惩罚的问题。
        density_error = (density - density_gt).abs().flatten(1).sum(dim=1)
        density_loss = (density_error / (count_gt + 1.0)).mean()

        predicted_count = density.flatten(1).sum(dim=1)
        # 除以 (count_gt + 1) 把绝对误差转为近似相对误差：大场景的绝对误差不再
        # 主导整体 loss，各样本量级可比；+1 平滑避免空图（count_gt=0）除零，
        # 此时该项退化为纯绝对误差。
        count_loss = ((predicted_count - count_gt).abs() / (count_gt + 1.0)).mean()

        predicted_local = self._local_region_sums(density)
        target_local = self._local_region_sums(density_gt)
        # 全局 count 只约束总人数，可能出现"人数正确但位置全错"；local 项按
        # grid×grid 子区域分别做相对误差，把密度监督细化到空间分布。
        local_loss = ((predicted_local - target_local).abs() / (target_local + 1.0)).mean()

        # 原始未归一化 MAE（绝对人数误差）：供监控训练集实际人数偏差与日志记录。
        mae = (predicted_count - count_gt).abs().mean()

        # 四项按 LossWeights 加权求和；权重集中在 dataclass 中配置，便于实验调参。
        total = (
            self.weights.probability * probability_loss
            + self.weights.density * density_loss
            + self.weights.count * count_loss
            + self.weights.local * local_loss
        )
        return {
            "total": total,
            "loss": total,
            "probability": probability_loss,
            "density": density_loss,
            "count": count_loss,
            "local": local_loss,
            "mae": mae.detach(),
            "predicted_count": predicted_count.detach(),
            "pred_count": predicted_count.mean().detach(),
            "gt_count": count_gt.mean().detach(),
            "density_mean": density.mean().detach(),
            "density_max": density.max().detach(),
        }

    def forward(self, outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.compute(outputs, targets)["total"]


def global_count_loss(predicted_density: torch.Tensor, count_gt: torch.Tensor) -> torch.Tensor:
    """用于实验的独立归一化全局人数损失。"""

    predicted_count = predicted_density.flatten(1).sum(dim=1)
    count_gt = count_gt.to(device=predicted_density.device, dtype=predicted_density.dtype).reshape(-1)
    # 与 CrowdLoss 中的 count 项同构：仅做全局人数监督，供对比实验单独使用；
    # 目标同样先转到预测张量的设备/精度，再做 (gt+1) 相对误差归一。
    return ((predicted_count - count_gt).abs() / (count_gt + 1.0)).mean()
