import pytest
import torch

from data.target_generator import TargetConfig, generate_targets
from losses.crowd_loss import CrowdLoss
from models.crowd_counter import CrowdCounter


def test_crowd_loss_is_finite_and_differentiable() -> None:
    model = CrowdCounter(use_ultralytics=False)
    outputs = model(torch.randn(1, 3, 640, 640))
    # 故意用贴边点（12,12 与 639,639）：边界处高斯被截断、密度图很稀疏，
    # 恰好能暴露除零、NaN 类数值问题。
    generated = generate_targets([[12.0, 12.0], [639.0, 639.0]], TargetConfig())
    targets = {
        # 生成器返回单样本 (H,W) 图，这里手动补上 batch 与通道两个维度；
        # count 则包成 (1,) 张量，匹配 CrowdLoss._as_count 的形状校验。
        "probability_gt": generated["probability"][None, None],
        "density_gt": generated["density"][None, None],
        "count_gt": torch.tensor([generated["count"]]),
    }
    loss = CrowdLoss()(outputs, targets)
    # 有限性：clamp/eps 平滑必须保证任何路径都不会产生 NaN/Inf（含空图场景）。
    assert torch.isfinite(loss)
    loss.backward()
    # 可反传性：四项加权损失中至少有一条路径把梯度送到模型参数，
    # 防止某个分支被 detach/.item() 无意截断。
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_density_power_and_scale_validation() -> None:
    """验证非法 density_power (<1.0 或 >2.0) 与 density_scale (<=0) 会抛出 ValueError。"""
    with pytest.raises(ValueError, match=r"density_power must be in \[1\.0, 2\.0\]"):
        CrowdLoss(density_power=0.9)

    with pytest.raises(ValueError, match=r"density_power must be in \[1\.0, 2\.0\]"):
        CrowdLoss(density_power=2.5)

    with pytest.raises(ValueError, match=r"density_scale must be positive"):
        CrowdLoss(density_scale=0.0)

    with pytest.raises(ValueError, match=r"density_scale must be positive"):
        CrowdLoss(density_scale=-1.0)


def test_density_power_one_scale_one_matches_l1() -> None:
    """验证 power=1.0, scale=1.0 时 density_loss 严格等价于标准 L1 损失。"""
    torch.manual_seed(42)
    b, c, h, w = 2, 1, 32, 32
    density_pred = torch.rand(b, c, h, w, requires_grad=True)
    density_gt = torch.rand(b, c, h, w)
    count_gt = density_gt.flatten(1).sum(dim=1)

    outputs = {
        "probability": torch.sigmoid(torch.randn(b, c, h, w)),
        "density": density_pred,
        "count": density_pred.flatten(1).sum(dim=1),
    }
    targets = {
        "probability_gt": torch.zeros(b, c, h, w),
        "density_gt": density_gt,
        "count_gt": count_gt,
    }

    criterion = CrowdLoss(
        probability_weight=0.0,
        density_weight=1.0,
        count_weight=0.0,
        local_weight=0.0,
        density_power=1.0,
        density_scale=1.0,
    )
    details = criterion.compute(outputs, targets)
    loss = details["density"]

    expected_l1 = ((density_pred - density_gt).abs().flatten(1).sum(dim=1) / (count_gt + 1.0)).mean()
    assert torch.allclose(loss, expected_l1, atol=1e-6)

    loss.backward()
    assert density_pred.grad is not None

    # L1 梯度解析解：sign(pred - gt) / (count_gt + 1) / B
    expected_grad = torch.sign(density_pred.detach() - density_gt) / (count_gt[:, None, None, None] + 1.0) / b
    assert torch.allclose(density_pred.grad, expected_grad, atol=1e-5)


def test_density_power_gradient_scaling_behavior() -> None:
    """验证 power=1.5, scale=10.0 在小误差处梯度明显弱于 L1，大误差处梯度强于 L1。"""
    b, c, h, w = 1, 1, 16, 16
    density_gt = torch.full((b, c, h, w), 0.05)
    count_gt = density_gt.flatten(1).sum(dim=1)

    targets = {
        "probability_gt": torch.zeros(b, c, h, w),
        "density_gt": density_gt,
        "count_gt": count_gt,
    }

    # Case 1: 小误差 e = 0.0001
    pred_small = (density_gt + 0.0001).clone().detach().requires_grad_(True)
    outputs_small = {"probability": torch.zeros(b, c, h, w), "density": pred_small, "count": pred_small.flatten(1).sum(dim=1)}

    loss_l1 = CrowdLoss(probability_weight=0.0, count_weight=0.0, local_weight=0.0, density_power=1.0, density_scale=1.0)
    loss_power15 = CrowdLoss(probability_weight=0.0, count_weight=0.0, local_weight=0.0, density_power=1.5, density_scale=10.0)

    loss_l1(outputs_small, targets).backward()
    grad_l1_small = pred_small.grad.clone()

    pred_small.grad = None
    loss_power15(outputs_small, targets).backward()
    grad_power15_small = pred_small.grad.clone()

    # d/de(10 * e^1.5) = 15 * sqrt(e) = 15 * 0.01 = 0.15 vs L1 的 1.0
    ratio_small = (grad_power15_small / grad_l1_small).mean().item()
    assert 0.14 <= ratio_small <= 0.16, f"Expected small error grad ratio ~0.15, got {ratio_small}"

    # Case 2: 大误差 e = 0.04
    pred_large = (density_gt + 0.04).clone().detach().requires_grad_(True)
    outputs_large = {"probability": torch.zeros(b, c, h, w), "density": pred_large, "count": pred_large.flatten(1).sum(dim=1)}

    pred_large.grad = None
    loss_l1(outputs_large, targets).backward()
    grad_l1_large = pred_large.grad.clone()

    pred_large.grad = None
    loss_power15(outputs_large, targets).backward()
    grad_power15_large = pred_large.grad.clone()

    # d/de(10 * e^1.5) = 15 * sqrt(0.04) = 15 * 0.2 = 3.0 vs L1 的 1.0
    ratio_large = (grad_power15_large / grad_l1_large).mean().item()
    assert 2.95 <= ratio_large <= 3.05, f"Expected large error grad ratio ~3.0, got {ratio_large}"


def test_anti_collapse_gradient_when_pred_near_zero() -> None:
    """防塌缩测试：当模型预测 Pred 接近 0 且 Pred << GT 时，总损失梯度有能力将预测推高（梯度为负）。"""
    # 构造含前景（人头）与背景的真实样本 target
    points = [[50.0, 50.0], [50.0, 100.0], [100.0, 50.0], [100.0, 100.0]]
    generated = generate_targets(points, TargetConfig(output_size=160, output_stride=1))
    prob_gt = generated["probability"][None, None]
    density_gt = generated["density"][None, None]
    count_gt = torch.tensor([generated["count"]])

    targets = {
        "probability_gt": prob_gt,
        "density_gt": density_gt,
        "count_gt": count_gt,
    }

    # 模拟接近 collapse 的状态：密度与概率预测极其微小（接近 0）
    pred_density = torch.full_like(density_gt, 1e-4, requires_grad=True)
    pred_prob = torch.full_like(prob_gt, 1e-3, requires_grad=True)

    outputs = {
        "probability": pred_prob,
        "density": pred_density,
        "count": pred_density.flatten(1).sum(dim=1),
    }

    criterion = CrowdLoss(
        probability_weight=1.0,
        density_weight=1.0,
        count_weight=0.5,
        local_weight=0.25,
        density_power=1.5,
        density_scale=10.0,
    )

    loss = criterion(outputs, targets)
    loss.backward()

    assert pred_density.grad is not None
    assert torch.isfinite(pred_density.grad).all()

    # 前景人头区域 (density_gt > 0.01)：梯度必须强烈为负（负梯度让梯度下降更新时 pred 变大）
    fg_mask = density_gt > 0.01
    fg_grad = pred_density.grad[fg_mask]
    assert (fg_grad < 0).all(), "Foreground density gradients must be negative to pull predictions up towards GT"

    # 背景区域 (density_gt == 0)：pred = 1e-4 时的压制梯度非常微弱
    bg_mask = density_gt == 0.0
    bg_grad = pred_density.grad[bg_mask]

    # 前景拉升梯度的平均幅值应该远大于背景压制梯度的平均幅值
    fg_mag = fg_grad.abs().mean()
    bg_mag = bg_grad.abs().mean()
    assert fg_mag > 5.0 * bg_mag, f"Foreground upward pull ({fg_mag:.4f}) should dominate background push ({bg_mag:.4f})"

    # 整图方向上的全局梯度总和为负：网络整体受到向上拉升的总冲量
    total_density_grad_sum = pred_density.grad.sum()
    assert total_density_grad_sum < 0, "Net gradient over the entire map must be negative to increase overall predicted count"
