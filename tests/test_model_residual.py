import torch
from data.target_generator import TargetConfig, generate_targets
from losses.crowd_loss import CrowdLoss
from models.crowd_counter import CrowdCounter
from models.density_head import DensityHead


def test_density_residual_alpha_zero_equivalence() -> None:
    """阶段二测试 1：alpha=0 时模型输出等价于无 long skip。"""
    torch.manual_seed(42)
    model = CrowdCounter(use_ultralytics=False)
    x = torch.randn(2, 3, 640, 640)

    # 默认 alpha=0
    assert torch.allclose(model.density_residual_alpha, torch.zeros(1))
    out_zero = model(x)

    # 验证 alpha=0 时计算图输出与无 skip 等价
    p2, p3, p4 = model.backbone(x)
    features = model.fusion(p2, p3, p4)
    prob = model.probability_head(features)
    attended, _ = model.attention(features, prob)
    refined = model.refinement(attended)
    expected_density = model.density_head(refined)

    assert torch.allclose(out_zero["density"], expected_density, atol=1e-6)


def test_density_residual_alpha_nonzero_impact() -> None:
    """阶段二测试 2：alpha≠0 时 features 确实能直接影响 Density 分支。"""
    torch.manual_seed(42)
    model = CrowdCounter(use_ultralytics=False)
    x = torch.randn(1, 3, 640, 640)

    out_alpha_0 = model(x)

    # 设置 alpha != 0
    with torch.no_grad():
        model.density_residual_alpha.fill_(0.5)

    out_alpha_nonzero = model(x)

    # 密度图与人数预测必须发生显著变化
    assert not torch.allclose(out_alpha_0["density"], out_alpha_nonzero["density"])
    assert not torch.allclose(out_alpha_0["count"], out_alpha_nonzero["count"])


def test_density_residual_alpha_backward_gradient() -> None:
    """阶段二测试 3：backward 必须使 density_residual_alpha 获得有效有限梯度。"""
    torch.manual_seed(42)
    model = CrowdCounter(use_ultralytics=False)
    x = torch.randn(1, 3, 640, 640)

    # 设置非零让残差参与梯度，或在0处由 loss 反传梯度
    with torch.no_grad():
        model.density_residual_alpha.fill_(0.1)

    outputs = model(x)
    generated = generate_targets([[100.0, 100.0], [200.0, 200.0]], TargetConfig())
    targets = {
        "probability_gt": generated["probability"][None, None],
        "density_gt": generated["density"][None, None],
        "count_gt": torch.tensor([generated["count"]]),
    }

    loss = CrowdLoss()(outputs, targets)
    loss.backward()

    assert model.density_residual_alpha.grad is not None
    assert torch.isfinite(model.density_residual_alpha.grad).all()
    assert float(model.density_residual_alpha.grad.abs().item()) > 0.0


def test_density_head_initialization_scale() -> None:
    """阶段三测试：DensityHead bias=-5.0 初始化使初始 count 处于正常量级（几十~几百，非上万）。"""
    head = DensityHead(in_channels=128)
    # 模拟输入特征
    feats = torch.randn(1, 128, 160, 160)
    density = head(feats)
    initial_count = float(density.flatten(1).sum(dim=1).item())

    # 160x160 网格上，Softplus(-5.0) ≈ 0.006737 -> 160*160*0.006737 ≈ 172.5
    # 初始人数应在 [50, 1000] 范围内，绝不能是 Softplus(0) 产生的 ~17740
    assert 10.0 < initial_count < 1000.0, f"Expected initial count in (10, 1000), got {initial_count}"


def test_density_loss_normalized_scale() -> None:
    """阶段四测试：全零预测时 count-normalized density loss 能够提供显著的梯度惩罚（非 0.000x）。"""
    criterion = CrowdLoss()
    zero_density = torch.zeros(1, 1, 160, 160)
    prob = torch.zeros(1, 1, 160, 160)

    # 100 个人头真实标签
    generated = generate_targets([[float(i * 5), float(i * 5)] for i in range(100)], TargetConfig())
    targets = {
        "probability_gt": generated["probability"][None, None],
        "density_gt": generated["density"][None, None],
        "count_gt": torch.tensor([100.0]),
    }
    outputs = {
        "probability": prob,
        "density": zero_density,
    }

    details = criterion.compute(outputs, targets)
    density_loss = float(details["density"].item())

    # 100 个人头时，全零预测的整图绝对空间误差之和为 100，除以 (100+1) 约等于 0.99
    # 绝不能塌缩为旧 SmoothL1 的 100 / 25600 ≈ 0.0039
    assert 0.5 < density_loss <= 1.0, f"Expected normalized density loss ~0.99, got {density_loss}"
