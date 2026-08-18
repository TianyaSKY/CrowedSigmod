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
