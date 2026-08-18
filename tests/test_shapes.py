import torch

from models.crowd_counter import CrowdCounter


def test_crowd_counter_contract_and_density_nonnegative() -> None:
    model = CrowdCounter(use_ultralytics=False)
    outputs = model(torch.randn(2, 3, 640, 640))
    # 输出契约：除三张 stride=4 的特征图（640→160）外，还必须给出标量 count。
    assert set(outputs) == {"probability", "attention", "density", "count"}
    assert outputs["probability"].shape == (2, 1, 160, 160)
    assert outputs["attention"].shape == (2, 1, 160, 160)
    assert outputs["density"].shape == (2, 1, 160, 160)
    assert outputs["count"].shape == (2,)
    # 密度激活必须非负：下游 SmoothL1 与守恒损失都假定密度是"质量"而非有符号残差。
    assert torch.all(outputs["density"] >= 0)
    # 不变量：count 就是密度图的积分。锁住这条，防止未来改成独立的回归头
    # 而破坏"预测人数 = 密度求和"的语义。
    assert torch.allclose(outputs["count"], outputs["density"].flatten(1).sum(dim=1))
