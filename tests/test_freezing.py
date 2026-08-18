from engine.freeze_scheduler import FreezeScheduler, TrainingPhase
from models.crowd_counter import CrowdCounter


def test_freeze_partial_full_transitions() -> None:
    model = CrowdCounter(use_ultralytics=False)
    schedule = FreezeScheduler(10, 10, 30)
    # apply 有副作用：既返回阶段，又就地改写模型参数的 requires_grad，
    # 断言必须沿 epoch 顺序调用以观察真实状态迁移。
    assert schedule.apply(model, 0) is TrainingPhase.FROZEN
    # FROZEN：骨干全部冻结、头部仍可训练 —— 预训练阶段只学头，不动预训练权重。
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.probability_head.parameters())
    # 临界 epoch 采用左闭区间：epoch 10 起进入 PARTIAL、epoch 30 起进入 FULL。
    assert schedule.apply(model, 10) is TrainingPhase.PARTIAL
    # PARTIAL 只解冻骨干后半段（深层特征），用 any 验证"部分"而非"全部"解冻。
    assert any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert schedule.apply(model, 30) is TrainingPhase.FULL
    # FULL：骨干完全解冻，进入全量微调。
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
