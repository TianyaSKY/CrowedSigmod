from engine.freeze_scheduler import FreezeScheduler, TrainingPhase
from models.crowd_counter import CrowdCounter


def test_freeze_partial_full_transitions() -> None:
    model = CrowdCounter(use_ultralytics=False)
    schedule = FreezeScheduler(10, 10, 30)
    assert schedule.apply(model, 0) is TrainingPhase.FROZEN
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.probability_head.parameters())
    assert schedule.apply(model, 10) is TrainingPhase.PARTIAL
    assert any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert schedule.apply(model, 30) is TrainingPhase.FULL
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
