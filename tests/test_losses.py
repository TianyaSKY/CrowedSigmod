import torch

from data.target_generator import TargetConfig, generate_targets
from losses.crowd_loss import CrowdLoss
from models.crowd_counter import CrowdCounter


def test_crowd_loss_is_finite_and_differentiable() -> None:
    model = CrowdCounter(use_ultralytics=False)
    outputs = model(torch.randn(1, 3, 640, 640))
    generated = generate_targets([[12.0, 12.0], [639.0, 639.0]], TargetConfig())
    targets = {
        "probability_gt": generated["probability"][None, None],
        "density_gt": generated["density"][None, None],
        "count_gt": torch.tensor([generated["count"]]),
    }
    loss = CrowdLoss()(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
