import torch

from models.crowd_counter import CrowdCounter


def test_crowd_counter_contract_and_density_nonnegative() -> None:
    model = CrowdCounter(use_ultralytics=False)
    outputs = model(torch.randn(2, 3, 640, 640))
    assert set(outputs) == {"probability", "attention", "density", "count"}
    assert outputs["probability"].shape == (2, 1, 160, 160)
    assert outputs["attention"].shape == (2, 1, 160, 160)
    assert outputs["density"].shape == (2, 1, 160, 160)
    assert outputs["count"].shape == (2,)
    assert torch.all(outputs["density"] >= 0)
    assert torch.allclose(outputs["count"], outputs["density"].flatten(1).sum(dim=1))
