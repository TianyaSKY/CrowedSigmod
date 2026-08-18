import torch

from data.target_generator import crop_points, generate_density_target, generate_probability_target, validate_density_conservation


def test_density_conserves_points_at_boundaries_and_overlap() -> None:
    points = torch.tensor([[0.0, 0.0], [639.0, 639.0], [320.0, 320.0], [321.0, 320.0]])
    density = generate_density_target(points, output_size=160, output_stride=4, sigma=2.0)
    validate_density_conservation(density, 4.0, tolerance=2e-4)
    assert torch.all(density >= 0)


def test_probability_is_not_mass_normalized() -> None:
    points = torch.tensor([[100.0, 100.0], [104.0, 100.0]])
    probability = generate_probability_target(points, output_size=160, output_stride=4, sigma=2.0)
    assert probability.shape == (160, 160)
    assert float(probability.max()) <= 1.0
    assert float(probability.max()) == 1.0


def test_crop_uses_half_open_membership() -> None:
    points = torch.tensor([[0.0, 0.0], [640.0, 100.0], [639.0, 639.0]])
    cropped = crop_points(points, (0, 0), 640)
    assert torch.equal(cropped, torch.tensor([[0.0, 0.0], [639.0, 639.0]]))
