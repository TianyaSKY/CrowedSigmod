import torch
from torch import nn

from inference.tiled_inference import predict_tiled


class UnitDensity(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"density": torch.ones(images.shape[0], 1, images.shape[-2] // 4, images.shape[-1] // 4)}


def test_tiled_blending_covers_image_without_overlap_counting() -> None:
    image = torch.zeros(1, 3, 1000, 1000)
    result = predict_tiled(UnitDensity(), image, tile_size=640, tile_stride=512)
    assert result.density.shape == (1, 1, 250, 250)
    assert torch.all(result.weight > 0)
    assert torch.allclose(result.count, torch.tensor([250.0 * 250.0]), atol=1e-3)
