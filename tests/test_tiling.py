import torch
from torch import nn

from inference.tiled_inference import predict_tiled


class UnitDensity(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # 恒为 1 的密度图：count 必然等于输出网格点数，作为检验融合"守恒"的天然标尺。
        return {"density": torch.ones(images.shape[0], 1, images.shape[-2] // 4, images.shape[-1] // 4)}


def test_tiled_blending_covers_image_without_overlap_counting() -> None:
    # 1000 不能被 tile_size=640 整除，且 stride 512 产生块间重叠：
    # 一次覆盖右/下边界填充与重叠区融合两条路径。
    image = torch.zeros(1, 3, 1000, 1000)
    result = predict_tiled(UnitDensity(), image, tile_size=640, tile_stride=512)
    assert result.density.shape == (1, 1, 250, 250)
    # 权重必须严格为正：余弦窗的 clamp_min(1e-3) 保证任何单元（含边界）在
    # 归一化时都不会除零 —— 这正是选余弦窗而非 Hann 窗的原因。
    assert torch.all(result.weight > 0)
    # 守恒：重叠区按权重凸组合，质量不会被重复累加，故单位密度模型的
    # count 精确等于 250×250 个网格点（atol 只吸收浮点舍入误差）。
    assert torch.allclose(result.count, torch.tensor([250.0 * 250.0]), atol=1e-3)
