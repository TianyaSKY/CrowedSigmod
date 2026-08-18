"""对整图进行分块人群计数。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from inference.tiled_inference import DensityTiler
from models.crowd_counter import CrowdCounter


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        # frombuffer 零拷贝复用像素缓冲区，避免额外数据拷贝。
        raw = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        return raw.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1).float().div_(255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--output-stride", type=int, default=4)
    args = parser.parse_args()
    # 推理入口：模型按默认结构构造（不加载预训练权重），
    # 权重完全来自 --checkpoint；瓦片参数直接由命令行覆盖。
    model = CrowdCounter().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = DensityTiler(args.tile_size, args.tile_stride, args.output_stride)(
        model, image_tensor(args.image), device=args.device
    )
    print(f"count={float(result.count.item()):.4f}")


if __name__ == "__main__":
    main()
