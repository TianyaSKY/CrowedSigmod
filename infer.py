import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
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
    parser.add_argument("--output", type=Path, default=None, help="Directory or json file to save inference result")
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    if args.output is not None:
        out_dir = args.output if args.output.is_dir() or not args.output.suffix else args.output.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(out_dir / "infer.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO",
            rotation="10 MB",
            encoding="utf-8",
        )

    # 推理入口：模型按默认结构构造（不加载预训练权重），
    # 权重完全来自 --checkpoint；瓦片参数直接由命令行覆盖。
    model = CrowdCounter().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = DensityTiler(args.tile_size, args.tile_stride, args.output_stride)(
        model, image_tensor(args.image), device=args.device
    )
    predicted_count = float(result.count.item())
    logger.info(f"Inference image '{args.image}' -> Estimated count: {predicted_count:.4f}")
    if args.output is not None:
        json_path = args.output if args.output.suffix == ".json" else out_dir / f"{args.image.stem}_result.json"
        json_path.write_text(
            json.dumps({"image": str(args.image), "count": predicted_count}, indent=2, ensure_ascii=False)
        )
        logger.info(f"Result saved to {json_path}")
    print(f"count={predicted_count:.4f}")


if __name__ == "__main__":
    main()
