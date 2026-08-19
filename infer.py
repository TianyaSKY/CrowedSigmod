import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

from inference.tiled_inference import DensityTiler
from models.crowd_counter import CrowdCounter
from utils.visualization import (
    create_composite_figure,
    density_to_heatmap,
    overlay_density_on_image,
    save_figure,
)


def image_tensor(path: Path) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        raw = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        tensor = raw.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1).float().div_(255.0)
        return tensor, rgb.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile crop size for high-res inference")
    parser.add_argument("--tile-stride", type=int, default=512, help="Tile sliding stride")
    parser.add_argument("--output-stride", type=int, default=4, help="Model downsampling stride")
    parser.add_argument("--output", type=Path, default=None, help="Directory or file path to save inference results")
    parser.add_argument("--save-vis", action="store_true", default=True, help="Save visualization figures (default: True)")
    parser.add_argument("--no-vis", dest="save_vis", action="store_false", help="Disable visual figure saving")
    parser.add_argument("--colormap", default="jet", help="Colormap for density heatmap (jet, viridis, magma, etc.)")
    parser.add_argument("--alpha", type=float, default=0.55, help="Alpha transparency for overlay blending")
    args = parser.parse_args()

    out_dir: Path | None = None
    if args.output is not None:
        out_dir = args.output if args.output.is_dir() or not args.output.suffix else args.output.parent
    elif args.save_vis:
        out_dir = Path("runs/infer") / args.image.stem

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    if out_dir is not None:
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

    img_tensor, pil_image = image_tensor(args.image)
    result = DensityTiler(args.tile_size, args.tile_stride, args.output_stride)(
        model, img_tensor, device=args.device
    )
    predicted_count = float(result.count.item())
    logger.info(f"Inference image '{args.image}' -> Estimated count: {predicted_count:.4f}")

    if out_dir is not None:
        # 保存预测结果 JSON
        json_path = (
            args.output
            if args.output is not None and args.output.suffix == ".json"
            else out_dir / f"{args.image.stem}_result.json"
        )
        json_path.write_text(
            json.dumps(
                {
                    "image": str(args.image),
                    "count": predicted_count,
                    "height": pil_image.height,
                    "width": pil_image.width,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        logger.info(f"Result JSON saved to {json_path}")

        # 生成并保存可视化图像
        if args.save_vis:
            # 1. 组合对比大图 (Original | Density Heatmap | Overlay)
            composite_fig = create_composite_figure(
                pil_image,
                result.density,
                pred_count=predicted_count,
                title=f"{args.image.name} | Predicted Count: {predicted_count:.2f}",
                colormap=args.colormap,
                alpha=args.alpha,
            )
            composite_path = out_dir / f"{args.image.stem}_composite.jpg"
            save_figure(composite_fig, composite_path)

            # 2. 单独纯热力图
            heatmap_np = density_to_heatmap(result.density, colormap=args.colormap)
            heatmap_path = out_dir / f"{args.image.stem}_heatmap.jpg"
            Image.fromarray(heatmap_np).save(heatmap_path, quality=95)

            # 3. 单独叠加图
            overlay_np = overlay_density_on_image(pil_image, result.density, alpha=args.alpha, colormap=args.colormap)
            overlay_path = out_dir / f"{args.image.stem}_overlay.jpg"
            Image.fromarray(overlay_np).save(overlay_path, quality=95)

            logger.success(f"Visualizations saved to {out_dir} (*_composite.jpg, *_heatmap.jpg, *_overlay.jpg)")

    print(f"count={predicted_count:.4f}")


if __name__ == "__main__":
    main()

