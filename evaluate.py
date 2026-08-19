import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml
from loguru import logger

from data.crowd_dataset import CrowdDataset
from data.target_generator import generate_density_target
from engine.evaluator import evaluate_tiled
from inference.tiled_inference import DensityTiler
from models.crowd_counter import CrowdCounter
from utils.visualization import create_composite_figure, plot_count_scatter, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/crowd.yaml"), help="Path to config yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Evaluation device")
    parser.add_argument("--output", type=Path, default=None, help="Directory to save eval logs and metrics")
    parser.add_argument("--visualize-dir", type=Path, default=None, help="Directory to save qualitative sample comparisons")
    parser.add_argument("--num-vis", type=int, default=8, help="Number of qualitative sample images to visualize")
    parser.add_argument("--save-scatter", action="store_true", default=True, help="Save GT vs Pred scatter plot (default: True)")
    parser.add_argument("--no-scatter", dest="save_scatter", action="store_false", help="Disable scatter plot saving")
    parser.add_argument("--colormap", default="jet", help="Colormap for density heatmaps")
    args = parser.parse_args()

    out_dir = args.output or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_log_file = out_dir / "eval.log"

    vis_dir: Path | None = args.visualize_dir
    if vis_dir is None and args.num_vis > 0:
        vis_dir = out_dir / "eval_vis"
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )
    logger.add(
        str(eval_log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="10 MB",
        encoding="utf-8",
    )

    config = yaml.safe_load(args.config.read_text())
    model_cfg = config["model"]
    model = CrowdCounter(
        backbone_name=model_cfg.get("backbone", "yolo11n.yaml"),
        use_ultralytics=bool(model_cfg.get("use_ultralytics", True)),
        fusion_channels=int(model_cfg.get("fusion_channels", 128)),
        projection_channels=int(model_cfg.get("projection_channels", 64)),
        msr_blocks=int(model_cfg.get("msr_blocks", 3)),
        msr_dilations=tuple(model_cfg.get("dilations", [1, 2, 3])),
    ).to(args.device)
    # weights_only=False 兼容旧版 torch.save 保存的完整 checkpoint 结构。
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # 关闭动态裁剪与增强，保证同一检查点每次评估得到完全相同的结果。
    dataset = CrowdDataset(
        config["data"]["root"],
        config["data"].get("val_split", "val"),
        crop_size=int(config["image_size"]),
        output_stride=int(config["output_stride"]),
        dynamic_crop=False,
        augment=False,
    )
    logger.info(
        f"Evaluating checkpoint '{args.checkpoint}' on split '{config['data'].get('val_split', 'val')}' ({len(dataset)} images)"
    )

    def _samples() -> Iterator[dict[str, Any]]:
        for index in range(len(dataset)):
            yield dataset.full_image(index)

    metrics, detailed_records = evaluate_tiled(
        model,
        _samples(),
    tiler = DensityTiler(
        tile_size=int(config["inference"]["tile_size"]),
        tile_stride=int(config["inference"]["tile_stride"]),
        output_stride=int(config["output_stride"]),
        batch_size=8,
        use_amp=True,
    )

    metrics, detailed_records = evaluate_tiled(
        model,
        _samples(),
        tiler=tiler,
        device=args.device,
        total_samples=len(dataset),
        return_details=True,
    )
    metric_str = " | ".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items())
    logger.success(f"Evaluation finished -> {metric_str}")

    metrics_out = out_dir / "eval_metrics.json"
    metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info(f"Saved evaluation metrics to {metrics_out}")

    # 1. 生成并保存 GT vs Pred 回归散点图
    if args.save_scatter and detailed_records:
        gt_counts = [r["target_count"] for r in detailed_records]
        pred_counts = [r["pred_count"] for r in detailed_records]
        scatter_fig = plot_count_scatter(
            gt_counts,
            pred_counts,
            metrics=metrics,
            title=f"Evaluation ({config['data'].get('val_split', 'val')}): GT vs. Predicted Count",
        )
        scatter_path = out_dir / "gt_vs_pred_scatter.png"
        save_figure(scatter_fig, scatter_path)
        logger.info(f"Scatter plot saved to {scatter_path}")

    # 2. 导出定性样本对比大图（兼顾最大误差样本与最小误差样本，按需重新生成避免内存爆炸）
    if vis_dir is not None and args.num_vis > 0 and detailed_records:
        # 按绝对误差从大到小排序
        sorted_records = sorted(detailed_records, key=lambda r: abs(r["error"]), reverse=True)
        selected_records: list[tuple[str, dict[str, Any]]] = []

        half = max(1, args.num_vis // 2)
        # 前 half 个为难例（最大误差）
        for rank, r in enumerate(sorted_records[:half]):
            selected_records.append((f"worst_{rank + 1:02d}", r))
        # 后 half 个为优秀样本（最小误差）
        for rank, r in enumerate(sorted_records[-half:]):
            selected_records.append((f"best_{rank + 1:02d}", r))

        out_stride = int(config.get("output_stride", 4))
        density_sigma = float(config.get("targets", {}).get("density_sigma", 2.0))

        for tag, record in selected_records:
            sample_idx = record.get("index", 0)
            sample_item = dataset.full_image(sample_idx)
            sample_image = sample_item["image"]
            sample_pred = tiler(model, sample_image, device=args.device)

            img_id = record.get("image_id") or "sample"
            clean_id = img_id.replace("/", "_").replace("\\", "_")
            pred_cnt = record["pred_count"]
            tgt_cnt = record["target_count"]
            pts = sample_item.get("points")
            gt_density = None
            if pts is not None and len(pts) > 0:
                img_h, img_w = sample_image.shape[-2:]
                out_h = max(1, img_h // out_stride)
                out_w = max(1, img_w // out_stride)
                gt_density = generate_density_target(
                    pts,
                    output_size=(out_h, out_w),
                    output_stride=out_stride,
                    sigma=density_sigma,
                )
            fig = create_composite_figure(
                image=sample_image,
                pred_density=sample_pred.density,
                gt_density=gt_density,
                points=pts,
                pred_count=pred_cnt,
                gt_count=tgt_cnt,
                title=f"[{tag.upper()}] {clean_id} | GT: {tgt_cnt:.1f} | Pred: {pred_cnt:.1f} (Err: {abs(pred_cnt - tgt_cnt):.1f})",
                colormap=args.colormap,
            )
            vis_path = vis_dir / f"{tag}_{clean_id}_gt{tgt_cnt:.0f}_pred{pred_cnt:.0f}.jpg"
            save_figure(fig, vis_path)

        logger.success(f"Saved {len(selected_records)} qualitative evaluation comparison images to {vis_dir}")

    print(metrics)


if __name__ == "__main__":
    main()

