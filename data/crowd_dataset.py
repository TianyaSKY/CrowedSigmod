"""面向点标注人群图像的动态裁剪数据集。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from .target_generator import TargetConfig, crop_points, generate_targets, validate_density_conservation
from .transforms import apply_augmentations, image_to_tensor, pad_to_size


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CrowdRecord:
    image_path: Path
    points_path: Path | None
    image_id: str


class CrowdDataset(torch.utils.data.Dataset):
    """加载原始图像/点标注，并在线生成一个固定尺寸的训练裁剪。

    点文件可包含 ``x y`` 归一化坐标、标准 YOLO 行
    ``class x_center y_center width height`` 或像素坐标。数据集将点标注
    视为 ``count_gt`` 的唯一来源。
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        crop_size: int = 640,
        output_stride: int = 4,
        points_dir: str = "points",
        labels_dir: str = "labels",
        dynamic_crop: bool | None = None,
        crop_probabilities: dict[str, float] | None = None,
        augment: bool | None = None,
        probability_sigma: float = 2.0,
        density_sigma: float = 2.0,
        adaptive_density: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.crop_size = int(crop_size)
        self.output_stride = int(output_stride)
        # crop_size 必须能被 output_stride 整除，标签网格尺寸才是整数，
        # 裁剪图与标签图才能保持严格的空间对齐。
        if self.crop_size <= 0 or self.crop_size % self.output_stride != 0:
            raise ValueError("crop_size must be positive and divisible by output_stride")
        # 训练集默认开启动态裁剪与增强，验证/测试集默认关闭，
        # 保证评估时每张图只对应一个确定性的输入。
        self.dynamic_crop = split == "train" if dynamic_crop is None else bool(dynamic_crop)
        self.augment = split == "train" if augment is None else bool(augment)
        self.rng = random.Random(seed)
        self.crop_probabilities = crop_probabilities or {
            "random": 0.30,
            "head-centered": 0.30,
            "high-density": 0.30,
            "background": 0.10,
        }
        self._validate_crop_probabilities()
        self.target_config = TargetConfig(
            output_size=self.crop_size // self.output_stride,
            output_stride=self.output_stride,
            probability_sigma=probability_sigma,
            density_sigma=density_sigma,
            adaptive_density=adaptive_density,
        )
        image_root = self._resolve_image_root(self.root, split)
        if not image_root.exists():
            raise FileNotFoundError(f"image split does not exist: {image_root}")
        self.records = self._build_records(image_root, self.root, split, points_dir, labels_dir)
        if not self.records:
            raise FileNotFoundError(f"no images found under {image_root}")

    @staticmethod
    def _resolve_image_root(root: Path, split: str) -> Path:
        """多策略解析图像目录，兼容标准规范与各类开源数据集组织形式。"""
        candidates = [
            root / "images" / split,
            root / split / "images",
            root / f"{split}_data" / "images",
            root / split,
            root / split.capitalize(),
            root / split.lower(),
        ]
        if split in {"", ".", "root"}:
            candidates.append(root)

        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("*")):
                # 检查该目录下是否有图片
                if any(p.suffix.lower() in _IMAGE_SUFFIXES for p in candidate.rglob("*")):
                    return candidate
        return root / "images" / split

    def _validate_crop_probabilities(self) -> None:
        if any(value < 0 for value in self.crop_probabilities.values()):
            raise ValueError("crop probabilities must be non-negative")
        total = sum(self.crop_probabilities.values())
        if total <= 0:
            raise ValueError("at least one crop probability must be positive")

    @staticmethod
    def _build_records(
        image_root: Path,
        root: Path,
        split: str,
        points_dir: str = "points",
        labels_dir: str = "labels",
    ) -> list[CrowdRecord]:
        paths = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES)
        records: list[CrowdRecord] = []
        points_root = root / points_dir / split
        labels_root = root / labels_dir / split

        for image_path in paths:
            relative_stem = image_path.relative_to(image_root).with_suffix("")
            point_path = points_root / relative_stem.with_suffix(".txt")
            label_path = labels_root / relative_stem.with_suffix(".txt")

            # 候选标注文件查找策略：
            # 1. points/{split}/{stem}.txt
            # 2. labels/{split}/{stem}.txt
            # 3. 同目录下的 {stem}_ann.mat (UCF-QNRF / UCF_CC_50)
            # 4. 同目录下的 {stem}.txt / {stem}.mat
            # 5. 上级目录 ground_truth/GT_{stem}.mat (ShanghaiTech)
            same_dir_ann_mat = image_path.with_name(f"{image_path.stem}_ann.mat")
            same_dir_txt = image_path.with_suffix(".txt")
            same_dir_mat = image_path.with_suffix(".mat")
            shanghai_gt_mat = image_path.parent.parent / "ground_truth" / f"GT_{image_path.stem}.mat"

            chosen_pts: Path | None = None
            if point_path.exists():
                chosen_pts = point_path
            elif label_path.exists():
                chosen_pts = label_path
            elif same_dir_ann_mat.exists():
                chosen_pts = same_dir_ann_mat
            elif same_dir_txt.exists():
                chosen_pts = same_dir_txt
            elif same_dir_mat.exists():
                chosen_pts = same_dir_mat
            elif shanghai_gt_mat.exists():
                chosen_pts = shanghai_gt_mat

            records.append(
                CrowdRecord(
                    image_path=image_path,
                    points_path=chosen_pts,
                    image_id=str(relative_stem),
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _parse_annotation(path: Path | None, width: int, height: int) -> torch.Tensor:
        if path is None or not path.exists():
            return torch.empty((0, 2), dtype=torch.float32)

        # 支持 MATLAB .mat 标注文件 (UCF-QNRF, UCF_CC_50, ShanghaiTech 等)
        if path.suffix.lower() == ".mat":
            try:
                import scipy.io

                mat = scipy.io.loadmat(str(path))
                if "annPoints" in mat:
                    pts = np.asarray(mat["annPoints"], dtype=np.float32)
                elif "image_info" in mat:
                    pts = np.asarray(mat["image_info"][0, 0]["location"][0, 0], dtype=np.float32)
                elif "point" in mat:
                    pts = np.asarray(mat["point"], dtype=np.float32)
                else:
                    # 查找包含 2 列浮点数据的第一个数组键
                    pts = np.empty((0, 2), dtype=np.float32)
                    for k, v in mat.items():
                        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2:
                            pts = v.astype(np.float32)
                            break

                if pts.size == 0:
                    return torch.empty((0, 2), dtype=torch.float32)
                points = torch.from_numpy(pts).float()
                # 裁剪到图像物理范围 [0, W-1] x [0, H-1]
                points[:, 0].clamp_(0, max(width - 1, 0))
                points[:, 1].clamp_(0, max(height - 1, 0))
                return points
            except Exception as e:
                raise ValueError(f"failed to parse .mat annotation at {path}: {e}") from e

        rows: list[tuple[float, float]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            fields = line.replace(",", " ").split()
            if not fields:
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError as exc:
                raise ValueError(f"invalid annotation at {path}:{line_number}") from exc
            if len(values) >= 5:
                # 标准 YOLO 行：class, x_center, y_center, width, height。
                x, y = values[1], values[2]
            elif len(values) == 4:
                # 像素/绝对坐标边界框：x1, y1, x2, y2。
                x, y = (values[0] + values[2]) / 2.0, (values[1] + values[3]) / 2.0
            elif len(values) >= 2:
                # 纯坐标行：仅 x y。
                x, y = values[0], values[1]
            else:
                raise ValueError(f"annotation row needs at least two values at {path}:{line_number}")
            rows.append((x, y))
        if not rows:
            return torch.empty((0, 2), dtype=torch.float32)
        points = torch.tensor(rows, dtype=torch.float32)
        if float(points.abs().max()) <= 1.000001:
            points[:, 0] *= max(width - 1, 1)
            points[:, 1] *= max(height - 1, 1)
        points[:, 0].clamp_(0, max(width - 1, 0))
        points[:, 1].clamp_(0, max(height - 1, 0))
        return points

    def _sample_origin(self, points: torch.Tensor, width: int, height: int) -> tuple[int, int, str]:
        max_x = max(width - self.crop_size, 0)
        max_y = max(height - self.crop_size, 0)
        if not self.dynamic_crop:
            return 0, 0, "fixed"
        labels = list(self.crop_probabilities)
        weights = [self.crop_probabilities[label] for label in labels]
        mode = self.rng.choices(labels, weights=weights, k=1)[0]
        # 点中心采样：随机选一个头作为窗口中心并施加抖动；head-centered
        # 抖动更大，让窗口覆盖更广的人群区域。
        if mode in {"head-centered", "high-density"} and len(points):
            index = self.rng.randrange(len(points))
            px, py = points[index].tolist()
            # 高密度裁剪作为保守的一级近似保持以某个点为中心；
            # 后续可用难例挖掘替换该采样器。
            jitter = self.crop_size * (0.15 if mode == "high-density" else 0.30)
            x0 = int(round(px - self.crop_size / 2 + self.rng.uniform(-jitter, jitter)))
            y0 = int(round(py - self.crop_size / 2 + self.rng.uniform(-jitter, jitter)))
            return max(0, min(x0, max_x)), max(0, min(y0, max_y)), mode
        # 背景采样采用拒绝法：最多尝试 12 个随机位置，直到窗口内不含
        # 任何点；全部失败则退化为普通随机裁剪，避免无限循环。
        if mode == "background" and len(points) and (max_x or max_y):
            for _ in range(12):
                x0 = self.rng.randint(0, max_x)
                y0 = self.rng.randint(0, max_y)
                inside = (
                    (points[:, 0] >= x0)
                    & (points[:, 0] < x0 + self.crop_size)
                    & (points[:, 1] >= y0)
                    & (points[:, 1] < y0 + self.crop_size)
                )
                if not bool(inside.any()):
                    return x0, y0, mode
        return self.rng.randint(0, max_x), self.rng.randint(0, max_y), mode

    def full_image(self, index: int) -> dict[str, Any]:
        """返回未裁剪的图像及其完整人数，用于分块评估。"""

        record = self.records[index]
        with Image.open(record.image_path) as loaded:
            image = loaded.convert("RGB")
        width, height = image.width, image.height
        points = self._parse_annotation(record.points_path, width, height)
        # 评估路径需要整图：模型只接受固定尺寸输入，整图须由外部
        # 瓦片推理后再拼接，因此这里不做裁剪、也不生成标签图。
        return {
            "image": image_to_tensor(image),
            "count_gt": torch.tensor(float(len(points)), dtype=torch.float32),
            "image_id": record.image_id,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as loaded:
            image = loaded.convert("RGB")
        width, height = image.width, image.height
        points = self._parse_annotation(record.points_path, width, height)
        x0, y0, crop_mode = self._sample_origin(points, width, height)
        crop_points_local = crop_points(points, (x0, y0), self.crop_size)
        # 靠近图像右下边缘的窗口实际裁剪尺寸小于 crop_size，用黑色
        # 填充补齐到定尺寸，保证 batch 内图像与标签网格大小一致；
        # 填充区不含任何点，不产生头部质量。
        image = image.crop((x0, y0, min(x0 + self.crop_size, width), min(y0 + self.crop_size, height)))
        image = pad_to_size(image, (self.crop_size, self.crop_size))
        if self.augment:
            augmented = apply_augmentations(image, crop_points_local, rng=self.rng)
            image, crop_points_local = augmented
        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = image_to_tensor(image)
        targets = generate_targets(crop_points_local, self.target_config)
        # 为标签图补上通道维，与图像 [C,H,W] 布局一致，
        # 便于按 batch 堆叠并与网络输出的通道逐位比较。
        probability_gt = targets["probability"].unsqueeze(0)
        density_gt = targets["density"].unsqueeze(0)
        count_gt = torch.tensor(targets["count"], dtype=torch.float32)
        # 在线断言守恒（容差放宽到 2e-4，容纳 float32 累加误差）；
        # 训练早期即可暴露标签生成的坐标/平移类 bug。
        validate_density_conservation(density_gt, count_gt, tolerance=2e-4)
        return {
            "image": image_tensor,
            "points": crop_points_local,
            "probability_gt": probability_gt,
            "density_gt": density_gt,
            "count_gt": count_gt,
            "image_id": record.image_id,
            "crop_info": {
                "x0": x0,
                "y0": y0,
                "width": width,
                "height": height,
                "mode": crop_mode,
            },
        }


def crowd_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """堆叠张量标签的同时，整理变长点列表。"""

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "points": [item["points"] for item in batch],
        "probability_gt": torch.stack([item["probability_gt"] for item in batch]),
        "density_gt": torch.stack([item["density_gt"] for item in batch]),
        "count_gt": torch.stack([item["count_gt"] for item in batch]),
        "image_id": [item["image_id"] for item in batch],
        "crop_info": [item["crop_info"] for item in batch],
    }
