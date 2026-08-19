"""面向人群裁剪的小型、与点同步的数据增强。"""

from __future__ import annotations

import random
from typing import Tuple

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """将 RGB PIL 图像转换为取值在 ``[0, 1]`` 的浮点 ``[C, H, W]`` 张量。"""

    rgb = image.convert("RGB")
    # 使用 ``torch.frombuffer`` 可避免数据路径中对 NumPy 的额外依赖。
    raw = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    tensor = raw.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1).contiguous()
    return tensor.float().div_(255.0)


def pad_to_size(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """在图像右侧/底部用黑色像素填充至 ``(height, width)``。"""

    target_h, target_w = int(size[0]), int(size[1])
    if image.height > target_h or image.width > target_w:
        raise ValueError("pad_to_size cannot crop an image")
    if image.height == target_h and image.width == target_w:
        return image
    # 只在右侧/底部填充：裁剪窗口的缺失部分恰好位于右下区域，
    # 原点不动，填充后图像坐标与点坐标保持对齐。
    return ImageOps.expand(image, border=(0, 0, target_w - image.width, target_h - image.height), fill=0)


def apply_augmentations(
    image: Image.Image,
    points: torch.Tensor,
    *,
    horizontal_flip: bool = True,
    color_jitter: bool = True,
    blur_probability: float = 0.0,
    noise_probability: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[Image.Image, torch.Tensor]:
    """应用几何与外观变换，同时保持点坐标对齐。"""

    rng = rng or random
    points = points.clone().to(dtype=torch.float32)
    if horizontal_flip and rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        # 几何增强必须同步变换点：水平翻转后 x' = W-1-x（W-1 保证
        # 翻转前后像素索引空间一致），y 坐标不变。
        if len(points):
            points[:, 0] = image.width - 1 - points[:, 0]
    # 颜色扰动在 PIL 域完成，避免张量-图像反复转换；扰动幅度
    # 取小范围随机，保持人群外观与光照统计大致不变。
    if color_jitter:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.9, 1.1))
    if blur_probability > 0 and rng.random() < blur_probability:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    if noise_probability > 0 and rng.random() < noise_probability:
        # 加噪路径直接返回张量而非 PIL 图像：0.02 标准差的高斯噪声
        # 只对张量施加，调用方据此跳过 image_to_tensor 转换；
        # clamp 保证像素仍在 [0, 1]。
        tensor = image_to_tensor(image)
        noise = torch.randn_like(tensor) * 0.02
        return (tensor.add(noise).clamp_(0.0, 1.0), points)
    return image, points
