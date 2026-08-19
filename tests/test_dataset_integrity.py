import glob
from pathlib import Path
import numpy as np
import pytest

from data.crowd_dataset import CrowdDataset


def test_dataset_txt_annotations_integrity() -> None:
    """确保 datasets 目录下所有 txt 点标注文件均不存在对角线串位异常 (相关系数过高)。"""
    base_dir = Path("datasets")
    if not base_dir.exists():
        pytest.skip("datasets directory not found")

    txt_files = list(base_dir.glob("*/points/*/*.txt"))
    assert len(txt_files) > 0, "No point annotation txt files found in datasets/"

    # 抽样检测 50 个标注文件
    rng = np.random.RandomState(42)
    sample_files = rng.choice(txt_files, size=min(50, len(txt_files)), replace=False)

    high_corr_count = 0
    for p in sample_files:
        data = np.loadtxt(p)
        if data.ndim == 2 and data.shape[1] == 2 and len(data) >= 10:
            corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
            # 正常分布的人群标注 x 与 y 相关系数不会极度接近 1.0 (例如 > 0.85)
            if abs(corr) > 0.85:
                high_corr_count += 1

    assert high_corr_count == 0, f"Found {high_corr_count} files with abnormal diagonal correlation (>0.85)!"


def test_full_image_returns_points() -> None:
    """测试 full_image 返回 points 字段供整图 GT 密度生成。"""
    ds = CrowdDataset("datasets/ucf_qnrf", "val", crop_size=640)
    full = ds.full_image(0)
    assert "points" in full
    assert isinstance(full["points"], np.ndarray) or hasattr(full["points"], "shape")
    assert "count_gt" in full
    assert "image" in full
