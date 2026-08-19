import json
import tempfile
from pathlib import Path
import pytest
import torch
from unittest.mock import MagicMock, patch

from train_all import (
    DatasetRunResult,
    DatasetSpec,
    discover_available_datasets,
    generate_benchmark_summary,
    plot_benchmark_bar_chart,
)


def test_discover_available_datasets() -> None:
    specs = discover_available_datasets(
        datasets_arg="all",
        base_datasets_dir=Path("datasets"),
        base_data_dir=Path("data"),
        ucf_cc50_folds="0",
    )
    assert len(specs) >= 3
    names = [s.name for s in specs]
    assert "ucf_qnrf" in names
    assert "shanghaitech_AB" in names
    assert "jhu_crowd" in names


def test_discover_cc50_all_folds() -> None:
    specs = discover_available_datasets(
        datasets_arg=["ucf_cc50"],
        base_datasets_dir=Path("datasets"),
        base_data_dir=Path("data"),
        ucf_cc50_folds="all",
    )
    assert len(specs) == 5
    assert all(s.name.startswith("ucf_cc50_fold") for s in specs)


def test_generate_benchmark_summary() -> None:
    results = [
        DatasetRunResult(
            name="dataset_a",
            status="SUCCESS",
            train_samples=100,
            val_samples=20,
            test_samples=30,
            best_epoch=10,
            best_val_mae=12.5,
            best_val_rmse=18.3,
            best_val_nae=0.15,
            test_mae=14.2,
            test_rmse=20.1,
            test_nae=0.18,
            duration_seconds=120.0,
            checkpoint_path="runs/dataset_a/best.pt",
        ),
        DatasetRunResult(
            name="dataset_b",
            status="SUCCESS",
            train_samples=200,
            val_samples=40,
            test_samples=50,
            best_epoch=15,
            best_val_mae=8.4,
            best_val_rmse=11.2,
            best_val_nae=0.09,
            test_mae=9.1,
            test_rmse=13.0,
            test_nae=0.10,
            duration_seconds=180.0,
            checkpoint_path="runs/dataset_b/best.pt",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        args = MagicMock()
        args.device = "cpu"
        args.config = "configs/crowd.yaml"

        generate_benchmark_summary(results, out_dir, mode="sequential", args=args)

        assert (out_dir / "summary_report.md").exists()
        assert (out_dir / "summary_metrics.json").exists()
        assert (out_dir / "summary_metrics.csv").exists()
        assert (out_dir / "benchmark_comparison.png").exists()

        # 检查 json 内容
        data = json.loads((out_dir / "summary_metrics.json").read_text())
        assert data["mode"] == "sequential"
        assert len(data["results"]) == 2
        assert data["results"][0]["name"] == "dataset_a"

        # 检查 md 内容
        md_text = (out_dir / "summary_report.md").read_text()
        assert "Benchmark Report" in md_text
        assert "dataset_a" in md_text
        assert "dataset_b" in md_text
        assert "AVERAGE" in md_text
