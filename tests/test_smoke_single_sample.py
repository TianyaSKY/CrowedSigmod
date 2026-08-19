import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data.crowd_dataset import CrowdDataset, crowd_collate
from engine.freeze_scheduler import FreezeScheduler
from engine.trainer import CrowdTrainer
from losses.crowd_loss import CrowdLoss
from models.crowd_counter import CrowdCounter


def test_single_real_sample_smoke_and_checkpoint_reload() -> None:
    """阶段六完整验证：1个真实样本、1 epoch、bs=1 全链路与断点重载。"""
    root = Path("datasets/ucf_qnrf")
    if not root.exists():
        return

    dataset = CrowdDataset(
        root,
        split="train",
        crop_size=640,
        output_stride=4,
        dynamic_crop=True,
    )
    # 取单样本子集
    dataset.records = [dataset.records[0]]
    assert len(dataset) == 1

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=crowd_collate,
    )

    # 1. 验证数据加载
    sample_batch = next(iter(loader))
    assert sample_batch["image"].shape == (1, 3, 640, 640)
    assert sample_batch["probability_gt"].shape == (1, 1, 160, 160)
    assert sample_batch["density_gt"].shape == (1, 1, 160, 160)
    assert sample_batch["count_gt"].numel() == 1

    # 2. 构建模型与训练器
    model = CrowdCounter(use_ultralytics=False)
    criterion = CrowdLoss()
    freeze = FreezeScheduler(freeze_epochs=0, partial_unfreeze_epoch=0, full_unfreeze_epoch=0)
    trainer = CrowdTrainer(
        model,
        criterion,
        device="cpu",
        base_lr=1e-3,
        freeze_scheduler=freeze,
    )

    # 3. 前向与损失验证
    initial_alpha = float(model.density_residual_alpha.item())
    assert initial_alpha == 0.0

    outputs = model(sample_batch["image"])
    details = criterion.compute(outputs, sample_batch)
    loss = details["total"]

    assert torch.isfinite(loss)
    assert "pred_count" in details
    assert "gt_count" in details
    assert "density_mean" in details
    assert "density_max" in details

    # 4. 反向传播与梯度验证
    loss.backward()
    assert model.density_residual_alpha.grad is not None
    assert torch.isfinite(model.density_residual_alpha.grad).all()

    # 5. 训练 1 epoch 与 checkpoint 保存与加载
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpt"
        history = trainer.fit(
            loader,
            epochs=1,
            checkpoint_dir=ckpt_dir,
            show_pbar=False,
        )

        assert len(history) == 1
        last_pt = ckpt_dir / "last.pt"
        best_pt = ckpt_dir / "best.pt"
        assert last_pt.exists()
        assert best_pt.exists()

        # 6. 新模型重新加载 checkpoint 验证
        new_model = CrowdCounter(use_ultralytics=False)
        new_trainer = CrowdTrainer(new_model, criterion, device="cpu", freeze_scheduler=freeze)
        loaded = new_trainer.load_checkpoint(best_pt)

        assert loaded["epoch"] == 0
        assert torch.allclose(new_model.density_residual_alpha, model.density_residual_alpha)
        # 验证新模型输出与原模型一致
        with torch.no_grad():
            out_orig = model(sample_batch["image"])
            out_new = new_model(sample_batch["image"])
            assert torch.allclose(out_orig["count"], out_new["count"], atol=1e-5)
