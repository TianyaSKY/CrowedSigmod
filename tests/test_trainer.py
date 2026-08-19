import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from engine.freeze_scheduler import FreezeScheduler
from engine.trainer import CrowdTrainer
from losses.crowd_loss import CrowdLoss
from models.crowd_counter import CrowdCounter


class DummyCrowdDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.randn(3, 128, 128),
            "probability_gt": torch.zeros(1, 32, 32),
            "density_gt": torch.zeros(1, 32, 32),
            "count_gt": torch.tensor([0.0]),
        }


def test_trainer_fit_with_tensorboard_and_tqdm() -> None:
    model = CrowdCounter(use_ultralytics=False)
    criterion = CrowdLoss()
    freeze = FreezeScheduler(freeze_epochs=1, partial_unfreeze_epoch=1, full_unfreeze_epoch=2)
    trainer = CrowdTrainer(
        model,
        criterion,
        device="cpu",
        freeze_scheduler=freeze,
    )
    dataset = DummyCrowdDataset()
    loader = DataLoader(dataset, batch_size=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        tb_dir = Path(tmpdir) / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))

        history = trainer.fit(
            loader,
            epochs=2,
            checkpoint_dir=ckpt_dir,
            writer=writer,
            show_pbar=True,
        )
        writer.close()

        assert len(history) == 2
        assert (ckpt_dir / "epoch_000.pt").exists()
        assert (ckpt_dir / "epoch_001.pt").exists()
        assert (ckpt_dir / "metrics.json").exists()
        assert any(tb_dir.glob("events.out.tfevents.*"))
