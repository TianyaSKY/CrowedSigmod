"""预热加余弦学习率调度。"""

from __future__ import annotations

import math

import torch


def warmup_cosine_factor(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """返回 ``epoch`` 开始时的标量学习率乘数。"""

    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if warmup_epochs < 0 or warmup_epochs > total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs]")
    if warmup_epochs and epoch < warmup_epochs:
        # 预热期线性升至 1.0。(epoch+1) 保证第 0 个 epoch 也有非零学习率，
        # 避免训练初期大学习率破坏预训练权重；下限 1e-6 防学习率归零。
        return max((epoch + 1) / warmup_epochs, 1e-6)
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(max(epoch - warmup_epochs, 0) / cosine_epochs, 1.0)
    # 余弦退火：progress∈[0,1] 时因子从 1.0 平滑单调降至 0，末期学习率
    # 趋近于零而非骤降，利于收敛到平坦极小值；clamp 处理 epoch 越界。
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 0,
) -> list[float]:
    """对所有参数组应用相同的调度，保持学习率比例不变。"""

    factor = warmup_cosine_factor(epoch, total_epochs, warmup_epochs)
    learning_rates: list[float] = []
    for group in optimizer.param_groups:
        # 以各组自身的 initial_lr 为基准（而非当前 lr）乘同一因子，
        # 保证头部/骨干各组间的学习率比例在调度全程保持不变。
        base_lr = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = base_lr * factor
        learning_rates.append(group["lr"])
    return learning_rates
