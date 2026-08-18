"""YOLO 风格的多尺度编码器，可选使用 Ultralytics 后端。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """后备编码器使用的紧凑 Conv-BN-SiLU 基础模块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        # padding 取 kernel_size // 2：stride=1 时空间分辨率保持不变（same 卷积）
        padding = kernel_size // 2
        # bias=False：BatchNorm 自带可学习平移项，卷积偏置会被其吸收，属冗余参数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualC2f(nn.Module):
    """仿照 YOLO C2f/C3 模块设计的小型残差块：1×1 降维 → 3×3 变换 → 1×1 投影回原通道。"""

    def __init__(self, channels: int, expansion: float = 0.5) -> None:
        super().__init__()
        # expansion 控制瓶颈宽度；max(8, ...) 兜底，避免极窄通道下瓶颈退化为空
        hidden = max(8, int(channels * expansion))
        self.cv1 = ConvBNAct(channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden, hidden, 3)
        self.cv3 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.cv1(x)
        x = self.cv2(x)
        # cv3 用 1×1 把瓶颈投影回 channels；BN 与激活放在残差相加之后，
        # 恒等路径上无任何变换，梯度可以无损直通
        return self.act(self.bn(self.cv3(x)) + residual)


class LightweightYOLOBackbone(nn.Module):
    """Ultralytics 不可用时使用的无依赖 P2/P3/P4 编码器。"""

    # 三档特征通道数（P2/P3/P4），上层融合与各 head 依此构造
    feature_channels = (64, 128, 256)
    # P2 相对原图的下采样倍数：stem 与 stage1 各 stride 2，共 4 倍
    output_stride = 4

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        # 首层通道取基宽一半（最小 16），控制 stem 阶段的计算量
        c1 = max(16, base_channels // 2)
        # 此后通道逐级翻倍、分辨率逐级减半，构成 stride 4/8/16 的特征金字塔
        c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage1 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), ResidualC2f(c2))
        self.stage2 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), ResidualC2f(c3))
        self.stage3 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), ResidualC2f(c4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        # 每个 stage 输出一档金字塔特征：p2/p3/p4 对应 stride 4/8/16
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        return p2, p3, p4


class UltralyticsYOLOBackbone(nn.Module):
    """复用来自 Ultralytics YOLO YAML 或检查点的骨干网络层。

    YOLO11 的骨干网络在第 2/4/6 层之后输出 P2/P3/P4。检测颈部（neck）
    和 Detect 头有意不复制，因此人群模型拥有其余的计算，
    且永远不会产生边界框 logits。
    """

    output_stride = 4
    # YOLO11 骨干中第 2/4/6 层恰好输出 stride 4/8/16 的 P2/P3/P4 特征
    out_indices = (2, 4, 6)

    def __init__(self, model_name: str = "yolo11n.yaml", pretrained: str | Path | None = None) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise ImportError("Ultralytics is required for the Ultralytics YOLO backend") from exc

        source_name = str(pretrained or model_name)
        source = YOLO(source_name, verbose=False).model
        if len(source.model) <= max(self.out_indices):
            raise ValueError(f"YOLO model has no P2/P3/P4 layers: {source_name}")
        # 只截取骨干部分（前 7 层）而丢弃 neck 与 Detect 头：人群计数不需要
        # 边界框 logits，且省去 neck 的额外计算开销
        self.layers = nn.ModuleList(list(source.model[: max(self.out_indices) + 1]))
        # 输出通道数由 YAML 结构在运行时决定，用 dummy forward 实测推断而非硬编码
        self.feature_channels = self._infer_channels()

    def _forward_layers(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        features: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            # layer.f 描述跨层连接：-1 用当前输入，整数引用指定前层输出，
            # 列表则按各索引拼接（等价于 Ultralytics 的 concat 语义）
            if layer.f != -1:
                if isinstance(layer.f, int):
                    x = outputs[layer.f]
                else:
                    x = [x if index == -1 else outputs[index] for index in layer.f]
            x = layer(x)
            outputs.append(x)
            # 只缓存目标索引处的特征，最后按 out_indices 顺序返回
            if layer.i in self.out_indices:
                features[layer.i] = x
        return tuple(features[index] for index in self.out_indices)

    @torch.no_grad()
    def _infer_channels(self) -> tuple[int, int, int]:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        # 64×64 的 dummy 输入可被 output_stride=4 整除，且远小于真实图像，
        # 仅用于读取各输出层的通道数（shape[1]）
        features = self._forward_layers(torch.zeros(1, 3, 64, 64, device=device))
        # 恢复调用方原有的训练/评估状态，避免污染 BN 的统计行为
        if was_training:
            self.train()
        return tuple(int(feature.shape[1]) for feature in features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_layers(x)  # type: ignore[return-value]


class YOLOBackbone(nn.Module):
    """可用时选择 Ultralytics，并提供确定性的本地后备方案。"""

    output_stride = 4

    def __init__(
        self,
        model_name: str = "yolo11n.yaml",
        *,
        pretrained: str | Path | None = None,
        use_ultralytics: bool = True,
        fallback_base_channels: int = 64,
    ) -> None:
        super().__init__()
        backend: nn.Module
        if use_ultralytics:
            try:
                backend = UltralyticsYOLOBackbone(model_name=model_name, pretrained=pretrained)
            except Exception as exc:  # 保证本地测试与离线运行可用
                # 刻意捕获所有初始化异常（缺依赖、权重下载失败、YAML 不兼容），
                # 统一降级为本地轻量编码器，保证模型在离线环境仍可构建与复现
                warnings.warn(
                    f"由于 Ultralytics 初始化失败，回退到本地 YOLO 风格编码器：{exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                backend = LightweightYOLOBackbone(base_channels=fallback_base_channels)
        else:
            backend = LightweightYOLOBackbone(base_channels=fallback_base_channels)
        self.backend = backend
        self.feature_channels = tuple(int(channel) for channel in getattr(backend, "feature_channels"))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.backend(x)  # type: ignore[return-value]

    @property
    def layers(self) -> nn.ModuleList | None:
        return getattr(self.backend, "layers", None)

    def backbone_stage_parameters(self, high_only: bool = False) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """返回用于分阶段解冻的低层/高层参数列表。"""

        # 按特征金字塔语义切层：低层负责 P2/P3（边缘、纹理等通用特征），
        # 高层负责 P4（密度等任务相关语义）；分阶段解冻时先训高层再逐步放开低层，
        # 既符合迁移学习惯例，也降低首阶段优化器的参数规模
        if isinstance(self.backend, LightweightYOLOBackbone):
            low_modules = (self.backend.stem, self.backend.stage1, self.backend.stage2)
            high_modules = (self.backend.stage3,)
        else:
            layers = self.backend.layers
            assert layers is not None
            # 前 6 层输出 P2/P3，第 6 层（索引 6）输出 P4，与 out_indices 对齐
            low_modules = (nn.ModuleList(list(layers[:6])),)
            high_modules = (nn.ModuleList(list(layers[6:])),)
        low_ids = {id(parameter) for module in low_modules for parameter in module.parameters()}
        high_ids = {id(parameter) for module in high_modules for parameter in module.parameters()}
        low = [parameter for parameter in self.parameters() if id(parameter) in low_ids]
        high = [parameter for parameter in self.parameters() if id(parameter) in high_ids]
        return ([] if high_only else low), high
