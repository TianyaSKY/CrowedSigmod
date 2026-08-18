"""YOLO-style multi-scale encoders with an optional Ultralytics backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """The compact Conv-BN-SiLU primitive used by the fallback encoder."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualC2f(nn.Module):
    """A small residual block shaped like a YOLO C2f/C3 block."""

    def __init__(self, channels: int, expansion: float = 0.5) -> None:
        super().__init__()
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
        return self.act(self.bn(self.cv3(x)) + residual)


class LightweightYOLOBackbone(nn.Module):
    """Dependency-free P2/P3/P4 encoder used when Ultralytics is unavailable."""

    feature_channels = (64, 128, 256)
    output_stride = 4

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c1 = max(16, base_channels // 2)
        c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage1 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), ResidualC2f(c2))
        self.stage2 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), ResidualC2f(c3))
        self.stage3 = nn.Sequential(ConvBNAct(c3, c4, 3, 2), ResidualC2f(c4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        return p2, p3, p4


class UltralyticsYOLOBackbone(nn.Module):
    """Reuse the backbone layers from an Ultralytics YOLO YAML or checkpoint.

    YOLO11's backbone emits P2/P3/P4 after layers 2/4/6.  The detection neck
    and Detect head are intentionally not copied, so the crowd model owns the
    rest of the computation and never produces box logits.
    """

    output_stride = 4
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
        self.layers = nn.ModuleList(list(source.model[: max(self.out_indices) + 1]))
        self.feature_channels = self._infer_channels()

    def _forward_layers(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        features: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            if layer.f != -1:
                if isinstance(layer.f, int):
                    x = outputs[layer.f]
                else:
                    x = [x if index == -1 else outputs[index] for index in layer.f]
            x = layer(x)
            outputs.append(x)
            if layer.i in self.out_indices:
                features[layer.i] = x
        return tuple(features[index] for index in self.out_indices)

    @torch.no_grad()
    def _infer_channels(self) -> tuple[int, int, int]:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        features = self._forward_layers(torch.zeros(1, 3, 64, 64, device=device))
        if was_training:
            self.train()
        return tuple(int(feature.shape[1]) for feature in features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_layers(x)  # type: ignore[return-value]


class YOLOBackbone(nn.Module):
    """Select Ultralytics when available, with a deterministic local fallback."""

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
            except Exception as exc:  # keep local tests and offline runs usable
                warnings.warn(
                    f"Falling back to the local YOLO-style encoder because Ultralytics initialization failed: {exc}",
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
        """Return low/high parameter lists for staged unfreezing."""

        if isinstance(self.backend, LightweightYOLOBackbone):
            low_modules = (self.backend.stem, self.backend.stage1, self.backend.stage2)
            high_modules = (self.backend.stage3,)
        else:
            layers = self.backend.layers
            assert layers is not None
            low_modules = (nn.ModuleList(list(layers[:6])),)
            high_modules = (nn.ModuleList(list(layers[6:])),)
        low_ids = {id(parameter) for module in low_modules for parameter in module.parameters()}
        high_ids = {id(parameter) for module in high_modules for parameter in module.parameters()}
        low = [parameter for parameter in self.parameters() if id(parameter) in low_ids]
        high = [parameter for parameter in self.parameters() if id(parameter) in high_ids]
        return ([] if high_only else low), high
