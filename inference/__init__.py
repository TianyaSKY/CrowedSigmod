"""整图分块密度推理。"""

# 对外暴露分块推理接口：predict_tiled 为单次推理入口，DensityTiler
# 为保留分块配置的可复用包装器，TiledPrediction 为结果结构，
# cosine_blend_window 供需要自定义融合窗的场景直接使用。
from .tiled_inference import DensityTiler, TiledPrediction, cosine_blend_window, predict_tiled

__all__ = ["DensityTiler", "TiledPrediction", "cosine_blend_window", "predict_tiled"]
