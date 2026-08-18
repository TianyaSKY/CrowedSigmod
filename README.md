# YOLO-PGMD 人群计数器

基于 YOLO 多尺度特征的概率引导密度图人群计数（Crowd Counting）实现。主线逻辑：

```text
YOLO 多尺度特征 → 人群概率图 → 概率引导注意力 → 多尺度密度细化 → 密度图 → ΣD = 人数
```

人数由密度图积分直接得到，不经过全连接回归器。设计优先保证数据守恒、网络可解释、训练稳定，并支持逐项消融。

## 特性

- **守恒密度标签**：每个标注点恰好贡献 1 个单位质量，含被裁剪边界截断的高斯核（可见部分重归一化）；数据集返回前断言 `ΣD = N`。
- **概率图与密度图分离**：概率图表达"哪里可能有人头"（不守恒，BCE+Dice 监督）；密度图负责精确计数（守恒，SmoothL1 监督）。
- **动态在线裁剪**：`Dataset.__getitem__` 实时生成 640×640 crop，支持随机 / 以点为中心 / 高密度 / 背景四模式采样（背景模式含密度避免逻辑）。
- **三阶段冻结训练**：冻结 → 部分解冻 → 全解冻，冻结期 BN 统计不更新，学习率分层（head 1.0× / neck 0.3× / backbone 高 0.1× / backbone 低 0.03×）。
- **重叠 tile 推理**：整图滑窗 + 余弦权重密度拼接（累加密度除以累加权重，重叠区域不重复计数）。
- **统一接口**：训练与推理共享同一个 `model(images) → dict` 契约，各组件可独立开关做消融。

## 架构

```text
Input [B,3,640,640]
        │
        ▼
YOLO Backbone（P2 / P3 / P4，stride 4/8/16，去掉 Detect head）
        │
        ▼
Multi-Scale Fusion（1×1 投影 → 上采样到 P2 → concat → C2f 风格融合）
        │  F: [B,128,160,160]
        ├───────────────► Probability Head → P: [B,1,160,160]
        │                                    （保存用于可视化）
        ▼
Probability-Guided Attention（ECA 通道注意力 + P/Avg/Max 空间注意力，α 初始为 0）
        │
        ▼
MSR ×3（膨胀卷积 1/2/3 多尺度残差块）
        │
        ▼
Density Head（Softplus）→ D: [B,1,160,160]，D ≥ 0
        │
        ▼
count = Σ D（不取整，评估直接用浮点值）
```

输出步长固定为 4（stride-4），密度解码全程保持 160×160，不做进一步降采样，对小头部更友好。

## 目录结构

```text
.
├── configs/crowd.yaml        # V1 实验配置（实验参数全部移出代码）
├── data/
│   ├── crowd_dataset.py      # 动态裁剪数据集、四模式采样器、整图采样
│   ├── target_generator.py   # 概率图 / 守恒密度图生成
│   └── transforms.py         # 点同步的图像增强（翻转/亮度/对比度/颜色/模糊/噪声）
├── models/
│   ├── yolo_encoder.py       # Ultralytics P2/P3/P4 复用 + 无依赖轻量编码器兜底
│   ├── feature_fusion.py     # 多尺度融合
│   ├── probability_head.py   # 概率头
│   ├── attention.py          # ECA + 概率引导空间注意力
│   ├── msr.py                # 多尺度残差块
│   ├── density_head.py       # Softplus 密度头
│   └── crowd_counter.py      # CrowdCounter 总装（统一 forward 契约）
├── losses/crowd_loss.py      # 四项损失：概率(BCE+0.2Dice) / 密度(SmoothL1) / 全局计数 / 4×4 局部计数
├── engine/
│   ├── trainer.py            # AdamW + warmup+cosine + checkpoint
│   ├── freeze_scheduler.py   # 三阶段冻结调度 + 分层学习率
│   ├── schedules.py          # 预热余弦调度
│   └── evaluator.py          # MAE / RMSE / NAE
├── inference/tiled_inference.py  # 重叠 tile + 余弦权重密度拼接
├── tests/                    # 8 个行为测试（守恒/形状/损失/冻结/拼接/调度）
├── train.py                  # 训练入口
├── evaluate.py               # 整图固定 tiling 验证
└── infer.py                  # 单图滑窗推理
```

## 安装

```bash
pip install torch torchvision Pillow PyYAML
pip install ultralytics        # 可选：YOLO 后端，缺失时自动回退到内置轻量编码器
```

实测环境：Python 3.13、torch 2.12、ultralytics 8.4。

## 数据准备

数据集不入库（`data/`、`datasets/` 已被 `.gitignore` 排除）。按以下目录约定放置：

```text
datasets/<name>/
├── dataset.yaml          # Ultralytics 风格配置（train/val 指向 images 下目录）
├── images/{train,val,test}/*.jpg
├── points/{train,val,test}/*.txt      # 点标注
└── labels/{train,val,test}/*.txt      # 可选：YOLO 标签（points 缺失时回退）
```

点标注支持三种格式（自动识别）：

| 格式 | 示例 | 说明 |
| --- | --- | --- |
| 归一化点 | `0.029839 0.126829` | 相对图像宽高的点坐标 |
| YOLO 标签 | `0 0.5 0.5 0.01 0.01` | `class cx cy w h`，取中心 |
| 像素 bbox | `x1 y1 x2 y2` | 取 bbox 中心（推荐头部中心） |

裁剪归属规则固定为半开区间：点 `(x, y)` 属于 `[x0, x0+640) × [y0, y0+640)`，边界上的点不会同时属于相邻两个 crop。`count_gt` 唯一来源是 `len(points_crop)`。

## 配置

实验参数集中在 `configs/crowd.yaml`，改动配置即可切换实验，无需改源码：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `image_size` / `output_stride` | 640 / 4 | 输入尺寸与输出步长 |
| `model.msr_blocks` / `dilations` | 3 / [1,2,3] | 细化块数量与膨胀率 |
| `targets.density_sigma` | 2.0 | 固定高斯 σ（自适应 σ 为第二轮实验项） |
| `loss.*` | 1.0 / 1.0 / 0.5 / 0.25 | 四项损失权重 |
| `training.freeze_epochs` 等 | 10 / 10 / 30 | 三阶段切换点 |
| `inference.tile_*` | 640 / 512 | 滑窗尺寸与步长（128px 重叠） |

消融开关通过 `CrowdCounter` 构造参数控制（对应 E0–E6 实验序列）：`use_probability`、`use_attention`、`use_msr`。

## 训练

```bash
python train.py --config configs/crowd.yaml --device cuda
python train.py --config configs/crowd.yaml --epochs 10 --output runs/exp1   # 覆盖轮数/输出目录
```

训练策略：

| Epoch | Backbone | 其他模块 | 学习率 |
| --- | --- | --- | --- |
| 0–10 | 冻结（BN 统计也不更新） | 训练 | head 1.0× |
| 10–30 | 解冻高层（Stage3/4） | 训练 | + backbone 高 0.1× |
| 30–100 | 全解冻 | 训练 | + backbone 低 0.03× |

优化器 AdamW（base_lr 1e-3，wd 1e-4），调度 warmup + cosine。检查点保存到输出目录 `epoch_NNN.pt`（模型 / 优化器 / 调度状态），可随时断点续训。

## 评估

```bash
python evaluate.py --config configs/crowd.yaml --checkpoint runs/crowd/epoch_099.pt --device cuda
```

验证不做随机裁剪：整图固定 tiling → 每块推理 → 密度拼接 → 整图求和，保证每次评估完全一致。输出 MAE / RMSE / NAE。

## 推理

```bash
python infer.py image.jpg --checkpoint runs/crowd/epoch_099.pt --tile-size 640 --tile-stride 512
```

高分辨率原图不做整体 resize（会丢失小人头）。滑窗重叠区域使用二维余弦权重融合：

```text
D_final = Σ w·D_tile / (Σ w + ε)，N = Σ D_final
```

## 目标生成

- **概率图**：`σ=2.0` 小高斯、逐点取 max、clamp 到 [0,1]，不要求积分守恒。
- **密度图**：每个点独立绘制高斯并归一化 `G /= G.sum()` 后再累加——地图边缘截断不损失质量，点重叠也不重复计权，保证 `ΣD = N`。
- 生成后逐样本断言守恒（`assert |ΣD − N| ≤ tolerance`），守恒问题会在数据层立即暴露。

## 测试

```bash
pytest -q tests
```

8 个行为测试：密度守恒（边界/角落/重叠点）、模型形状契约、损失可微、冻结梯度（backbone 为 None / head 非 None）、BN 统计冻结、tile 拼接守恒、预热余弦调度。测试不依赖数据集，可直接运行。

## 接口约定

- 模型：`outputs = model(images)` 返回 `{"probability", "attention", "density", "count"}`，训练与推理共用。
- 数据集：`__getitem__` 返回 `{"image", "points", "probability_gt", "density_gt", "count_gt", "image_id", "crop_info"}`；`full_image(i)` 返回未裁剪整图与总人数，供验证使用。
- 损失：`criterion(outputs, targets) → 标量`；`criterion.compute(...)` 额外返回逐项 loss 供日志。

## 实验路线

V1 之后建议按顺序推进：固定 σ → KNN 自适应 σ；stride 4 → 2/8；MSR 膨胀 1/2/3 → 1/3/5；ECA → SE/CBAM；随机 crop → 密度感知 crop → hard crop mining；最后再考虑 Deformable Conv / Transformer。
