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
├── utils/visualization.py    # 密度热力图 / 叠加图 / 多面板对比 / 散点回归图生成
├── tools/visualize_dataset.py # 数据集与高斯标签质检可视化脚本
├── tests/                    # 16 个行为测试（守恒/形状/损失/冻结/拼接/调度/可视化）
├── train.py                  # 单数据集训练入口
├── train_all.py              # 全数据集一键训练/联合训练与跨数据集基准评测
├── evaluate.py               # 整图固定 tiling 验证与误差散点/定性图导出
└── infer.py                  # 单图滑窗推理与多视角热力图保存
```

## 安装

```bash
conda activate dl
pip install torch torchvision Pillow PyYAML loguru tqdm tensorboard
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
python train.py --config configs/crowd.yaml --epochs 10   # 覆盖轮数（自动按时间戳保存至 runs/train_<timestamp>）
```

训练策略：

| Epoch | Backbone | 其他模块 | 学习率 |
| --- | --- | --- | --- |
| 0–10 | 冻结（BN 统计也不更新） | 训练 | head 1.0× |
| 10–30 | 解冻高层（Stage3/4） | 训练 | + backbone 高 0.1× |
| 30–100 | 全解冻 | 训练 | + backbone 低 0.03× |

优化器 AdamW（base_lr 1e-3，wd 1e-4），调度 warmup + cosine。训练过程中定期执行整图验证，检查点仅保存 `last.pt`（最新轮次）与 `best.pt`（最优验证 MAE 轮次），大幅节省磁盘空间，并支持无缝断点续训。

## 评估

```bash
python evaluate.py --config configs/crowd.yaml --checkpoint runs/train_20260819_140000/best.pt --device cuda
# 可选生成回归散点图与定性样本对比：
python evaluate.py --checkpoint runs/train_20260819_140000/best.pt --visualize-dir runs/eval_vis --num-vis 10
```

验证不做随机裁剪：整图固定 tiling → 每块推理 → 密度拼接 → 整图求和，保证每次评估完全一致。输出 MAE / RMSE / NAE，并自动保存真实人数 vs 预测人数的回归散点图 `gt_vs_pred_scatter.png`。

## 全数据集训练与跨数据集基准评测 (Train & Benchmark All Datasets)

通过 `train_all.py` 可以一键对仓库内所有人群计数数据集（`ucf_qnrf`、`shanghaitech_AB`、`jhu_crowd`、`ucf_cc50`）进行批量自动化训练与验证评测：

```bash
# 1. 顺序基准模式（默认）：对所有数据集依次训练并在测试集评估，生成跨数据集对比总表与柱状图
python train_all.py --device cuda

# 2. 针对指定数据集子集快速训练（覆盖轮数与批大小）
python train_all.py --datasets ucf_qnrf shanghaitech_AB --epochs 50 --batch-size 4 --device cuda

# 3. UCF-CC-50 5 折交叉验证完整测试
python train_all.py --datasets ucf_cc50 --ucf-cc50-folds all --epochs 50 --device cuda

# 4. 联合混合训练模式：将所有数据集 train split 拼接联合训练单一通用模型，并在各个测试集上评估泛化能力
python train_all.py --mode joint --epochs 80 --device cuda

# 5. 跨数据集批量验证模式：加载已有检查点直接在所有数据集上做基准测试
python train_all.py --mode eval_only --checkpoint runs/joint_20260819_140000/joint_model/best.pt
```

评测产物自动输出至以时间戳命名的独立目录（如 `runs/joint_<timestamp>` 或 `runs/sequential_<timestamp>`），包含：
- **`summary_report.md`**：Markdown 格式的跨数据集综合 Benchmark 评测报告。
- **`summary_metrics.json` / `summary_metrics.csv`**：结构化指标大表（含 Macro-Average 平均指标）。
- **`benchmark_comparison.png`**：MAE / RMSE 跨数据集横向对比高清柱状图。
- 逐数据集专属子目录：`best.pt`、`last.pt`、`metrics.json`、`test_scatter.png` 与 `test_vis/` 定性样本图。


## 推理与可视化

```bash
# 默认自动输出预测 JSON + 组合对比图 (*_composite.jpg) + 纯热力图 (*_heatmap.jpg) + 原图叠加图 (*_overlay.jpg)
python infer.py image.jpg --checkpoint runs/crowd/best.pt --output runs/infer_results
```

高分辨率原图不做整体 resize（会丢失小人头）。滑窗重叠区域使用二维余弦权重融合：

```text
D_final = Σ w·D_tile / (Σ w + ε)，N = Σ D_final
```

## 数据集质检可视化

在模型训练或数据准备阶段，可使用质检工具直接检查点标注解析、高斯核密度积分守恒与概率图质量：

```bash
python tools/visualize_dataset.py --data-root data/UCF-QNRF_ECCV18 --split train --num-samples 5 --output-dir runs/dataset_vis
```
产物包含：`[裁剪原图 + 点标注] | [守恒密度图 GT (ΣD = N)] | [概率图 GT] | [密度图叠加图]` 4 面板高分辨率大图。


## 目标生成

- **概率图**：`σ=2.0` 小高斯、逐点取 max、clamp 到 [0,1]，不要求积分守恒。
- **密度图**：每个点独立绘制高斯并归一化 `G /= G.sum()` 后再累加——地图边缘截断不损失质量，点重叠也不重复计权，保证 `ΣD = N`。
- 生成后逐样本断言守恒（`assert |ΣD − N| ≤ tolerance`），守恒问题会在数据层立即暴露。

## 测试

```bash
pytest -q tests
```

9 个行为测试：密度守恒（边界/角落/重叠点）、模型形状契约、损失可微、冻结梯度（backbone 为 None / head 非 None）、BN 统计冻结、tile 拼接守恒、预热余弦调度、Trainer 循环与 TensorBoard/tqdm 集成。测试不依赖数据集，可直接运行。

## 接口约定

- 模型：`outputs = model(images)` 返回 `{"probability", "attention", "density", "count"}`，训练与推理共用。
- 数据集：`__getitem__` 返回 `{"image", "points", "probability_gt", "density_gt", "count_gt", "image_id", "crop_info"}`；`full_image(i)` 返回未裁剪整图与总人数，供验证使用。
- 损失：`criterion(outputs, targets) → 标量`；`criterion.compute(...)` 额外返回逐项 loss 供日志。

## 实验路线

V1 之后建议按顺序推进：固定 σ → KNN 自适应 σ；stride 4 → 2/8；MSR 膨胀 1/2/3 → 1/3/5；ECA → SE/CBAM；随机 crop → 密度感知 crop → hard crop mining；最后再考虑 Deformable Conv / Transformer。
