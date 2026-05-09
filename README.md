# KuaiRand CTR Reranking

基于 **KuaiRand-Pure** 的点击率预估与个性化重排实验项目。项目以用户历史行为、候选视频、用户画像、场景上下文、视频统计特征和离线语义特征为输入，训练 PyTorch CTR/rerank 模型，并以 **GAUC** 作为核心排序指标。

本仓库不是一个只保留最终模型的 demo，而是一个可复现实验工作区：包含数据预处理、特征构建、模型实现、训练评估、实验配置和结果追踪，适合用于研究短视频推荐场景下的用户内排序优化。

## Highlights

- **完整 CTR/rerank pipeline**：从 KuaiRand-Pure 原始 CSV 到 parquet/pickle 特征数据，再到训练、评估和结果汇总。
- **以 GAUC 为主的排序评估**：除全局 AUC 和 LogLoss 外，重点关注按 `user_id` 分组的用户内排序能力。
- **ADS 系列模型实现**：包含 ADS baseline、Transformer/SideInfo 增强、DIN-style target attention、EMA、MBC slices、Static MBC 等模块。
- **多源特征增强**：支持用户画像、历史行为序列、视频统计特征、作者特征、位置/时间上下文、离线视频语义 embedding 与 SimTier 匹配特征。
- **严格实验管理**：每个实验使用独立 YAML 配置和 `experiment_name`，结果记录在 `experiments/` 与 `outputs/model_comparison.json`。
- **GitHub 友好**：源码、配置、文档和精简结果可提交；原始数据、checkpoint、日志和大体积输出默认忽略。

## Task

给定用户、候选视频、历史行为序列和侧信息，模型预测二分类点击标签：

```text
label = is_click
```

主要指标：

| Metric | Meaning |
|---|---|
| `AUC` | 全局点击预测区分能力 |
| `GAUC` | 按用户分组后的加权 AUC，本项目主指标 |
| `LogLoss` | 概率校准质量 |

GAUC 的计算逻辑位于 `src/utils/metrics.py`：对每个用户单独计算 AUC，跳过只有单一标签类别的用户，再按用户样本数加权平均。

## Current Best Result

截至 `2026-05-08`，当前记录中的最强 protected candidate 为：

```text
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged.yaml
```

测试集结果：

| AUC | GAUC | LogLoss |
|---:|---:|---:|
| 0.754048 | 0.663378 | 0.582084 |

该结果使用 `merge_valid_into_train=true` 和 `train_without_validation=true`，属于 train+valid merged 的单 seed protected follow-up。若要作为严格稳定模型发布，建议继续补充多 seed 确认。

更多实验记录：

- `docs/README.md`
- `docs/experiment_summary_2026-05-08.md`
- `experiments/results_tracking.md`
- `outputs/model_comparison.json`

## Repository Structure

```text
.
├── configs/                 # YAML 训练配置与实验配置
├── docs/                    # 实验总结、优化路线和交接文档
├── experiments/             # 人类可读的逐实验追踪记录
├── outputs/                 # 精简结构化结果，保留 model_comparison.*
├── scripts/                 # 预处理、训练、评估、语义特征和实验汇总脚本
├── src/
│   ├── data/                # Dataset、sampler、feature map、semantic loader
│   ├── models/              # ADS、SideInfo、MBC、语义特征、融合与校准模块
│   ├── training/            # 训练循环、EMA、辅助 loss、评估逻辑
│   └── utils/               # IO、日志、指标和随机种子工具
└── README.md
```

本地生成目录默认不提交：

```text
data/
checkpoints/
logs/
```

## Main Model Families

### ADS baseline

`src/models/ads.py`

实现基础 ADS 建模流程：

- 历史序列表示 `E_S`
- 目标物品表示 `E_Q`
- 场景/域表示 `E_D`
- PSRG/PCRG 个性化表示生成
- target-aware attention 聚合兴趣向量

### ADS + Transformer + SideInfo

`src/models/ads_transformer_side.py`

当前主要实验主线，在 ADS 主干上加入：

- 行为侧信息编码：动作向量、播放比例、时间间隔、历史 tab、位置编码
- Transformer 序列上下文编码
- DIN-style target attention
- Side Attention Bias
- 视频统计 dense branch 与 residual logit
- EMA checkpoint
- MBC semantic slices
- 视频语义 embedding、SimTier、semantic slice gates
- 可选 calibration、position bias、pairwise loss、CCSS 等实验模块

### Static MBC branch

`src/models/static_mbc.py` 与 `src/models/ads_transformer_side_mbc.py`

实现静态字段多分支交互，包括 EFGC、Cross、Deep 等分支。该方向已在若干实验中验证，但当前 protected 主线仍以 MBC slices 方案为准。

## Data Preparation

请先下载 KuaiRand-Pure，并将原始文件放到：

```text
data/raw/
```

预处理脚本会查找以下文件：

```text
data/raw/log_standard_4_08_to_4_21_pure.csv
data/raw/log_standard_4_22_to_5_08_pure.csv
data/raw/user_features_pure.csv
data/raw/video_features_basic_pure.csv
data/raw/video_features_statistic_pure.csv
data/raw/kuairand_video_categories.csv
data/raw/kuairand_video_captions.csv
```

其中 `video_features_statistic_pure.csv` 和 `kuairand_video_captions.csv` 属于可选增强文件。若数据位于 `data/raw/KuaiRand-Pure/data/`，脚本也会兼容该路径。

默认时间切分：

| Split | Date range |
|---|---|
| train | `<= 20220421` |
| valid | `20220422 ~ 20220430` |
| test | `>= 20220501` |

部分配置支持 `random_aux_split`，会额外构建随机流量辅助训练/验证/测试集合。

## Environment

项目没有固定提交 `requirements.txt`。推荐使用 Python 3.10+，并安装以下依赖：

```bash
pip install torch numpy pandas scikit-learn pyyaml tqdm pyarrow
```

说明：

- `torch` 请根据你的 CUDA 环境从 PyTorch 官网选择合适安装命令。
- `pyarrow` 用于 parquet 读写；如果环境缺少 parquet engine，项目 IO 会尝试回退到 pickle。
- 语义 embedding 生成脚本支持 mock debug；真实 API/模型调用需要你自行配置对应环境和密钥，训练阶段不会在线调用外部 LLM API。

## Quick Start

### 1. Preprocess

Debug 小样本：

```bash
python scripts/preprocess.py --config configs/ads.yaml --debug
```

完整预处理：

```bash
python scripts/preprocess.py --config configs/ads.yaml
```

常见输出：

```text
data/processed/train.parquet
data/processed/valid.parquet
data/processed/test.parquet
data/processed/feature_maps.pkl
data/processed/preprocess_summary.json
```

### 2. Train a baseline

训练 ADS baseline：

```bash
python scripts/train.py --config configs/ads.yaml
```

Debug 训练：

```bash
python scripts/train.py --config configs/ads.yaml --debug
```

训练 SideInfo baseline：

```bash
python scripts/train.py --config configs/ads_transformer_side.yaml
```

### 3. Train the current protected candidate

```bash
python scripts/train.py \
  --config configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged.yaml
```

该配置会合并 train+valid，并固定 epoch 训练，不使用验证集早停。

### 4. Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --config configs/ads.yaml \
  --checkpoint checkpoints/<checkpoint_name>.pt \
  --split test
```

训练后重点查看：

```text
checkpoints/
logs/
outputs/
```

## Configuration

所有实验由 `configs/*.yaml` 控制。常用字段包括：

| Field | Meaning |
|---|---|
| `model_name` | 模型类型，如 `ads`、`ads_transformer_side` |
| `experiment_name` | 实验唯一名称，用于日志、checkpoint 和输出文件命名 |
| `data_dir` | 预处理后数据目录 |
| `seed` | 随机种子 |
| `batch_size` / `epochs` / `lr` | 基础训练超参数 |
| `use_ema` | 是否使用 EMA 权重评估与保存 |
| `use_video_stat` | 是否启用视频统计 dense 特征 |
| `attention_type` | 注意力类型，如 `din_mlp` |
| `use_side_attention_bias` | 是否启用 Side Attention Bias |
| `use_mbc_slices` | 是否启用 MBC semantic slices |
| `use_video_semantic_emb` | 是否读取离线视频语义 embedding |
| `use_simtier_features` | 是否启用语义相似度/分层匹配特征 |

推荐新实验复制一个相近配置，修改 `experiment_name`，并只改变一个主要变量，便于追踪结论。

## Semantic Features

项目支持离线视频语义增强。训练和评估阶段只读取固定 embedding 文件，不在线调用 Qwen/LLM API。

流程概览：

```text
video metadata/caption
  -> semantic text
  -> video semantic embedding
  -> preprocess joins embedding/index/features
  -> model consumes target_semantic_emb / simtier_features
```

Debug 示例：

```bash
python scripts/build_video_semantic_text.py --debug
python scripts/generate_qwen_video_embeddings.py --mock_debug --debug
```

已验证方向包括：

- target semantic embedding
- SimTier semantic matching features
- semantic long-short interest
- semantic late fusion
- MBC slice gate 与 gate regularization

当前结果中，`semantic_simtier_sem48_slicegate_reg_mid` 在四 seed 公平确认中胜出，并派生出当前 protected candidate。

## Experiment Tracking

本项目强调受控实验，而不是随意堆叠模块。推荐遵守：

- 每个实验只改变一个主要变量。
- 每个实验使用独立 `config` 和 `experiment_name`。
- 输出文件带实验名，避免覆盖。
- 高风险模块默认关闭，只由对应配置显式启用。
- 以 test GAUC 为主，同时参考 AUC、LogLoss 和多 seed 稳定性。

主要记录文件：

| File | Purpose |
|---|---|
| `docs/README.md` | 文档阅读顺序与当前状态索引 |
| `experiments/results_tracking.md` | 人类可读的逐实验记录 |
| `outputs/model_comparison.json` | 结构化指标汇总 |
| `docs/experiment_summary_2026-05-08.md` | 当前阶段总结 |
| `docs/optimization_roadmap.md` | 历史优化路线 |

## Notes for GitHub

建议提交：

- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `experiments/`
- `outputs/model_comparison.json`
- `outputs/model_comparison.md`

不要提交：

- `data/`
- `checkpoints/`
- `logs/`
- `*.pt` / `*.pth` / `*.ckpt`
- 大量单实验临时 metrics
- 本地虚拟环境和编辑器文件

## Status

项目当前处于研究实验状态，重点是 KuaiRand CTR/rerank 任务上的可复现实验与模型迭代。当前 protected candidate 已在记录中取得较优单次结果，但严格稳定发布仍建议补充更多 seed 与环境复现验证。
