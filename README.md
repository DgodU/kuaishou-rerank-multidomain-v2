# KuaiRand-Pure CTR（ADS / ADS-Transformer-SideInfo / ADS-Transformer-SideInfo-MBC）

这是用于 KuaiRand-Pure 的 PyTorch CTR 预测项目：

- `label = is_click`
- 指标：`AUC`、`GAUC`、`LogLoss`
- 主要指标：`GAUC`

## 数据目录

将原始数据放在 `data/raw/` 下。

预处理脚本会查找：

- `data/raw/log_standard_4_08_to_4_21_pure.csv`
- `data/raw/log_standard_4_22_to_5_08_pure.csv`
- `data/raw/user_features_pure.csv`
- `data/raw/video_features_basic_pure.csv`
- 可选：`data/raw/video_features_statistic_pure.csv`
- `data/raw/kuairand_video_categories.csv`
- 可选：`data/raw/kuairand_video_captions.csv`

如果文件位于 `data/raw/KuaiRand-Pure/data/`，脚本也支持该路径。

## 预处理

Debug 小样本：

```bash
python scripts/preprocess.py --config configs/ads.yaml --debug
```

完整预处理：

```bash
python scripts/preprocess.py --config configs/ads.yaml
```

输出文件：

- `data/processed/train.parquet`
- `data/processed/valid.parquet`
- `data/processed/test.parquet`
- `data/processed/train_debug.parquet`
- `data/processed/valid_debug.parquet`
- `data/processed/test_debug.parquet`
- `data/processed/feature_maps.pkl`
- `data/processed/preprocess_summary.json`

如果没有安装 parquet engine（`pyarrow` / `fastparquet`），流程会自动回退到 `.pkl` 文件，训练/评估脚本仍会透明读取。

`kuairand_video_categories.csv` 默认使用 pandas C engine 分块读取；若遇到解析失败，会自动回退到 Python engine 并跳过坏行。

## 当前代码级扩展配置

当前最强保护候选是 `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged.yaml`，对应测试 AUC/GAUC/LogLoss 为 `0.754048/0.663378/0.582084`。该结果基于公平确认 winner `semantic_simtier_sem48_slicegate_reg_mid`，并将 train+valid 合并训练、关闭验证集早停；它是当前最强单 seed protected candidate，但若要严格评估稳定性，仍建议补充 merged-train 多 seed。

四 seed 公平确认 winner 是 `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid.yaml`，四 seed test GAUC mean/min 为 `0.659845/0.656514`。上一代保护族 `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices.yaml` 四 seed test GAUC mean/min 为 `0.657891/0.654963`，seed-2025/2026 GAUC 为 `0.660225/0.659100`。

以下扩展均通过独立 config flag 控制，默认不影响未显式启用的实验配置：

- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_debias_protocol_baseline.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_random_aux.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_pal.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_pal_random_aux.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_rank_calib_split.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_history_dense.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_long_short.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_pcrg_token.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_pcrg_token_tfusion.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_dynamic_mbc_gate.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_target.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_match.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_long_short.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_long_short.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_mbcgate005.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_ccss.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_static_mbc_residual.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_static_mbc_main_input.yaml`
- `configs/sidebias_dinatt_click_only_video_stat_ema_static_mbc_main_input_residual.yaml`

## Qwen/LLM 视频语义 embedding 实验

训练和评估阶段不会在线调用 Qwen/API。先离线生成语义文本与 embedding：

```bash
python scripts/build_video_semantic_text.py --debug
python scripts/generate_qwen_video_embeddings.py --mock_debug --debug
```

正式 embedding 文件路径：

- `data/semantic/video_semantic_text.parquet`
- `data/semantic/video_semantic_emb.parquet`
- `data/semantic/video_semantic_emb_v4_full.pkl`

语义增强实验主要通过 MBC slices 注入 `semantic_target`、`simtier`、`semantic_interest`；另外尝试过独立 late fusion residual head。完整语义增强和 semantic long-short full 结果已记录在 `experiments/results_tracking.md`、`outputs/model_comparison.json` 和 `docs/semantic_simtier_experiment_plan.md`。最终结论：普通 semantic long-short 与 late fusion 不推广；四 seed 公平确认后，`semantic_simtier_sem48_slicegate_reg_mid` 是当前语义高点 winner，并进一步派生出当前最强 no-validation protected candidate。

## 训练 ADS 基线

Debug：

```bash
python scripts/train.py --config configs/ads.yaml --debug
```

完整训练：

```bash
python scripts/train.py --config configs/ads.yaml
```

## 训练 ADS-Transformer-SideInfo

Debug：

```bash
python scripts/train.py --config configs/ads_transformer_side.yaml --debug
```

完整训练：

```bash
python scripts/train.py --config configs/ads_transformer_side.yaml
```

## 训练 ADS-Transformer-SideInfo-MBC

Debug：

```bash
python scripts/train.py --config configs/ads_transformer_side_mbc.yaml --debug
```

完整训练：

```bash
python scripts/train.py --config configs/ads_transformer_side_mbc.yaml
```

### MBC 模型说明

`ADS-Transformer-SideInfo-MBC` 包含两个分支：

- 序列兴趣分支：通过 PSRG + side info + Transformer + PCRG + position-wise target attention 建模目标感知的序列兴趣。
- 静态 MBC 分支：通过 EFGC + low-rank CrossNet + DeepNet + shared top 建模 field-level 静态交互。

最终预测使用：

- `final_input = concat(interest_enhanced, static_mbc_vector)`

当配置启用时，支持可选分支辅助损失和 diversity regularization。

## 评估

```bash
python scripts/evaluate.py \
  --config configs/ads.yaml \
  --checkpoint checkpoints/best_ads_full.pt \
  --split test
```

## GAUC 定义

GAUC 按 `user_id` 分组计算：

- 跳过只有单一 label 类别的用户；
- 每个用户的权重为该用户样本数；
- `GAUC = sum(user_auc * user_weight) / sum(user_weight)`。

该定义与 `src/utils/metrics.py` 中的实现一致。

## 当前保留基线与实验规则

历史主保留基线是 `ADS-Transformer-SideInfo + Side Attention Bias`：

- 配置别名：`configs/ads_transformer_side_sidebias.yaml`
- 历史最佳测试 GAUC：`0.645878`

当前实验主线已经推进到语义 SimTier sem48 + per-slice gate regularization：

- 当前最强 protected candidate：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged.yaml`
- 测试 AUC/GAUC/LogLoss：`0.754048/0.663378/0.582084`
- 公平确认 winner：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid.yaml`
- 四 seed test GAUC mean/min/std：`0.659845/0.656514/0.002195`
- 原 protected MBC slices 四 seed test GAUC mean/min/std：`0.657891/0.654963/0.001991`
- CCSS 版 `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_ccss.yaml` 测试 AUC/GAUC/LogLoss 为 `0.747419/0.662450/0.590168`，低于 no-validation protected，不推广为当前 protected。
- `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices.yaml` 仍作为上一代保护族参考；其 seed-2025/2026/2027/2028 test GAUC 为 `0.660225/0.659100/0.657276/0.654963`。

新实验必须遵循单变量纪律：

- 每次只改变一个变量。
- 使用独立配置和 `experiment_name`。
- 输出文件必须带实验名，避免互相覆盖。
- 高风险模块默认关闭，只有对应阶段实验配置才能显式启用。
- 后续不再把资源利用率统计写入实验文档；训练加速应作为单独吞吐优化任务处理。

路线图：

```bash
docs/optimization_roadmap.md
```

实验结果跟踪：

```bash
experiments/results_tracking.md
```
