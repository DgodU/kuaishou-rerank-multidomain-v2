# 变更日志

本文档记录项目创建以来的主要工程改动、实验结果和阶段性结论。实验名、配置路径和指标字段保留英文，便于脚本和日志检索。

## 2026-05-06 - Qwen/LLM 视频语义 SimTier/long-short 实验实现

### Full 结果更新

- `semantic_long_short` full 完成：best_valid_epoch=4，valid AUC/GAUC/LogLoss=0.761015/0.673342/0.576493，test AUC/GAUC/LogLoss=0.746075/0.659765/0.592911。链路有效但排序和校准均未超过关键对照，不推广。
- `semantic_simtier_long_short` full 完成：best_valid_epoch=3，valid AUC/GAUC/LogLoss=0.760972/0.673135/0.575097，test AUC/GAUC/LogLoss=0.745698/0.660579/0.592307。相比 `semantic_long_short` 略有提升，但 GAUC 仍低于 `latefusion` 0.660865、`sem48` seed2025 0.662346、`mbcgate005` seed2025 0.662555 和 `reg_mid` seed2025 0.662360；不推广。
- `semantic_simtier_long_short` 首次 full preprocess 曾因 embedding 路径指向缺失的 `data/semantic/video_semantic_emb.parquet` 失败；已修正为 `data/semantic/video_semantic_emb_v4_full.pkl` 后完成重跑。
- 语义增强方向总体结论：现有 semantic target、SimTier、semantic long-short 和 late fusion 尝试均未达到推广标准，除非明确重启新方向，否则停止继续扩展语义/LLM sweep。

### 新增内容

- 新增 `src/data/semantic_loader.py`，支持离线 `video_semantic_emb.parquet` 的 list/string/emb_0 多列格式读取、NaN/Inf 清理、L2 normalize 和 unknown zero padding。
- 新增 `scripts/build_video_semantic_text.py`，只生成视频语义文本，不调用 API。
- 新增 `scripts/generate_qwen_video_embeddings.py`，用于离线 Qwen embedding 生成，并支持 `--mock_debug` deterministic embedding。
- 新增 `src/models/semantic_features.py`，实现 `VideoSemanticEncoder`、`SimTierEncoder`、`SemanticLongShortInterest`。
- `scripts/preprocess.py` 支持 `use_video_semantic_emb`、`use_simtier_features`、`use_semantic_long_short`，输出 semantic matrix/index、SimTier features、feature names 和 scaler。
- `src/data/dataset.py` 支持从 semantic index 动态查 `target_semantic_emb` 与 `hist_semantic_emb`。
- `ADSTransformerSideModel` 支持 MBC slices 注入 `semantic_target`、`simtier`、`semantic_interest`，不新增语义 residual logit head。
- 新增三套配置：
  - `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier`
  - `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_long_short`
  - `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_long_short`

### 状态

- 实现状态：implemented，语义 config 已完成 mock embedding debug。
- Debug 状态：`semantic_simtier` / `semantic_long_short` / `semantic_simtier_long_short` 均已通过 preprocess debug 与 train debug。
- Full run：`semantic_long_short` 和 `semantic_simtier_long_short` 已完成，均不推广。
- 当前保护模型配置未修改；三套语义实验使用独立 processed 目录，避免覆盖保护模型数据。

## 2026-05-03 - 代码级扩展验收与预处理鲁棒性修复

### 验收范围

- 确认 random auxiliary BCE separate head、PAL、rank/calibration split、history dense、author、long-short interest、PCRG token、TransformerFusion、semantic placeholder、dynamic MBC gate 等模块均由独立 config flag 控制，默认不影响当前保护配置。
- 确认 13 个代码级扩展 config 均存在且 `experiment_name` 独立。
- 完成最低 debug 验收：
  - `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_debias_protocol_baseline`
  - `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_random_aux`
  - `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_pal`
  - `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_rank_calib_split`
  - `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_history_dense`
  - 当前保护配置 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices`
- 本次只运行 debug，不启动 full run。

### 修复

- `scripts/preprocess.py` 中 `load_categories_for_videos` 增加 CSV 读取 fallback：默认使用 pandas C engine 分块读取，遇到 `ParserError` 时回退到 Python engine 并跳过坏行，用于提升 debug/full 预处理可复现性。

## 2026-04-30 - 分阶段 SideBias 优化框架

### 新增内容

- 建立严格的单变量实验纪律，以当前最优 SideBias 基线为参照。
- 新增 `docs/optimization_roadmap.md`，规划历史过滤、DIN 目标注意力、时间上下文、PCRG token、TransformerFusion、MBC slices、dense history-only 等阶段。
- 新增 `experiments/results_tracking.md` 用于记录 debug/full 实验。
- README 中补充基线说明：历史保留 SideBias 测试 GAUC 为 `0.645878`。

### 预处理

- 在 `scripts/preprocess.py` 中加入 `history_mode`：
  - `all`
  - `click_only`
  - `click_or_long_view`
- 在预处理输出中加入历史长度统计。
- 加入时间上下文特征：
  - `hour_of_day`
  - `day_of_week`
  - `is_weekend`
- 增加 dense history-only 实验兼容开关，并默认避免全局先验 dense 特征影响主线。

### 数据集与模型

- `KuaiRandDataset` 增加可选时间上下文字段。
- `ADSModel` 在 `use_time_context` 下支持时间上下文 embedding。
- SideBias attention-bias MLP 在 `use_time_context` 时可接收时间上下文输入。
- 新增 `src/models/attention.py` 中的 `DINStyleTargetAttention`。
- 将 `attention_type: din_mlp` 集成到 `ADSTransformerSideModel`，并保留可选 SideBias score bias。
- 为新增 attention/time-context 路径加入 debug tensor shape 日志。

### 未来模块占位

- 新增 `src/models/pcrg_token.py` 中的 `PCRGTokenLayer`。
- 新增 `src/models/transformer_fusion.py` 中的 `TransformerFusion`。
- 新增 `src/models/mbc_slices.py` 中的 `MBCSemanticHead`。
- 这些模块默认不在基线配置中启用。

### 配置与训练输出

- 新增基线别名：`configs/ads_transformer_side_sidebias.yaml`。
- 新增 阶段 1/2/3 和未来模块占位配置。
- `scripts/train.py` 优先使用 `experiment_name` 生成日志、checkpoint 和输出指标文件名，避免实验互相覆盖。
- 启动日志增加关键实验开关，便于回溯配置。

## 早期完整实验历史

### Base ADS

- 测试 AUC：`0.737232`
- 测试 GAUC：`0.643964`
- 测试 LogLoss：`0.595815`
- 结论：作为非 SideBias ADS 参考。

### SideBias retained baseline

- 实验：`ads_transformer_side_sidebias`
- 测试 AUC：`0.739911`
- 测试 GAUC：`0.645878`
- 测试 LogLoss：`0.597871`
- 结论：早期保留的 SideBias GAUC 基线。

### SideBias + EMA

- 实验：`sidebias_ema`
- 测试 AUC：`0.740797`
- 测试 GAUC：`0.645829`
- 测试 LogLoss：`0.594567`
- 结论：AUC/LogLoss 较均衡，但 GAUC 未超过保留 SideBias。

### Full user-group pairwise

- 实验：`sidebias_userpair`
- 测试 AUC：`0.699547`
- 测试 GAUC：`0.637569`
- 测试 LogLoss：`0.624322`
- 结论：完整替换训练分布伤害泛化和校准，拒绝。

### Global prior dense residual

- 实验：`sidebias_dense`
- 测试 AUC：`0.754205`
- 测试 GAUC：`0.639251`
- 测试 LogLoss：`0.582311`
- 结论：AUC/LogLoss 改善，但 GAUC 明显下降，拒绝。

## 当前实验策略

- 实验级别串行运行，避免多个 full run 干扰结果和资源。
- 模型对比坚持单变量纪律。
- 后续不再把资源利用率统计写入实验文档；资源信息只用于单独的训练吞吐优化。
- 当前主要目标是提升 GAUC，同时不能让 AUC/LogLoss 实质性崩塌。

## 2026-04-30 - 阶段 1 历史过滤实验

### `sidebias_history_click_only`

- 配置：`configs/sidebias_history_click_only.yaml`
- 测试 AUC：`0.736680`
- 测试 GAUC：`0.646484`
- 测试 LogLoss：`0.603616`
- 结论：相对历史 SideBias 略有 GAUC 信号，但校准较差；在后续 DIN 路径中继续验证。

### `sidebias_history_click_or_long_view`

- 配置：`configs/sidebias_history_click_or_long_view.yaml`
- 测试 AUC：`0.731172`
- 测试 GAUC：`0.641942`
- 测试 LogLoss：`0.604102`
- 结论：拒绝。

## 2026-04-30 - 阶段 2 DIN attention

### `sidebias_dinatt`

- 配置：`configs/sidebias_dinatt.yaml`
- 测试 AUC：`0.742062`
- 测试 GAUC：`0.648728`
- 测试 LogLoss：`0.597294`
- 结论：成为当时最佳完整实验，需要 seed 确认。

### `sidebias_dinatt_confirm_seed2026`

- 配置：`configs/sidebias_dinatt_confirm_seed2026.yaml`
- 测试 AUC：`0.741810`
- 测试 GAUC：`0.650264`
- 测试 LogLoss：`0.594934`
- 结论：seed 确认后提升稳定，DIN 被提升为保护基线家族。

### 保护配置

- 新增 `configs/ads_transformer_side_dinatt_protected.yaml`。
- 该配置保留已确认 DIN 架构，供后续单变量实验继承。

## 2026-04-30 - DIN + click-only 历史

### `sidebias_dinatt_history_click_only`

- 配置：`configs/sidebias_dinatt_history_click_only.yaml`
- 测试 AUC：`0.740477`
- 测试 GAUC：`0.654016`
- 测试 LogLoss：`0.601096`
- 结论：GAUC 明显提升，进入 seed 确认。

### `sidebias_dinatt_history_click_only_confirm_seed2026`

- 配置：`configs/sidebias_dinatt_history_click_only_confirm_seed2026.yaml`
- 测试 AUC：`0.740356`
- 测试 GAUC：`0.652649`
- 测试 LogLoss：`0.594600`
- 结论：确认保留，DIN + click-only 成为后续保护参考。

### 保护配置

- 新增 `configs/ads_transformer_side_dinatt_click_only_protected.yaml`。
- 后续单变量实验以 DIN + click-only 为参考族。

## 2026-04-30 至 2026-05-01 - DIN + click-only 后续单变量实验

### 拒绝方向

- `sidebias_dinatt_click_only_pcrg_token`：测试 GAUC `0.652005`，低于保护确认 `0.652649`，拒绝。
- `sidebias_dinatt_click_only_dense_history_only`：测试 GAUC `0.651980`，低于保护确认，拒绝。
- `sidebias_dinatt_click_only_tfusion`：测试 GAUC `0.649408`，LogLoss `0.604401`，拒绝。
- `sidebias_dinatt_click_only_interactions`：测试 GAUC `0.649730`，LogLoss `0.603755`，拒绝。
- `sidebias_dinatt_click_only_side_gate`：测试 GAUC `0.651850`，低于保护确认，拒绝。
- `sidebias_dinatt_click_only_target_tag`：测试 GAUC `0.652278`，低于保护确认，拒绝。
- `sidebias_dinatt_click_only_user_profile`：测试 GAUC `0.649580`，拒绝。
- `sidebias_dinatt_click_only_user_onehot`：测试 GAUC `0.650862`，拒绝。
- `sidebias_dinatt_click_only_no_side_attention_bias`：测试 GAUC `0.650352`，说明 side-attention bias 有用，拒绝。
- `sidebias_dinatt_click_only_no_din_side_bias`：测试 GAUC `0.650953`，说明 DIN 消费 side-bias 有用，拒绝。
- `sidebias_dinatt_click_only_no_behavior_side`：测试 GAUC `0.650112`，LogLoss `0.603077`，拒绝。
- `sidebias_dinatt_click_only_no_psrg`：测试 GAUC `0.647920`，拒绝。
- `sidebias_dinatt_click_only_no_category`：测试 GAUC `0.646873`，拒绝。

### 不确定方向

- `sidebias_dinatt_click_only_mbc_slices`：测试 GAUC `0.652958`，仅小幅高于保护确认，不推广。
- `sidebias_dinatt_click_only_match_features`：测试 GAUC `0.653410`，但低于 seed-2025 最佳 DIN + click-only，且 LogLoss `0.603938`，不推广。
- `sidebias_dinatt_click_only_zero_init_bias`：测试 GAUC `0.653579`，LogLoss `0.600785`，不推广。
- `sidebias_dinatt_click_only_auxrank`：测试 GAUC `0.652887`，LogLoss `0.601402`，不推广。
- `sidebias_dinatt_click_only_auxrank_hard`：测试 GAUC `0.653189`，LogLoss `0.602080`，不推广。
- `sidebias_dinatt_click_only_ema`：测试 GAUC `0.653318`，未超过最佳 DIN + click-only 单次结果，不单独推广。
- `sidebias_dinatt_click_only_no_pcrg`：测试 GAUC `0.653015`，收益小且校准回退，不推广。

## 2026-05-01 至 2026-05-02 - Video-stat 系列

### `sidebias_dinatt_click_only_video_stat`

- 配置：`configs/sidebias_dinatt_click_only_video_stat.yaml`
- 单变量：启用 `use_video_stat: true`
- 测试 AUC：`0.740214`
- 测试 GAUC：`0.656082`
- 测试 LogLoss：`0.595241`
- 结论：显著提升 GAUC，进入 seed 确认。

### `sidebias_dinatt_click_only_video_stat_confirm_seed2026`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_confirm_seed2026.yaml`
- 单变量：仅将 seed 改为 `2026`
- 测试 AUC：`0.741851`
- 测试 GAUC：`0.655718`
- 测试 LogLoss：`0.600709`
- 结论：GAUC 方向确认保留，但 LogLoss 需关注。

### `sidebias_dinatt_click_only_video_stat_caption`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_caption.yaml`
- 单变量：在 video_stat 上启用 caption SVD dense 特征
- 测试 AUC：`0.740404`
- 测试 GAUC：`0.652855`
- 测试 LogLoss：`0.599250`
- 结论：低于已确认 video_stat，拒绝，保持 `use_caption` 默认关闭。

### `sidebias_dinatt_click_only_video_stat_no_pcrg`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_no_pcrg.yaml`
- 单变量：在 video_stat 上设置 `use_pcrg: false`
- 测试 AUC：`0.740559`
- 测试 GAUC：`0.656795`
- 测试 LogLoss：`0.594320`
- 结论：单次强，进入 seed 确认。

### `sidebias_dinatt_click_only_video_stat_no_pcrg_confirm_seed2026`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_no_pcrg_confirm_seed2026.yaml`
- 单变量：仅将 seed 改为 `2026`
- 测试 AUC：`0.744102`
- 测试 GAUC：`0.655680`
- 测试 LogLoss：`0.598598`
- 结论：seed-2026 未保持 seed-2025 的 GAUC 提升，不推广。

## 2026-05-02 - Video-stat + EMA 系列

### `sidebias_dinatt_click_only_video_stat_ema`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema.yaml`
- 单变量：启用 `use_ema: true`，`ema_decay: 0.995`，`ema_warmup_steps: 300`
- 测试 AUC：`0.744170`
- 测试 GAUC：`0.658104`
- 测试 LogLoss：`0.594204`
- 结论：相对 video_stat 明显提升，进入 seed 确认。

### `sidebias_dinatt_click_only_video_stat_ema_confirm_seed2026`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_confirm_seed2026.yaml`
- 单变量：仅将 seed 改为 `2026`
- 测试 AUC：`0.744078`
- 测试 GAUC：`0.658566`
- 测试 LogLoss：`0.594494`
- 结论：已确认保留；`video_stat_ema` 成为当前已确认保护族候选。

### `sidebias_dinatt_click_only_video_stat_ema_auxrank`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_auxrank.yaml`
- 单变量：启用 `use_aux_rank_loader: true`
- 测试 AUC：`0.743504`
- 测试 GAUC：`0.658159`
- 测试 LogLoss：`0.593323`
- 结论：LogLoss 改善，但测试 GAUC 未超过 seed-2026 EMA 确认，不推广。

### `sidebias_dinatt_click_only_video_stat_ema_zero_init_bias`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_zero_init_bias.yaml`
- 单变量：设置 `zero_init_side_attention_bias: true`
- 测试 AUC：`0.744292`
- 测试 GAUC：`0.657954`
- 测试 LogLoss：`0.593893`
- 结论：测试 GAUC 不及已确认 EMA，不推广。

### `sidebias_dinatt_click_only_video_stat_ema_match_features`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_match_features.yaml`
- 单变量：启用 `use_side_attention_match_features: true`
- 测试 AUC：`0.743173`
- 测试 GAUC：`0.658858`
- 测试 LogLoss：`0.593194`
- 结论：当前单次最佳测试 GAUC；相对 seed-2026 EMA 确认提升 `+0.000292`，但幅度小且验证 GAUC 较低，因此进入 seed-2026 确认。

### `sidebias_dinatt_click_only_video_stat_ema_match_features_confirm_seed2026`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_match_features_confirm_seed2026.yaml`
- 单变量：仅将 seed 改为 `2026`
- 最佳验证轮次：`3`
- Valid AUC：`0.758491`
- Valid GAUC：`0.672785`
- Valid LogLoss：`0.577886`
- 测试 AUC：`0.743220`
- 测试 GAUC：`0.657493`
- 测试 LogLoss：`0.593418`
- 结论：确认未通过，不推广。确认测试 GAUC 比 seed-2025 match_features 低 `-0.001365`，也比 `video_stat_ema_confirm_seed2026` 低 `-0.001073`。

### `sidebias_dinatt_click_only_video_stat_ema_dense128`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_dense128.yaml`
- 单变量：将 `dense_hidden_dim` 从 `64` 扩大到 `128`
- 最佳验证轮次：`3`
- Valid AUC：`0.758837`
- Valid GAUC：`0.672454`
- Valid LogLoss：`0.577065`
- 测试 AUC：`0.742641`
- 测试 GAUC：`0.657414`
- 测试 LogLoss：`0.593419`
- 结论：拒绝，不推广。测试 GAUC 比 `video_stat_ema_confirm_seed2026` 低 `-0.001152`，虽然 LogLoss 改善，但主要指标未达标。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices.yaml`
- 单变量：启用 `use_mbc_slices: true`
- 最佳验证轮次：`3`
- Valid AUC：`0.760391`
- Valid GAUC：`0.672036`
- Valid LogLoss：`0.575273`
- 测试 AUC：`0.745756`
- 测试 GAUC：`0.660225`
- 测试 LogLoss：`0.590654`
- 结论：候选保留，需要 seed-2026 确认。相对 `video_stat_ema_confirm_seed2026` 测试 GAUC 提升 `+0.001659`，测试 AUC 和 LogLoss 也同时改善；确认通过前不推广。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2026`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2026.yaml`
- 单变量：仅将 seed 改为 `2026`
- 最佳验证轮次：`4`
- Valid AUC：`0.761254`
- Valid GAUC：`0.673646`
- Valid LogLoss：`0.575920`
- 测试 AUC：`0.745534`
- 测试 GAUC：`0.659100`
- 测试 LogLoss：`0.592021`
- 结论：确认通过，推广为当前保护族候选。相对 `video_stat_ema_confirm_seed2026` 测试 GAUC 提升 `+0.000534`，测试 AUC 和 LogLoss 同时改善。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate02`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate02.yaml`
- 单变量：将 `mbc_gate_init` 从 `0.1` 提高到 `0.2`
- 最佳验证轮次：`4`
- Valid AUC：`0.761660`
- Valid GAUC：`0.671367`
- Valid LogLoss：`0.575925`
- 测试 AUC：`0.746569`
- 测试 GAUC：`0.659271`
- 测试 LogLoss：`0.592276`
- 结论：不推广。测试 AUC 提升，但同 seed 测试 GAUC 比 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.000954`，LogLoss 也变差。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate005`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate005.yaml`
- 单变量：将 `mbc_gate_init` 从 `0.1` 降到 `0.05`
- 最佳验证轮次：`3`
- Valid AUC：`0.760387`
- Valid GAUC：`0.671978`
- Valid LogLoss：`0.575269`
- 测试 AUC：`0.745785`
- 测试 GAUC：`0.659455`
- 测试 LogLoss：`0.590722`
- 结论：不推广。优于 gate02，但同 seed 测试 GAUC 比 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.000770`，不进入 seed 确认。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64.yaml`
- 单变量：将 `mbc_branch_dim` 从 `128` 降到 `64`
- 最佳验证轮次：`4`
- Valid AUC：`0.761129`
- Valid GAUC：`0.672376`
- Valid LogLoss：`0.576527`
- 测试 AUC：`0.745126`
- 测试 GAUC：`0.657912`
- 测试 LogLoss：`0.593888`
- 结论：拒绝。测试 GAUC 比 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.002313`，也低于上一代 `video_stat_ema_confirm_seed2026` 的 `0.658566`。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch256`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch256.yaml`
- 单变量：将 `mbc_branch_dim` 从 `128` 提高到 `256`
- 最佳验证轮次：`3`
- Valid AUC：`0.760838`
- Valid GAUC：`0.671585`
- Valid LogLoss：`0.574909`
- 测试 AUC：`0.745939`
- 测试 GAUC：`0.659231`
- 测试 LogLoss：`0.590988`
- 结论：不推广。测试 AUC 略升，但测试 GAUC 比 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.000994`，不进入 seed 确认。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_auxloss`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_auxloss.yaml`
- 单变量：将 `use_mbc_aux_loss` 从 `false` 改为 `true`
- 最佳验证轮次：`3`
- Valid AUC：`0.759911`
- Valid GAUC：`0.672734`
- Valid LogLoss：`0.576098`
- 测试 AUC：`0.744818`
- 测试 GAUC：`0.658578`
- 测试 LogLoss：`0.592351`
- 结论：不推广。辅助损失生效，但测试 GAUC 比 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.001647`，也低于 MBC seed-2026 确认值 `0.659100`。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion128`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion128.yaml`
- 单变量：将 `mbc_fusion_dim` 从 `64` 提高到 `128`
- 最佳验证轮次：`4`
- Valid AUC：`0.761672`
- Valid GAUC：`0.672226`
- Valid LogLoss：`0.576092`
- 测试 AUC：`0.746084`
- 测试 GAUC：`0.658569`
- 测试 LogLoss：`0.593069`
- 结论：不推广。测试 AUC 略升，但测试 GAUC 比 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.001656`，LogLoss 也明显变差。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32.yaml`
- 单变量：将 `mbc_fusion_dim` 从 `64` 降到 `32`
- 最佳验证轮次：`4`
- Valid AUC：`0.760805`
- Valid GAUC：`0.672489`
- Valid LogLoss：`0.576542`
- 测试 AUC：`0.745559`
- 测试 GAUC：`0.658648`
- 测试 LogLoss：`0.593635`
- 结论：不推广。测试 GAUC 比 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.001577`，LogLoss 明显变差。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015.yaml`
- 单变量：将 `mbc_gate_init` 从 `0.1` 提高到 `0.15`
- 最佳验证轮次：`4`
- Valid AUC：`0.761807`
- Valid GAUC：`0.671644`
- Valid LogLoss：`0.575751`
- 测试 AUC：`0.746850`
- 测试 GAUC：`0.658795`
- 测试 LogLoss：`0.591825`
- 结论：不推广。测试 AUC 提升，但测试 GAUC 比 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.001430`，主指标未达标。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012.yaml`
- 单变量：将 `mbc_gate_init` 从 `0.1` 提高到 `0.12`
- 最佳验证轮次：`3`
- Valid AUC：`0.760379`
- Valid GAUC：`0.671161`
- Valid LogLoss：`0.575321`
- 测试 AUC：`0.745757`
- 测试 GAUC：`0.659435`
- 测试 LogLoss：`0.590688`
- 结论：不推广。测试 GAUC 高于 MBC seed-2026 确认值 `0.659100`，但比同 seed `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.000790`。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008.yaml`
- 单变量：将 `mbc_gate_init` 从 `0.1` 降低到 `0.08`
- 最佳验证轮次：`3`
- Valid AUC：`0.760360`
- Valid GAUC：`0.671379`
- Valid LogLoss：`0.575457`
- 测试 AUC：`0.745761`
- 测试 GAUC：`0.659784`
- 测试 LogLoss：`0.590957`
- 结论：不推广。测试 GAUC 高于 MBC seed-2026 确认值 `0.659100`，但比同 seed `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.000441`。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009.yaml`
- 单变量：将 `mbc_gate_init` 从 `0.1` 降低到 `0.09`
- 最佳验证轮次：`3`
- Valid AUC：`0.760413`
- Valid GAUC：`0.671844`
- Valid LogLoss：`0.575353`
- 测试 AUC：`0.745790`
- 测试 GAUC：`0.659383`
- 测试 LogLoss：`0.590782`
- 结论：不推广。测试 GAUC 低于 gate008，且比同 seed `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.000842`。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256.yaml`
- 单变量：将 `attn_hidden_dim` 从 `128` 提高到 `256`
- 最佳验证轮次：`4`
- Valid AUC：`0.761234`
- Valid GAUC：`0.671796`
- Valid LogLoss：`0.575987`
- 测试 AUC：`0.745455`
- 测试 GAUC：`0.657895`
- 测试 LogLoss：`0.592844`
- 结论：拒绝。测试 GAUC 比同 seed `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.002330`，且低于 MBC seed-2026 确认值。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64.yaml`
- 单变量：将 `attn_hidden_dim` 从 `128` 降到 `64`
- 最佳验证轮次：`4`
- Valid AUC：`0.761276`
- Valid GAUC：`0.672118`
- Valid LogLoss：`0.575886`
- 测试 AUC：`0.745364`
- 测试 GAUC：`0.659065`
- 测试 LogLoss：`0.592487`
- 结论：不推广。测试 GAUC 比同 seed `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.001160`，且略低于 MBC seed-2026 确认值。

### `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005`

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005.yaml`
- 单变量：将 `side_attention_bias_scale` 从 `0.1` 降到 `0.05`
- 最佳验证轮次：`5`
- Valid AUC：`0.761726`
- Valid GAUC：`0.671685`
- Valid LogLoss：`0.577559`
- 测试 AUC：`0.746158`
- 测试 GAUC：`0.656681`
- 测试 LogLoss：`0.594088`
- 结论：拒绝。测试 GAUC 比同 seed `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 低 `-0.003544`，LogLoss 也明显变差。本轮配置级单变量调参停止。

## 当前推荐动作

1. 将 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 作为当前已确认保护族候选。
2. 后续实验必须以该确认候选为基线继续做单变量比较。
3. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate02`，因为同 seed GAUC 未超过当前保护候选。
4. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate005`，因为同 seed GAUC 仍未超过当前保护候选。
5. 拒绝 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64`，因为测试 GAUC 明显退化。
6. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch256`，因为同 seed GAUC 未超过当前保护候选。
7. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_auxloss`，因为辅助分支监督没有提升测试主指标。
8. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion128`，因为扩大 fusion bottleneck 后测试 GAUC 和 LogLoss 退化。
9. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32`，因为缩小 fusion bottleneck 后测试 GAUC 仍未超过保护候选，LogLoss 也退化。
10. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015`，因为测试 AUC 提升但 GAUC 未达标。
11. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012`，因为测试 GAUC 未超过同 seed 保护候选。
12. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008`，因为测试 GAUC 仍未超过同 seed 保护候选。
13. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009`，因为测试 GAUC 低于 gate008 和同 seed 保护候选。
14. 拒绝 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256`，因为测试 GAUC 明显低于当前保护候选和 MBC seed-2026 确认值。
15. 不推广 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64`，因为测试 GAUC 仍低于当前保护候选和 MBC seed-2026 确认值。
16. 拒绝 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005`，因为测试 GAUC 和 LogLoss 明显退化。
17. 不推广 `sidebias_dinatt_click_only_video_stat_ema_match_features`，因为 seed-2026 确认 GAUC 未超过 EMA 确认基线。
18. 不推广 `sidebias_dinatt_click_only_video_stat_ema_dense128`，因为测试 GAUC 低于 EMA 确认基线。
19. 本轮配置级单变量调参已经停止；后续新实验应先经过方案复盘和用户确认，不能叠加 `auxrank`、`zero_init_bias`、caption、no-PCRG、match_features、dense128 等未确认或已拒绝组件。
20. 资源使用统计不再作为实验文档内容；训练加速应单独做吞吐优化实验。
