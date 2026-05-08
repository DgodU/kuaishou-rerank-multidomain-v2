# KuaiRand CTR/Rerank 实验与模型变更总结

日期：2026-05-01

本文档汇总 `/root/autodl-tmp/kuaishou-rerank-multidomain-shared` 中主要实验、模型/预处理改动、阶段性结论和当前保护模型状态。

> 2026-05-08 更新：当前最强 protected candidate 已推进为 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged`，测试 AUC/GAUC/LogLoss=`0.754048/0.663378/0.582084`。四 seed 公平确认 winner 为 `semantic_simtier_sem48_slicegate_reg_mid`，mean/min GAUC=`0.659845/0.656514`。下文保留 2026-05-01 阶段历史结论，最新完整总结见 `docs/experiment_summary_2026-05-08.md`。

## 2026-05-01 阶段结论

当前已确认的保护族主线是：

```text
SideBias + DIN 风格目标注意力 + click-only 历史 + video_stat dense 特征 + EMA + MBC semantic slices
```

关键参考：

- 历史保留 SideBias 测试 GAUC：`0.645878`
- DIN + click_only 保护确认测试 GAUC：`0.652649`
- DIN + click_only seed-2025 最佳观测测试 GAUC：`0.654016`
- `video_stat` seed-2025 测试 GAUC：`0.656082`
- `video_stat` seed-2026 确认测试 GAUC：`0.655718`
- `video_stat_ema` seed-2025 测试 GAUC：`0.658104`
- `video_stat_ema` seed-2026 确认测试 GAUC：`0.658566`
- `video_stat_ema_match_features` seed-2025 测试 GAUC：`0.658858`；seed-2026 确认测试 GAUC：`0.657493`，确认未通过
- `video_stat_ema_mbc_slices` seed-2025 测试 GAUC：`0.660225`，当前为单次最佳观测；seed-2026 确认测试 GAUC：`0.659100`，确认通过

结论：

- 截至 2026-05-01，`video_stat_ema_mbc_slices` 是当时已确认保护族候选。
- `video_stat_ema_match_features` 确认未通过，不推广；`video_stat_ema_dense128` 也不推广。
- `auxrank`、`zero_init_bias`、`no_pcrg` 等方向目前不推广，也不要与其他不确定/拒绝组件叠加。
- 后续资源监控不再作为实验文档记录重点；如需加速训练，应单独做训练吞吐优化实验。

## 主要模型与预处理变更

### 实验基础设施

- 增加基于 `experiment_name` 的日志、checkpoint 和输出命名，避免多个实验互相覆盖。
- 增加/使用实验追踪文件：
  - `experiments/results_tracking.md`
  - `outputs/model_comparison.md`
  - `outputs/model_comparison.json`
  - `docs/optimization_roadmap.md`
  - `CHANGELOG.md`

### 历史行为预处理变体

实现 `history_mode` 支持：

- `all`
- `click_only`
- `click_or_long_view`

同时在预处理摘要中加入历史长度统计。

结论：

- `click_only` 成为保护 DIN 家族的一部分。
- `click_or_long_view` 被拒绝。

### DIN 风格目标注意力

在配置项下加入 `DINStyleTargetAttention`：

```yaml
attention_type: din_mlp
```

DIN scorer 使用基于目标-历史匹配特征的 MLP 注意力打分，替代保护路径中的简单点积打分。

结论：

- DIN 风格注意力为 `confirmed_keep`。
- 它相对原始保留 SideBias 基线有提升，并进入保护族。

### Side attention bias 控制项

加入或验证以下独立控制项：

- `use_side_attention_bias`
- `use_side_bias`
- `use_side_attention_interactions`
- `use_side_attention_match_features`
- `use_side_attention_gate`
- `zero_init_side_attention_bias`

重要区别：

- `use_side_attention_bias` 控制是否生成基于 side 信息的 attention score bias。
- `use_side_bias` 控制 `DINStyleTargetAttention` 是否消费该 bias。

结论：

- 移除 side-attention bias 会伤害 GAUC。
- 禁用 DIN 对生成 side-bias 的消费也会伤害 GAUC。
- 当前保护族保持二者开启。

### 时间上下文支持

加入预处理/模型支持：

- `hour_of_day`
- `day_of_week`
- `is_weekend`

结论：

- 在保留 SideBias 路径下的完整实验被拒绝。
- 没有重新设计前，不并入当前 DIN 保护族。

### PCRG token、TransformerFusion、MBC slices

加入并验证模块化组件/配置：

- `PCRGTokenLayer`
- `TransformerFusion`
- `MBCSemanticHead` / semantic slice path

结论：

- `pcrg_token`：拒绝。
- `transformer_fusion`：拒绝。
- `mbc_slices`：不确定，不推广。

### 辅助排序目标

实现保守的辅助排序训练：

- 主 BCE DataLoader 保持普通随机训练。
- 可选的同用户辅助排序 batch 每 N step 插入一次。
- 测试过 all-pairs 和 hardest-pair 变体。

结论：

- 辅助排序在 DIN + click_only 和 video_stat_ema 上都只带来很小或不稳定的 GAUC 信号。
- 虽然部分 LogLoss 改善，但 GAUC 未超过已确认 EMA 基线，因此不推广、不叠加。

### Category 消融支持

在 `src/models/ads.py` 中加入有效的 `use_category` 控制。

当 `use_category: false` 时：

- 序列 item 表征移除类别 embedding。
- 目标 item 表征移除类别 embedding。
- domain 表征移除类别 embedding。
- 输出维度 `E_S`、`E_Q`、`E_D` 保持固定。

结论：

- 移除 category 造成较大 GAUC 下降。
- 保持 `use_category: true`。

### Video statistic dense 特征

让 `use_video_stat` 成为有效单变量开关：

- `scripts/preprocess.py` 读取 `video_features_statistic_pure.csv`。
- 生成 50 个 `dense_video_stat_*` 特征，使用数值/log 缩放变换。
- 将其加入 `dense_cols` 和处理后的模型列。
- `src/models/ads_transformer_side.py` 在 `use_video_stat: true` 时启用 dense projection/residual 路径。

结论：

- `video_stat` 成为新的 GAUC 领先候选。
- seed 确认在 GAUC 方向上稳定。
- 后续 EMA 将其进一步提升为当前已确认保护族候选。

### Caption dense 特征

在 `video_stat` 确认后，让 `use_caption` 成为有效开关：

- 读取 `data/raw/kuairand_video_captions.csv`。
- 对 caption 文本构建 TF-IDF。
- 应用 `TruncatedSVD(n_components=64)`。
- 生成 `dense_caption_caption_svd_*` 特征。
- 将 caption dense 列加入 `dense_cols`。
- `use_caption: true` 时启用 dense projection/residual 路径。

`sidebias_dinatt_click_only_video_stat_caption` 调试结果：

- `dense_dim = 114`
  - 50 个 video-stat dense 特征
  - 64 个 caption SVD dense 特征
- `dense_features = (256, 114)`
- `final_input = (256, 288)`
- 参数量：`2,426,105`

完整结果：

- 最佳验证轮次：`4`
- Valid AUC：`0.757339`
- Valid GAUC：`0.669494`
- Valid LogLoss：`0.582670`
- Test AUC：`0.740404`
- Test GAUC：`0.652855`
- Test LogLoss：`0.599250`
- 结论：拒绝/不推广。caption 叠加在 video-stat 上后，测试 GAUC 相对 seed-2026 确认 video-stat 下降 `-0.002863`，相对 seed-2025 video-stat 下降 `-0.003227`。

### `sidebias_dinatt_click_only_video_stat_no_pcrg`

目的：

- 在已确认 video-stat 候选上重新评估早期不确定的 `no_pcrg` 消融。
- 相对 `sidebias_dinatt_click_only_video_stat` 保持单变量：只把 `use_pcrg` 从 `true` 改为 `false`。

配置：

- `configs/sidebias_dinatt_click_only_video_stat_no_pcrg.yaml`
- `base_model: sidebias_dinatt_click_only_video_stat`
- `data_dir: data/processed_click_only_video_stat`
- `use_video_stat: true`
- `use_pcrg: false`
- `use_caption: false`

调试结果：

- 复用现有 `processed_click_only_video_stat` debug 数据。
- `dense_features = (256, 50)`
- `final_input = (256, 288)`
- 参数量：`2,421,881`

完整结果：

- 最佳验证轮次：`3`
- Valid AUC：`0.756581`
- Valid GAUC：`0.671029`
- Valid LogLoss：`0.579429`
- Test AUC：`0.740559`
- Test GAUC：`0.656795`
- Test LogLoss：`0.594320`
- Seed-2025 确认前结论：单次最佳 GAUC，等待 seed 确认。

Seed 确认：

- 配置：`configs/sidebias_dinatt_click_only_video_stat_no_pcrg_confirm_seed2026.yaml`
- Seed：`2026`
- 最佳验证轮次：`5`
- Valid AUC：`0.759574`
- Valid GAUC：`0.670878`
- Valid LogLoss：`0.581021`
- Test AUC：`0.744102`
- Test GAUC：`0.655680`
- Test LogLoss：`0.598598`
- 最终结论：不确定/不推广。seed-2026 确认未保持 seed-2025 的 GAUC 提升：相对 seed-2025 no-PCRG 为 `-0.001115`，相对 seed-2026 video-stat 确认为 `-0.000038`。

### `sidebias_dinatt_click_only_video_stat_ema`

目的：

- 在已确认 video-stat 候选上重新评估 EMA 训练/评估。
- 相对 `sidebias_dinatt_click_only_video_stat` 保持单变量：只把 `use_ema` 从 `false` 改为 `true`。

配置：

- `configs/sidebias_dinatt_click_only_video_stat_ema.yaml`
- `base_model: sidebias_dinatt_click_only_video_stat`
- `data_dir: data/processed_click_only_video_stat`
- `use_video_stat: true`
- `use_ema: true`
- `ema_decay: 0.995`
- `ema_warmup_steps: 300`

调试结果：

- 复用现有 `processed_click_only_video_stat` debug 数据。
- `dense_features = (256, 50)`
- `final_input = (256, 288)`
- 参数量：`2,421,881`

完整结果：

- 最佳验证轮次：`4`
- Valid AUC：`0.760282`
- Valid GAUC：`0.671989`
- Valid LogLoss：`0.577461`
- Test AUC：`0.744170`
- Test GAUC：`0.658104`
- Test LogLoss：`0.594204`
- Seed-2025 结论：候选保留。EMA 相对 seed-2025 video-stat 测试 GAUC 提升 `+0.002022`，相对 seed-2026 video-stat 确认提升 `+0.002386`，同时改善 AUC 和 LogLoss。

Seed 确认：

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_confirm_seed2026.yaml`
- Seed：`2026`
- 最佳验证轮次：`4`
- Valid AUC：`0.759424`
- Valid GAUC：`0.672000`
- Valid LogLoss：`0.578526`
- Test AUC：`0.744078`
- Test GAUC：`0.658566`
- Test LogLoss：`0.594494`
- 最终结论：已确认保留/已推广。确认实验相对 seed-2025 EMA 测试 GAUC 提升 `+0.000461`，相对 seed-2026 不含 EMA 的 video-stat 提升 `+0.002848`。

## 完整实验结果汇总

### 基线与已确认保留

| 实验 | 主要改动 | 测试GAUC | 测试AUC | 测试LogLoss | 结论 |
|---|---|---:|---:|---:|---|
| `best_sidebias` | 历史保留 SideBias | `0.645878` | - | - | 保留参考 |
| `sidebias_dinatt` | DIN 风格注意力，seed 2025 | `0.648728` | `0.742062` | `0.597294` | 候选保留 |
| `sidebias_dinatt_confirm_seed2026` | DIN 确认 | `0.650264` | `0.741810` | `0.594934` | 已确认保留 |
| `sidebias_dinatt_history_click_only` | DIN + click-only，seed 2025 | `0.654016` | `0.740477` | `0.601096` | 候选保留 |
| `sidebias_dinatt_history_click_only_confirm_seed2026` | DIN + click-only 确认 | `0.652649` | `0.740356` | `0.594600` | 保护参考 |
| `sidebias_dinatt_click_only_video_stat` | 增加 video_stat dense 特征 | `0.656082` | `0.740214` | `0.595241` | 候选保留 |
| `sidebias_dinatt_click_only_video_stat_confirm_seed2026` | video_stat seed 确认 | `0.655718` | `0.741851` | `0.600709` | 已确认保留 |
| `sidebias_dinatt_click_only_video_stat_ema` | video_stat 上启用 EMA | `0.658104` | `0.744170` | `0.594204` | 已确认保留 |
| `sidebias_dinatt_click_only_video_stat_ema_confirm_seed2026` | EMA seed 确认 | `0.658566` | `0.744078` | `0.594494` | 已确认保留 |

### 拒绝实验

| 实验 | 主要改动 | 测试GAUC | 测试AUC | 测试LogLoss | 原因 |
|---|---|---:|---:|---:|---|
| `sidebias_userpair` | 完整 user-group pairwise 替代训练 | `0.637569` | `0.699547` | `0.624322` | 伤害校准和 GAUC |
| `sidebias_auxrank_hard` | 旧 SideBias 上 hard 辅助排序 | `0.644389` | `0.737734` | `0.597000` | 低于 all-pairs auxrank 和保留 SideBias |
| `sidebias_dense` | 全局先验 dense residual | `0.639251` | `0.754205` | `0.582311` | AUC/LogLoss 改善但 GAUC 下降 |
| `sidebias_history_click_or_long_view` | click_or_long_view 历史 | `0.641942` | `0.731172` | `0.604102` | 低于基线/保护族 |
| `sidebias_time_context` | 小时/星期/周末上下文 | `0.644237` | `0.737989` | `0.600681` | 低于 DIN 保护族 |
| `sidebias_dinatt_click_only_pcrg_token` | PCRG token attention | `0.652005` | `0.742101` | `0.599881` | 低于保护 DIN + click_only |
| `sidebias_dinatt_click_only_dense_history_only` | Dense history-only 特征 | `0.651980` | `0.739373` | `0.597513` | 低于保护 DIN + click_only |
| `sidebias_dinatt_click_only_tfusion` | TransformerFusion | `0.649408` | `0.740293` | `0.604401` | GAUC 下降且 LogLoss 变差 |
| `sidebias_dinatt_click_only_interactions` | Side-attention Q/K 交互特征 | `0.649730` | `0.740526` | `0.603755` | GAUC 下降且 LogLoss 变差 |
| `sidebias_dinatt_click_only_side_gate` | 可学习 side-bias 标量 gate | `0.651850` | `0.740426` | `0.601276` | 低于保护确认 |
| `sidebias_dinatt_click_only_target_tag` | Target tag embedding | `0.652278` | `0.741647` | `0.599354` | 略低于保护且 LogLoss 变差 |
| `sidebias_dinatt_click_only_user_profile` | 静态用户 profile 上下文 | `0.649580` | `0.738541` | `0.597468` | GAUC 回退 |
| `sidebias_dinatt_click_only_user_onehot` | 用户 onehot 上下文 | `0.650862` | `0.740881` | `0.599247` | GAUC 回退 |
| `sidebias_dinatt_click_only_no_side_attention_bias` | 移除 side-attention bias 生成 | `0.650352` | `0.741327` | `0.600727` | side-attention bias 有用 |
| `sidebias_dinatt_click_only_no_din_side_bias` | 禁用 DIN 对 side-bias 的消费 | `0.650953` | `0.740166` | `0.602755` | side-bias 消费有用 |
| `sidebias_dinatt_click_only_no_behavior_side` | 移除 behavior-side 输入 | `0.650112` | `0.739494` | `0.603077` | behavior side 有用 |
| `sidebias_dinatt_click_only_no_psrg` | 移除 PSRG | `0.647920` | `0.736732` | `0.601005` | GAUC 大幅下降 |
| `sidebias_dinatt_click_only_no_category` | 移除 category 特征 | `0.646873` | `0.738465` | `0.600461` | GAUC 大幅下降 |
| `sidebias_dinatt_click_only_video_stat_caption` | video_stat 上叠加 Caption SVD dense 特征 | `0.652855` | `0.740404` | `0.599250` | 低于已确认 video_stat，不推广 |

### 不确定实验

这些实验相对保护确认有小幅收益或指标混合，但未充分超过已确认基线，或 LogLoss/AUC 有回退。没有重设计或 seed 确认前不应叠加。

| 实验 | 主要改动 | 测试GAUC | 测试AUC | 测试LogLoss | 状态 |
|---|---|---:|---:|---:|---|
| `sidebias_auxrank` | 旧 SideBias 上辅助排序 | `0.645426` | `0.738183` | `0.597461` | 不确定 |
| `sidebias_history_click_only` | dot attention 下 click-only 历史 | `0.646484` | `0.736680` | `0.603616` | 相对旧 SideBias 不确定 |
| `sidebias_dinatt_click_only_mbc_slices` | MBC semantic slices | `0.652958` | `0.740041` | `0.597156` | 不确定 |
| `sidebias_dinatt_click_only_match_features` | Side-attention match features | `0.653410` | `0.741693` | `0.603938` | 不确定 |
| `sidebias_dinatt_click_only_zero_init_bias` | Zero-init side-attention bias 输出 | `0.653579` | `0.740534` | `0.600785` | 不确定 |
| `sidebias_dinatt_click_only_auxrank` | DIN + click-only 上辅助排序 | `0.652887` | `0.741138` | `0.601402` | 不确定 |
| `sidebias_dinatt_click_only_auxrank_hard` | Hard auxiliary rank loader | `0.653189` | `0.740841` | `0.602080` | 不确定 |
| `sidebias_dinatt_click_only_ema` | EMA 训练 | `0.653318` | `0.741080` | `0.597654` | 不确定 |
| `sidebias_dinatt_click_only_no_pcrg` | 移除 PCRG | `0.653015` | `0.741535` | `0.601045` | 不确定 |
| `sidebias_dinatt_click_only_video_stat_no_pcrg` | video_stat 上移除 PCRG | `0.656795` | `0.740559` | `0.594320` | 单次强但未确认 |
| `sidebias_dinatt_click_only_video_stat_no_pcrg_confirm_seed2026` | no-PCRG seed 确认 | `0.655680` | `0.744102` | `0.598598` | 不确定/不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_auxrank` | video_stat EMA 上辅助排序 | `0.658159` | `0.743504` | `0.593323` | 不确定/不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_zero_init_bias` | video_stat EMA 上 zero-init side-attention bias | `0.657954` | `0.744292` | `0.593893` | 不确定/不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_match_features` | video_stat EMA 上 side-attention match features | `0.658858` | `0.743173` | `0.593194` | 单次最佳但确认未通过 |
| `sidebias_dinatt_click_only_video_stat_ema_match_features_confirm_seed2026` | match_features seed 确认 | `0.657493` | `0.743220` | `0.593418` | 确认未通过/不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_dense128` | video_stat EMA 上扩大 dense_hidden_dim 到 128 | `0.657414` | `0.742641` | `0.593419` | 拒绝/不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` | video_stat EMA 上启用 MBC semantic slices | `0.660225` | `0.745756` | `0.590654` | 已确认保留 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2026` | MBC semantic slices seed 确认 | `0.659100` | `0.745534` | `0.592021` | 确认通过/当前保护族 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate02` | 当前保护族上将 mbc_gate_init 提高到 0.2 | `0.659271` | `0.746569` | `0.592276` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate005` | 当前保护族上将 mbc_gate_init 降到 0.05 | `0.659455` | `0.745785` | `0.590722` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64` | 当前保护族上将 mbc_branch_dim 降到 64 | `0.657912` | `0.745126` | `0.593888` | 拒绝 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch256` | 当前保护族上将 mbc_branch_dim 提高到 256 | `0.659231` | `0.745939` | `0.590988` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_auxloss` | 当前保护族上启用 MBC 辅助分支监督 | `0.658578` | `0.744818` | `0.592351` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion128` | 当前保护族上将 mbc_fusion_dim 提高到 128 | `0.658569` | `0.746084` | `0.593069` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32` | 当前保护族上将 mbc_fusion_dim 降到 32 | `0.658648` | `0.745559` | `0.593635` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015` | 当前保护族上将 mbc_gate_init 提高到 0.15 | `0.658795` | `0.746850` | `0.591825` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012` | 当前保护族上将 mbc_gate_init 提高到 0.12 | `0.659435` | `0.745757` | `0.590688` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008` | 当前保护族上将 mbc_gate_init 降到 0.08 | `0.659784` | `0.745761` | `0.590957` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009` | 当前保护族上将 mbc_gate_init 降到 0.09 | `0.659383` | `0.745790` | `0.590782` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256` | 当前保护族上将 attn_hidden_dim 提高到 256 | `0.657895` | `0.745455` | `0.592844` | 拒绝 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64` | 当前保护族上将 attn_hidden_dim 降到 64 | `0.659065` | `0.745364` | `0.592487` | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005` | 当前保护族上将 side_attention_bias_scale 降到 0.05 | `0.656681` | `0.746158` | `0.594088` | 拒绝/停止调参 |

## 历史资源观察

以下内容仅保留为历史运行观察，用于判断训练吞吐瓶颈；后续实验记录不再统计资源使用情况。

近期 `video_stat` 相关运行整体表现为 CPU/RAM/数据管线偏重，GPU 利用率偏低。该信息应服务于单独的训练加速优化，而不是作为模型优劣结论。

## 磁盘清理记录

数据盘曾达到 100% 使用率，主要来源是 `/root/autodl-tmp/data`。

删除了可重新生成的 processed 数据集：

- `/root/autodl-tmp/data/processed_dense`
- `/root/autodl-tmp/data/processed_time_context`
- `/root/autodl-tmp/data/processed_history_click_or_long_view`
- `/root/autodl-tmp/data/processed_click_only_dense_history_only`
- `/root/autodl-tmp/data/processed_history_all`

保留：

- `/root/autodl-tmp/data/raw`
- `/root/autodl-tmp/data/processed`
- `/root/autodl-tmp/data/processed_history_click_only`
- `/root/autodl-tmp/data/processed_click_only_video_stat`
- `/root/autodl-tmp/data/processed_click_only_video_stat_caption`

清理后，`/root/autodl-tmp` 曾有约 `30GB` 可用空间，使用率约 `51%`。

## 推荐下一步

1. 将 `video_stat_ema_mbc_slices` 视为当前已确认保护族候选。
2. 将不含 MBC semantic slices 的 `video_stat_ema` 保留为上一代已确认参考。
3. 不推广 `video_stat_no_pcrg`：单次 GAUC 强，但 seed-2026 未确认收益。
4. 不推广 `video_stat_ema_auxrank`：LogLoss 改善，但未超过 seed-2026 EMA 确认 GAUC。
5. 不推广 `video_stat_ema_zero_init_bias`：验证 GAUC 和 LogLoss 改善，但测试 GAUC 未超过已确认 EMA。
6. 不推广 `video_stat_ema_match_features`：seed-2026 确认测试 GAUC 为 `0.657493`，低于 `video_stat_ema_confirm_seed2026` 的 `0.658566`。
7. 不推广 `video_stat_ema_dense128`：测试 GAUC 为 `0.657414`，低于 `video_stat_ema_confirm_seed2026` 的 `0.658566`。
8. 不推广 `video_stat_ema_mbc_slices_gate02`：同 seed 测试 GAUC 低于当前保护候选，且 LogLoss 变差。
9. 不推广 `video_stat_ema_mbc_slices_gate005`：优于 gate02，但同 seed 测试 GAUC 仍低于当前保护候选。
10. 拒绝 `video_stat_ema_mbc_slices_branch64`：测试 GAUC 低于当前保护候选和上一代 EMA 确认基线。
11. 不推广 `video_stat_ema_mbc_slices_branch256`：测试 GAUC 仍低于同 seed 当前保护候选。
12. 不推广 `video_stat_ema_mbc_slices_auxloss`：辅助分支监督生效但测试 GAUC/AUC/LogLoss 均未超过当前保护候选。
13. 不推广 `video_stat_ema_mbc_slices_fusion128`：扩大 fusion bottleneck 后测试 AUC 略升，但 GAUC 和 LogLoss 均未达标。
14. 不推广 `video_stat_ema_mbc_slices_fusion32`：缩小 fusion bottleneck 仍未超过当前保护候选，LogLoss 明显变差。
15. 不推广 `video_stat_ema_mbc_slices_gate015`：测试 AUC 提升，但主指标 GAUC 低于当前确认值。
16. 不推广 `video_stat_ema_mbc_slices_gate012`：测试 GAUC 高于 MBC seed-2026 确认值，但仍低于同 seed 保护候选。
17. 不推广 `video_stat_ema_mbc_slices_gate008`：测试 GAUC 继续接近但仍低于同 seed 保护候选。
18. 不推广 `video_stat_ema_mbc_slices_gate009`：测试 GAUC 低于 gate008 和同 seed 保护候选，结束 gate 细扫。
19. 拒绝 `video_stat_ema_mbc_slices_attn256`：测试 GAUC 低于当前保护候选和 MBC seed-2026 确认值。
20. 不推广 `video_stat_ema_mbc_slices_attn64`：测试 GAUC 优于 attn256，但仍低于当前保护候选和 MBC seed-2026 确认值。
21. 拒绝 `video_stat_ema_mbc_slices_sideattnscale005`：测试 GAUC 和 LogLoss 明显退化，仅 AUC 略高不构成推广依据。
22. 结束本轮配置级单变量调参；后续不再自动启动新的调参实验，应转向外部复盘、特征侧、训练目标或结构级方案设计。
23. 不要叠加已拒绝、不确定或未确认组件。
24. 保持 `use_caption` 默认关闭，因为 `sidebias_dinatt_click_only_video_stat_caption` 低于已确认 `video_stat`。
