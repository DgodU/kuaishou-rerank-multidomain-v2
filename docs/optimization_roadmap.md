# 优化路线图

<!-- roadmap_reader_note:start -->
## Reader Note

This roadmap explains how the model family evolved. It includes many rejected or historical directions because they are useful for avoiding repeated experiments.

Current high-level path:

```text
SideBias baseline
  -> DIN-style target attention
  -> click-only history
  -> video-stat dense features
  -> EMA
  -> MBC slices
  -> semantic target + SimTier + regularized slice gates
  -> train+valid merged protected follow-up
```

Current protected candidate: `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged`.

Latest compact summary: `experiment_summary_2026-05-08.md`.
<!-- roadmap_reader_note:end -->


## 当前主模型

当前最强 protected candidate 是：SideBias + DIN 风格目标注意力 + click-only 历史 + video-stat dense 特征 + EMA + MBC slices + semantic SimTier sem48 + trainable per-slice gates with regularization，并在最终 protected follow-up 中合并 train+valid 训练。

- 模型：ADS-Transformer-SideInfo + Side Attention Bias + DIN 风格目标注意力 + click-only 历史过滤 + 目标视频统计 dense 特征 + EMA 评估/保存 + semantic target/SimTier MBC slices + per-slice gates
- 当前 protected candidate 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged.yaml`
- 当前 protected candidate 测试 AUC/GAUC/LogLoss：`0.754048/0.663378/0.582084`
- 公平确认 winner 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid.yaml`
- 公平确认 winner 四 seed test GAUC mean/min/std：`0.659845/0.656514/0.002195`
- 上一代 MBC slices 四 seed test GAUC mean/min/std：`0.657891/0.654963/0.001991`
- 主要目标：提升以 GAUC 衡量的用户内排序质量

2026-05-08 工程状态：random auxiliary、PAL、rank/calibration split、history dense、author、long-short interest、PCRG token、TransformerFusion、semantic target/SimTier、dynamic MBC gate、Static MBC residual/main-input、no-validation training、CCSS loss 均由独立 config flag 控制，默认不影响未显式启用的实验配置。CCSS 当前未超过 no-validation protected，不作为保护模型。

上一代不含 EMA 的 video-stat 家族仍作为已确认参考：

- 候选配置：`configs/sidebias_dinatt_click_only_video_stat.yaml`
- 确认配置：`configs/sidebias_dinatt_click_only_video_stat_confirm_seed2026.yaml`
- Seed-2025 测试 GAUC：0.656082
- Seed-2026 确认测试 GAUC：0.655718
- 状态：已确认保留，但已被 EMA 版本替代为当前主参考

之前 no-PCRG 的单次收益未被确认：

- 候选配置：`configs/sidebias_dinatt_click_only_video_stat_no_pcrg.yaml`
- 确认配置：`configs/sidebias_dinatt_click_only_video_stat_no_pcrg_confirm_seed2026.yaml`
- Seed-2025 测试 GAUC：0.656795
- Seed-2026 确认测试 GAUC：0.655680
- 状态：不确定，不推广

之前 DIN + click-only 保护族仍是稳定基线参考：

- 保护配置别名：`configs/ads_transformer_side_dinatt_click_only_protected.yaml`
- 架构模板：`configs/sidebias_dinatt_history_click_only.yaml`
- 确认配置：`configs/sidebias_dinatt_history_click_only_confirm_seed2026.yaml`
- Seed-2025 测试 GAUC：0.654016
- Seed-2026 确认测试 GAUC：0.652649

更早的 DIN-only 保护族保留为历史参考：

- 保护配置别名：`configs/ads_transformer_side_dinatt_protected.yaml`
- Seed-2026 确认测试 GAUC：0.650264

更早保留的 SideBias 基线保留为历史参考：

- 配置：`configs/ads_transformer_side_sidebias.yaml`
- 测试 GAUC：0.645878

平衡备份模型是 `sidebias_ema`：

- 测试 AUC：0.740797
- 测试 GAUC：0.645829
- 测试 LogLoss：0.594567

## 暂不优先的方向

以下方向暂不作为下一轮主线：

- 完整 user-group pairwise 替代训练，历史测试 GAUC 仅 0.637569。
- hardest-pair 辅助排序，在旧 SideBias 上泛化更差，测试 GAUC 0.644389。
- dense 全局先验 residual，虽然改善 AUC/LogLoss，但测试 GAUC 降至 0.639251。
- `sidebias_history_click_or_long_view`，测试 GAUC 仅 0.641942。
- `sidebias_time_context`，测试 GAUC 仅 0.644237。
- `sidebias_dinatt_click_only_pcrg_token`，测试 GAUC 0.652005，低于保护的 DIN + click-only。
- `sidebias_dinatt_click_only_dense_history_only`，测试 GAUC 0.651980，低于保护的 DIN + click-only。
- `sidebias_dinatt_click_only_tfusion`，测试 GAUC 0.649408，且 LogLoss 变差。
- `sidebias_dinatt_click_only_video_stat_caption`，caption SVD dense 特征叠加在已确认 video-stat 上后测试 GAUC 降至 0.652855。

## 推荐单变量阶段

### 阶段 1：历史过滤

完整实验后的状态：

- `click_only`：在 DIN 上复跑后确认保留，seed-2025 测试 GAUC 0.654016，seed-2026 确认测试 GAUC 0.652649。
- `click_or_long_view`：拒绝，测试 GAUC 0.641942。

结论：click-only 历史是当前保护 DIN 家族的一部分。

### 阶段 2：DIN 风格注意力打分 + SideBias

状态：已确认保留。

DIN 风格目标注意力保留 SideBias，并将注意力打分从点积改为基于 `[Q, K, Q - K, Q * K]` 的 MLP。

结论：该模块仍是后续单变量实验的参考架构组成。

### 阶段 3：时间上下文特征

状态：相对 SideBias 完整实验后拒绝。

仅加入：

- `hour_of_day`
- `day_of_week`
- `is_weekend`

默认注入 domain embedding 和 SideBias 生成分支。

结论：除非有重新设计后的独立证据，否则不并入 DIN 保护族。

### 阶段 4：PCRG multi-interest token

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_pcrg_token` 测试 GAUC 0.652005，低于保护的 `DIN + click_only` 确认值 0.652649。

结论：默认关闭 `use_pcrg_token`。

### 阶段 5：兴趣 token 上的 TransformerFusion

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_tfusion` 测试 GAUC 0.649408，低于保护的 `DIN + click_only` 确认值 0.652649。

结论：除非重设计后的 PCRG token 先独立证明收益，否则不要叠加 TransformerFusion；默认关闭 `use_transformer_fusion`。

### 阶段 6：MBC semantic slices

状态：相对当前保护族完整实验后不确定，不推广。

`sidebias_dinatt_click_only_mbc_slices` 测试 GAUC 0.652958，仅比保护的 `DIN + click_only` 确认值 0.652649 高 +0.000309，但低于最佳观测 `DIN + click_only` 0.654016。

结论：除非确认或重设计能带来更大、更稳定的测试 GAUC 收益且不损害 AUC/LogLoss，否则不纳入保护族。

### 阶段 6b：Side-attention match features

状态：相对旧保护族完整实验后不确定，不推广。

`sidebias_dinatt_click_only_match_features` 测试 GAUC 0.653410，比保护的 `DIN + click_only` 确认值 0.652649 高 +0.000761，但低于最佳观测 `DIN + click_only` 0.654016，且测试 LogLoss 变差到 0.603938。

结论：在 seed 确认或校准重设计前，不推广到保护族。

### 阶段 6c：Side-attention interaction features

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_interactions` 测试 GAUC 0.649730，低于保护的 `DIN + click_only` 确认值 0.652649，也低于最佳观测 `DIN + click_only` 0.654016。

结论：不要把 `Q*K` / `abs(Q-K)` side-attention 交互特征纳入保护族。

### 阶段 6d：Side-attention scalar gate

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_side_gate` 测试 GAUC 0.651850，低于保护确认值 0.652649，也低于最佳观测 0.654016。

结论：不推广可学习的 Side Attention Bias 标量 gate。

### 阶段 6e：Zero-init side-attention bias

状态：相对当前保护族完整实验后不确定，不推广。

`sidebias_dinatt_click_only_zero_init_bias` 测试 GAUC 0.653579，比保护确认值 0.652649 高 +0.000930，但低于最佳观测 0.654016，且测试 LogLoss 变差到 0.600785。

结论：没有 seed 确认或校准重设计前，不推广 zero-init Side Attention Bias。

### 阶段 7：Dense history-only 特征

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_dense_history_only` 测试 GAUC 0.651980，低于保护确认值 0.652649。

结论：不要把全局先验 CTR/count 特征直接放入主排序 logit；除非重设计并独立验证，否则不推广。

### 阶段 8：Auxiliary rank loader

状态：相对当前保护族完整实验后不确定，不推广。

`sidebias_dinatt_click_only_auxrank` 测试 GAUC 0.652887，比保护确认值 0.652649 仅高 +0.000238，低于最佳观测 0.654016，且测试 LogLoss 变差到 0.601402。

结论：同用户辅助排序改善验证 GAUC，但测试增益太小且校准变差。没有确认或重新设计 objective schedule/weight 前，不推广、不叠加。

### 阶段 8b：Hard auxiliary rank loader

状态：相对当前保护族完整实验后不确定，不推广。

`sidebias_dinatt_click_only_auxrank_hard` 测试 GAUC 0.653189，比保护确认值 0.652649 高 +0.000540，但低于最佳观测 0.654016，且测试 LogLoss 变差到 0.602080。

结论：hardest-pair 在测试 GAUC 上略优于 all-pairs auxrank，但收益仍小、校准更差。没有确认或重设计前，不推广、不叠加。

### 阶段 9：保护 DIN + click-only 上的 EMA

状态：相对当前保护族完整实验后不确定，不推广。

`sidebias_dinatt_click_only_ema` 测试 GAUC 0.653318，比保护确认值 0.652649 高 +0.000669，但低于最佳观测 0.654016，且测试 LogLoss 为 0.597654。

结论：EMA 改善验证指标并减少部分 LogLoss 回退，但测试 GAUC 仍未超过最佳单 seed 结果。没有 seed 确认前不推广。

### 阶段 10：Target tag 特征

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_target_tag` 测试 GAUC 0.652278，比保护确认值 0.652649 低 -0.000371，比最佳观测 0.654016 低 -0.001738，且测试 LogLoss 变差到 0.599354。

结论：保持 `use_target_tag` 默认关闭，不推广、不叠加 target tag embedding。

### 阶段 11：User profile context

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_user_profile` 测试 GAUC 0.649580，比保护确认值低 -0.003069，比最佳观测低 -0.004436。

结论：向 `E_D` 加入用户静态 bucket 上下文伤害排序质量，保持 `use_user_profile_context` 默认关闭。

### 阶段 12：User onehot context

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_user_onehot` 测试 GAUC 0.650862，比保护确认值低 -0.001787，比最佳观测低 -0.003154。

结论：用户 onehot 稀疏字段优于静态 profile bucket 变体，但仍低于保护确认，保持 `use_user_onehot_context` 默认关闭。

### 阶段 13：Side-attention-bias 消融

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_no_side_attention_bias` 测试 GAUC 0.650352，比保护确认值低 -0.002297，比最佳观测低 -0.003664。

结论：去掉 side-attention bias 会伤害排序质量，保护族保持 `use_side_attention_bias: true`。

### 阶段 14：Behavior side 消融

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_no_behavior_side` 测试 GAUC 0.650112，比保护确认值低 -0.002537，比最佳观测低 -0.003904，且 LogLoss 明显变差。

结论：去掉历史行为侧输入会伤害排序质量并恶化 LogLoss，保护族保持 `use_behavior_side: true`。

### 阶段 15：PSRG 消融

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_no_psrg` 测试 GAUC 0.647920，比保护确认值低 -0.004729，比最佳观测低 -0.006096。

结论：去掉 PSRG 造成较大排序质量下降，保护族保持 `use_psrg: true`。

### 阶段 16：PCRG 消融

状态：相对当前保护族完整实验后不确定。

`sidebias_dinatt_click_only_no_pcrg` 测试 GAUC 0.653015，比保护确认值 0.652649 高 +0.000366，但比最佳观测 0.654016 低 -0.001001，且 LogLoss 变差到 0.601045。

结论：去掉 PCRG 可作为确认候选，但因为增益小、未超过最佳观测且校准回退，不推广，也不作为可叠加 keep。

### 阶段 17：Category 特征消融

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_no_category` 测试 GAUC 0.646873，比保护确认值低 -0.005776，比最佳观测低 -0.007143。

结论：去掉类别特征造成近期消融中最大的下降之一，保护族保持 `use_category: true`。

### 阶段 18：DIN side-bias 消费消融

状态：相对当前保护族完整实验后拒绝。

`sidebias_dinatt_click_only_no_din_side_bias` 测试 GAUC 0.650953，比保护确认值低 -0.001696，比最佳观测低 -0.003063。

结论：该消融保留 `use_side_attention_bias: true`，仅禁用 `DINStyleTargetAttention` 对生成 side-bias 的消费。结果支持在 side-attention-bias 分支启用时保持 `use_side_bias: true`。

### 阶段 19：Video statistic dense 特征实验

状态：相对当前保护族完整实验后候选保留。

`sidebias_dinatt_click_only_video_stat` 测试 GAUC 0.656082，比保护 `DIN + click_only` 确认值 0.652649 高 +0.003433，比最佳观测 `DIN + click_only` 0.654016 高 +0.002066。测试 LogLoss 为 0.595241，只比保护确认差 +0.000641。

该实验让 `use_video_stat` 成为有效单变量开关：通过现有 dense 投影路径输入 50 个 log-scaled 视频统计 dense 特征，同时保持 `use_dense_features: false` 和 `use_dense_history_only: false`。

Seed 确认：

- `sidebias_dinatt_click_only_video_stat_confirm_seed2026` 测试 GAUC 0.655718，比保护确认值 0.652649 高 +0.003069，只比 seed-2025 video-stat 低 -0.000364。
- Valid GAUC 稳定：0.670370 vs 0.670363。
- 测试 AUC 提升到 0.741851。
- 测试 LogLoss 回退到 0.600709，后续组合实验需关注。

结论：按 GAUC 确认保留。将 video-stat dense 特征提升为下一代保护族候选，但继续叠加前需要显式考虑 LogLoss tradeoff。

### 阶段 20：Caption SVD dense 特征实验

状态：相对当前 video-stat 候选完整实验后拒绝。

`sidebias_dinatt_click_only_video_stat_caption` 最佳验证轮次 4，验证 AUC 0.757339，验证 GAUC 0.669494，验证 LogLoss 0.582670。

测试指标：AUC 0.740404，GAUC 0.652855，LogLoss 0.599250。

相比已确认 video-stat，caption 相对 seed-2026 video-stat 确认值 0.655718 降低测试 GAUC 0.002863，相对 seed-2025 video-stat 0.656082 降低 0.003227。它仅比更早的 `DIN + click_only` 确认值 0.652649 高 +0.000206，不支持叠加到当前候选。

结论：拒绝作为保护族新增组件。保持 `use_caption` 默认关闭，除非重设计并独立复测。

### 阶段 21：Video-stat 候选上的 PCRG 消融

状态：相对当前已确认 video-stat 候选 seed 确认后不确定，不推广。

`sidebias_dinatt_click_only_video_stat_no_pcrg` 最佳验证轮次 3，验证 AUC 0.756581，验证 GAUC 0.671029，验证 LogLoss 0.579429。

测试指标：AUC 0.740559，GAUC 0.656795，LogLoss 0.594320。

相比 seed-2025 video-stat，禁用 PCRG 测试 GAUC 提升 0.000713，测试 LogLoss 改善 0.000921。相比 seed-2026 video-stat 确认，测试 GAUC 提升 0.001077，LogLoss 改善 0.006389，但测试 AUC 低 0.001292。

Seed 确认：

- `sidebias_dinatt_click_only_video_stat_no_pcrg_confirm_seed2026` 最佳验证轮次 5，验证 AUC 0.759574，验证 GAUC 0.670878，验证 LogLoss 0.581021。
- 测试指标：AUC 0.744102，GAUC 0.655680，LogLoss 0.598598。
- 相比 seed-2025 no-PCRG，确认测试 GAUC 下降 0.001115。
- 相比 seed-2026 video-stat 确认，no-PCRG 确认测试 GAUC 低 0.000038，虽然 AUC 和 LogLoss 改善。

结论：不把 no-PCRG 推广为已确认保护候选。除非未来重设计提供更强 GAUC 证据，否则已确认 video-stat 家族保持 `use_pcrg: true`。

### 阶段 22：Video-stat 候选上的 EMA

状态：相对当前已确认 video-stat 候选 seed 确认后已确认保留并推广。

`sidebias_dinatt_click_only_video_stat_ema` 仅改变训练/评估权重策略：启用 `use_ema: true`，`ema_decay: 0.995`，`ema_warmup_steps: 300`。

最佳验证轮次 4，验证 AUC 0.760282，验证 GAUC 0.671989，验证 LogLoss 0.577461。

测试指标：AUC 0.744170，GAUC 0.658104，LogLoss 0.594204。

相比 seed-2025 video-stat，EMA 测试 GAUC 提升 0.002022，测试 LogLoss 改善 0.001037。相比 seed-2026 video-stat 确认，EMA 测试 GAUC 提升 0.002386，测试 AUC 提升 0.002319，测试 LogLoss 改善 0.006505。

Seed 确认：

- `sidebias_dinatt_click_only_video_stat_ema_confirm_seed2026` 最佳验证轮次 4，验证 AUC 0.759424，验证 GAUC 0.672000，验证 LogLoss 0.578526。
- 测试指标：AUC 0.744078，GAUC 0.658566，LogLoss 0.594494。
- 相比 seed-2025 EMA，确认测试 GAUC 提升 0.000461。
- 相比 seed-2026 不含 EMA 的 video-stat，确认测试 GAUC 提升 0.002848。

结论：将 video-stat + EMA 推广为当前已确认保护族候选。

### 阶段 23：Video-stat EMA 候选上的 Auxiliary rank loader

状态：相对当前已确认 video-stat EMA 候选完整实验后不确定，不推广。

`sidebias_dinatt_click_only_video_stat_ema_auxrank` 仅改变训练目标调度：启用 `use_aux_rank_loader: true`，`aux_rank_loss_weight: 0.01`，`aux_rank_every_n_steps: 8`，`pairwise_loss_mode: all_pairs`。

最佳验证轮次 3，验证 AUC 0.758734，验证 GAUC 0.672425，验证 LogLoss 0.577514。

测试指标：AUC 0.743504，GAUC 0.658159，LogLoss 0.593323。

相比 seed-2025 EMA，辅助排序测试 GAUC 仅提升 0.000055，LogLoss 改善 0.000881，测试 AUC 下降 0.000666。相比 seed-2026 EMA 确认，测试 GAUC 低 0.000407，测试 AUC 低 0.000574，LogLoss 改善 0.001171。

结论：不在已确认 video-stat EMA 候选上推广或叠加辅助排序。除非重新设计的用户内排序目标给出更强跨 seed GAUC 证据，否则保持不确定。

### 阶段 24：Video-stat EMA 候选上的 zero-initialized side-attention bias

状态：相对当前已确认 video-stat EMA 候选完整实验后不确定，不推广。

`sidebias_dinatt_click_only_video_stat_ema_zero_init_bias` 仅改变 side-attention-bias 初始化：设置 `zero_init_side_attention_bias: true`，其余 EMA、video-stat dense、PCRG、PSRG、DIN 风格目标注意力保持不变。

最佳验证轮次 4，验证 AUC 0.760187，验证 GAUC 0.672747，验证 LogLoss 0.577564。

测试指标：AUC 0.744292，GAUC 0.657954，LogLoss 0.593893。

相比 seed-2025 EMA，zero-init side-attention bias 测试 GAUC 降低 0.000150，同时测试 AUC 提升 0.000122、LogLoss 改善 0.000311。相比 seed-2026 EMA 确认，测试 GAUC 低 0.000612，同时 AUC 提升 0.000214、LogLoss 改善 0.000601。

结论：不推广 video-stat EMA 候选上的 zero-initialized side-attention bias。验证 GAUC 和校准可接受，但主要测试 GAUC 未超过确认基线。

### 阶段 25：Video-stat EMA 候选上的 side-attention match features

状态：seed-2026 确认未通过，不推广。

`sidebias_dinatt_click_only_video_stat_ema_match_features` 仅改变 side-attention-bias 输入：设置 `use_side_attention_match_features: true`，向 side-attention bias MLP 加入 target-history 类别、tab 和时长匹配指示特征。

Seed-2025 结果：

- 最佳验证轮次：3
- 验证 AUC：0.758741
- 验证 GAUC：0.671261
- 验证 LogLoss：0.577314
- 测试 AUC：0.743173
- 测试 GAUC：0.658858
- 测试 LogLoss：0.593194

相比 seed-2025 EMA，match features 测试 GAUC 提升 0.000754，LogLoss 改善 0.001010，但 AUC 下降 0.000997。相比 seed-2026 EMA 确认，测试 GAUC 提升 0.000292，LogLoss 改善 0.001300，但 AUC 下降 0.000905。

Seed-2026 确认结果：

- 配置：`configs/sidebias_dinatt_click_only_video_stat_ema_match_features_confirm_seed2026.yaml`
- 最佳验证轮次：3
- 验证 AUC：0.758491
- 验证 GAUC：0.672785
- 验证 LogLoss：0.577886
- 测试 AUC：0.743220
- 测试 GAUC：0.657493
- 测试 LogLoss：0.593418

确认结论：seed-2026 确认未复现 seed-2025 的测试 GAUC 优势。确认测试 GAUC 比 seed-2025 match_features 低 0.001365，也比 `sidebias_dinatt_click_only_video_stat_ema_confirm_seed2026` 的 0.658566 低 0.001073。虽然 LogLoss 仍优于 EMA 确认基线，但主要指标 GAUC 未通过确认。

结论：不推广 `use_side_attention_match_features` 到当前保护族。当前已确认保护族候选保持 `sidebias_dinatt_click_only_video_stat_ema`。

### 阶段 26：Video-stat EMA 候选上的 dense hidden 128

状态：完整实验后拒绝，不推广。

`sidebias_dinatt_click_only_video_stat_ema_dense128` 仅改变 video-stat dense 分支容量：将 `dense_hidden_dim` 从 64 扩大到 128，其余 EMA、video-stat dense、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 3，验证 AUC 0.758837，验证 GAUC 0.672454，验证 LogLoss 0.577065。

测试指标：AUC 0.742641，GAUC 0.657414，LogLoss 0.593419。

相比 seed-2025 EMA，dense128 测试 GAUC 降低 0.000690，测试 AUC 降低 0.001529，LogLoss 改善 0.000785。相比 seed-2026 EMA 确认，测试 GAUC 低 0.001152，测试 AUC 低 0.001437，LogLoss 改善 0.001075。

结论：不推广 `dense_hidden_dim: 128`。扩大 video-stat dense 分支容量改善了 LogLoss，但主要测试 GAUC 和 AUC 均低于已确认 EMA 基线，不能进入保护族或组合实验。

### 阶段 27：Video-stat EMA 候选上的 MBC semantic slices

状态：seed-2026 确认通过，推广为当前已确认保护族候选。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 仅改变 MBC 语义切片残差：设置 `use_mbc_slices: true`，并保持 `mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、`mbc_gate_init: 0.1`。其余 EMA、video-stat dense、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

Seed-2025 结果：最佳验证轮次 3，验证 AUC 0.760391，验证 GAUC 0.672036，验证 LogLoss 0.575273；测试 AUC 0.745756，测试 GAUC 0.660225，测试 LogLoss 0.590654。

Seed-2026 确认结果：配置 `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2026.yaml`，最佳验证轮次 4，验证 AUC 0.761254，验证 GAUC 0.673646，验证 LogLoss 0.575920；测试 AUC 0.745534，测试 GAUC 0.659100，测试 LogLoss 0.592021。

相比 seed-2026 EMA 确认，MBC semantic slices 确认测试 GAUC 提升 0.000534，测试 AUC 提升 0.001456，LogLoss 改善 0.002473。相比 seed-2025 MBC semantic slices，确认测试 GAUC 回落 0.001125，但仍稳定高于旧保护候选。

结论：确认通过，推广 `use_mbc_slices` 到当前保护族。后续单变量实验应基于 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices`，但不得叠加未确认组件。

### 阶段 28：MBC semantic slices 的 gate 初始化 0.2

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate02` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_gate_init` 从 0.1 提高到 0.2。其余 EMA、video-stat dense、MBC semantic slices、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 4，验证 AUC 0.761660，验证 GAUC 0.671367，验证 LogLoss 0.575925。

测试指标：AUC 0.746569，GAUC 0.659271，LogLoss 0.592276。

相比 seed-2025 当前保护候选，gate02 测试 GAUC 降低 0.000954，LogLoss 变差 0.001622，但测试 AUC 提升 0.000813。相比 seed-2026 确认结果，gate02 测试 GAUC 高 0.000171，测试 AUC 高 0.001035，但 LogLoss 变差 0.000255。

结论：不推广 `mbc_gate_init: 0.2`。该调参没有在同 seed 下提升主指标 GAUC，且 LogLoss 变差；后续可尝试更保守的 gate 初始化，但不能把 0.2 纳入保护族或组合实验。

### 阶段 29：MBC semantic slices 的 gate 初始化 0.05

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate005` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_gate_init` 从 0.1 降到 0.05。其余 EMA、video-stat dense、MBC semantic slices、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 3，验证 AUC 0.760387，验证 GAUC 0.671978，验证 LogLoss 0.575269。

测试指标：AUC 0.745785，GAUC 0.659455，LogLoss 0.590722。

相比 seed-2025 当前保护候选，gate005 测试 GAUC 降低 0.000770，测试 AUC 提升 0.000029，LogLoss 变差 0.000068。相比 seed-2026 确认结果，gate005 测试 GAUC 高 0.000355，测试 AUC 高 0.000251，LogLoss 改善 0.001299。

结论：不推广 `mbc_gate_init: 0.05`，也不进入 seed 确认。gate005 优于 gate02，但仍没有在同 seed 下超过当前保护候选；gate 初始化方向暂时停止，下一步转向 MBC 分支容量单变量实验。

### 阶段 30：MBC semantic slices 的 branch_dim 64

状态：完整实验后拒绝。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_branch_dim` 从 128 降到 64。其余 EMA、video-stat dense、MBC semantic slices、`mbc_fusion_dim: 64`、`mbc_gate_init: 0.1`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 4，验证 AUC 0.761129，验证 GAUC 0.672376，验证 LogLoss 0.576527。

测试指标：AUC 0.745126，GAUC 0.657912，LogLoss 0.593888。

相比 seed-2025 当前保护候选，branch64 测试 GAUC 降低 0.002313，测试 AUC 降低 0.000630，LogLoss 变差 0.003234。相比 seed-2026 EMA 确认基线，branch64 测试 GAUC 也低 0.000654。

结论：拒绝 `mbc_branch_dim: 64`。降低 MBC 分支容量会明显损害主指标；下一步可测试相反方向 `mbc_branch_dim: 256`，但仍必须作为独立单变量实验，不得与 gate、aux loss 或 diversity loss 叠加。

### 阶段 31：MBC semantic slices 的 branch_dim 256

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch256` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_branch_dim` 从 128 提高到 256。其余 EMA、video-stat dense、MBC semantic slices、`mbc_fusion_dim: 64`、`mbc_gate_init: 0.1`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 3，验证 AUC 0.760838，验证 GAUC 0.671585，验证 LogLoss 0.574909。

测试指标：AUC 0.745939，GAUC 0.659231，LogLoss 0.590988。

相比 seed-2025 当前保护候选，branch256 测试 GAUC 降低 0.000994，测试 AUC 提升 0.000183，LogLoss 变差 0.000334。相比 seed-2026 MBC 确认结果，branch256 测试 GAUC 高 0.000131，测试 AUC 高 0.000405，LogLoss 改善 0.001033，但仍没有超过同 seed 保护候选。

结论：不推广 `mbc_branch_dim: 256`，也不进入 seed 确认。MBC 分支容量方向中，64 明显退化，256 也未提升主指标；下一步转向尚未测试的 MBC branch auxiliary supervision 单变量实验。

### 阶段 32：MBC semantic slices 的辅助分支监督

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_auxloss` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `use_mbc_aux_loss` 从 false 改为 true，使用训练器默认 `mbc_aux_loss_weight: 0.1`。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、`mbc_gate_init: 0.1`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 3，验证 AUC 0.759911，验证 GAUC 0.672734，验证 LogLoss 0.576098。

测试指标：AUC 0.744818，GAUC 0.658578，LogLoss 0.592351。

相比 seed-2025 当前保护候选，auxloss 测试 GAUC 降低 0.001647，测试 AUC 降低 0.000938，LogLoss 变差 0.001697。相比 seed-2026 MBC 确认结果，auxloss 测试 GAUC 低 0.000522，测试 AUC 低 0.000716，LogLoss 变差 0.000330。

结论：不推广 `use_mbc_aux_loss: true`，也不进入 seed 确认。辅助分支监督虽然在训练日志中生效，但没有带来测试主指标收益；下一步转向 MBC fusion bottleneck 容量的单变量实验。

### 阶段 33：MBC semantic slices 的 fusion_dim 128

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion128` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_fusion_dim` 从 64 提高到 128。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_gate_init: 0.1`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 4，验证 AUC 0.761672，验证 GAUC 0.672226，验证 LogLoss 0.576092。

测试指标：AUC 0.746084，GAUC 0.658569，LogLoss 0.593069。

相比 seed-2025 当前保护候选，fusion128 测试 GAUC 降低 0.001656，测试 AUC 提高 0.000328，LogLoss 变差 0.002415。相比 seed-2026 MBC 确认结果，fusion128 测试 GAUC 低 0.000531，测试 AUC 高 0.000550，LogLoss 变差 0.001048。

结论：不推广 `mbc_fusion_dim: 128`，也不进入 seed 确认。扩大 fusion bottleneck 虽然略提高测试 AUC，但损伤主指标 GAUC 和 LogLoss；下一步尝试相反方向的 `mbc_fusion_dim: 32` 单变量实验。

### 阶段 34：MBC semantic slices 的 fusion_dim 32

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_fusion_dim` 从 64 降到 32。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_gate_init: 0.1`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 4，验证 AUC 0.760805，验证 GAUC 0.672489，验证 LogLoss 0.576542。

测试指标：AUC 0.745559，GAUC 0.658648，LogLoss 0.593635。

相比 seed-2025 当前保护候选，fusion32 测试 GAUC 降低 0.001577，测试 AUC 降低 0.000197，LogLoss 变差 0.002981。相比 seed-2026 MBC 确认结果，fusion32 测试 GAUC 低 0.000452，测试 AUC 高 0.000025，LogLoss 变差 0.001614。

结论：不推广 `mbc_fusion_dim: 32`，也不进入 seed 确认。fusion bottleneck 扩大到 128 或缩小到 32 都没有带来主指标收益；下一步回到 gate 初始化，在 `0.1` 与已失败的 `0.2` 之间尝试更保守的 `mbc_gate_init: 0.15` 单变量实验。

### 阶段 35：MBC semantic slices 的 gate_init 0.15

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_gate_init` 从 0.1 提高到 0.15。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 4，验证 AUC 0.761807，验证 GAUC 0.671644，验证 LogLoss 0.575751。

测试指标：AUC 0.746850，GAUC 0.658795，LogLoss 0.591825。

相比 seed-2025 当前保护候选，gate015 测试 GAUC 降低 0.001430，测试 AUC 提高 0.001094，LogLoss 变差 0.001171。相比 seed-2026 MBC 确认结果，gate015 测试 GAUC 低 0.000305，测试 AUC 高 0.001316，LogLoss 改善 0.000196。

结论：不推广 `mbc_gate_init: 0.15`，也不进入 seed 确认。该设置提升了测试 AUC，但主指标 GAUC 未超过当前确认值；下一步尝试更接近基线的 `mbc_gate_init: 0.12`，观察能否保留 AUC 改善同时恢复 GAUC。

### 阶段 36：MBC semantic slices 的 gate_init 0.12

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_gate_init` 从 0.1 提高到 0.12。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 3，验证 AUC 0.760379，验证 GAUC 0.671161，验证 LogLoss 0.575321。

测试指标：AUC 0.745757，GAUC 0.659435，LogLoss 0.590688。

相比 seed-2025 当前保护候选，gate012 测试 GAUC 降低 0.000790，测试 AUC 提高 0.000001，LogLoss 变差 0.000034。相比 seed-2026 MBC 确认结果，gate012 测试 GAUC 高 0.000335，测试 AUC 高 0.000223，LogLoss 改善 0.001333。

结论：不推广 `mbc_gate_init: 0.12`，也不进入 seed 确认。该结果高于 seed-2026 确认值但未超过同 seed 保护候选，说明轻微增大 gate 仍不如原始 `0.1`；下一步尝试轻微降低 gate 到 `0.08`，测试 0.05 与 0.1 之间是否存在更稳的点。

### 阶段 37：MBC semantic slices 的 gate_init 0.08

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_gate_init` 从 0.1 降低到 0.08。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 3，验证 AUC 0.760360，验证 GAUC 0.671379，验证 LogLoss 0.575457。

测试指标：AUC 0.745761，GAUC 0.659784，LogLoss 0.590957。

相比 seed-2025 当前保护候选，gate008 测试 GAUC 降低 0.000441，测试 AUC 基本持平，LogLoss 变差 0.000303。相比 seed-2026 MBC 确认结果，gate008 测试 GAUC 高 0.000684，测试 AUC 高 0.000227，LogLoss 改善 0.001064。

结论：不推广 `mbc_gate_init: 0.08`，也不进入 seed 确认。该设置比 gate012 更接近同 seed 保护候选，但仍未超过 `mbc_gate_init: 0.1`；下一步尝试更贴近原始 gate 的 `mbc_gate_init: 0.09`，验证轻微降低 gate 是否能进一步贴近或超过保护候选。

### 阶段 38：MBC semantic slices 的 gate_init 0.09

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `mbc_gate_init` 从 0.1 降低到 0.09。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 3，验证 AUC 0.760413，验证 GAUC 0.671844，验证 LogLoss 0.575353。

测试指标：AUC 0.745790，GAUC 0.659383，LogLoss 0.590782。

相比 seed-2025 当前保护候选，gate009 测试 GAUC 降低 0.000842，测试 AUC 提高 0.000034，LogLoss 变差 0.000128。相比 seed-2026 MBC 确认结果，gate009 测试 GAUC 高 0.000283，测试 AUC 高 0.000256，LogLoss 改善 0.001239。

结论：不推广 `mbc_gate_init: 0.09`，也不进入 seed 确认。该设置低于 gate008，说明当前 gate 微调未找到超过同 seed 保护候选的设置；下一步结束 gate 细扫，转向 `mbc_branch_dim: 64` 的单变量容量收缩实验。

### 阶段 39：MBC semantic slices 的 DIN attention hidden 256

状态：完整实验后拒绝。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `attn_hidden_dim` 从 128 提高到 256。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、`mbc_gate_init: 0.1`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 4，验证 AUC 0.761234，验证 GAUC 0.671796，验证 LogLoss 0.575987。

测试指标：AUC 0.745455，GAUC 0.657895，LogLoss 0.592844。

相比 seed-2025 当前保护候选，attn256 测试 GAUC 降低 0.002330，测试 AUC 降低 0.000301，LogLoss 变差 0.002190。相比 seed-2026 MBC 确认结果，attn256 测试 GAUC 低 0.001205，测试 AUC 低 0.000079，LogLoss 变差 0.000823。

结论：拒绝 `attn_hidden_dim: 256`。扩大 DIN attention MLP 容量提升了部分验证指标，但测试主指标明显退化；下一步若继续检查注意力容量，应只做 `attn_hidden_dim: 64` 的单变量缩容实验，不与任何已拒绝 MBC 调参叠加。

### 阶段 40：MBC semantic slices 的 DIN attention hidden 64

状态：完整实验后不推广。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `attn_hidden_dim` 从 128 降低到 64。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、`mbc_gate_init: 0.1`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 4，验证 AUC 0.761276，验证 GAUC 0.672118，验证 LogLoss 0.575886。

测试指标：AUC 0.745364，GAUC 0.659065，LogLoss 0.592487。

相比 seed-2025 当前保护候选，attn64 测试 GAUC 降低 0.001160，测试 AUC 降低 0.000392，LogLoss 变差 0.001833。相比 seed-2026 MBC 确认结果，attn64 测试 GAUC 低 0.000035，测试 AUC 低 0.000170，LogLoss 变差 0.000466。

结论：不推广 `attn_hidden_dim: 64`，也不进入 seed 确认。该设置明显优于 attn256，但仍未超过当前保护候选和确认基线；DIN attention hidden 容量方向暂时停止，下一步转向 side-attention bias 强度的单变量实验。

### 阶段 41：MBC semantic slices 的 side-attention bias scale 0.05

状态：完整实验后拒绝；配置级单变量调参阶段停止。

`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005` 以当前已确认保护族 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 为基线，仅将 `side_attention_bias_scale` 从 0.1 降低到 0.05。其余 EMA、video-stat dense、MBC semantic slices、`mbc_branch_dim: 128`、`mbc_fusion_dim: 64`、`mbc_gate_init: 0.1`、`attn_hidden_dim: 128`、PCRG、PSRG、DIN 风格目标注意力和 side-attention bias 设置保持不变。

最佳验证轮次 5，验证 AUC 0.761726，验证 GAUC 0.671685，验证 LogLoss 0.577559。

测试指标：AUC 0.746158，GAUC 0.656681，LogLoss 0.594088。

相比 seed-2025 当前保护候选，sideattnscale005 测试 GAUC 降低 0.003544，测试 AUC 提高 0.000402，LogLoss 变差 0.003434。相比 seed-2026 MBC 确认结果，sideattnscale005 测试 GAUC 低 0.002419，测试 AUC 高 0.000624，LogLoss 变差 0.002067。

结论：拒绝 `side_attention_bias_scale: 0.05`。降低 side-attention bias 强度显著损害 GAUC 和 LogLoss，仅 AUC 略高不构成推广理由。本轮从 MBC gate、branch/fusion 容量、aux loss、DIN attention hidden 到 side-attention scale 的配置级单变量调参均未产生稳定实质收益，因此停止继续调参，后续应转向特征侧、训练目标或结构级重新设计。

## 实验纪律

每个阶段必须作为独立单变量实验评估。不要在单个模型中同时合并多个新想法；只有当某个组件已经稳定证明收益后，才允许进入组合实验。未来实验应与当前已确认保护族候选比较，而不是旧的 dot-product SideBias 基线。

## 融合规则

一个改动只有满足以下条件，才可以进入组合实验：

- 验证 GAUC 至少提升 0.001；
- 测试 GAUC 不下降；
- AUC 和 LogLoss 不出现实质性崩塌；
- 最好跨 seed 方向一致。

如果 GAUC 只提升 0.0002 到 0.0005，应标记为不确定，而不是立即融合。
