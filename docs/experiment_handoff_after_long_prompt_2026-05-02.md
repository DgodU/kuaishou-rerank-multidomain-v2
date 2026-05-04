# KuaiRand 实验交接文档：超长 prompt 之后的工作汇总

生成时间：2026-05-02

用途：给外部大模型快速更新当前项目、实验状态、已做工作、结论和后续建议。

## 1. 项目与当前任务背景

项目路径：

```text
/root/autodl-tmp/kuaishou-rerank-multidomain-shared
```

任务目标是在 KuaiRand CTR/rerank 场景下，围绕 `ads_transformer_side` 模型持续做低风险迭代，核心评价指标为测试集 GAUC，同时参考 AUC 和 LogLoss。

本轮实验从一个已经比较强的保护族候选开始，遵循严格实验纪律：

- 只做单变量实验。
- 每个实验先 debug，确认张量形状、参数量、loss 路径正常，再启动 full run。
- 以测试 GAUC 为主指标，AUC 和 LogLoss 为辅助判断。
- 只有同 seed 下超过当前保护候选，才考虑进入 seed 确认。
- 不叠加未确认或已拒绝组件。
- 不把资源使用统计作为本轮实验文档内容。
- 每个完整实验结束后，同步更新中文文档和 JSON 记录。

## 2. 当前确认保护基线

当前确认保护族为：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices
```

其基于：

- click-only 历史行为。
- video-stat dense 特征。
- EMA。
- PCRG/PSRG/Transformer 主干。
- DIN 风格目标注意力。
- side attention bias。
- MBC semantic slices。

核心配置：

```yaml
model_name: ads_transformer_side
history_mode: click_only
data_dir: data/processed_click_only_video_stat
attention_type: din_mlp
use_side_bias: true
use_side_attention_bias: true
use_video_stat: true
use_ema: true
use_mbc_slices: true
mbc_branch_dim: 128
mbc_fusion_dim: 64
mbc_gate_init: 0.1
use_mbc_aux_loss: false
use_mbc_diversity_loss: false
attn_hidden_dim: 128
side_attention_bias_scale: 0.1
side_attention_bias_hidden_dim: 64
```

保护候选指标：

| 实验 | seed | test AUC | test GAUC | test LogLoss | 结论 |
|---|---:|---:|---:|---:|---|
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` | 2025 | 0.745756 | 0.660225 | 0.590654 | 当前同 seed 保护候选 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2026` | 2026 | 0.745534 | 0.659100 | 0.592021 | seed 确认通过 |

上一代 `video_stat_ema` seed-2026 确认 GAUC 为 `0.658566`。因此 MBC semantic slices 在 seed-2026 上仍有 `+0.000534` GAUC 增益，已推广为当前确认保护族。

## 3. 超长 prompt 之后完成的主要工作

### 3.1 完成并记录 MBC fusion 缩容实验：`fusion32`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32
```

单变量：

```yaml
mbc_fusion_dim: 64 -> 32
```

结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 4 |
| valid GAUC | 0.672489 |
| test AUC | 0.745559 |
| test GAUC | 0.658648 |
| test LogLoss | 0.593635 |

结论：不推广。测试 GAUC 低于 MBC seed-2026 确认值 `0.659100`，也低于同 seed 保护候选 `0.660225`，LogLoss 明显变差。

### 3.2 完成并记录 gate 初始化实验：`gate015`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015
```

单变量：

```yaml
mbc_gate_init: 0.1 -> 0.15
```

结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 4 |
| valid GAUC | 0.671644 |
| test AUC | 0.746850 |
| test GAUC | 0.658795 |
| test LogLoss | 0.591825 |

结论：不推广。AUC 提升，但 GAUC 低于保护候选和确认基线。

### 3.3 完成并记录 gate 初始化实验：`gate012`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012
```

单变量：

```yaml
mbc_gate_init: 0.1 -> 0.12
```

结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 3 |
| valid GAUC | 0.671161 |
| test AUC | 0.745757 |
| test GAUC | 0.659435 |
| test LogLoss | 0.590688 |

结论：不推广。高于 MBC seed-2026 确认值 `0.659100`，但仍低于同 seed 保护候选 `0.660225`，不进入 seed 确认。

### 3.4 完成并记录 gate 初始化实验：`gate008`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008
```

单变量：

```yaml
mbc_gate_init: 0.1 -> 0.08
```

结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 3 |
| valid GAUC | 0.671379 |
| test AUC | 0.745761 |
| test GAUC | 0.659784 |
| test LogLoss | 0.590957 |

结论：不推广。它是 gate 微调方向里相对最接近保护候选的实验，但仍低于同 seed `0.660225`，不进入 seed 确认。

### 3.5 完成并记录 gate 初始化实验：`gate009`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009
```

单变量：

```yaml
mbc_gate_init: 0.1 -> 0.09
```

结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 3 |
| valid AUC | 0.760413 |
| valid GAUC | 0.671844 |
| valid LogLoss | 0.575353 |
| test AUC | 0.745790 |
| test GAUC | 0.659383 |
| test LogLoss | 0.590782 |

结论：不推广。低于 `gate008`，也低于同 seed 保护候选 `0.660225`。虽然高于 MBC seed-2026 确认值 `0.659100`，但同 seed 未达标，不进入 seed 确认。

### 3.6 确认并记录 branch 缩容实验：`branch64`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64
```

单变量：

```yaml
mbc_branch_dim: 128 -> 64
```

结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 4 |
| valid AUC | 0.761129 |
| valid GAUC | 0.672376 |
| valid LogLoss | 0.576527 |
| test AUC | 0.745126 |
| test GAUC | 0.657912 |
| test LogLoss | 0.593888 |

结论：拒绝。测试 GAUC 明显低于当前保护候选 `0.660225`，也低于上一代 EMA seed-2026 确认值 `0.658566`。

### 3.7 完成并记录 DIN attention hidden 扩容实验：`attn256`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256
```

单变量：

```yaml
attn_hidden_dim: 128 -> 256
```

配置文件：

```text
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256.yaml
```

debug 信息：

| 项 | 数值 |
|---|---:|
| 参数量 | 2,496,123 |
| final_input | `(256, 288)` |
| side_bias | `(256, 50)` |
| mbc_vector | `(256, 64)` |
| debug test GAUC | 0.604965 |

full run 结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 4 |
| valid AUC | 0.761234 |
| valid GAUC | 0.671796 |
| valid LogLoss | 0.575987 |
| test AUC | 0.745455 |
| test GAUC | 0.657895 |
| test LogLoss | 0.592844 |

结论：拒绝。测试 GAUC 比同 seed 保护候选低 `0.002330`，也低于 MBC seed-2026 确认值。

### 3.8 完成并记录 DIN attention hidden 缩容实验：`attn64`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64
```

单变量：

```yaml
attn_hidden_dim: 128 -> 64
```

配置文件：

```text
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64.yaml
```

debug 信息：

| 项 | 数值 |
|---|---:|
| 参数量 | 2,446,587 |
| final_input | `(256, 288)` |
| side_bias | `(256, 50)` |
| din_attn_feat | `(256, 50, 256)` |
| mbc_vector | `(256, 64)` |
| debug test GAUC | 0.603361 |

full run 结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 4 |
| valid AUC | 0.761276 |
| valid GAUC | 0.672118 |
| valid LogLoss | 0.575886 |
| test AUC | 0.745364 |
| test GAUC | 0.659065 |
| test LogLoss | 0.592487 |

结论：不推广。优于 `attn256`，但仍低于同 seed 保护候选 `0.660225`，且略低于 MBC seed-2026 确认值 `0.659100`。

### 3.9 完成并记录 side-attention bias scale 实验：`sideattnscale005`

实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005
```

单变量：

```yaml
side_attention_bias_scale: 0.1 -> 0.05
```

配置文件：

```text
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005.yaml
```

debug 信息：

| 项 | 数值 |
|---|---:|
| 参数量 | 2,463,099 |
| final_input | `(256, 288)` |
| side_bias | `(256, 50)` |
| attention_score | `(256, 50)` |
| din_attn_feat | `(256, 50, 256)` |
| mbc_vector | `(256, 64)` |
| debug test GAUC | 0.616718 |

full run 结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 5 |
| valid AUC | 0.761726 |
| valid GAUC | 0.671685 |
| valid LogLoss | 0.577559 |
| test AUC | 0.746158 |
| test GAUC | 0.656681 |
| test LogLoss | 0.594088 |

结论：拒绝。测试 AUC 略高，但测试 GAUC 比同 seed 保护候选低 `0.003544`，LogLoss 也明显变差。该实验作为本轮配置级单变量调参的最后一次实验，完成后停止继续调参。

## 4. 本轮相关实验汇总表

| 实验 | 单变量 | test AUC | test GAUC | test LogLoss | 结论 |
|---|---|---:|---:|---:|---|
| `mbc_slices` | 启用 MBC semantic slices | 0.745756 | 0.660225 | 0.590654 | 当前同 seed 保护候选 |
| `mbc_slices_confirm_seed2026` | seed 2026 确认 | 0.745534 | 0.659100 | 0.592021 | 确认通过 |
| `mbc_slices_branch64` | `mbc_branch_dim: 128 -> 64` | 0.745126 | 0.657912 | 0.593888 | 拒绝 |
| `mbc_slices_branch256` | `mbc_branch_dim: 128 -> 256` | 0.745939 | 0.659231 | 0.590988 | 不推广 |
| `mbc_slices_auxloss` | `use_mbc_aux_loss: false -> true` | 0.744818 | 0.658578 | 0.592351 | 不推广 |
| `mbc_slices_fusion128` | `mbc_fusion_dim: 64 -> 128` | 0.746084 | 0.658569 | 0.593069 | 不推广 |
| `mbc_slices_fusion32` | `mbc_fusion_dim: 64 -> 32` | 0.745559 | 0.658648 | 0.593635 | 不推广 |
| `mbc_slices_gate02` | `mbc_gate_init: 0.1 -> 0.2` | 0.746569 | 0.659271 | 0.592276 | 不推广 |
| `mbc_slices_gate005` | `mbc_gate_init: 0.1 -> 0.05` | 0.745785 | 0.659455 | 0.590722 | 不推广 |
| `mbc_slices_gate015` | `mbc_gate_init: 0.1 -> 0.15` | 0.746850 | 0.658795 | 0.591825 | 不推广 |
| `mbc_slices_gate012` | `mbc_gate_init: 0.1 -> 0.12` | 0.745757 | 0.659435 | 0.590688 | 不推广 |
| `mbc_slices_gate008` | `mbc_gate_init: 0.1 -> 0.08` | 0.745761 | 0.659784 | 0.590957 | 不推广 |
| `mbc_slices_gate009` | `mbc_gate_init: 0.1 -> 0.09` | 0.745790 | 0.659383 | 0.590782 | 不推广 |
| `mbc_slices_attn256` | `attn_hidden_dim: 128 -> 256` | 0.745455 | 0.657895 | 0.592844 | 拒绝 |
| `mbc_slices_attn64` | `attn_hidden_dim: 128 -> 64` | 0.745364 | 0.659065 | 0.592487 | 不推广 |
| `mbc_slices_sideattnscale005` | `side_attention_bias_scale: 0.1 -> 0.05` | 0.746158 | 0.656681 | 0.594088 | 拒绝/停止调参 |

## 5. 主要结论

### 5.1 当前保护族仍然是最佳可确认方案

当前保留方案仍是：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices
```

理由：

- seed-2025 单次测试 GAUC 为 `0.660225`。
- seed-2026 确认测试 GAUC 为 `0.659100`。
- 相比上一代 `video_stat_ema_confirm_seed2026` 的 `0.658566`，MBC seed-2026 仍有 `+0.000534` 提升。
- 后续大量单变量调参均未超过同 seed 保护候选。

### 5.2 配置级单变量调参收益已经很低

已尝试方向包括：

- MBC branch 容量。
- MBC fusion bottleneck 容量。
- MBC auxiliary branch supervision。
- MBC gate 初始化。
- DIN attention hidden 容量。
- side-attention bias scale。

这些方向均没有产生稳定、可推广的 GAUC 提升。

最接近保护候选的是：

```text
mbc_slices_gate008: test_gauc = 0.659784
```

但仍低于同 seed 保护候选：

```text
0.660225
```

因此未进入 seed 确认。

### 5.3 单变量调参不应被期待带来 0.5% 相对提升

若以 `0.659100` 为当前确认 GAUC，0.5% 相对提升约为：

```text
0.659100 * 0.005 ≈ 0.003296
```

目标大约是：

```text
0.6624
```

本轮配置级单变量实验的实际波动主要在：

```text
0.6567 ~ 0.6598
```

没有任何实验接近 `0.6624`。因此，如果目标是 0.5% 相对 GAUC 提升，应转向更高信息量或更大改造的方向，而不是继续配置级微调。

## 6. 已更新的项目文件

本轮每次实验完成后，按要求同步更新了以下记录文件：

```text
experiments/results_tracking.md
outputs/model_comparison.md
outputs/model_comparison.json
docs/optimization_roadmap.md
docs/experiment_model_change_summary_2026-05-01.md
CHANGELOG.md
README.md
```

新增或确认使用的配置文件包括：

```text
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32.yaml
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015.yaml
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012.yaml
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008.yaml
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009.yaml
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64.yaml
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256.yaml
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64.yaml
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005.yaml
```

关键日志文件包括：

```text
logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009_full.log
logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64_full.log
logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256_full.log
logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64_full.log
logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005_full.log
```

关键测试指标 JSON 文件包括：

```text
outputs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009_test_metrics.json
outputs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64_test_metrics.json
outputs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256_test_metrics.json
outputs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64_test_metrics.json
outputs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005_test_metrics.json
```

## 7. 当前代码与模型相关信息

核心模型文件：

```text
src/models/ads_transformer_side_mbc.py
```

该文件中的 MBC 相关配置包括：

```python
self.use_mbc_aux_loss = bool(config.get("use_mbc_aux_loss", False))
self.use_mbc_diversity_loss = bool(config.get("use_mbc_diversity_loss", False))
self.mbc_branch_dim = int(config.get("mbc_branch_dim", 128))
self.mbc_fusion_dim = int(config.get("mbc_fusion_dim", 64))
mbc_gate_init = float(config.get("mbc_gate_init", 0.1))
```

训练损失文件：

```text
src/training/trainer.py
```

其中：

- `use_mbc_aux_loss` 已确认能正常生效，但实验不提升。
- `use_mbc_diversity_loss` 当前不适合直接打开，因为 trainer 期望 `branch_vectors` 中存在 `efgc`、`cross`、`deep` 三类向量，而当前 semantic MBC 路径并不按该结构输出；若要使用 diversity loss，需要先做代码级适配，不应作为纯配置实验硬开。

## 8. 当前状态

当前没有需要继续的配置级调参实验。

最后一次实验：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005
```

已完成并记录为拒绝。

当前推荐保留模型仍然是：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices
```

当前推荐保护指标：

```text
seed-2025 test_gauc = 0.660225
seed-2026 test_gauc = 0.659100
```

配置级单变量调参阶段已经停止，不应继续自动启动新的微调实验。

## 9. 给外部大模型的建议问题

如果外部大模型要继续提供方案，建议它不要继续建议简单扫以下变量：

- `mbc_gate_init`
- `mbc_branch_dim`
- `mbc_fusion_dim`
- `use_mbc_aux_loss`
- `attn_hidden_dim`
- `side_attention_bias_scale`

这些方向已经被充分探索，未产生可推广收益。

更值得考虑的方向是：

### 9.1 特征侧增强

可能方向：

- 更细粒度用户历史统计特征。
- 用户长期/短期兴趣分离。
- 视频侧多窗口统计。
- author/creator 维度统计。
- 曝光频次、最近行为强度、时间衰减特征。
- item/user 交互统计。

这类方向可能比继续调模型维度更有机会带来实质 GAUC 增益。

### 9.2 训练目标重设计

可能方向：

- 更贴近 GAUC 的 user-level surrogate loss。
- calibration-aware loss。
- 更稳的 pairwise/listwise objective。
- hard negative mining 重新设计，但必须避免之前 auxrank 类实验中的 LogLoss 恶化问题。

### 9.3 MBC 结构代码级改造

可能方向：

- 让 MBC branch vectors 明确输出可用于 diversity loss 的语义向量。
- user-conditioned gate，而不是固定 gate init 微调。
- side-attention bias 与 MBC 分支解耦或共享机制重设计。
- 更明确的 history semantic slicing 方式。

这些属于代码级方案，需要重新设计、实现和 debug，不应视为纯配置调参。

### 9.4 多 seed 稳定性策略

当前 seed-2025 和 seed-2026 之间本身存在约 `0.001125` 的 GAUC 差异：

```text
0.660225 - 0.659100 = 0.001125
```

因此任何小于 `0.001` 的单次提升都可能是 seed 波动。后续若出现候选提升，应尽早做多 seed 均值确认，而不是只看单次 seed-2025。

## 10. 最终结论

本轮超长 prompt 之后的实验目标已经完成：

- 所有完成实验都已提取指标。
- 所有结果都已同步到中文文档和 JSON。
- 当前保护候选明确。
- 未推广实验均有结论和原因。
- 配置级单变量调参已经停止。

当前最重要的判断是：

```text
继续配置级单变量调参的边际收益已经很低；若目标是 0.5% 相对 GAUC 提升，应转向特征、训练目标或结构级改造。
```

## 11. 2026-05-02 晚间代码级扩展实现记录

本次根据开发 prompt 转向代码级扩展，没有启动任何 full run。核心原则仍是保护当前确认模型、单变量、默认关闭高风险功能、先 debug。

已实现并 debug 通过的能力：

- `random auxiliary BCE`：支持 `use_random_aux`、`use_random_head`、独立 random loader、separate random head、`train_random_aux_loss` 日志。
- `PAL / position-bias tower`：新增 `src/models/position_bias.py`，支持 context-only bias tower；当前原始日志未发现显式 position/rank 字段，预处理摘要会记录提示。
- `rank/calibration split`：新增 `src/models/calibration_head.py`，支持 `ranking_logit`、`calibration_logit`、`final_logit` 和两套评估指标。
- `history-only dense features`：预处理生成 37 维 `history_dense_features`，模型侧 LayerNorm + Linear + GELU + Dropout 后拼入 final input。
- `author features`：检测到 `video_features_basic_pure.csv` 中存在 `author_id`，预处理输出 `target_author_id` / `hist_author_id`，模型侧支持 target author embedding 与 3 维 history match features。
- `long-short interest`：支持短期窗口 gated_sum，与原长兴趣分支融合，并在 debug 中打印 `short_interest`、`long_interest`、`long_short_gate`。
- `PCRG token` 与 `TransformerFusion`：已有模块路径已接入并 debug 通过，仍作为实验配置单独开启。
- `semantic target/match`：已支持预处理零值占位列、Dataset 读取、模型投影拼接和 debug shape；真实语义 embedding 仍需离线生成后 join。
- `user-conditioned MBC gate`：已用样本级 gate 替代固定 scalar gate，可在 debug 中看到 `mbc_gate=(B,)`。
- `semantic feature_maps metadata`：预处理已写出 `semantic_cols`、`semantic_dim`、`semantic_match_cols`、`semantic_match_dim`，模型可从 `feature_maps.pkl` 读取维度。

新增配置文件：

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

Debug 验收：

- `preprocess debias_protocol_baseline --debug` 通过，输出 `data/processed_click_only_video_stat_debias/`，各 split 最多 5000 行。
- `train debias_protocol_baseline --debug` 通过。
- `train random_aux --debug` 通过，打印 `random_logit` 与 `random_aux_loss`。
- `train pal --debug` 通过，输出 relevance / observed 两套指标。
- `train pal_random_aux --debug` 通过，同时打印 PAL 多 logit 与 random auxiliary loss。
- `train rank_calib_split --debug` 通过，输出 ranking / final 两套指标。
- `preprocess history_dense --debug` 与 `train history_dense --debug` 通过，`history_dense_features=(256,37)`。
- 额外 debug 通过：`author`、`long_short`、`pcrg_token`、`pcrg_token_tfusion`、`semantic_target`、`semantic_match`、`dynamic_mbc_gate`。
- 当前保护配置 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices.yaml --debug` 在最新代码下仍可运行，debug 输出使用 `_debug` checkpoint/metrics 后缀。
- 当前目录未检测到 `.git`，无法用 `git status`/`git diff` 做版本差异摘要；已用 py_compile、YAML/JSON parse、debug run 做一致性验证。

重要风险记录：

- 发现原 `scripts/train.py --debug` 会写入与 full 相同的 checkpoint 和 metrics 文件名。本次已修复为 debug checkpoint/metrics 加 `_debug` 后缀。
- 修复前第一次保护配置 debug 曾写过 `checkpoints/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_best.pt` 和对应 best-AUC checkpoint；full metrics JSON 已恢复为 0.660225 的正式结果。若后续需要严格恢复该 checkpoint 文件本身，请重新跑当前保护模型 full 或从外部备份恢复；本次没有自动 full run。

仍为占位或未完整实现的方向：

- semantic embedding 训练中不会在线调用 Qwen/API；当前 debug 使用预处理零值占位列，`semantic_emb_path` / `semantic_emb_dim` 仅为后续离线 join 保留，真实收益需要后续离线生成 embedding 并 join。
- semantic match features 已接入零值列和模型投影，但未实现真实 cosine/相似度统计。
- `use_author_prior`、`use_random_debiased_prior` 仅保留 flag，未实现统计。
- `calibration_features` 当前为配置记录字段，实际 calibration head 使用 dense/calibration dense tensor。
- `random_head_type` 当前固定为 separate head 路径，配置值用于实验记录。
- IPS/SNIPS 未实现。

推荐下一步：先不要 full run 所有配置。按 prompt 顺序从同协议 baseline 开始，只选择一个方向做 full run，并与同协议 baseline 比较。
