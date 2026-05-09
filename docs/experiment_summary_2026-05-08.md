# KuaiRand 实验总结：语义高点公平确认、Protected 更新与 CCSS

<!-- external_reader_context:start -->
## How to Read This Document

This is the main current-state summary for external readers. It compresses the long experiment history into the latest decision: which model is currently recommended, how that decision was made, and which follow-up directions did not replace it.

For navigation:

- Read `README.md` first for setup and commands.
- Read `docs/README.md` for the documentation map.
- Use `experiments/results_tracking.md` only when you need the full per-experiment audit trail.
- Use `outputs/model_comparison.json` for structured metrics.

The key result is `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged` with test AUC/GAUC/LogLoss = `0.754048 / 0.663378 / 0.582084`.
<!-- external_reader_context:end -->


日期：2026-05-08

本文档是推送 GitHub 前的当前状态总结。更细的逐实验记录见 `experiments/results_tracking.md`，结构化结果见 `outputs/model_comparison.json`。

## 当前推荐结论

当前最强 protected candidate：

```text
sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged
```

配置：

```text
configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged.yaml
```

测试指标：

| AUC | GAUC | LogLoss |
|---:|---:|---:|
| 0.754048 | 0.663378 | 0.582084 |

注意：该结果使用 `merge_valid_into_train=true` 和 `train_without_validation=true`，属于 train+valid merged 单 seed protected follow-up。若要做严格稳定性发布，建议继续补 `seed2026/2027/2028` 的 merged-train 确认。

## 四 seed 公平确认

公平确认要求每个候选具备 `2025-2028` 四个 seed，排序规则为：

1. mean test GAUC
2. min test GAUC
3. lower mean LogLoss

| 候选 | AUC mean | GAUC mean | GAUC min | GAUC std | LogLoss mean | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `protected_mbc_slices` | 0.745561 | 0.657891 | 0.654963 | 0.001991 | 0.590395 | 上一代 protected |
| `semantic_simtier_sem48` | 0.747106 | 0.659736 | 0.656649 | 0.002066 | 0.590212 | 排名第二 |
| `semantic_simtier_sem48_mbcgate005` | 0.746799 | 0.659390 | 0.656960 | 0.002177 | 0.589860 | min GAUC/LogLoss 较好，但 mean GAUC 不最高 |
| `semantic_simtier_sem48_slicegate_reg_mid` | 0.747148 | 0.659845 | 0.656514 | 0.002195 | 0.590712 | 公平确认 winner |

公平确认 winner：

```text
semantic_simtier_sem48_slicegate_reg_mid
```

## Protected follow-up

| 实验 | AUC | GAUC | LogLoss | 结论 |
|---|---:|---:|---:|---|
| `semantic_simtier_sem48_slicegate_reg_mid` 四 seed mean | 0.747148 | 0.659845 | 0.590712 | 公平确认 winner |
| `semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged` | 0.754048 | 0.663378 | 0.582084 | 当前最强 protected candidate |
| `semantic_simtier_sem48_slicegate_reg_mid_protected_ccss` | 0.747419 | 0.662450 | 0.590168 | 不替代 no-validation protected |

`protected_train_valid_merged` 相对公平确认 winner 四 seed mean：

- GAUC：`+0.003533`
- LogLoss：`-0.008628`

CCSS 相对 no-validation protected：

- GAUC：`-0.000929`
- LogLoss：`+0.008084`

结论：当前保守版 CCSS 有一定 GAUC 信号，但没有超过 no-validation protected，不作为新 protected。

## 主要工程改动

- `scripts/train.py` 支持：
  - `merge_valid_into_train`
  - `train_without_validation`
- `src/training/trainer.py` 新增：
  - `fit_without_validation`
  - `use_ccss` 训练损失
- CCSS 当前实现为保守版：
  - 对 `dense_features` 中的数值特征做 factual/counterfactual 扰动
  - 加入 pairwise monotonic contrast loss
  - 可选 factual BCE augmentation
- 新增自动后处理脚本：
  - `scripts/summarize_semantic_highpoint_confirm.py`
  - `scripts/report_disk_cleanup_candidates.py`
  - `scripts/run_after_semantic_highpoint_queue.sh`
  - `scripts/run_protected_followup_after_summary.py`
  - `scripts/run_semantic_highpoint_fair_confirm_queue.sh`

## Static MBC 方向结论

Static MBC 替代当前 MBC slices 的三种尝试均未通过：

| 实验 | test AUC | test GAUC | test LogLoss | 结论 |
|---|---:|---:|---:|---|
| `static_mbc_residual` | 0.742560 | 0.656525 | 0.594724 | 拒绝 |
| `static_mbc_main_input` | 0.742727 | 0.657588 | 0.593793 | 拒绝 |
| `static_mbc_main_input_residual` | 0.742400 | 0.655734 | 0.593903 | 拒绝 |

结论：在当前 ADS-Transformer-SideInfo 主线中，Static MBC residual/main-input 替代均低于 MBC slices protected，不继续推广。

## 磁盘清理状态

已生成磁盘清理候选报告：

```text
outputs/disk_cleanup_candidates.md
outputs/disk_cleanup_candidates.json
```

报告只列出候选，不执行删除。当前观察到的大头主要在 `/root/autodl-tmp/data` 的多个 processed/raw 数据目录，以及 shared repo 的 checkpoint。任何删除前都需要人工确认，避免误删可复现实验数据。

## 推送前建议

- 将源码、配置、脚本、文档和 `outputs/model_comparison.json` 一起提交。
- 不建议提交大体积数据、checkpoint、日志和 ignored outputs 报告。
- 若要把 no-validation protected 作为正式稳定模型，建议后续补多 seed 确认。
