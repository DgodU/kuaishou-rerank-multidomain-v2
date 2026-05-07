# Qwen/LLM 视频语义 SimTier 实验计划

## 为什么不再做普通 long_short

普通 `long_short` 只在已有 ID/category 行为表示上切分近期与长期兴趣，之前已经没有稳定超过 `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` 保护模型。继续围绕同一 ID embedding 路径做小开关，边际收益很低，也容易只改善 LogLoss 或验证集而不改善 test GAUC。

## 为什么 SimTier 比直接拼 target semantic embedding 更有价值

直接把 target semantic embedding 拼到 final MLP，本质上主要学习候选视频自身语义，容易退化成一个 target-side dense slice。CTR/rerank 的排序增益更依赖“候选视频是否匹配该用户过去点击兴趣”，所以本轮重点构造 target semantic 与历史点击视频 semantic 的相似度分布、top-k、recent/long contrast、same category/author similarity 等低维匹配特征。

## 为什么 semantic history matching 对 GAUC 更重要

GAUC 按用户分组衡量排序能力。跨用户的全局热门度、目标视频先验或纯 target 表示，可能提升 AUC/LogLoss，但未必改善同一用户内部候选排序。Semantic history matching 明确以用户历史为参照，建模当前候选与用户近期/长期兴趣的语义贴合度，更符合提升 per-user ranking 的目标。

## 为什么训练时不能在线调用 Qwen/API

训练和评估期间在线调用 Qwen/API 会带来不可复现、吞吐不可控、成本不可控、失败重试污染训练状态等问题，也可能引入随时间变化的外部模型输出。本项目只允许离线生成 `data/semantic/video_semantic_emb.parquet`，训练时只读取固定 embedding matrix。

## 三个实验区别

- `semantic_simtier`：加入 `semantic_target` slice 和 `simtier` slice，重点验证低维语义相似度分层特征。
- `semantic_long_short`：加入 `semantic_target` slice 和 `semantic_interest` slice，使用 target semantic query 分别 attend 近期与长期历史语义兴趣。
- `semantic_simtier_long_short`：同时加入 `semantic_target`、`simtier`、`semantic_interest`，只建议在单独 SimTier 或 semantic long-short 有正向信号后运行。

## 成功标准

- `test_gauc <= 0.660225`：reject。
- `0.660225 < test_gauc < 0.661225`：uncertain。
- `test_gauc >= 0.661225 且 test_logloss <= 0.593`：candidate，进入 seed 确认。
- `test_gauc >= 0.661225 且 test_logloss > 0.593`：calibration_risk，不能直接替换保护模型。

## 推荐运行顺序

1. `semantic_simtier`
2. `semantic_long_short`
3. `semantic_simtier_long_short`，仅当前两个单实验至少一个超过保护模型后再运行。

## Full 结果与最终结论

| 实验 | best_valid_epoch | valid AUC | valid GAUC | valid LogLoss | test AUC | test GAUC | test LogLoss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `semantic_long_short` | 4 | 0.761015 | 0.673342 | 0.576493 | 0.746075 | 0.659765 | 0.592911 | 不推广 |
| `semantic_simtier_long_short` | 3 | 0.760972 | 0.673135 | 0.575097 | 0.745698 | 0.660579 | 0.592307 | 不推广 |

`semantic_simtier_long_short` 相比 `semantic_long_short` 有小幅提升，但仍低于 `latefusion` test GAUC 0.660865、`sem48` seed2025 test GAUC 0.662346、`mbcgate005` seed2025 test GAUC 0.662555 和 `reg_mid` seed2025 test GAUC 0.662360。两个 semantic long-short full 实验均未达到推广标准。

`semantic_simtier_long_short` 首次 full preprocess 曾因 embedding 路径指向缺失的 `data/semantic/video_semantic_emb.parquet` 失败，修正为 `data/semantic/video_semantic_emb_v4_full.pkl` 后完成重跑。最终结论是语义 long-short 链路有效但收益不足，现阶段停止继续扩展语义/LLM sweep。
