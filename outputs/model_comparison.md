# 模型对比汇总

<!-- model_comparison_reader_summary:start -->
## Reader Summary

This file is a compact metrics table for tracked runs. Some late-stage protected follow-up results are summarized here before the legacy table so external readers do not have to infer the current project state from the long run list.

Current headline result:

| Experiment | Test AUC | Test GAUC | Test LogLoss | Status |
|---|---:|---:|---:|---|
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged` | 0.754048 | 0.663378 | 0.582084 | Current protected candidate |
| `semantic_simtier_sem48_slicegate_reg_mid` four-seed mean | 0.747148 | 0.659845 | 0.590712 | Fair-confirmation winner |
| `semantic_simtier_sem48_slicegate_reg_mid_protected_ccss` | 0.747419 | 0.662450 | 0.590168 | Strong, but does not replace no-validation protected |
| `protected_mbc_slices` four-seed mean | 0.745561 | 0.657891 | 0.590395 | Previous protected reference |

Selection rule used for fair confirmation: candidates must have seeds 2025-2028 and are ranked by mean test GAUC, then min test GAUC, then lower mean LogLoss.

The detailed table below is useful for historical comparison, but the current recommendation is the protected candidate above.
<!-- model_comparison_reader_summary:end -->

## Detailed Run Table

| 实验名 | 模型名 | 模式 | 最佳验证轮次 | 最佳验证GAUC | 测试AUC | 测试GAUC | 测试LogLoss | 日志来源 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ads_debug | ads | debug | 7 | 0.626946 | 0.65628 | 0.613512 | 0.646217 | logs/train_ads_debug.log |
| ads_transformer_side_debug | ads_transformer_side | debug | 13 | 0.616737 | 0.711584 | 0.611522 | 0.628776 | logs/train_ads_transformer_side_debug.log |
| ads_full_serial | ads | full | 3 | 0.660602 | 0.737232 | 0.643964 | 0.595815 | logs/nohup_ads_full.log |
| ads_no_psrg_no_pcrg | ads | full | 6 | 0.660902 | 0.739413 | 0.647874 | 0.599534 | logs/nohup_ads_no_psrg_no_pcrg_full_train.log |
| ads_paper_psrg_pcrg | ads | full | 3 | 0.659772 | 0.737912 | 0.645720 | 0.599150 | logs/nohup_ads_paper_psrg_pcrg_full_train.log |
| ads_transformer_side_full_serial | ads_transformer_side | full | 6 | 0.639355 | 0.746008 | 0.623646 | 0.596008 | logs/nohup_ads_transformer_side_full_serial.log |
| ads_transformer_side_mbc_full_serial | ads_transformer_side_mbc | full | 4 | 0.635036 | 0.75595 | 0.620099 | 0.582299 | logs/nohup_ads_transformer_side_mbc_full_serial.log |
| ads_transformer_side_sidebias | ads_transformer_side | full | 3 | 0.660045 | 0.739911 | 0.645878 | 0.597871 | logs/nohup_ads_transformer_side_sidebias_full.log |
| sidebias_history_click_only | ads_transformer_side | full | 4 | 0.661993 | 0.736680 | 0.646484 | 0.603616 | logs/nohup_sidebias_history_click_only_full.log |
| sidebias_history_click_or_long_view | ads_transformer_side | full | 2 | 0.660002 | 0.731172 | 0.641942 | 0.604102 | logs/nohup_sidebias_history_click_or_long_view_full.log |
| sidebias_dinatt | ads_transformer_side | full | 4 | 0.664079 | 0.742062 | 0.648728 | 0.597294 | logs/nohup_sidebias_dinatt_full.log |
| sidebias_dinatt_confirm_seed2026 | ads_transformer_side | full | 4 | 0.665345 | 0.741810 | 0.650264 | 0.594934 | logs/nohup_sidebias_dinatt_confirm_seed2026_full.log |
| sidebias_dinatt_history_click_only | ads_transformer_side | full | 4 | 0.666839 | 0.740477 | 0.654016 | 0.601096 | logs/nohup_sidebias_dinatt_history_click_only_full.log |
| sidebias_dinatt_history_click_only_confirm_seed2026 | ads_transformer_side | full | 4 | 0.665631 | 0.740356 | 0.652649 | 0.594600 | logs/nohup_sidebias_dinatt_history_click_only_confirm_seed2026_full.log |
| sidebias_dinatt_click_only_video_stat | ads_transformer_side | full | 3 | 0.670363 | 0.740214 | 0.656082 | 0.595241 | logs/nohup_sidebias_dinatt_click_only_video_stat_full.log |
| sidebias_dinatt_click_only_video_stat_confirm_seed2026 | ads_transformer_side | full | 3 | 0.670370 | 0.741851 | 0.655718 | 0.600709 | logs/nohup_sidebias_dinatt_click_only_video_stat_confirm_seed2026_full.log |
| sidebias_dinatt_click_only_video_stat_caption | ads_transformer_side | full | 4 | 0.669494 | 0.740404 | 0.652855 | 0.599250 | logs/nohup_sidebias_dinatt_click_only_video_stat_caption_full.log |
| sidebias_dinatt_click_only_video_stat_no_pcrg | ads_transformer_side | full | 3 | 0.671029 | 0.740559 | 0.656795 | 0.594320 | logs/nohup_sidebias_dinatt_click_only_video_stat_no_pcrg_full.log |
| sidebias_dinatt_click_only_video_stat_no_pcrg_confirm_seed2026 | ads_transformer_side | full | 5 | 0.670878 | 0.744102 | 0.655680 | 0.598598 | logs/nohup_sidebias_dinatt_click_only_video_stat_no_pcrg_confirm_seed2026_full.log |
| sidebias_dinatt_click_only_video_stat_ema | ads_transformer_side | full | 4 | 0.671989 | 0.744170 | 0.658104 | 0.594204 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_full.log |
| sidebias_dinatt_click_only_video_stat_ema_confirm_seed2026 | ads_transformer_side | full | 4 | 0.672000 | 0.744078 | 0.658566 | 0.594494 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_confirm_seed2026_full.log |
| sidebias_dinatt_click_only_video_stat_ema_auxrank | ads_transformer_side | full | 3 | 0.672425 | 0.743504 | 0.658159 | 0.593323 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_auxrank_full.log |
| sidebias_dinatt_click_only_video_stat_ema_zero_init_bias | ads_transformer_side | full | 4 | 0.672747 | 0.744292 | 0.657954 | 0.593893 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_zero_init_bias_full.log |
| sidebias_dinatt_click_only_video_stat_ema_match_features | ads_transformer_side | full | 3 | 0.671261 | 0.743173 | 0.658858 | 0.593194 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_match_features_full.log |
| sidebias_dinatt_click_only_video_stat_ema_match_features_confirm_seed2026 | ads_transformer_side | full | 3 | 0.672785 | 0.743220 | 0.657493 | 0.593418 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_match_features_confirm_seed2026_full.log |
| sidebias_dinatt_click_only_video_stat_ema_dense128 | ads_transformer_side | full | 3 | 0.672454 | 0.742641 | 0.657414 | 0.593419 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_dense128_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices | ads_transformer_side | full | 3 | 0.672036 | 0.745756 | 0.660225 | 0.590654 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_no_psrg_no_pcrg | ads_transformer_side | full | 5 | 0.672363 | 0.746665 | 0.659488 | 0.593512 | logs/nohup_protected_mbc_slices_no_psrg_no_pcrg_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_no_psrg_no_pcrg_confirm_seed2026 | ads_transformer_side | full | 4 | 0.673532 | 0.746648 | 0.659494 | 0.590837 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_no_psrg_no_pcrg_confirm_seed2026_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_no_psrg_no_pcrg_confirm_seed2027 | ads_transformer_side | full | 4 | 0.672892 | 0.746330 | 0.657936 | 0.590910 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_no_psrg_no_pcrg_confirm_seed2027_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_no_psrg_no_pcrg_confirm_seed2028 | ads_transformer_side | full | 3 | 0.673685 | 0.745065 | 0.654995 | 0.589592 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_no_psrg_no_pcrg_confirm_seed2028_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_paper_psrg_pcrg | ads_transformer_side | full | 3 | 0.671236 | 0.744852 | 0.655875 | 0.589997 | logs/nohup_protected_mbc_slices_paper_psrg_pcrg_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2026 | ads_transformer_side | full | 4 | 0.673646 | 0.745534 | 0.659100 | 0.592021 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2026_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author_pcrg_token_token2 | ads_transformer_side | full | 4 | 0.673118 | 0.743049 | 0.657053 | 0.591938 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author_pcrg_token_token2_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author_pcrg_token_dim32 | ads_transformer_side | full | 4 | 0.675857 | 0.743860 | 0.657978 | 0.591525 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author_pcrg_token_dim32_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author_prior | ads_transformer_side | full | 5 | 0.675240 | 0.743827 | 0.656625 | 0.593985 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author_prior_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate02 | ads_transformer_side | full | 4 | 0.671367 | 0.746569 | 0.659271 | 0.592276 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate02_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate005 | ads_transformer_side | full | 3 | 0.671978 | 0.745785 | 0.659455 | 0.590722 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate005_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64 | ads_transformer_side | full | 4 | 0.672376 | 0.745126 | 0.657912 | 0.593888 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch64_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch256 | ads_transformer_side | full | 3 | 0.671585 | 0.745939 | 0.659231 | 0.590988 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_branch256_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_auxloss | ads_transformer_side | full | 3 | 0.672734 | 0.744818 | 0.658578 | 0.592351 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_auxloss_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion128 | ads_transformer_side | full | 4 | 0.672226 | 0.746084 | 0.658569 | 0.593069 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion128_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32 | ads_transformer_side | full | 4 | 0.672489 | 0.745559 | 0.658648 | 0.593635 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_fusion32_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015 | ads_transformer_side | full | 4 | 0.671644 | 0.746850 | 0.658795 | 0.591825 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate015_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012 | ads_transformer_side | full | 3 | 0.671161 | 0.745757 | 0.659435 | 0.590688 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate012_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008 | ads_transformer_side | full | 3 | 0.671379 | 0.745761 | 0.659784 | 0.590957 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008_confirm_seed2026 | ads_transformer_side | full | 4 | 0.673774 | 0.746262 | 0.658206 | 0.591404 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008_confirm_seed2026_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008_confirm_seed2027 | ads_transformer_side | full | 3 | 0.672931 | 0.745996 | 0.656504 | 0.589095 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008_confirm_seed2027_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008_confirm_seed2028 | ads_transformer_side | full | 4 | 0.673366 | 0.745569 | 0.655682 | 0.591146 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate008_confirm_seed2028_full_train.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009 | ads_transformer_side | full | 3 | 0.671844 | 0.745790 | 0.659383 | 0.590782 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_gate009_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256 | ads_transformer_side | full | 4 | 0.671796 | 0.745455 | 0.657895 | 0.592844 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn256_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64 | ads_transformer_side | full | 4 | 0.672118 | 0.745364 | 0.659065 | 0.592487 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_attn64_full.log |
| sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005 | ads_transformer_side | full | 5 | 0.671685 | 0.746158 | 0.656681 | 0.594088 | logs/nohup_sidebias_dinatt_click_only_video_stat_ema_mbc_slices_sideattnscale005_full.log |
| sidebias_dinatt_click_only_pcrg_token | ads_transformer_side | full | 6 | 0.666977 | 0.742101 | 0.652005 | 0.599881 | logs/nohup_sidebias_dinatt_click_only_pcrg_token_full.log |
| sidebias_dinatt_click_only_dense_history_only | ads_transformer_side | full | 4 | 0.666765 | 0.739373 | 0.651980 | 0.597513 | logs/nohup_sidebias_dinatt_click_only_dense_history_only_full.log |
| sidebias_dinatt_click_only_tfusion | ads_transformer_side | full | 5 | 0.663918 | 0.740293 | 0.649408 | 0.604401 | logs/nohup_sidebias_dinatt_click_only_tfusion_full.log |
| sidebias_dinatt_click_only_mbc_slices | ads_transformer_side | full | 4 | 0.663652 | 0.740041 | 0.652958 | 0.597156 | logs/nohup_sidebias_dinatt_click_only_mbc_slices_full.log |
| sidebias_dinatt_click_only_match_features | ads_transformer_side | full | 5 | 0.666178 | 0.741693 | 0.653410 | 0.603938 | logs/nohup_sidebias_dinatt_click_only_match_features_full.log |
| sidebias_dinatt_click_only_interactions | ads_transformer_side | full | 5 | 0.664292 | 0.740526 | 0.649730 | 0.603755 | logs/nohup_sidebias_dinatt_click_only_interactions_full.log |
| sidebias_dinatt_click_only_side_gate | ads_transformer_side | full | 4 | 0.666046 | 0.740426 | 0.651850 | 0.601276 | logs/nohup_sidebias_dinatt_click_only_side_gate_full.log |
| sidebias_dinatt_click_only_zero_init_bias | ads_transformer_side | full | 4 | 0.666135 | 0.740534 | 0.653579 | 0.600785 | logs/nohup_sidebias_dinatt_click_only_zero_init_bias_full.log |
| sidebias_dinatt_click_only_auxrank | ads_transformer_side | full | 4 | 0.667872 | 0.741138 | 0.652887 | 0.601402 | logs/nohup_sidebias_dinatt_click_only_auxrank_full.log |
| sidebias_dinatt_click_only_auxrank_hard | ads_transformer_side | full | 4 | 0.666860 | 0.740841 | 0.653189 | 0.602080 | logs/nohup_sidebias_dinatt_click_only_auxrank_hard_full.log |
| sidebias_dinatt_click_only_ema | ads_transformer_side | full | 4 | 0.667586 | 0.741080 | 0.653318 | 0.597654 | logs/nohup_sidebias_dinatt_click_only_ema_full.log |
| sidebias_dinatt_click_only_target_tag | ads_transformer_side | full | 5 | 0.664267 | 0.741647 | 0.652278 | 0.599354 | logs/nohup_sidebias_dinatt_click_only_target_tag_full.log |
| sidebias_dinatt_click_only_user_profile | ads_transformer_side | full | 4 | 0.664130 | 0.738541 | 0.649580 | 0.597468 | logs/nohup_sidebias_dinatt_click_only_user_profile_full.log |
| sidebias_dinatt_click_only_user_onehot | ads_transformer_side | full | 4 | 0.664557 | 0.740881 | 0.650862 | 0.599247 | logs/nohup_sidebias_dinatt_click_only_user_onehot_full.log |
| sidebias_dinatt_click_only_no_side_attention_bias | ads_transformer_side | full | 5 | 0.665059 | 0.741327 | 0.650352 | 0.600727 | logs/nohup_sidebias_dinatt_click_only_no_side_attention_bias_full.log |
| sidebias_dinatt_click_only_no_din_side_bias | ads_transformer_side | full | 5 | 0.664464 | 0.740166 | 0.650953 | 0.602755 | logs/nohup_sidebias_dinatt_click_only_no_din_side_bias_full.log |
| sidebias_dinatt_click_only_no_behavior_side | ads_transformer_side | full | 5 | 0.663845 | 0.739494 | 0.650112 | 0.603077 | logs/nohup_sidebias_dinatt_click_only_no_behavior_side_full.log |
| sidebias_dinatt_click_only_no_psrg | ads_transformer_side | full | 3 | 0.662635 | 0.736732 | 0.647920 | 0.601005 | logs/nohup_sidebias_dinatt_click_only_no_psrg_full.log |
| sidebias_dinatt_click_only_no_pcrg | ads_transformer_side | full | 5 | 0.665886 | 0.741535 | 0.653015 | 0.601045 | logs/nohup_sidebias_dinatt_click_only_no_pcrg_full.log |
| sidebias_dinatt_click_only_no_category | ads_transformer_side | full | 5 | 0.661802 | 0.738465 | 0.646873 | 0.600461 | logs/nohup_sidebias_dinatt_click_only_no_category_full.log |
| sidebias_time_context | ads_transformer_side | full | 4 | 0.658256 | 0.737989 | 0.644237 | 0.600681 | logs/nohup_sidebias_time_context_full.log |

Historical note for the legacy table: before the late semantic protected follow-up, the best listed full experiment by test GAUC in this older table was `sidebias_dinatt_click_only_video_stat_ema_mbc_slices` with test GAUC `0.660225`. That older MBC-slices line is no longer the headline recommendation; the current protected candidate is summarized at the top of this file.

多 seed 候选对比（seed2025-2028，Test GAUC mean/stdev）：protected MBC slices = 0.657891/0.002299；author = 0.658165/0.001113；author+pcrg_token = 0.658521/0.001190；pcrg_token = 0.657438/0.000295。当前按均值看 author+pcrg_token 最高，但相对 protected 仅 +0.000630，差距小于 protected seed 标准差；pcrg_token 最稳定但均值较低。后续若要推广，应优先做统计检验或更多 seed，而非仅看单次最高。

原始 ADS PSRG/PCRG 消融（seed2025）：`ads_no_psrg_no_pcrg` 测试 GAUC=0.647874，相对原始 `ads_full_serial` 提升 +0.003910，但测试 LogLoss 从 0.595815 退化到 0.599534；`ads_paper_psrg_pcrg` 测试 GAUC=0.645720，相对原始 ADS 提升 +0.001756，但仍低于 no-PSRG-no-PCRG，且 LogLoss 退化到 0.599150。结论：原始 ADS 中关闭 PSRG/PCRG 单次 GAUC 更强，但校准变差；论文输入路径版 PSRG/PCRG 未超过关闭模块版本，暂不推广，建议至少做 seed 确认。

当前 protected MBC slices 上的 PSRG/PCRG 消融（seed2025）：`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_no_psrg_no_pcrg` 测试 GAUC=0.659488，低于同 seed protected baseline 0.660225（-0.000736），且 LogLoss 明显变差到 0.593512；`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_paper_psrg_pcrg` 测试 GAUC=0.655875，明显低于 protected baseline（-0.004350），虽然 LogLoss 略优于 baseline。结论：在当前 protected 版本中，保留原有 PSRG/PCRG 路径的 protected baseline 仍最好，不推广关闭 PSRG/PCRG 或 paper 输入路径版。

author+pcrg_token 稳定化尝试（seed2025）：`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author_pcrg_token_token2` 将 `num_interest_tokens` 从 4 降到 2，测试 GAUC=0.657053，低于原 `author_pcrg_token` 0.660004、单独 `author` 0.658705 和 MBC seed-2026 确认值 0.659100；不推广。

author+pcrg_token 降维尝试（seed2025）：`sidebias_dinatt_click_only_video_stat_ema_mbc_slices_author_pcrg_token_dim32` 将 `pcrg_token_dim` 从 64 降到 32，测试 GAUC=0.657978，高于 token2 但低于原 `author_pcrg_token` 0.660004、单独 `author` 0.658705 和 MBC seed-2026 确认值 0.659100；不推广。

author prior 实现尝试（seed2025）：在 author 特征基础上新增 train-only author CTR/count 先验并拼接 16 维投影，best_epoch=5，验证 GAUC=0.675240，但测试 GAUC=0.656625，低于单独 author、author+pcrg_token 和 MBC seed-2026 确认值；不推广。

稳定性确认队列（seed2025-2028）：`gate008` Test GAUC values=[0.659784, 0.658206, 0.656504, 0.655682]，mean/stdev=0.657544/0.001826，低于 protected mean 0.657891；`no_psrg_no_pcrg` values=[0.659488, 0.659494, 0.657936, 0.654995]，mean/stdev=0.657978/0.002120，与 protected 0.657891/0.002299 基本持平但未超过噪声，且平均 LogLoss 更差。因此 gate008 和 no_psrg_no_pcrg 均不推广，保留当前 protected。

## 语义增强与 semantic long-short 补充结果

| 实验名 | 最佳验证轮次 | 验证AUC | 验证GAUC | 验证LogLoss | 测试AUC | 测试GAUC | 测试LogLoss | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_latefusion` | 4 | 0.762430 | 0.673508 | 0.575052 | 0.747199 | 0.660865 | 0.591680 | 新架构未达标 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_long_short` | 4 | 0.761015 | 0.673342 | 0.576493 | 0.746075 | 0.659765 | 0.592911 | 不推广 |
| `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_long_short` | 3 | 0.760972 | 0.673135 | 0.575097 | 0.745698 | 0.660579 | 0.592307 | 不推广 |

`semantic_simtier_long_short` 相比 `semantic_long_short` 的 test GAUC 从 0.659765 提升到 0.660579，LogLoss 从 0.592911 改善到 0.592307，但仍低于 `latefusion` test GAUC 0.660865、`sem48` seed2025 test GAUC 0.662346、`mbcgate005` seed2025 test GAUC 0.662555 和 `reg_mid` seed2025 test GAUC 0.662360。当前语义增强、semantic long-short 与 late fusion 尝试均未达到推广标准。
