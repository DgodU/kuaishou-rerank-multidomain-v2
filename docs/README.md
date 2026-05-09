# Documentation Guide

This directory contains the human-readable records for the KuaiRand CTR/reranking experiments. If you are new to the project, read the documents in this order.

## Recommended Reading Order

1. **Project entry**: `../README.md`
   - Explains the task, repository structure, data preparation, training commands, and current best result.
2. **Current experiment summary**: `experiment_summary_2026-05-08.md`
   - The best compact summary of the latest protected candidate, fair seed confirmation, and rejected follow-ups.
3. **Experiment roadmap**: `optimization_roadmap.md`
   - Explains how the model family evolved and which directions were kept or rejected.
4. **Detailed experiment log**: `../experiments/results_tracking.md`
   - A long chronological table of individual experiment results. Use it as a reference, not as the first document to read.
5. **Structured result table**: `../outputs/model_comparison.md` and `../outputs/model_comparison.json`
   - Machine-/table-friendly experiment metrics. The Markdown file includes a reader summary and the legacy full table.
6. **Change history**: `../CHANGELOG.md`
   - Engineering and experiment milestones by date.

## Current Status at a Glance

| Item | Value |
|---|---|
| Dataset | KuaiRand-Pure |
| Task | CTR prediction / personalized reranking |
| Main metric | GAUC grouped by `user_id` |
| Current protected candidate | `sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged` |
| Config | `configs/sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_protected_train_valid_merged.yaml` |
| Test AUC / GAUC / LogLoss | `0.754048 / 0.663378 / 0.582084` |
| Fair-confirmation winner | `semantic_simtier_sem48_slicegate_reg_mid` |
| Important caveat | The current protected candidate is a train+valid merged single-seed follow-up; more seeds are recommended for strict stability claims. |

## Document Types

- **Current summary documents** describe the latest recommended state.
- **Historical handoff documents** preserve earlier reasoning and may mention older protected models. They now include warning notes at the top.
- **Tracking tables** are intentionally verbose because they preserve negative results and failed directions.

## What Not to Expect in Git

The repository intentionally does not include local raw data, preprocessed data, checkpoints, logs, or large generated reports. See `.gitignore` and `../README.md` for details.
