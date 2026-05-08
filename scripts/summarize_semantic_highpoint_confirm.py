from __future__ import annotations

import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TRACKING = ROOT / "experiments" / "results_tracking.md"

CANDIDATES = {
    "protected_mbc_slices": {
        2025: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices",
        2026: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2026",
        2027: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2027",
        2028: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_confirm_seed2028",
    },
    "semantic_simtier_sem48": {
        2025: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48",
        2026: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_confirm_seed2026",
        2027: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_confirm_seed2027",
        2028: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_confirm_seed2028",
    },
    "semantic_simtier_sem48_mbcgate005": {
        2025: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_mbcgate005",
        2026: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_mbcgate005_confirm_seed2026",
        2027: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_mbcgate005_confirm_seed2027",
        2028: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_mbcgate005_confirm_seed2028",
    },
    "semantic_simtier_sem48_slicegate_reg_mid": {
        2025: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid",
        2026: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_confirm_seed2026",
        2027: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_confirm_seed2027",
        2028: "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_confirm_seed2028",
    },
}


def load_metrics(run_name: str) -> dict[str, Any] | None:
    path = OUTPUTS / f"{run_name}_test_metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    metrics = data.get("test_metrics", data)
    log_path = ROOT / "logs" / f"ads_transformer_side_{run_name}.log"
    best_valid_epoch = None
    best_valid_gauc = None
    best_valid_auc = None
    best_valid_logloss = None
    if log_path.exists():
        text = log_path.read_text(errors="ignore")
        epoch_matches = re.findall(
            r"Epoch (\d+) .*?valid_auc=([0-9.]+) \| valid_gauc=([0-9.]+) \| valid_logloss=([0-9.]+)",
            text,
        )
        if epoch_matches:
            best = max(epoch_matches, key=lambda x: float(x[2]))
            best_valid_epoch = int(best[0])
            best_valid_auc = float(best[1])
            best_valid_gauc = float(best[2])
            best_valid_logloss = float(best[3])
    return {
        "run_name": run_name,
        "metrics_path": str(path.relative_to(ROOT)),
        "test_auc": float(metrics["auc"]),
        "test_gauc": float(metrics["gauc"]),
        "test_logloss": float(metrics["logloss"]),
        "best_valid_epoch": best_valid_epoch,
        "best_valid_auc": best_valid_auc,
        "best_valid_gauc": best_valid_gauc,
        "best_valid_logloss": best_valid_logloss,
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def build_summary() -> dict[str, Any]:
    candidates = {}
    for name, seed_runs in CANDIDATES.items():
        seed_metrics = {}
        missing = []
        for seed, run_name in seed_runs.items():
            item = load_metrics(run_name)
            if item is None:
                missing.append(seed)
            else:
                seed_metrics[str(seed)] = item
        complete = len(missing) == 0
        row = {
            "complete": complete,
            "missing_seeds": missing,
            "seed_metrics": seed_metrics,
        }
        if complete:
            metrics_list = list(seed_metrics.values())
            row["test_auc"] = summarize_values([x["test_auc"] for x in metrics_list])
            row["test_gauc"] = summarize_values([x["test_gauc"] for x in metrics_list])
            row["test_logloss"] = summarize_values([x["test_logloss"] for x in metrics_list])
        candidates[name] = row
    complete_items = [(name, data) for name, data in candidates.items() if data["complete"]]
    ranking = sorted(
        complete_items,
        key=lambda x: (
            x[1]["test_gauc"]["mean"],
            x[1]["test_gauc"]["min"],
            -x[1]["test_logloss"]["mean"],
        ),
        reverse=True,
    )
    winner = ranking[0][0] if ranking else None
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": "eligible candidates require all seeds 2025-2028; rank by mean test GAUC, then min test GAUC, then lower mean LogLoss",
        "required_seeds": [2025, 2026, 2027, 2028],
        "candidates": candidates,
        "ranking": [name for name, _ in ranking],
        "recommended_protected_model": winner,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("# Semantic High-Point Fair Confirmation Summary")
    lines.append("")
    lines.append(f"Generated at: `{summary['generated_at']}`")
    lines.append("")
    lines.append(f"Selection rule: {summary['selection_rule']}.")
    lines.append("")
    lines.append("| Candidate | Complete | Missing seeds | GAUC mean | GAUC min | GAUC std | AUC mean | LogLoss mean | Recommendation |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---|")
    winner = summary.get("recommended_protected_model")
    for name, data in summary["candidates"].items():
        complete = data["complete"]
        missing = ",".join(str(x) for x in data["missing_seeds"]) if data["missing_seeds"] else "-"
        if complete:
            gauc = data["test_gauc"]
            auc = data["test_auc"]
            logloss = data["test_logloss"]
            rec = "new protected recommendation" if name == winner else "candidate compared"
            lines.append(
                f"| `{name}` | yes | {missing} | {gauc['mean']:.6f} | {gauc['min']:.6f} | {gauc['std']:.6f} | {auc['mean']:.6f} | {logloss['mean']:.6f} | {rec} |"
            )
        else:
            lines.append(f"| `{name}` | no | {missing} | - | - | - | - | - | wait for missing runs |")
    lines.append("")
    if winner:
        lines.append(f"Recommended protected model: `{winner}`.")
    else:
        lines.append("Recommended protected model: unavailable until all required seed metrics exist.")
    lines.append("")
    lines.append("Next planned steps after recommendation: create merged train+valid full-training config for the recommended model, then implement a conservative CCSS variant and compare against that latest protected model.")
    lines.append("")
    return "\n".join(lines)


def update_tracking(markdown: str) -> None:
    start = "<!-- semantic_highpoint_fair_confirm_summary:start -->"
    end = "<!-- semantic_highpoint_fair_confirm_summary:end -->"
    block = f"{start}\n{markdown}\n{end}"
    text = TRACKING.read_text()
    if start in text and end in text:
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        text = pattern.sub(block, text)
    else:
        text = text.replace("\n| 实验名 |", f"\n{block}\n\n| 实验名 |", 1)
    TRACKING.write_text(text)


def main() -> None:
    summary = build_summary()
    markdown = render_markdown(summary)
    (OUTPUTS / "semantic_highpoint_fair_confirm_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUTPUTS / "semantic_highpoint_fair_confirm_summary.md").write_text(markdown)
    update_tracking(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
