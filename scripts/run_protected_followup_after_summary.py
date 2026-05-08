from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs" / "semantic_highpoint_fair_confirm_summary.json"
CONFIG_DIR = ROOT / "configs"
LOG_DIR = ROOT / "logs"

BASE_CONFIG_BY_CANDIDATE = {
    "protected_mbc_slices": "sidebias_dinatt_click_only_video_stat_ema_mbc_slices",
    "semantic_simtier_sem48": "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48",
    "semantic_simtier_sem48_mbcgate005": "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_mbcgate005",
    "semantic_simtier_sem48_slicegate_reg_mid": "sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid",
}


def wait_for_summary() -> dict:
    while True:
        if SUMMARY_PATH.exists():
            summary = json.loads(SUMMARY_PATH.read_text())
            winner = summary.get("recommended_protected_model")
            if winner and winner in BASE_CONFIG_BY_CANDIDATE:
                return summary
        print(f"{datetime.now().isoformat(timespec='seconds')} waiting for complete semantic summary")
        time.sleep(120)


def write_config(base_name: str, suffix: str, extra: dict) -> str:
    source = CONFIG_DIR / f"{base_name}.yaml"
    target_name = f"{base_name}_{suffix}"
    target = CONFIG_DIR / f"{target_name}.yaml"
    config = yaml.safe_load(source.read_text())
    config["experiment_name"] = target_name
    config["base_model"] = base_name
    config.update(extra)
    target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return target_name


def run_train(config_name: str) -> None:
    cmd = ["conda", "run", "-n", "kuaishou-rerank-multidomain", "python", "scripts/train.py", "--config", f"configs/{config_name}.yaml"]
    print(f"{datetime.now().isoformat(timespec='seconds')} START {config_name}")
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f"{datetime.now().isoformat(timespec='seconds')} END {config_name} status={proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def append_tracking(winner: str, no_valid_name: str, ccss_name: str) -> None:
    tracking = ROOT / "experiments" / "results_tracking.md"
    text = tracking.read_text()
    line = (
        f"\n2026-05-08 protected follow-up queued: semantic fair-confirmation winner `{winner}` selected from "
        f"`outputs/semantic_highpoint_fair_confirm_summary.json`. Created and serially ran/queued no-validation config "
        f"`{no_valid_name}` with `merge_valid_into_train=true` and `train_without_validation=true`, then CCSS config "
        f"`{ccss_name}` with conservative dense-feature counterfactual contrastive loss.\n"
    )
    if line not in text:
        text = text.replace("\n| 实验名 |", line + "\n| 实验名 |", 1)
        tracking.write_text(text)


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    summary = wait_for_summary()
    winner = summary["recommended_protected_model"]
    base_name = BASE_CONFIG_BY_CANDIDATE[winner]
    no_valid_name = write_config(
        base_name,
        "protected_train_valid_merged",
        {
            "merge_valid_into_train": True,
            "train_without_validation": True,
            "epochs": 4,
            "early_stop_patience": 0,
            "seed": 2025,
        },
    )
    ccss_name = write_config(
        base_name,
        "protected_ccss",
        {
            "use_ccss": True,
            "ccss_feature_tensor": "dense_features",
            "ccss_sample_ratio": 0.25,
            "ccss_delta_scale": 0.5,
            "ccss_loss_weight": 0.05,
            "ccss_factual_loss_weight": 0.05,
            "ccss_margin": 0.0,
            "seed": 2025,
        },
    )
    append_tracking(winner, no_valid_name, ccss_name)
    run_train(no_valid_name)
    run_train(ccss_name)


if __name__ == "__main__":
    main()
