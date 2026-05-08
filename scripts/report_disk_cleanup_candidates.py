from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTODL_TMP = Path("/root/autodl-tmp")
OUTPUTS = ROOT / "outputs"

KEEP_CHECKPOINT_KEYWORDS = [
    "mbc_slices_best",
    "mbc_slices_confirm_seed2026",
    "mbc_slices_confirm_seed2027",
    "mbc_slices_confirm_seed2028",
    "semantic_simtier_sem48",
    "mbcgate005",
    "slicegate_reg_mid",
]


def folder_size(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            fp = Path(dirpath) / filename
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def human(size: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def top_dirs(base: Path, depth: int = 2) -> list[dict]:
    rows = []
    if not base.exists():
        return rows
    candidates = [base]
    for _ in range(depth):
        next_candidates = []
        for path in candidates:
            try:
                for child in path.iterdir():
                    if child.is_dir():
                        next_candidates.append(child)
            except OSError:
                continue
        candidates.extend(next_candidates)
    seen = []
    for path in candidates:
        if path not in seen:
            seen.append(path)
    for path in seen:
        try:
            size = folder_size(path)
        except OSError:
            continue
        rows.append({"path": str(path), "bytes": size, "human": human(size)})
    return sorted(rows, key=lambda x: x["bytes"], reverse=True)


def checkpoint_candidates() -> list[dict]:
    ckpt_dir = ROOT / "checkpoints"
    rows = []
    if not ckpt_dir.exists():
        return rows
    for path in ckpt_dir.glob("*.pt"):
        name = path.name
        keep = any(keyword in name for keyword in KEEP_CHECKPOINT_KEYWORDS)
        try:
            stat = path.stat()
        except OSError:
            continue
        if not keep:
            rows.append({"path": str(path), "bytes": stat.st_size, "human": human(stat.st_size), "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")})
    return sorted(rows, key=lambda x: x["bytes"], reverse=True)


def render(report: dict) -> str:
    lines = []
    lines.append("# Disk Cleanup Candidate Report")
    lines.append("")
    lines.append(f"Generated at: `{report['generated_at']}`")
    lines.append("")
    lines.append("This report does not delete files. Review before running any cleanup command.")
    lines.append("")
    lines.append("## Largest directories under /root/autodl-tmp")
    lines.append("")
    lines.append("| Size | Path |")
    lines.append("|---:|---|")
    for row in report["top_dirs"][:30]:
        lines.append(f"| {row['human']} | `{row['path']}` |")
    lines.append("")
    lines.append("## Shared-repo checkpoint deletion candidates")
    lines.append("")
    lines.append("| Size | Modified | Path |")
    lines.append("|---:|---|---|")
    for row in report["checkpoint_candidates"][:80]:
        lines.append(f"| {row['human']} | {row['mtime']} | `{row['path']}` |")
    lines.append("")
    lines.append(f"Estimated checkpoint cleanup bytes from listed non-protected candidates: `{human(report['checkpoint_candidate_total_bytes'])}`")
    lines.append("")
    lines.append("Large external candidates observed from directory sizes should be handled separately because they may be raw or reusable processed datasets.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    candidates = checkpoint_candidates()
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_dirs": top_dirs(AUTODL_TMP, depth=2),
        "checkpoint_candidates": candidates,
        "checkpoint_candidate_total_bytes": sum(x["bytes"] for x in candidates),
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "disk_cleanup_candidates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    markdown = render(report)
    (OUTPUTS / "disk_cleanup_candidates.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
