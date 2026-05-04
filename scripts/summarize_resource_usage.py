from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def to_float(value: str) -> float | None:
    try:
        value = value.strip()
        if not value or value.lower() == "[not supported]":
            return None
        return float(value)
    except Exception:
        return None


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None, "min": None}
    return {"count": len(values), "mean": mean(values), "max": max(values), "min": min(values)}


def read_gpu(path: Path) -> dict[str, dict[str, float | int | None]]:
    if not path.exists():
        return {}
    cols = ["timestamp", "util_gpu", "util_mem", "mem_used_mb", "mem_total_mb", "power_w", "temp_c"]
    buckets: dict[str, list[float]] = {c: [] for c in cols[1:]}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < len(cols):
                continue
            for idx, col in enumerate(cols[1:], start=1):
                v = to_float(row[idx])
                if v is not None:
                    buckets[col].append(v)
    return {k: summarize(v) for k, v in buckets.items()}


def read_proc(path: Path) -> dict[str, dict[str, float | int | None]]:
    if not path.exists():
        return {}
    buckets = {"child_count": [], "cpu_pct": [], "mem_pct": [], "rss_mb": [], "vsz_mb": []}
    with path.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            if len(parts) >= 7 and parts[2].replace(".", "", 1).isdigit():
                child_count = to_float(parts[2])
                cpu = to_float(parts[3])
                mem = to_float(parts[4])
                rss = to_float(parts[5])
                vsz = to_float(parts[6])
            else:
                child_count = None
                cpu = to_float(parts[3])
                mem = to_float(parts[4])
                rss = to_float(parts[5])
                vsz = to_float(parts[6])
            if child_count is not None:
                buckets["child_count"].append(child_count)
            if cpu is not None:
                buckets["cpu_pct"].append(cpu)
            if mem is not None:
                buckets["mem_pct"].append(mem)
            if rss is not None:
                buckets["rss_mb"].append(rss / 1024.0)
            if vsz is not None:
                buckets["vsz_mb"].append(vsz / 1024.0)
    return {k: summarize(v) for k, v in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--resource-dir", default="logs/resource")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    resource_dir = Path(args.resource_dir)
    exp = args.experiment
    summary = {
        "experiment": exp,
        "gpu": read_gpu(resource_dir / f"{exp}_gpu.csv"),
        "process": read_proc(resource_dir / f"{exp}_proc.tsv"),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
