#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/kuaishou-rerank-multidomain-shared"
QUEUE_PID_FILE="${1:-${ROOT}/logs/nohup_semantic_highpoint_fair_confirm_queue_20260508_170012.pid}"
cd "$ROOT"

if [ -f "$QUEUE_PID_FILE" ]; then
  queue_pid="$(cat "$QUEUE_PID_FILE")"
  while kill -0 "$queue_pid" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for semantic queue pid=${queue_pid}"
    sleep 120
  done
else
  echo "$(date '+%F %T') queue pid file not found: ${QUEUE_PID_FILE}; continue with post-processing"
fi

while pgrep -f 'sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_.*confirm_seed202[78].*scripts/train.py' >/dev/null; do
  echo "$(date '+%F %T') waiting for semantic high-point train processes"
  sleep 120
done

echo "$(date '+%F %T') summarizing semantic fair confirmation"
python scripts/summarize_semantic_highpoint_confirm.py

echo "$(date '+%F %T') writing disk cleanup candidate report"
python scripts/report_disk_cleanup_candidates.py

echo "$(date '+%F %T') post-queue tasks complete; review outputs/semantic_highpoint_fair_confirm_summary.md and outputs/disk_cleanup_candidates.md before cleanup or new protected follow-up runs"
