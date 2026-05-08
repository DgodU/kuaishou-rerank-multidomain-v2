#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/kuaishou-rerank-multidomain-shared"
cd "$ROOT"

current="sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_mbcgate005_confirm_seed2028"
while ps -ef | grep -F "python scripts/train.py --config configs/${current}.yaml" | grep -v grep >/dev/null; do
  echo "$(date '+%F %T') waiting for ${current} to finish"
  sleep 60
done

for name in \
  sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_confirm_seed2027 \
  sidebias_dinatt_click_only_video_stat_ema_mbc_slices_semantic_simtier_sem48_slicegate_reg_mid_confirm_seed2028
 do
  if [ -f "outputs/${name}_test_metrics.json" ]; then
    echo "$(date '+%F %T') SKIP ${name}: metrics already exists"
    continue
  fi
  echo "$(date '+%F %T') START ${name}"
  conda run -n kuaishou-rerank-multidomain python scripts/train.py --config "configs/${name}.yaml"
  status=$?
  echo "$(date '+%F %T') END ${name} status=${status}"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done

echo "$(date '+%F %T') queue complete"
