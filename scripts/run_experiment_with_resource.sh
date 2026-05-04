#!/usr/bin/env bash
set -euo pipefail

EXP="${1:?experiment name required}"
CONFIG="${2:?config path required}"
CONDA_ENV="${CONDA_ENV:-kuaishou-rerank-multidomain}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  OMP_NUM_THREADS=4
fi
export OMP_NUM_THREADS

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

mkdir -p logs/resource
MAIN_LOG="logs/nohup_${EXP}_full.log"
GPU_LOG="logs/resource/${EXP}_gpu.csv"
PROC_LOG="logs/resource/${EXP}_proc.tsv"
STATUS_LOG="logs/resource/${EXP}_status.log"

exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "[$(date '+%F %T')] preprocess start ${EXP} config=${CONFIG}"
python scripts/preprocess.py --config "${CONFIG}"
echo "[$(date '+%F %T')] train start ${EXP} config=${CONFIG}"

echo "timestamp,util_gpu,util_mem,mem_used_mb,mem_total_mb,power_w,temp_c" > "${GPU_LOG}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits -l 10 >> "${GPU_LOG}" 2>/dev/null &
  GPU_MON_PID=$!
else
  GPU_MON_PID=""
fi

python scripts/train.py --config "${CONFIG}" &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${STATUS_LOG}"
echo -e "timestamp\tpid\tchild_count\tcpu_pct_total\tmem_pct_total\trss_kb_total\tvsz_kb_total" > "${PROC_LOG}"

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  ps -eo pid=,ppid=,%cpu=,%mem=,rss=,vsz= | awk -v root="${TRAIN_PID}" -v ts="$(date '+%F %T')" '
    $1 == root || $2 == root {
      count += 1
      cpu += $3
      mem += $4
      rss += $5
      vsz += $6
    }
    END {
      if (count > 0) {
        printf "%s\t%s\t%d\t%.6f\t%.6f\t%d\t%d\n", ts, root, count - 1, cpu, mem, rss, vsz
      }
    }
  ' >> "${PROC_LOG}"
  sleep 10
done

set +e
wait "${TRAIN_PID}"
TRAIN_RC=$?
set -e
if [[ -n "${GPU_MON_PID}" ]]; then
  kill "${GPU_MON_PID}" 2>/dev/null || true
  wait "${GPU_MON_PID}" 2>/dev/null || true
fi

echo "[$(date '+%F %T')] train done ${EXP} rc=${TRAIN_RC}"
exit "${TRAIN_RC}"
