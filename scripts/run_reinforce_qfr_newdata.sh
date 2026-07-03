#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/alpha_1203/gva_factor_experiments/runs_newdata/reinforce_qfr_$(date +%Y%m%d_%H%M%S)
LOGDIR="$RUN_ROOT/logs"
mkdir -p "$LOGDIR"
cd "$ROOT"

COMMON=(
  --random_seeds=0,1,2
  --steps=30000
  --eval_every_steps=1000
  --pool_capacity=10
  --instruments=csi300
  --qlib_data_path=/root/autodl-tmp/cn_data_akshare_2010_2026
  --qlib_kernels=1
  --device_str=auto
  --train_start=2010-01-04
  --train_end=2021-12-31
  --valid_start=2022-01-04
  --valid_end=2023-12-29
  --test_start=2024-01-02
  --test_end=2026-05-28
  --output_root="$RUN_ROOT"
)

echo "[START] $(date '+%F %T')"
echo "RUN_ROOT=$RUN_ROOT"
echo "SPLIT=train:2010-01-04..2021-12-31 valid:2022-01-04..2023-12-29 test:2024-01-02..2026-05-28"

PYTHONPATH="$ROOT" /root/miniconda3/bin/python -u scripts/reinforce_qfr_newdata.py \
  --algo=reinforce \
  --method_name=reinforce \
  "${COMMON[@]}" > "$LOGDIR/reinforce.log" 2>&1 &
PID_REINFORCE=$!

PYTHONPATH="$ROOT" /root/miniconda3/bin/python -u scripts/reinforce_qfr_newdata.py \
  --algo=qfr \
  --method_name=qfr \
  "${COMMON[@]}" > "$LOGDIR/qfr.log" 2>&1 &
PID_QFR=$!

echo "PID_REINFORCE=$PID_REINFORCE"
echo "PID_QFR=$PID_QFR"

wait "$PID_REINFORCE"
echo "[DONE] reinforce status=$? time=$(date '+%F %T')"
wait "$PID_QFR"
echo "[DONE] qfr status=$? time=$(date '+%F %T')"
echo "[END] $(date '+%F %T')"
