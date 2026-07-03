#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/alpha_1203/AlphaForge-master/alphagen-master
OUT=/root/alpha_1203/gva_factor_experiments/runs_newdata/qfr_fixed_$(date +%Y%m%d_%H%M%S)
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"
cd "$ROOT"
echo "[START] $(date '+%F %T')"
echo "OUT=$OUT"
echo "SEEDS=0,1,2 STEPS=30000"
PYTHONPATH="$ROOT" /root/miniconda3/bin/python -u scripts/reinforce_qfr_newdata.py \
  --algo=qfr \
  --method_name=qfr \
  --random_seeds=0,1,2 \
  --steps=30000 \
  --eval_every_steps=1000 \
  --pool_capacity=10 \
  --instruments=csi300 \
  --qlib_data_path=/root/autodl-tmp/cn_data_akshare_2010_2026 \
  --qlib_kernels=1 \
  --device_str=auto \
  --train_start=2010-01-04 \
  --train_end=2021-12-31 \
  --valid_start=2022-01-04 \
  --valid_end=2023-12-29 \
  --test_start=2024-01-02 \
  --test_end=2026-05-28 \
  --output_root="$OUT" > "$LOGDIR/qfr.log" 2>&1
echo "[END] $(date '+%F %T')"
