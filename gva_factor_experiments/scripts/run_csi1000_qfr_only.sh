#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
SCRIPT_ROOT=/root/alpha_1203/gva_factor_experiments/scripts
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_qfr_only_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"
echo "UNIVERSE=CSI1000"
echo "METHODS=QFR"

/root/miniconda3/bin/python "$SCRIPT_ROOT/reinforce_qfr_newdata.py" \
  --instruments=csi1000 \
  --steps=30000 \
  --pool_capacity=10 \
  --qlib_kernels=48 \
  --train_start=2010-01-04 \
  --train_end=2021-12-31 \
  --valid_start=2022-01-04 \
  --valid_end=2023-12-29 \
  --test_start=2024-01-02 \
  --test_end=2026-05-28 \
  --random_seeds=0,1,2 \
  --output_root="$RUN_ROOT" \
  --algo=qfr \
  > "$RUN_ROOT/logs/qfr.log" 2>&1

echo "END=$(date '+%F %T') STATUS=$?"
