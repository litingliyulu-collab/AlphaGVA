#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
SCRIPT_ROOT=/root/alpha_1203/gva_factor_experiments/scripts
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_rl_advanced_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"
echo "UNIVERSE=CSI500"
echo "METHODS=A2C"

/root/miniconda3/bin/python "$SCRIPT_ROOT/rl_advanced_baselines.py" \
  --algos=a2c \
  --random_seeds=0,1,2 \
  --instruments=csi500 \
  --steps=30000 \
  --n_steps=64 \
  --pool_capacity=10 \
  --qlib_kernels=64 \
  --train_start=2010-01-04 \
  --train_end=2021-12-31 \
  --valid_start=2022-01-04 \
  --valid_end=2023-12-29 \
  --test_start=2024-01-02 \
  --test_end=2026-05-28 \
  --output_root="$RUN_ROOT" \
  > "$RUN_ROOT/logs/a2c.log" 2>&1

echo "END=$(date '+%F %T') STATUS=$?"
