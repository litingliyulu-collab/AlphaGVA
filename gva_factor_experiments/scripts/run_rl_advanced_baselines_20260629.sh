#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/alpha_1203/AlphaForge-master/alphagen-master
OUT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629
LOGDIR=${OUT}/logs
mkdir -p "$LOGDIR"
cd "$ROOT"
echo "[START] $(date '+%F %T')"
echo "[RESOURCE_BEFORE]"
free -h || true
df -h / /root/autodl-tmp || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv || true
PYTHONPATH="$ROOT" /root/miniconda3/bin/python -u /root/alpha_1203/gva_factor_experiments/scripts/rl_advanced_baselines.py \
  --algos=a2c,dqn,qrdqn \
  --random_seeds="[0,1,2]" \
  --steps=30000 \
  --pool_capacity=10 \
  --instruments=csi300 \
  --n_steps=64 \
  --qlib_kernels=1 \
  --device_str=auto \
  --train_start=2010-01-04 \
  --train_end=2021-12-31 \
  --valid_start=2022-01-04 \
  --valid_end=2023-12-29 \
  --test_start=2024-01-02 \
  --test_end=2026-05-28 \
  --output_root="$OUT"
echo "[END] $(date '+%F %T')"
