#!/usr/bin/env bash
set -euo pipefail
cd /root/alpha_1203/AlphaForge-master/alphagen-master
export PYTHONPATH=/root/alpha_1203/AlphaForge-master/alphagen-master:${PYTHONPATH:-}
PY=/root/miniconda3/bin/python
ADV=/root/alpha_1203/gva_factor_experiments/scripts/rl_advanced_baselines.py
ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/rl_advanced_baselines_20260629
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
COMMON="--steps 30000 --qlib_kernels 32 --model_checkpoint_interval 5000 --keep_model_checkpoints 1"

echo "[A2C] resume seed1" | tee -a "$LOGDIR/a2c_resume_and_s2.log"
$PY -u "$ADV" --algos a2c --random_seeds 1 $COMMON \
  --resume_checkpoint "$ROOT/a2c_baseline_s1/results/csi300_10_1_20260629195426_a2c_original/23552_steps.zip" \
  --resume_warm_pool_json "$ROOT/a2c_baseline_s1/results/csi300_10_1_20260629195426_a2c_original/23552_steps_pool.json" \
  >> "$LOGDIR/a2c_resume_and_s2.log" 2>&1

echo "[A2C] fresh seed2" | tee -a "$LOGDIR/a2c_resume_and_s2.log"
$PY -u "$ADV" --algos a2c --random_seeds 2 $COMMON \
  >> "$LOGDIR/a2c_resume_and_s2.log" 2>&1

echo "[A2C] done" | tee -a "$LOGDIR/a2c_resume_and_s2.log"

