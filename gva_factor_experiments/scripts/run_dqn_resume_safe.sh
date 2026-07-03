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

echo "[DQN] resume seed0" | tee -a "$LOGDIR/dqn_resume_all.log"
$PY -u "$ADV" --algos dqn --random_seeds 0 $COMMON \
  --resume_checkpoint "$ROOT/dqn_baseline_s0/results/csi300_10_0_20260629195426_dqn_original/5804_steps.zip" \
  --resume_warm_pool_json "$ROOT/dqn_baseline_s0/results/csi300_10_0_20260629195426_dqn_original/5800_steps_pool.json" \
  >> "$LOGDIR/dqn_resume_all.log" 2>&1

echo "[DQN] fresh seed1,2" | tee -a "$LOGDIR/dqn_resume_all.log"
$PY -u "$ADV" --algos dqn --random_seeds 1,2 $COMMON \
  >> "$LOGDIR/dqn_resume_all.log" 2>&1

echo "[DQN] done" | tee -a "$LOGDIR/dqn_resume_all.log"

