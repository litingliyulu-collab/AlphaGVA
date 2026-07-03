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

echo "[DQN] fresh seed0,1,2" | tee -a "$LOGDIR/dqn_fresh_all.log"
$PY -u "$ADV" --algos dqn --random_seeds 0,1,2 $COMMON \
  >> "$LOGDIR/dqn_fresh_all.log" 2>&1

echo "[DQN] done" | tee -a "$LOGDIR/dqn_fresh_all.log"
