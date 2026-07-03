#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/alpha_1203/gva_factor_experiments/runs_newdata/stage1_probe_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --random_seeds=0
  --pool_capacity=10
  --instruments=csi300
  --steps=512
  --n_steps=32
  --ppo_epochs=1
  --qlib_kernels=8
  --device_str=auto
  --train_start=2010-01-04
  --train_end=2021-12-31
  --valid_start=2022-01-04
  --valid_end=2023-12-29
  --test_start=2024-01-02
  --test_end=2026-05-28
  --output_root="$RUN_ROOT"
)

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"

/root/miniconda3/bin/python scripts/rl_v1.py \
  "${COMMON[@]}" \
  --method_name=ppo_s0 \
  --use_custom_ppo=False \
  > "$RUN_ROOT/logs/ppo_s0.log" 2>&1 &
PID_PPO=$!
echo "PPO_PID=$PID_PPO"

/root/miniconda3/bin/python scripts/rl_v1.py \
  "${COMMON[@]}" \
  --method_name=critic_gva25_s0 \
  --use_custom_ppo=True \
  --custom_critic_loss=hybrid \
  --td_weight=0.9 \
  --baseline_weight=0.1 \
  --actor_gap_weight=0.0 \
  --gva_budget_ratio=0.25 \
  --gva_max_updates_per_rollout=1 \
  --gva_refresh_interval=5 \
  --gva_min_state_len=1 \
  > "$RUN_ROOT/logs/critic_gva25_s0.log" 2>&1 &
PID_GVA=$!
echo "GVA_PID=$PID_GVA"

echo "$PID_PPO" > "$RUN_ROOT/ppo.pid"
echo "$PID_GVA" > "$RUN_ROOT/critic_gva.pid"

set +e
wait "$PID_PPO"
PPO_STATUS=$?
echo "PPO_DONE status=$PPO_STATUS time=$(date '+%F %T')"
wait "$PID_GVA"
GVA_STATUS=$?
echo "GVA_DONE status=$GVA_STATUS time=$(date '+%F %T')"
set -e

echo "END=$(date '+%F %T')"
exit $((PPO_STATUS + GVA_STATUS))