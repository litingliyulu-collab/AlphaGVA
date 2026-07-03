#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/alpha_1203/gva_factor_experiments/runs_newdata/warmpool_gva_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

WARM_TEMPLATE='/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_20260628_223714/random_search_s{seed}/results/20260628224314_random/final_pool.json'

COMMON=(
  --pool_capacity=10
  --instruments=csi300
  --steps=30000
  --n_steps=64
  --ppo_epochs=2
  --qlib_kernels=126
  --device_str=auto
  --train_start=2010-01-04
  --train_end=2021-12-31
  --valid_start=2022-01-04
  --valid_end=2023-12-29
  --test_start=2024-01-02
  --test_end=2026-05-28
  --output_root="$RUN_ROOT"
  --use_custom_ppo=True
  --custom_critic_loss=hybrid
  --td_weight=0.9
  --baseline_weight=0.1
  --actor_gap_weight=0.5
  --actor_gap_clip=2.0
  --gva_budget_ratio=0.25
  --gva_max_updates_per_rollout=1
  --gva_refresh_interval=5
  --gva_min_state_len=1
  --warm_pool_json="$WARM_TEMPLATE"
)

launch() {
  local seed="$1"
  local name="warm_full_gva25_s${seed}"
  echo "LAUNCH $name $(date '+%F %T')"
  /root/miniconda3/bin/python scripts/rl_v1.py "${COMMON[@]}" --random_seeds="$seed" --method_name="$name" > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_ROOT/${name}.pid"
  echo "$name PID=$pid"
}

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "START=$(date '+%F %T')"
  echo "METHOD=WarmPool-Full-GVA"
  echo "SEEDS=0,1,2"
  echo "STEPS=30000 N_STEPS=64 PPO_EPOCHS=2 QLIB_KERNELS=126"
  echo "WARM_TEMPLATE=$WARM_TEMPLATE"
  for seed in 0 1 2; do
    launch "$seed"
  done

  set +e
  STATUS=0
  for pidfile in "$RUN_ROOT"/*.pid; do
    name=$(basename "$pidfile" .pid)
    pid=$(cat "$pidfile")
    wait "$pid"
    code=$?
    echo "DONE $name status=$code time=$(date '+%F %T')"
    STATUS=$((STATUS + code))
  done
  set -e
  echo "END=$(date '+%F %T') STATUS=$STATUS"
  exit "$STATUS"
} | tee "$RUN_ROOT/run.log"
