#!/usr/bin/env bash
# Continue cap20 run toward 30k total steps using warm pool from 10k checkpoint.
# Note: reloads factor expressions only; actor/critic re-init (rl_v1 has no custom-PPO resume).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ROOT_10K> [TOTAL_STEPS=30000]"
  exit 1
fi

SRC_ROOT="$1"
TOTAL_STEPS="${2:-30000}"
PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT="${SRC_ROOT}_continue_${TOTAL_STEPS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

find_pool() {
  local sub="$1"
  local seed="$2"
  local result
  result=$(find "$SRC_ROOT/$sub/results" -mindepth 1 -maxdepth 1 -type d | head -1)
  local pool
  pool=$(ls "$result"/*_steps_pool.json 2>/dev/null | sort -t_ -k1 -n | tail -1)
  echo "$pool"
}

COMMON=(
  --pool_capacity=20
  --instruments=csi500
  --steps="$TOTAL_STEPS"
  --n_steps=64
  --ppo_epochs=2
  --qlib_kernels=64
  --device_str=auto
  --train_start=2010-01-04
  --train_end=2021-12-31
  --valid_start=2022-01-04
  --valid_end=2023-12-29
  --test_start=2024-01-02
  --test_end=2026-05-28
  --output_root="$RUN_ROOT"
)

launch() {
  local name="$1"
  local warm="$2"
  shift 2
  echo "LAUNCH $name warm_pool=$warm"
  /root/miniconda3/bin/python scripts/rl_v1.py \
    "${COMMON[@]}" \
    --warm_pool_json="$warm" \
    "$@" > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  echo $! > "$RUN_ROOT/${name}.pid"
}

for seed in 0 1 2; do
  ppo_warm=$(find_pool "ppo_s${seed}" "$seed")
  gva_warm=$(find_pool "full_gva25_s${seed}" "$seed")
  launch "ppo_s${seed}" "$ppo_warm" --random_seeds="${seed}" --method_name="ppo_s${seed}" --use_custom_ppo=False
done
wait
for seed in 0 1 2; do
  gva_warm=$(find_pool "full_gva25_s${seed}" "$seed")
  launch "full_gva25_s${seed}" "$gva_warm" \
    --random_seeds="${seed}" \
    --method_name="full_gva25_s${seed}" \
    --use_custom_ppo=True \
    --custom_critic_loss=hybrid \
    --td_weight=0.9 \
    --baseline_weight=0.1 \
    --actor_gap_weight=0.5 \
    --actor_gap_clip=2.0 \
    --gva_budget_ratio=0.25 \
    --gva_max_updates_per_rollout=1 \
    --gva_refresh_interval=5 \
    --gva_min_state_len=1
done
wait
echo "RUN_ROOT=$RUN_ROOT"
