#!/usr/bin/env bash
# CSI500: pool_capacity=20, 10k steps, AlphaGen-PPO + Full-GVA only.
# Launch in two waves (3+3) to limit GPU/qlib memory pressure.
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_cap20_10k_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --pool_capacity=20
  --instruments=csi500
  --steps=10000
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
  shift
  echo "LAUNCH $name $(date '+%F %T')"
  /root/miniconda3/bin/python scripts/rl_v1.py "${COMMON[@]}" "$@" > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  echo $! > "$RUN_ROOT/${name}.pid"
  echo "$name PID=$(cat "$RUN_ROOT/${name}.pid")"
}

wait_wave() {
  local label="$1"
  shift
  local status=0
  for name in "$@"; do
    local pid
    pid=$(cat "$RUN_ROOT/${name}.pid")
    wait "$pid" || status=$?
    echo "DONE $name status=$? time=$(date '+%F %T')"
  done
  echo "WAVE $label status=$status"
  return "$status"
}

echo "RUN_ROOT=$RUN_ROOT" | tee "$RUN_ROOT/run_meta.txt"
{
  echo "START=$(date '+%F %T')"
  echo "UNIVERSE=CSI500"
  echo "POOL_CAPACITY=20"
  echo "STEPS=10000"
  echo "METHODS=AlphaGen-PPO,Full-GVA"
  echo "SEEDS=0,1,2"
  echo "QLIB_KERNELS=64 (memory-safe)"
  echo "WAVES=ppo x3 then full_gva x3"
  echo "RESUME_NOTE=rl_v1 custom PPO has no policy resume; use run_csi500_cap20_continue_30k.sh with warm_pool after 10k"
} >> "$RUN_ROOT/run_meta.txt"

# Wave 1: AlphaGen-PPO
for seed in 0 1 2; do
  launch "ppo_s${seed}" \
    --random_seeds="${seed}" \
    --method_name="ppo_s${seed}" \
    --use_custom_ppo=False
done
wait_wave "alphagen" ppo_s0 ppo_s1 ppo_s2

# Wave 2: Full-GVA
for seed in 0 1 2; do
  launch "full_gva25_s${seed}" \
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
wait_wave "full_gva" full_gva25_s0 full_gva25_s1 full_gva25_s2

echo "END=$(date '+%F %T')" | tee -a "$RUN_ROOT/run_meta.txt"
echo "RUN_ROOT=$RUN_ROOT"
echo "Next: bash .../run_csi500_cap20_gva_filter_from_run.sh $RUN_ROOT"
