#!/usr/bin/env bash
# CSI500 Full-GVA only: pool_capacity=20, 30k steps, seeds 1/2/3.
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_gva_cap20_30k_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --pool_capacity=20
  --instruments=csi500
  --steps=30000
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
  local seed="$2"
  echo "LAUNCH $name seed=$seed $(date '+%F %T')"
  /root/miniconda3/bin/python scripts/rl_v1.py \
    "${COMMON[@]}" \
    --random_seeds="${seed}" \
    --method_name="${name}" \
    --use_custom_ppo=True \
    --custom_critic_loss=hybrid \
    --td_weight=0.9 \
    --baseline_weight=0.1 \
    --actor_gap_weight=0.5 \
    --actor_gap_clip=2.0 \
    --gva_budget_ratio=0.25 \
    --gva_max_updates_per_rollout=1 \
    --gva_refresh_interval=5 \
    --gva_min_state_len=1 \
    > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  echo $! > "$RUN_ROOT/${name}.pid"
  echo "$name PID=$(cat "$RUN_ROOT/${name}.pid")"
}

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "START=$(date '+%F %T')"
  echo "METHOD=Full-GVA only"
  echo "POOL_CAPACITY=20"
  echo "STEPS=30000"
  echo "SEEDS=1,2,3"
} | tee "$RUN_ROOT/run_meta.txt"

for seed in 1 2 3; do
  launch "full_gva25_s${seed}" "${seed}"
done

STATUS=0
for seed in 1 2 3; do
  name="full_gva25_s${seed}"
  pid=$(cat "$RUN_ROOT/${name}.pid")
  wait "$pid" || STATUS=$?
  echo "DONE $name status=$? time=$(date '+%F %T')"
done

echo "END=$(date '+%F %T') STATUS=$STATUS" | tee -a "$RUN_ROOT/run_meta.txt"
exit "$STATUS"
