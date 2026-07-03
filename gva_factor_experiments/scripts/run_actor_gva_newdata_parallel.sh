#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/actor_gva_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

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
)

launch() {
  local name="$1"
  shift
  echo "LAUNCH $name $(date '+%F %T')"
  /root/miniconda3/bin/python scripts/rl_v1.py "${COMMON[@]}" "$@" > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_ROOT/${name}.pid"
  echo "$name PID=$pid"
}

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"
echo "METHODS=Actor-GVA"
echo "SEEDS=0,1,2"
echo "STEPS=30000 N_STEPS=64 PPO_EPOCHS=2 QLIB_KERNELS=126"
echo "DATA=/root/autodl-tmp/cn_data_akshare_2010_2026"
echo "SPLIT=train:2010-2021 valid:2022-2023 test:2024-01-02..2026-05-28"
echo "ACTOR_GVA=critic_loss=mse baseline_weight=0 actor_gap_weight=0.5 gva_budget_ratio=0.25 max_updates=1 refresh=5"

for seed in 0 1 2; do
  launch actor_gva25_s${seed} \
    --random_seeds=${seed} \
    --method_name=actor_gva25_s${seed} \
    --use_custom_ppo=True \
    --custom_critic_loss=mse \
    --td_weight=1.0 \
    --baseline_weight=0.0 \
    --actor_gap_weight=0.5 \
    --actor_gap_clip=2.0 \
    --gva_budget_ratio=0.25 \
    --gva_max_updates_per_rollout=1 \
    --gva_refresh_interval=5 \
    --gva_min_state_len=1
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
