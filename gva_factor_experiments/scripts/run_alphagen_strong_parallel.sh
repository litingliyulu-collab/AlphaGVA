#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/alphagen_strong_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --pool_capacity=10
  --instruments=csi300
  --steps=200000
  --n_steps=256
  --ppo_epochs=4
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
echo "METHODS=AlphaGen-PPO-strong"
echo "SEEDS=0,1,2"
echo "STEPS=200000 N_STEPS=256 PPO_EPOCHS=4 QLIB_KERNELS=126"
echo "DATA=/root/autodl-tmp/cn_data_akshare_2010_2026"
echo "SPLIT=train:2010-2021 valid:2022-2023 test:2024-01-02..2026-05-28"

for seed in 0 1 2; do
  launch alphagen_ppo_strong_s${seed} \
    --random_seeds=${seed} \
    --method_name=alphagen_ppo_strong_s${seed} \
    --use_custom_ppo=False
  sleep 2
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
