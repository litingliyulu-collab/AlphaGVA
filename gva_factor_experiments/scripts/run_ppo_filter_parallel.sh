#!/usr/bin/env bash
set -euo pipefail
PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/ppo_filter_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"
echo "METHOD=PPO_filter seeds=0,1,2 steps=30000 n_steps=64 ppo_epochs=2"

launch() {
  local seed="$1"
  local name="ppo_filter_s${seed}"
  echo "LAUNCH $name $(date '+%F %T')"
  /root/miniconda3/bin/python -u /root/alpha_1203/gva_factor_experiments/scripts/ppo_filter_newdata.py \
    --random_seeds=${seed} \
    --method_name=ppo_filter \
    --output_root="$RUN_ROOT" \
    --qlib_data_path=/root/autodl-tmp/cn_data_akshare_2010_2026 \
    --qlib_kernels=126 \
    --device_str=auto \
    --instruments=csi300 \
    --pool_capacity=10 \
    --steps=30000 \
    --n_steps=64 \
    --ppo_epochs=2 \
    --batch_size=64 \
    --mutual_threshold=0.99 \
    --candidate_limit=2000 \
    --train_start=2010-01-04 \
    --train_end=2021-12-31 \
    --valid_start=2022-01-04 \
    --valid_end=2023-12-29 \
    --test_start=2024-01-02 \
    --test_end=2026-05-28 \
    > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_ROOT/${name}.pid"
  echo "$name PID=$pid"
}

for seed in 0 1 2; do
  launch "$seed"
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
