#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
BASE=/root/alpha_1203/gva_factor_experiments/runs_newdata/locked_warmpool_gva_20260629_172252
RUN_ROOT=/root/alpha_1203/gva_factor_experiments/runs_newdata/locked_warmpool_gva_continue_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --pool_capacity=10
  --instruments=csi300
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
  --warm_lock_n=5
  --warm_sort_by_abs_weight=True
)

latest_pool_and_step() {
  local seed="$1"
  /root/miniconda3/bin/python - "$BASE" "$seed" <<'PY'
import glob, os, re, sys
base, seed = sys.argv[1], sys.argv[2]
paths = glob.glob(f"{base}/locked_warm_full_gva25_s{seed}/results/*/*_steps_pool.json")
items=[]
for p in paths:
    m=re.search(r'/(\d+)_steps_pool\.json$', p)
    if m:
        items.append((int(m.group(1)), p))
if not items:
    raise SystemExit(f'no pool for seed {seed}')
step, path=max(items)
print(step)
print(path)
PY
}

launch() {
  local seed="$1"
  local info step pool rem name
  info=$(latest_pool_and_step "$seed")
  step=$(echo "$info" | sed -n '1p')
  pool=$(echo "$info" | sed -n '2p')
  rem=$((30000 - step))
  if [ "$rem" -lt 64 ]; then rem=64; fi
  # keep rollout-aligned, custom trainer stops at or just above requested timesteps
  rem=$(( ((rem + 63) / 64) * 64 ))
  name="locked_warm_full_gva25_s${seed}_continue"
  echo "LAUNCH $name seed=$seed base_step=$step continue_steps=$rem pool=$pool time=$(date '+%F %T')"
  /root/miniconda3/bin/python scripts/rl_v1.py "${COMMON[@]}" --steps="$rem" --random_seeds="$seed" --method_name="$name" --warm_pool_json="$pool" > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_ROOT/${name}.pid"
  echo "$name PID=$pid"
}

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "START=$(date '+%F %T')"
  echo "METHOD=Locked-WarmPool-Full-GVA continuation from latest pool"
  echo "BASE=$BASE"
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
