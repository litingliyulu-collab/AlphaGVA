#!/usr/bin/env bash
# Run GVA-filter + PPO-filter (history) for a cap20 run root after RL finishes.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ROOT> [FILTER_OUT_ROOT]"
  exit 1
fi

MAIN_ROOT="$1"
FILTER_OUT="${2:-/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_cap20_filter_$(date +%Y%m%d_%H%M%S)}"
SCRIPT=/root/alpha_1203/gva_factor_experiments/scripts/csi500_chapter4/gva_filter_from_history.py
PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
mkdir -p "$FILTER_OUT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --instruments=csi500
  --pool_capacity=20
  --filter_mode=strong
  --mutual_threshold=0.99
  --l1_alpha=5e-3
  --qlib_kernels=1
  --device_str=auto
  --train_start=2010-01-04
  --train_end=2021-12-31
  --valid_start=2022-01-04
  --valid_end=2023-12-29
  --test_start=2024-01-02
  --test_end=2026-05-28
  --output_root="$FILTER_OUT"
)

echo "MAIN_ROOT=$MAIN_ROOT"
echo "FILTER_OUT=$FILTER_OUT"

for seed in 0 1 2; do
  echo "== GVA-filter seed $seed =="
  /root/miniconda3/bin/python "$SCRIPT" \
    "${COMMON[@]}" \
    --random_seeds="$seed" \
    --method_name="gva_filter" \
    --source_root="$MAIN_ROOT" \
    --source_pattern="full_gva25_s{seed}" \
    > "$FILTER_OUT/logs/gva_filter_s${seed}.log" 2>&1

  echo "== PPO-filter seed $seed =="
  /root/miniconda3/bin/python "$SCRIPT" \
    "${COMMON[@]}" \
    --random_seeds="$seed" \
    --method_name="ppo_filter" \
    --source_root="$MAIN_ROOT" \
    --source_pattern="ppo_s{seed}" \
    > "$FILTER_OUT/logs/ppo_filter_s${seed}.log" 2>&1
done

echo "FILTER_OUT=$FILTER_OUT"
