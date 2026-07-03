#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/root/alpha_1203/gva_factor_experiments/scripts
PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_BASE=/root/autodl-tmp/gva_factor_experiments/runs_newdata
LOG_DIR=/root/autodl-tmp/gva_factor_experiments/launch_logs
MAIN_ROOT=${1:-/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_main_compare_20260701_022717}
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"
cd "$PROJECT"
export PYTHONPATH=.

echo "START=$(date '+%F %T')"
echo "MAIN_ROOT=$MAIN_ROOT"
echo "SUITE=CSI1000_FAST_STAGE2"

PPO_FILTER_ROOT="$RUN_BASE/csi1000_ppo_history_filter_${STAMP}"
mkdir -p "$PPO_FILTER_ROOT/logs"
nohup env PYTHONPATH=. /root/miniconda3/bin/python -u "$SCRIPT_DIR/gva_filter_from_history.py" \
  --instruments=csi1000 \
  --qlib_data_path=/root/autodl-tmp/cn_data_akshare_2010_2026 \
  --train_start=2010-01-04 --train_end=2021-12-31 \
  --valid_start=2022-01-04 --valid_end=2023-12-29 \
  --test_start=2024-01-02 --test_end=2026-05-28 \
  --random_seeds=0,1,2 --method_name=ppo_filter --output_root="$PPO_FILTER_ROOT" \
  --source_root="$MAIN_ROOT" --source_pattern='ppo_s{seed}' \
  --pool_capacity=10 --filter_mode=strong --qlib_kernels=48 --device_str=auto \
  > "$PPO_FILTER_ROOT/logs/ppo_filter_from_alphagen_s012.log" 2>&1 &
echo "$!" > "$PPO_FILTER_ROOT/ppo_filter.pid"
echo "PPO_FILTER_ROOT=$PPO_FILTER_ROOT PID=$(cat "$PPO_FILTER_ROOT/ppo_filter.pid")"

GVA_FILTER_ROOT="$RUN_BASE/csi1000_gva_filter_${STAMP}"
mkdir -p "$GVA_FILTER_ROOT/logs"
nohup env PYTHONPATH=. /root/miniconda3/bin/python -u "$SCRIPT_DIR/gva_filter_from_history.py" \
  --instruments=csi1000 \
  --qlib_data_path=/root/autodl-tmp/cn_data_akshare_2010_2026 \
  --train_start=2010-01-04 --train_end=2021-12-31 \
  --valid_start=2022-01-04 --valid_end=2023-12-29 \
  --test_start=2024-01-02 --test_end=2026-05-28 \
  --random_seeds=0,1,2 --method_name=gva_filter --output_root="$GVA_FILTER_ROOT" \
  --source_root="$MAIN_ROOT" --source_pattern='full_gva25_s{seed}' \
  --pool_capacity=10 --filter_mode=strong --qlib_kernels=48 --device_str=auto \
  > "$GVA_FILTER_ROOT/logs/gva_filter_s012.log" 2>&1 &
echo "$!" > "$GVA_FILTER_ROOT/gva_filter.pid"
echo "GVA_FILTER_ROOT=$GVA_FILTER_ROOT PID=$(cat "$GVA_FILTER_ROOT/gva_filter.pid")"

REINFORCE_ROOT="$RUN_BASE/csi1000_reinforce_only_${STAMP}"
mkdir -p "$REINFORCE_ROOT/logs"
nohup /root/miniconda3/bin/python -u "$SCRIPT_DIR/reinforce_qfr_newdata.py" \
  --instruments=csi1000 \
  --steps=30000 \
  --pool_capacity=10 \
  --qlib_kernels=48 \
  --train_start=2010-01-04 \
  --train_end=2021-12-31 \
  --valid_start=2022-01-04 \
  --valid_end=2023-12-29 \
  --test_start=2024-01-02 \
  --test_end=2026-05-28 \
  --random_seeds=0,1,2 \
  --output_root="$REINFORCE_ROOT" \
  --algo=reinforce \
  > "$REINFORCE_ROOT/logs/reinforce.log" 2>&1 &
echo "$!" > "$REINFORCE_ROOT/reinforce.pid"
echo "REINFORCE_ROOT=$REINFORCE_ROOT PID=$(cat "$REINFORCE_ROOT/reinforce.pid")"

echo "LAUNCHED=$(date '+%F %T')"
