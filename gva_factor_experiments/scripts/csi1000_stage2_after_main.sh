#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/root/alpha_1203/gva_factor_experiments/scripts
PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_BASE=/root/autodl-tmp/gva_factor_experiments/runs_newdata
LOG_DIR=/root/autodl-tmp/gva_factor_experiments/launch_logs
MAIN_ROOT=${1:-/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_main_compare_20260701_022717}

mkdir -p "$LOG_DIR"

last_step() {
  local method=$1
  local f
  f=$(find "$MAIN_ROOT/$method" -name metrics.csv 2>/dev/null | head -1 || true)
  if [[ -z "$f" ]]; then
    echo 0
    return
  fi
  tail -1 "$f" | cut -d, -f1 | tr -dc '0-9'
}

echo "[STAGE2] Waiting for CSI1000 main compare: $MAIN_ROOT"
while true; do
  ready=1
  summary=""
  for method in ppo_s0 critic_gva25_s0 full_gva25_s0 ppo_s1 critic_gva25_s1 full_gva25_s1 ppo_s2 critic_gva25_s2 full_gva25_s2; do
    step=$(last_step "$method")
    summary="$summary $method=$step"
    if [[ "${step:-0}" -lt 30000 ]]; then
      ready=0
    fi
  done
  echo "[STAGE2] $(date '+%F %T')$summary"
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  sleep 180
done

echo "[STAGE2] Main compare complete. Launching CSI1000 stage2 at $(date '+%F %T')"

nohup "$SCRIPT_DIR/run_csi1000_actor_gva_parallel.sh" > "$LOG_DIR/csi1000_actor_gva_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "ACTOR_PID=$!"

nohup "$SCRIPT_DIR/run_csi1000_a2c.sh" > "$LOG_DIR/csi1000_a2c_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "A2C_PID=$!"

nohup "$SCRIPT_DIR/run_csi1000_reinforce_qfr.sh" > "$LOG_DIR/csi1000_reinforce_qfr_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "RQ_PID=$!"

nohup "$SCRIPT_DIR/run_csi1000_ml_baselines.sh" > "$LOG_DIR/csi1000_ml_baselines_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "ML_PID=$!"

nohup "$SCRIPT_DIR/run_csi1000_aff_official_adapted.sh" > "$LOG_DIR/csi1000_aff_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "AFF_PID=$!"

cd "$PROJECT"
PPO_FILTER_ROOT="$RUN_BASE/csi1000_ppo_history_filter_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PPO_FILTER_ROOT/logs"
nohup env PYTHONPATH=. /root/miniconda3/bin/python -u "$SCRIPT_DIR/gva_filter_from_history.py" \
  --instruments=csi1000 \
  --qlib_data_path=/root/autodl-tmp/cn_data_akshare_2010_2026 \
  --train_start=2010-01-04 --train_end=2021-12-31 \
  --valid_start=2022-01-04 --valid_end=2023-12-29 \
  --test_start=2024-01-02 --test_end=2026-05-28 \
  --random_seeds=0,1,2 --method_name=ppo_filter --output_root="$PPO_FILTER_ROOT" \
  --source_root="$MAIN_ROOT" --source_pattern='ppo_s{seed}' \
  --pool_capacity=10 --filter_mode=strong --qlib_kernels=64 --device_str=auto \
  > "$PPO_FILTER_ROOT/logs/ppo_filter_from_alphagen_s012.log" 2>&1 &
echo "PPO_HISTORY_FILTER_PID=$!"

GVA_FILTER_ROOT="$RUN_BASE/csi1000_gva_filter_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$GVA_FILTER_ROOT/logs"
nohup env PYTHONPATH=. /root/miniconda3/bin/python -u "$SCRIPT_DIR/gva_filter_from_history.py" \
  --instruments=csi1000 \
  --qlib_data_path=/root/autodl-tmp/cn_data_akshare_2010_2026 \
  --train_start=2010-01-04 --train_end=2021-12-31 \
  --valid_start=2022-01-04 --valid_end=2023-12-29 \
  --test_start=2024-01-02 --test_end=2026-05-28 \
  --random_seeds=0,1,2 --method_name=gva_filter --output_root="$GVA_FILTER_ROOT" \
  --source_root="$MAIN_ROOT" --source_pattern='full_gva25_s{seed}' \
  --pool_capacity=10 --filter_mode=strong --qlib_kernels=64 --device_str=auto \
  > "$GVA_FILTER_ROOT/logs/gva_filter_s012.log" 2>&1 &
echo "GVA_FILTER_PID=$!"

echo "[STAGE2] Stage2 launched at $(date '+%F %T')"
