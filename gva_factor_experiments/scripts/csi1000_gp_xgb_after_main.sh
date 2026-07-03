#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/root/alpha_1203/gva_factor_experiments/scripts
MAIN_ROOT=${1:-/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_main_compare_20260701_022717}
LOG_DIR=/root/autodl-tmp/gva_factor_experiments/launch_logs

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

echo "[GP_XGB] Waiting for CSI1000 main compare: $MAIN_ROOT"
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
  echo "[GP_XGB] $(date '+%F %T')$summary"
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  sleep 180
done

echo "[GP_XGB] Main compare complete. Launching GP/XGBoost at $(date '+%F %T')"
nohup "$SCRIPT_DIR/run_csi1000_gp_xgb.sh" > "$LOG_DIR/csi1000_gp_xgb_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo "GP_XGB_PID=$!"
