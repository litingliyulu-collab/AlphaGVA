#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=/root/autodl-tmp/gva_factor_experiments/launch_logs
SCRIPT_DIR=/root/alpha_1203/gva_factor_experiments/scripts
mkdir -p "$LOG_DIR"

PATTERNS=(
  "csi1000_actor_gva_20260701_154332"
  "csi1000_rl_advanced_20260701_154332"
  "csi1000_qfr_only_20260701_154332"
  "csi1000_ml_baselines_20260701_154332"
  "csi1000_gp_xgb_20260701_154332"
)

is_running() {
  local pat
  for pat in "${PATTERNS[@]}"; do
    if ps -eo args | grep "$pat" | grep -v grep >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

echo "[WATCH_AFF] start $(date '+%F %T')"
while is_running; do
  echo "[WATCH_AFF] waiting $(date '+%F %T')"
  df -h / /root/autodl-tmp | sed 's/^/[WATCH_AFF] /'
  sleep 300
done

echo "[WATCH_AFF] prior CSI1000 batch completed $(date '+%F %T')"
AFF_LOG="$LOG_DIR/csi1000_aff_$(date +%Y%m%d_%H%M%S).log"
nohup "$SCRIPT_DIR/run_csi1000_aff_official_adapted.sh" > "$AFF_LOG" 2>&1 &
echo "[WATCH_AFF] AFF_PID=$! LOG=$AFF_LOG"
