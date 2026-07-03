#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=/root/alpha_1203/gva_factor_experiments/scripts
LOG_DIR=/root/alpha_1203/gva_factor_experiments/logs
mkdir -p "$LOG_DIR"
chmod +x "$SCRIPT_DIR/run_stage2_confirm_worker_a.sh" "$SCRIPT_DIR/run_stage2_confirm_worker_b.sh"
nohup "$SCRIPT_DIR/run_stage2_confirm_worker_a.sh" > "$LOG_DIR/stage2_confirm_worker_a_20260627.log" 2>&1 &
echo $! > "$LOG_DIR/stage2_confirm_worker_a.pid"
nohup "$SCRIPT_DIR/run_stage2_confirm_worker_b.sh" > "$LOG_DIR/stage2_confirm_worker_b_20260627.log" 2>&1 &
echo $! > "$LOG_DIR/stage2_confirm_worker_b.pid"
echo "worker_a_pid=$(cat "$LOG_DIR/stage2_confirm_worker_a.pid")"
echo "worker_b_pid=$(cat "$LOG_DIR/stage2_confirm_worker_b.pid")"
