#!/usr/bin/env bash
set -euo pipefail

ROOT0=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_qfr_only_20260701_154332
SEQ_PID=10060

echo "[WATCH_QFR] start $(date '+%F %T') ROOT0=$ROOT0 SEQ_PID=$SEQ_PID"
while true; do
  f0=$(find "$ROOT0/qfr_s0" -name final_pool.json 2>/dev/null | head -1 || true)
  step="NA"
  metric=$(find "$ROOT0/qfr_s0" -name metrics.csv 2>/dev/null | head -1 || true)
  if [[ -n "$metric" ]]; then
    step=$(tail -1 "$metric" | cut -d, -f1)
  fi
  echo "[WATCH_QFR] $(date '+%F %T') seed0_final=${f0:-NA} step=$step"
  if [[ -n "$f0" ]]; then
    if kill -0 "$SEQ_PID" 2>/dev/null; then
      echo "[WATCH_QFR] seed0 done, killing sequential duplicate pid=$SEQ_PID"
      kill "$SEQ_PID" 2>/dev/null || true
    fi
    break
  fi
  sleep 120
done
echo "[WATCH_QFR] done $(date '+%F %T')"
