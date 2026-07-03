#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
SCRIPT_ROOT=/root/alpha_1203/gva_factor_experiments/scripts
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi500_reinforce_qfr_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --instruments=csi500
  --steps=30000
  --pool_capacity=10
  --qlib_kernels=64
  --train_start=2010-01-04
  --train_end=2021-12-31
  --valid_start=2022-01-04
  --valid_end=2023-12-29
  --test_start=2024-01-02
  --test_end=2026-05-28
  --random_seeds=0,1,2
  --output_root="$RUN_ROOT"
)

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"
echo "UNIVERSE=CSI500"
echo "METHODS=REINFORCE,QFR"

/root/miniconda3/bin/python "$SCRIPT_ROOT/reinforce_qfr_newdata.py" "${COMMON[@]}" --algo=reinforce > "$RUN_ROOT/logs/reinforce.log" 2>&1 &
echo "$!" > "$RUN_ROOT/reinforce.pid"

/root/miniconda3/bin/python "$SCRIPT_ROOT/reinforce_qfr_newdata.py" "${COMMON[@]}" --algo=qfr > "$RUN_ROOT/logs/qfr.log" 2>&1 &
echo "$!" > "$RUN_ROOT/qfr.pid"

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
