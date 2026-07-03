#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/alpha_1203/gva_factor_experiments/runs_newdata/horizontal_baselines_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --instruments=csi300
  --pool_capacity=10
  --qlib_kernels=64
  --device_str=auto
  --train_start=2010-01-04
  --train_end=2021-12-31
  --valid_start=2022-01-04
  --valid_end=2023-12-29
  --test_start=2024-01-02
  --test_end=2026-05-28
  --output_root="$RUN_ROOT"
)

launch() {
  local name="$1"
  shift
  echo "LAUNCH $name $(date '+%F %T')"
  /root/miniconda3/bin/python "$@" > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_ROOT/${name}.pid"
  echo "$name PID=$pid"
}

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"
echo "METHODS=Random Search, GP/gplearn"
echo "MODE=parallel"
echo "SCALE=random attempts=30000; gp population=1000 generations=30; seeds=0,1,2"
echo "DATA=/root/autodl-tmp/cn_data_akshare_2010_2026"
echo "SPLIT=train:2010-2021 valid:2022-2023 test:2024-01-02..2026-05-28"

for seed in 0 1 2; do
  launch random_s${seed} scripts/random_search_newdata.py \
    "${COMMON[@]}" \
    --random_seeds=${seed} \
    --method_name=random_search \
    --attempts=30000 \
    --eval_every=1000 \
    --max_depth=4

done

for seed in 0 1 2; do
  launch gp_s${seed} scripts/gp_newdata.py \
    "${COMMON[@]}" \
    --random_seeds=${seed} \
    --method_name=gp \
    --population_size=1000 \
    --generations=30 \
    --tournament_size=300 \
    --eval_every_generations=2 \
    --pool_candidate_multiplier=5

done

set +e
STATUS=0
for pidfile in "$RUN_ROOT"/*.pid; do
  name=$(basename "$pidfile" .pid)
  pid=$(cat "$pidfile")
  wait "$pid"
  code=$?
  echo "$code" > "$RUN_ROOT/${name}.status"
  echo "DONE $name status=$code time=$(date '+%F %T')"
  STATUS=$((STATUS + code))
done
set -e

echo "END=$(date '+%F %T') STATUS=$STATUS"
exit "$STATUS"
