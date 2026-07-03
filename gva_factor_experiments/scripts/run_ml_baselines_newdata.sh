#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/alpha_1203/gva_factor_experiments/runs_newdata/ml_baselines_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --instruments=csi300
  --num_lags=60
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

wait_group() {
  local pattern="$1"
  local status=0
  for pidfile in "$RUN_ROOT"/${pattern}.pid; do
    local name
    name=$(basename "$pidfile" .pid)
    local pid
    pid=$(cat "$pidfile")
    set +e
    wait "$pid"
    local code=$?
    set -e
    echo "$code" > "$RUN_ROOT/${name}.status"
    echo "DONE $name status=$code time=$(date '+%F %T')"
    status=$((status + code))
  done
  return "$status"
}

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"
echo "METHODS=LightGBM, MLP"
echo "SCALE=num_lags=60 seeds=0,1,2"
echo "DATA=/root/autodl-tmp/cn_data_akshare_2010_2026"
echo "SPLIT=train:2010-2021 valid:2022-2023 test:2024-01-02..2026-05-28"

for seed in 0 1 2; do
  launch lightgbm_s${seed} scripts/ml_baselines_newdata.py \
    "${COMMON[@]}" \
    --models=lightgbm \
    --random_seeds=${seed} \
    --num_threads=8 \
    --lgb_rounds=1000 \
    --early_stopping=100 \
    --log_every=50

done

STATUS=0
wait_group 'lightgbm_s*' || STATUS=$((STATUS + $?))

for seed in 0 1 2; do
  launch mlp_s${seed} scripts/ml_baselines_newdata.py \
    "${COMMON[@]}" \
    --models=mlp \
    --random_seeds=${seed} \
    --mlp_epochs=10 \
    --mlp_batch_size=2048 \
    --mlp_lr=0.001

done

wait_group 'mlp_s*' || STATUS=$((STATUS + $?))

echo "END=$(date '+%F %T') STATUS=$STATUS"
exit "$STATUS"
