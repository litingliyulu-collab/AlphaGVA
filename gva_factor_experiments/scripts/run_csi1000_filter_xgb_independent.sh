#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_filter_xgb_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH=.

COMMON=(
  --instruments=csi1000
  --qlib_data_path=/root/autodl-tmp/cn_data_akshare_2010_2026
  --train_start=2010-01-04
  --train_end=2021-12-31
  --valid_start=2022-01-04
  --valid_end=2023-12-29
  --test_start=2024-01-02
  --test_end=2026-05-28
)

launch() {
  local name="$1"
  shift
  echo "LAUNCH $name $(date '+%F %T')"
  "$@" > "$RUN_ROOT/logs/${name}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_ROOT/${name}.pid"
  echo "$name PID=$pid"
}

wait_pattern() {
  local pattern="$1"
  local status=0
  shopt -s nullglob
  for pidfile in "$RUN_ROOT"/${pattern}.pid; do
    local name pid code
    name=$(basename "$pidfile" .pid)
    pid=$(cat "$pidfile")
    set +e
    wait "$pid"
    code=$?
    set -e
    echo "$code" > "$RUN_ROOT/${name}.status"
    echo "DONE $name status=$code time=$(date '+%F %T')"
    status=$((status + code))
  done
  shopt -u nullglob
  return "$status"
}

echo "RUN_ROOT=$RUN_ROOT"
echo "START=$(date '+%F %T')"
echo "UNIVERSE=CSI500"
echo "SUITE=PPO-filter-weak, GP-filter-strong, XGBoost"
echo "NOTE=GVA-filter waits for CSI500 Full-GVA source history."

STATUS=0

for seed in 0 1 2; do
  launch ppo_filter_weak_s${seed} /root/miniconda3/bin/python -u /root/alpha_1203/gva_factor_experiments/scripts/ppo_filter_variants.py \
    "${COMMON[@]}" --random_seeds=${seed} --method_name=ppo_filter_weak --output_root="$RUN_ROOT" \
    --pool_capacity=10 --steps=30000 --n_steps=64 --ppo_epochs=2 --batch_size=64 \
    --candidate_limit=500 --filter_mode=weak --mutual_threshold=0.99 --qlib_kernels=64 --device_str=auto
done
wait_pattern 'ppo_filter_weak_s*' || STATUS=$((STATUS + $?))

for seed in 0 1 2; do
  launch gp_filter_strong_s${seed} /root/miniconda3/bin/python -u /root/alpha_1203/gva_factor_experiments/scripts/gp_filter_newdata.py \
    "${COMMON[@]}" --random_seeds=${seed} --method_name=gp_filter_strong --output_root="$RUN_ROOT" \
    --pool_capacity=10 --population_size=1000 --generations=30 --tournament_size=300 \
    --eval_every_generations=2 --pool_candidate_multiplier=5 --filter_mode=strong \
    --qlib_kernels=32 --device_str=auto --n_jobs=1
done
wait_pattern 'gp_filter_strong_s*' || STATUS=$((STATUS + $?))

for seed in 0 1 2; do
  launch xgboost_s${seed} /root/miniconda3/bin/python -u /root/alpha_1203/gva_factor_experiments/scripts/ml_baselines_xgb_newdata.py \
    "${COMMON[@]}" --models=xgboost --random_seeds=${seed} --output_root="$RUN_ROOT" \
    --num_lags=60 --num_threads=8 --xgb_rounds=600 --xgb_max_depth=6 --xgb_learning_rate=0.03 \
    --log_every=50 --qlib_kernels=32 --device_str=cpu
  wait_pattern "xgboost_s${seed}" || STATUS=$((STATUS + $?))
done

echo "END=$(date '+%F %T') STATUS=$STATUS"
exit "$STATUS"
