#!/usr/bin/env bash
set -euo pipefail
PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
RUN_ROOT=/root/alpha_1203/gva_factor_experiments/runs/stage2_confirm
PY=/root/miniconda3/bin/python
mkdir -p "$RUN_ROOT"
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

run_one() {
  local method="$1"
  local seed="$2"
  shift 2
  echo "[$(date '+%F %T')] START worker=B method=$method seed=$seed args=$*"
  "$PY" scripts/rl_v1.py \
    --random_seeds="$seed" \
    --pool_capacity=10 \
    --instruments=csi300 \
    --steps=20000 \
    --n_steps=256 \
    --method_name="$method" \
    --output_root="$RUN_ROOT" \
    "$@"
  echo "[$(date '+%F %T')] DONE worker=B method=$method seed=$seed"
}

for seed in 0 1 2; do
  run_one confirm_mse "$seed" --use_custom_ppo=True --custom_critic_loss=mse --td_weight=1.0 --baseline_weight=0.0 --actor_gap_weight=0.0
done

for seed in 0 1 2; do
  run_one confirm_full_gva_bw010_aw050 "$seed" --use_custom_ppo=True --custom_critic_loss=hybrid --td_weight=0.9 --baseline_weight=0.1 --actor_gap_weight=0.5 --actor_gap_clip=2.0
done

echo "[$(date '+%F %T')] WORKER B DONE"
