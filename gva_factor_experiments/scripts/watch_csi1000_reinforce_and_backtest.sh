#!/usr/bin/env bash
set -euo pipefail

ROOT0=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_reinforce_only_20260701_110038
ROOT12=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_reinforce_parallel_extra_20260701_111555
MAIN_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_main_compare_20260701_022717
PPO_FILTER_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_ppo_history_filter_20260701_110038
GVA_FILTER_ROOT=/root/autodl-tmp/gva_factor_experiments/runs_newdata/csi1000_gva_filter_20260701_110038
PROJECT=/root/alpha_1203/AlphaForge-master/alphagen-master
SCRIPT=/root/alpha_1203/gva_factor_experiments/scripts/backtest_csi1000_quick_selected.py
RUN_BASE=/root/autodl-tmp/gva_factor_experiments/runs_newdata
BT_BASE=/root/autodl-tmp/gva_factor_experiments/backtests
SEQ_PID=1918

find_final() {
  local root=$1
  local seed=$2
  find "$root/reinforce_s${seed}" -name final_pool.json 2>/dev/null | head -1 || true
}

echo "[WATCH] start $(date '+%F %T')"
while true; do
  f0=$(find_final "$ROOT0" 0)
  f1=$(find_final "$ROOT12" 1)
  f2=$(find_final "$ROOT12" 2)
  echo "[WATCH] $(date '+%F %T') s0=${f0:-NA} s1=${f1:-NA} s2=${f2:-NA}"

  if [[ -n "$f0" ]] && kill -0 "$SEQ_PID" 2>/dev/null; then
    echo "[WATCH] seed0 done, killing sequential duplicate pid=$SEQ_PID"
    kill "$SEQ_PID" 2>/dev/null || true
  fi

  if [[ -n "$f0" && -n "$f1" && -n "$f2" ]]; then
    break
  fi
  sleep 120
done

STAMP=$(date +%Y%m%d_%H%M%S)
COMBINED="$RUN_BASE/csi1000_reinforce_combined_fast_${STAMP}"
mkdir -p "$COMBINED"
ln -sfn "$ROOT0/reinforce_s0" "$COMBINED/reinforce_s0"
ln -sfn "$ROOT12/reinforce_s1" "$COMBINED/reinforce_s1"
ln -sfn "$ROOT12/reinforce_s2" "$COMBINED/reinforce_s2"

OUT="$BT_BASE/csi1000_quick_with_reinforce_${STAMP}"
cd "$PROJECT"
export PYTHONPATH=.
echo "[WATCH] launching final backtest OUT=$OUT COMBINED=$COMBINED"
/root/miniconda3/bin/python -u "$SCRIPT" \
  --output_dir="$OUT" \
  --main_root="$MAIN_ROOT" \
  --ppo_filter_root="$PPO_FILTER_ROOT" \
  --gva_filter_root="$GVA_FILTER_ROOT" \
  --reinforce_root="$COMBINED" \
  --instruments=csi1000 \
  --device=cpu

/root/miniconda3/bin/python /root/alpha_1203/gva_factor_experiments/scripts/plot_backtest_from_reports.py "$OUT"
echo "[WATCH] done $(date '+%F %T') OUT=$OUT"
