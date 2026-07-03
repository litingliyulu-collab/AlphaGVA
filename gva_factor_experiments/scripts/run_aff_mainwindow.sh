#!/usr/bin/env bash
set -euo pipefail
WORKSPACE=/root/autodl-tmp/aff_official_workspace_20260629
LOGDIR=$WORKSPACE/logs
mkdir -p "$LOGDIR"
exec > "$LOGDIR/aff_mainwindow.log" 2>&1
cd "$WORKSPACE"
echo "[START] $(date '+%F %T')"
PYTHONPATH=/root/alpha_1203/AlphaForge-master \
/root/miniconda3/bin/python -u /root/alpha_1203/gva_factor_experiments/scripts/evaluate_aff_main_window.py \
  --workspace=/root/autodl-tmp/aff_official_workspace_20260629 \
  --instruments=csi300 \
  --train-end-year=2023 \
  --seed=0 \
  --test-start=2024-01-02 \
  --test-end=2026-05-14 \
  --cuda=0
STATUS=$?
echo "[END] status=$STATUS time=$(date '+%F %T')"
exit $STATUS

