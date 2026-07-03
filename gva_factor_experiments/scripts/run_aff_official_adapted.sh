#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/root/autodl-tmp/aff_official_workspace_20260629"
SCRIPT_DIR="/root/alpha_1203/gva_factor_experiments/scripts"
ALPHAFORGE_ROOT="/root/alpha_1203/AlphaForge-master"
LOG_DIR="${WORKDIR}/logs"

mkdir -p "${WORKDIR}" "${LOG_DIR}"
cd "${WORKDIR}"

echo "[INFO] AFF official-adapted launcher started at $(date '+%F %T')"
echo "[INFO] Waiting for current GVA jobs if they are still alive: 2430 2432 2434"
for pid in 2430 2432 2434; do
  if kill -0 "${pid}" 2>/dev/null; then
    while kill -0 "${pid}" 2>/dev/null; do
      sleep 60
    done
  fi
done

echo "[INFO] Starting AFF stage1 at $(date '+%F %T')"
PYTHONPATH="${ALPHAFORGE_ROOT}" /root/miniconda3/bin/python -u "${SCRIPT_DIR}/train_AFF_official_adapted.py" \
  --instruments=csi300 \
  --train_end_year=2023 \
  --seeds="[0]" \
  --save_name=affofficial \
  --zoo_size=100 \
  --cuda=0 \
  --corr_thresh=0.7 \
  --ic_thresh=0.03 \
  --icir_thresh=0.1 \
  2>&1 | tee "${LOG_DIR}/aff_official_stage1.log"

echo "[INFO] Starting AFF stage2 at $(date '+%F %T')"
PYTHONPATH="${ALPHAFORGE_ROOT}" /root/miniconda3/bin/python -u "${SCRIPT_DIR}/combine_AFF_official_adapted.py" \
  --instruments=csi300 \
  --train_end_year=2023 \
  --seeds="[0]" \
  --save_name=affofficial \
  --n_factors=10 \
  --window=inf \
  --cuda=0 \
  2>&1 | tee "${LOG_DIR}/aff_official_stage2.log"

echo "[INFO] AFF official-adapted finished at $(date '+%F %T')"
