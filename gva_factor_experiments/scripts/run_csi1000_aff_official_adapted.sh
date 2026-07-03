#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/root/autodl-tmp/aff_official_workspace_csi1000_$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="/root/alpha_1203/gva_factor_experiments/scripts"
ALPHAFORGE_ROOT="/root/alpha_1203/AlphaForge-master"
LOG_DIR="${WORKDIR}/logs"

mkdir -p "${WORKDIR}" "${LOG_DIR}"
cd "${WORKDIR}"

echo "[INFO] AFF official-adapted CSI500 launcher started at $(date '+%F %T')"

PYTHONPATH="${ALPHAFORGE_ROOT}" /root/miniconda3/bin/python -u "${SCRIPT_DIR}/train_AFF_official_adapted.py" \
  --instruments=csi1000 \
  --train_end_year=2023 \
  --seeds="[0]" \
  --save_name=affofficial_csi1000 \
  --zoo_size=100 \
  --cuda=0 \
  --corr_thresh=0.7 \
  --ic_thresh=0.03 \
  --icir_thresh=0.1 \
  2>&1 | tee "${LOG_DIR}/aff_official_stage1.log"

PYTHONPATH="${ALPHAFORGE_ROOT}" /root/miniconda3/bin/python -u "${SCRIPT_DIR}/combine_AFF_official_adapted.py" \
  --instruments=csi1000 \
  --train_end_year=2023 \
  --seeds="[0]" \
  --save_name=affofficial_csi1000 \
  --n_factors=10 \
  --window=inf \
  --cuda=0 \
  2>&1 | tee "${LOG_DIR}/aff_official_stage2.log"

echo "[INFO] AFF official-adapted CSI500 finished at $(date '+%F %T')"
