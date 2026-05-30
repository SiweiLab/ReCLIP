#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FIG_DIR="${ROOT_DIR}/scripts/ptm/figure_plot"
LOG_DIR="${FIG_DIR}/logs"
CONDA_BIN="${CONDA_BIN:-/home/zhangzec/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mint}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
XGB_DEVICE="${XGB_DEVICE:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_PATH="${ROOT_DIR}/scripts/ptm/ReCLIP/esm2_ptm_reclip_prediction_save.py"
OUT_CSV="${FIG_DIR}/ReCLIP_ptm_for_figure.csv"
METRIC_TXT="${ROOT_DIR}/Results/result_ptm_reclip_XGB_tenfold_with_predictions.txt"
METRIC_JSON="${ROOT_DIR}/Results/result_ptm_reclip_XGB_tenfold_metadata.json"
BATCH_LOG="${LOG_DIR}/ptm_reclip_${RUN_TS}.log"
BATCH_PID_FILE="${LOG_DIR}/ptm_reclip_${RUN_TS}.pid"

mkdir -p "${FIG_DIR}" "${LOG_DIR}" "${ROOT_DIR}/Results"
echo $$ > "${BATCH_PID_FILE}"
echo "[ptm_reclip] START | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
echo "[ptm_reclip] PID_FILE ${BATCH_PID_FILE}" | tee -a "${BATCH_LOG}"
echo "[ptm_reclip] XGB_DEVICE=${XGB_DEVICE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${BATCH_LOG}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES

"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
  python "${SCRIPT_PATH}" \
  --classifier xgb \
  --xgb-device "${XGB_DEVICE}" \
  --out-csv "${OUT_CSV}" \
  --metric-txt "${METRIC_TXT}" \
  --metric-json "${METRIC_JSON}" 2>&1 | tee -a "${BATCH_LOG}"

rc=${PIPESTATUS[0]}
echo "[ptm_reclip] FINISHED rc=${rc} | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
exit "${rc}"
