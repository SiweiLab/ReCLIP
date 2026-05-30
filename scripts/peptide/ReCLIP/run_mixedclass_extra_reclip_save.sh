#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FIG_DIR="${ROOT_DIR}/scripts/peptide/figure_plot"
LOG_DIR="${FIG_DIR}/logs"
CONDA_BIN="${CONDA_BIN:-/home/zhangzec/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-mint}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
XGB_DEVICE="${XGB_DEVICE:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_PATH="${ROOT_DIR}/scripts/peptide/ReCLIP/esm2_peptide_reclip_crosspred_save.py"
BATCH_LOG="${LOG_DIR}/reclip_mixedclass_extra_${RUN_TS}.log"

mkdir -p "${FIG_DIR}" "${LOG_DIR}"
echo "[reclip_extra] START | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
echo "[reclip_extra] XGB_DEVICE=${XGB_DEVICE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${BATCH_LOG}"

DATASETS=(
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_crossval_HLA-A32:01_210.csv"
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_crossval_HLA-B38:01_210.csv"
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_crossval_HLA-C03:03_210.csv"
)

fail_count=0
for ds in "${DATASETS[@]}"; do
  tag="$(basename "${ds}" .csv)"
  out_csv="${FIG_DIR}/ReCLIP_${tag}_XGB_for_figure.csv"
  out_txt="${FIG_DIR}/ReCLIP_${tag}_XGB_metrics.txt"
  out_json="${FIG_DIR}/ReCLIP_${tag}_XGB_metadata.json"
  job_log="${LOG_DIR}/reclip_${tag}_${RUN_TS}.log"

  echo "[reclip_extra] START ${tag} | $(date '+%F %T') | log=${job_log}" | tee -a "${BATCH_LOG}"
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
      python "${SCRIPT_PATH}" \
      --data-set "${ds}" \
      --classifier xgb \
      --xgb-device "${XGB_DEVICE}" \
      --out-csv "${out_csv}" \
      --metric-txt "${out_txt}" \
      --metric-json "${out_json}" > "${job_log}" 2>&1; then
    echo "[reclip_extra] DONE ${tag} | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
  else
    rc=$?
    echo "[reclip_extra] FAIL ${tag} | $(date '+%F %T') | rc=${rc} | log=${job_log}" | tee -a "${BATCH_LOG}"
    fail_count=$((fail_count + 1))
  fi
done

echo "[reclip_extra] FINISHED | fail_count=${fail_count} | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
exit "${fail_count}"
