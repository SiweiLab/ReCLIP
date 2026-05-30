#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FIG_DIR="${ROOT_DIR}/scripts/peptide/figure_plot"
SCRIPT_PATH="${ROOT_DIR}/scripts/peptide/SWING/cross_pred_save.py"
ENV_NAME="mint"
CONDA_BIN="/home/zhangzec/miniconda3/bin/conda"

mkdir -p "${FIG_DIR}"

DATASETS=(
  "${ROOT_DIR}/data/ClassI_Model/ClassI_denovo_H-2-IAg7_210.csv"
  "${ROOT_DIR}/data/ClassI_Model/ClassI_denovo_H-2-IEk_210.csv"
  "${ROOT_DIR}/data/ClassII_Model/ClassII_denovo_H-2-IAg7_210.csv"
  "${ROOT_DIR}/data/ClassII_Model/ClassII_denovo_H-2-IEk_210.csv"
)

fail_count=0

for ds in "${DATASETS[@]}"; do
  tag="$(basename "${ds}" .csv)"
  out_csv="${FIG_DIR}/SWING_${tag}_for_figure.csv"
  out_txt="${FIG_DIR}/SWING_${tag}_metrics.txt"

  echo "[swing] START ${tag} | $(date '+%F %T')"
  if "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python "${SCRIPT_PATH}" \
    --data_set "${ds}" \
    --classifier XGBoost \
    --out_csv "${out_csv}" \
    --metric_txt "${out_txt}"; then
    echo "[swing] DONE  ${tag} | $(date '+%F %T')"
  else
    echo "[swing] FAIL  ${tag} | $(date '+%F %T')"
    fail_count=$((fail_count + 1))
  fi
done

echo "[swing] finished with fail_count=${fail_count}"
exit "${fail_count}"
