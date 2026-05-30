#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
TASK="${1:-all}"
DEVICE="${DEVICE:-auto}"
XGB_DEVICE="${XGB_DEVICE:-auto}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/scripts/ablation/runs/layer}"
LAYERS="${LAYERS:-0 23 32}"
TOP_K="${TOP_K:-5}"

if [[ "${TASK}" != "all" && "${TASK}" != "mutation" && "${TASK}" != "ptm" ]]; then
  echo "Usage: $0 [all|mutation|ptm]" >&2
  exit 2
fi

run_mutation() {
  local layer="$1"
  local out_dir="${RUN_ROOT}/mutation/L${layer}"
  mkdir -p "${out_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/four_classes_mutation/ReCLIP/run_reclip_prediction_save.py"
    --classifier xgb
    --hf-layer "${layer}"
    --top-k "${TOP_K}"
    --device "${DEVICE}"
    --output-prefix "mutation_reclip_L${layer}_k${TOP_K}"
    --results-dir "${out_dir}/results"
    --out-csv "${out_dir}/reclip_mutation_predictions.csv"
  )
  if [[ "${FORCE_REBUILD}" == "1" ]]; then
    cmd+=(--force-rebuild)
  fi
  "${cmd[@]}"
}

run_ptm() {
  local layer="$1"
  local out_dir="${RUN_ROOT}/ptm/L${layer}"
  mkdir -p "${out_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/ptm/ReCLIP/esm2_ptm_reclip_prediction_save.py"
    --classifier xgb
    --hf-layer "${layer}"
    --top-k "${TOP_K}"
    --device "${DEVICE}"
    --xgb-device "${XGB_DEVICE}"
    --cache-root "${out_dir}/cache"
    --out-csv "${out_dir}/reclip_ptm_predictions.csv"
    --metric-txt "${out_dir}/reclip_ptm_metrics.txt"
    --metric-json "${out_dir}/reclip_ptm_metadata.json"
  )
  if [[ "${FORCE_REBUILD}" == "1" ]]; then
    cmd+=(--force-rebuild)
  fi
  "${cmd[@]}"
}

for layer in ${LAYERS}; do
  if [[ "${TASK}" == "all" || "${TASK}" == "mutation" ]]; then
    run_mutation "${layer}"
  fi
  if [[ "${TASK}" == "all" || "${TASK}" == "ptm" ]]; then
    run_ptm "${layer}"
  fi
done
