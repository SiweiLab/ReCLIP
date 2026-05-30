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
BATCH_LOG="${LOG_DIR}/reclip_batch_${RUN_TS}.log"
BATCH_PID_FILE="${LOG_DIR}/reclip_batch_${RUN_TS}.pid"

mkdir -p "${FIG_DIR}" "${LOG_DIR}"
echo $$ > "${BATCH_PID_FILE}"
echo "[reclip] START | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
echo "[reclip] PID_FILE ${BATCH_PID_FILE}" | tee -a "${BATCH_LOG}"
echo "[reclip] XGB_DEVICE=${XGB_DEVICE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${BATCH_LOG}"

validate_csv() {
  local ds="$1"
  python - "$ds" <<'PY'
import csv
import sys
from pathlib import Path

p = Path(sys.argv[1])
required = ["Set", "Epitope", "Sequence", "Hit"]
if not p.exists():
    print(f"[validate] missing file: {p}")
    sys.exit(2)
with p.open("r", newline="") as f:
    rd = csv.DictReader(f)
    cols = rd.fieldnames or []
    miss = [c for c in required if c not in cols]
    if miss:
        print(f"[validate] missing columns {miss} in {p}")
        sys.exit(2)
    bad = {c: 0 for c in required}
    rows = 0
    for row in rd:
        rows += 1
        for c in required:
            v = row.get(c, "")
            s = "" if v is None else str(v).strip().lower()
            if s in {"", "nan", "na", "none", "null"}:
                bad[c] += 1
    print(f"[validate] ok {p} rows={rows} bad={bad}")
    if any(bad.values()):
        sys.exit(3)
PY
}

DATASETS=(
  "${ROOT_DIR}/data/ClassI_Model/ClassI_crossval_HLA-A02:02_210.csv"
  "${ROOT_DIR}/data/ClassI_Model/ClassI_crossval_HLA-A32:01_210.csv"
  "${ROOT_DIR}/data/ClassI_Model/ClassI_crossval_HLA-B38:01_210.csv"
  "${ROOT_DIR}/data/ClassI_Model/ClassI_crossval_HLA-B40:02_210.csv"
  "${ROOT_DIR}/data/ClassI_Model/ClassI_crossval_HLA-C03:03_210.csv"
  "${ROOT_DIR}/data/ClassI_Model/ClassI_crossval_HLA-C05:01_210.csv"
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_crossval_HLA-A02:02_210.csv"
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_crossval_HLA-A32:01_210.csv"
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_crossval_HLA-B38:01_210.csv"
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_crossval_HLA-B40:02_210.csv"
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_crossval_HLA-C03:03_210.csv"
  "${ROOT_DIR}/data/MixedClass_Model/MixedClass_HLA-C05:01_210.csv"
)

fail_count=0
for ds in "${DATASETS[@]}"; do
  tag="$(basename "${ds}" .csv)"
  out_csv="${FIG_DIR}/ReCLIP_${tag}_XGB_for_figure.csv"
  out_txt="${FIG_DIR}/ReCLIP_${tag}_XGB_metrics.txt"
  out_json="${FIG_DIR}/ReCLIP_${tag}_XGB_metadata.json"
  job_log="${LOG_DIR}/reclip_${tag}_${RUN_TS}.log"

  echo "[reclip] PRECHECK ${tag} | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
  if ! validate_csv "${ds}" >> "${BATCH_LOG}" 2>&1; then
    echo "[reclip] FAIL precheck:${tag} | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
    fail_count=$((fail_count + 1))
    continue
  fi

  echo "[reclip] START ${tag} | $(date '+%F %T') | log=${job_log}" | tee -a "${BATCH_LOG}"
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
      python "${SCRIPT_PATH}" \
      --data-set "${ds}" \
      --classifier xgb \
      --xgb-device "${XGB_DEVICE}" \
      --out-csv "${out_csv}" \
      --metric-txt "${out_txt}" \
      --metric-json "${out_json}" > "${job_log}" 2>&1; then
    echo "[reclip] DONE ${tag} | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
  else
    rc=$?
    echo "[reclip] FAIL ${tag} | $(date '+%F %T') | rc=${rc} | log=${job_log}" | tee -a "${BATCH_LOG}"
    fail_count=$((fail_count + 1))
  fi
done

echo "[reclip] FINISHED | fail_count=${fail_count} | $(date '+%F %T')" | tee -a "${BATCH_LOG}"
exit "${fail_count}"
