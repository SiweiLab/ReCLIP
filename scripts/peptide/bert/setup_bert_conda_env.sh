#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/home/zhangzec/miniconda3/bin/conda}"
TARGET_ENV="${TARGET_ENV:-bert-peptide}"
BASE_ENV="${BASE_ENV:-mint}"

echo "[setup_bert_env] conda=${CONDA_BIN}"
echo "[setup_bert_env] target_env=${TARGET_ENV} base_env=${BASE_ENV}"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | rg -x "${TARGET_ENV}" >/dev/null 2>&1; then
  if "${CONDA_BIN}" env list | awk '{print $1}' | rg -x "${BASE_ENV}" >/dev/null 2>&1; then
    echo "[setup_bert_env] cloning from ${BASE_ENV} -> ${TARGET_ENV}"
    "${CONDA_BIN}" create -y -n "${TARGET_ENV}" --clone "${BASE_ENV}"
  else
    echo "[setup_bert_env] base env ${BASE_ENV} not found, creating from package pins"
    "${CONDA_BIN}" create -y -n "${TARGET_ENV}" python=3.8 pip
    "${CONDA_BIN}" run --no-capture-output -n "${TARGET_ENV}" pip install --upgrade pip
    "${CONDA_BIN}" run --no-capture-output -n "${TARGET_ENV}" pip install \
      torch==1.12.1 \
      transformers==4.25.1 \
      datasets==2.13.2 \
      tokenizers==0.13.3 \
      numpy==1.21.2 \
      pandas==1.3.5 \
      scikit-learn==1.0.2 \
      xgboost==1.6.2 \
      gensim==4.2.0 \
      tqdm \
      wandb \
      sentencepiece
  fi
else
  echo "[setup_bert_env] target env already exists: ${TARGET_ENV}"
fi

echo "[setup_bert_env] validating imports in ${TARGET_ENV}"
"${CONDA_BIN}" run --no-capture-output -n "${TARGET_ENV}" python - <<'PY'
import importlib

mods = ["torch", "transformers", "datasets", "tokenizers", "numpy", "pandas", "sklearn", "xgboost", "gensim"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"{m}=={getattr(mod, '__version__', 'unknown')}")
PY

echo "[setup_bert_env] done"
