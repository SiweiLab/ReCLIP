#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/scripts/peptide/figure_plot/logs"
WRAP_LOG="${LOG_DIR}/peptide_reclip_tmux_${RUN_TS}.log"
BATCH_SCRIPT="${ROOT_DIR}/scripts/peptide/ReCLIP/run_all_reclip_save.sh"
AUTOSTOP_INSTANCE="${AUTOSTOP_INSTANCE:-1}"

mkdir -p "${LOG_DIR}"
echo "[tmux] START run_all_reclip_save_and_shutdown | $(date '+%F %T')" | tee -a "${WRAP_LOG}"
echo "[tmux] ROOT_DIR=${ROOT_DIR}" | tee -a "${WRAP_LOG}"
echo "[tmux] RUN_TS=${RUN_TS}" | tee -a "${WRAP_LOG}"
echo "[tmux] AUTOSTOP_INSTANCE=${AUTOSTOP_INSTANCE}" | tee -a "${WRAP_LOG}"

cd "${ROOT_DIR}" || exit 2
bash "${BATCH_SCRIPT}"
rc=$?

echo "[tmux] BATCH_FINISHED rc=${rc} | $(date '+%F %T')" | tee -a "${WRAP_LOG}"
sync

if [[ "${AUTOSTOP_INSTANCE}" != "0" ]]; then
  echo "[tmux] stopping instance via shutdown -h now | $(date '+%F %T')" | tee -a "${WRAP_LOG}"
  if command -v sudo >/dev/null 2>&1; then
    sudo -n /usr/sbin/shutdown -h now >/dev/null 2>&1 && exit "${rc}"
    sudo -n shutdown -h now >/dev/null 2>&1 && exit "${rc}"
  fi
  /usr/sbin/shutdown -h now >/dev/null 2>&1 && exit "${rc}"
  shutdown -h now >/dev/null 2>&1 && exit "${rc}"
  systemctl poweroff -i >/dev/null 2>&1 && exit "${rc}"
  echo "[tmux] WARNING: failed to stop instance automatically | $(date '+%F %T')" | tee -a "${WRAP_LOG}"
fi

exit "${rc}"
