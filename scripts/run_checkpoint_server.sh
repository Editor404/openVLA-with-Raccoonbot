#!/usr/bin/env bash
set -euo pipefail

# Launch the RaccoonBot-finetuned OpenVLA server against a local checkpoint.
# Usage:
#   CHECKPOINT_DIR=/path/to/checkpoint ./scripts/run_checkpoint_server.sh
#
# Defaults assume this script is run from the repository root and the checkpoint
# directory is located at ../checkpoint, which matches the current workspace.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/../checkpoint}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEFAULT_UNNORM_KEY="${DEFAULT_UNNORM_KEY:-raccoon_pick_place}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/baseline_inference_server.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] CHECKPOINT_DIR does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_DIR}/dataset_statistics.json" ]]; then
  echo "[ERROR] dataset_statistics.json is missing from checkpoint: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}/openvla"
export PYTHONPATH="${REPO_ROOT}/openvla:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] repo_root=${REPO_ROOT}"
  echo "[DRY_RUN] checkpoint_dir=${CHECKPOINT_DIR}"
  echo "[DRY_RUN] log_file=${LOG_FILE}"
  echo "[DRY_RUN] python_bin=${PYTHON_BIN}"
  echo "[DRY_RUN] command: ${PYTHON_BIN} openvla_server.py --model_path ${CHECKPOINT_DIR} --default-unnorm-key ${DEFAULT_UNNORM_KEY} --host ${HOST} --port ${PORT} --device ${DEVICE}"
  exit 0
fi

{
  echo "[INFO] repo_root=${REPO_ROOT}"
  echo "[INFO] checkpoint_dir=${CHECKPOINT_DIR}"
  echo "[INFO] host=${HOST} port=${PORT} device=${DEVICE} default_unnorm_key=${DEFAULT_UNNORM_KEY}"
  "${PYTHON_BIN}" openvla_server.py \
    --model_path "${CHECKPOINT_DIR}" \
    --default-unnorm-key "${DEFAULT_UNNORM_KEY}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --device "${DEVICE}"
} 2>&1 | tee "${LOG_FILE}"
