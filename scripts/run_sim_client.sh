#!/usr/bin/env bash
set -euo pipefail

# Run the MuJoCo OpenVLA client.
# Environment variables override the defaults below. Additional CLI arguments
# are appended, so they can override argparse options when needed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
CLIENT_DIR="${CLIENT_DIR:-${WORKSPACE_ROOT}/executeCode}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
XML_PATH="${XML_PATH:-${CLIENT_DIR}/Raccoon_colored_cylinder.xml}"
UNNORM_KEY="${UNNORM_KEY:-raccoon_pick_place}"
TARGET_COLOR="${TARGET_COLOR:-red}"
TARGET_OBJECT_TYPE="${TARGET_OBJECT_TYPE:-cylinder}"
OUTPUT_DIR="${OUTPUT_DIR:-${CLIENT_DIR}/rollout_outputs}"
EPISODE_ID="${EPISODE_ID:-1}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_DELTA_XYZ="${MAX_DELTA_XYZ:-0.005}"
USE_VIEWER="${USE_VIEWER:-1}"
DRY_RUN="${DRY_RUN:-0}"

CLIENT_SCRIPT="${CLIENT_DIR}/openvla_multicolor_client.py"

if [[ ! -f "${CLIENT_SCRIPT}" ]]; then
  echo "[ERROR] client script not found: ${CLIENT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${XML_PATH}" ]]; then
  echo "[ERROR] MuJoCo XML not found: ${XML_PATH}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" "${CLIENT_SCRIPT}"
  --server_url "${SERVER_URL}"
  --xml_path "${XML_PATH}"
  --unnorm_key "${UNNORM_KEY}"
  --target_color "${TARGET_COLOR}"
  --target_object_type "${TARGET_OBJECT_TYPE}"
  --output_dir "${OUTPUT_DIR}"
  --episode_id "${EPISODE_ID}"
  --max_steps "${MAX_STEPS}"
  --max_delta_xyz "${MAX_DELTA_XYZ}"
)

if [[ "${USE_VIEWER}" == "1" ]]; then
  cmd+=(--use_viewer)
fi

cmd+=("$@")

printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${CLIENT_DIR}"
exec "${cmd[@]}"
