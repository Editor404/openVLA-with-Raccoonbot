#!/usr/bin/env bash
set -euo pipefail

# Run the OpenVLA client against both MuJoCo and the physical RaccoonBot.
# Hardware motion requires the explicit confirmation:
#   USE_REAL_ROBOT=1 ./scripts/run_real_robot_client.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
CLIENT_DIR="${CLIENT_DIR:-${WORKSPACE_ROOT}/executeCode}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
XML_PATH="${XML_PATH:-${CLIENT_DIR}/Raccoon_colored_cylinder.xml}"
UNNORM_KEY="${UNNORM_KEY:-raccoon_pick_place}"
TARGET_COLOR="${TARGET_COLOR:-red}"
OUTPUT_DIR="${OUTPUT_DIR:-${CLIENT_DIR}/real_robot_rollout_outputs}"
EPISODE_ID="${EPISODE_ID:-1}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_DELTA_XYZ="${MAX_DELTA_XYZ:-0.005}"
USE_VIEWER="${USE_VIEWER:-1}"
USE_REAL_ROBOT="${USE_REAL_ROBOT:-0}"
REAL_GO_HOME_ON_EXIT="${REAL_GO_HOME_ON_EXIT:-1}"
DRY_RUN="${DRY_RUN:-0}"

CLIENT_SCRIPT="${CLIENT_DIR}/openvla_multicolor_client_real_robot.py"

if [[ ! -f "${CLIENT_SCRIPT}" ]]; then
  echo "[ERROR] real-robot client script not found: ${CLIENT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${XML_PATH}" ]]; then
  echo "[ERROR] MuJoCo XML not found: ${XML_PATH}" >&2
  exit 1
fi

if [[ "${USE_REAL_ROBOT}" != "1" && "${DRY_RUN}" != "1" ]]; then
  echo "[ERROR] physical robot execution is disabled." >&2
  echo "        Re-run with USE_REAL_ROBOT=1 after checking the robot workspace." >&2
  exit 2
fi

cmd=(
  "${PYTHON_BIN}" "${CLIENT_SCRIPT}"
  --server_url "${SERVER_URL}"
  --xml_path "${XML_PATH}"
  --unnorm_key "${UNNORM_KEY}"
  --target_color "${TARGET_COLOR}"
  --output_dir "${OUTPUT_DIR}"
  --episode_id "${EPISODE_ID}"
  --max_steps "${MAX_STEPS}"
  --max_delta_xyz "${MAX_DELTA_XYZ}"
  --use_real_robot
)

if [[ "${USE_VIEWER}" == "1" ]]; then
  cmd+=(--use_viewer)
fi

if [[ "${REAL_GO_HOME_ON_EXIT}" == "1" ]]; then
  cmd+=(--real_go_home_on_exit)
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
