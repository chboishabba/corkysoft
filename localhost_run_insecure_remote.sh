#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PID=""
SSH_KEY_PATH="${LOCALHOST_RUN_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_TARGET="${LOCALHOST_RUN_SSH_TARGET:-localhost.run}"

cleanup() {
    if [[ -n "${APP_PID}" ]] && kill -0 "${APP_PID}" 2>/dev/null; then
        kill "${APP_PID}" 2>/dev/null || true
        wait "${APP_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

# Force Streamlit onto the port expected by the localhost.run tunnel.
STREAMLIT_SERVER_PORT=8080 ./start_app.sh &
APP_PID=$!

sleep 8

if [[ ! -f "${SSH_KEY_PATH}" ]]; then
    echo "Missing SSH key: ${SSH_KEY_PATH}" >&2
    exit 1
fi

ssh -i "${SSH_KEY_PATH}" -R 80:localhost:8080 "${SSH_TARGET}"
