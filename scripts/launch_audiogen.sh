#!/usr/bin/env bash
# ==============================================================================
# AudioGen Desktop Launcher Script
#
# 1. Single-instance awareness: if already running on port 17000, opens browser
# 2. Background gateway launch via setsid
# 3. Waits for HTTP 200 on port 17000
# 4. Opens default browser at http://127.0.0.1:17000
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 1. Single-instance check on port 17000
if curl -sf http://127.0.0.1:17000/session/status >/dev/null 2>&1; then
    echo "[launcher] AudioGen is already active on port 17000. Opening UI in browser..."
    xdg-open "http://127.0.0.1:17000" >/dev/null 2>&1 || sensible-browser "http://127.0.0.1:17000" >/dev/null 2>&1 || true
    exit 0
fi

# Resolve python interpreter (.venv/bin/python preferred)
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

LOG_FILE="$PROJECT_ROOT/scratch/audiogen_launcher.log"
mkdir -p "$PROJECT_ROOT/scratch"

echo "[launcher] Launching AudioGen gateway on port 17000..."
if command -v setsid >/dev/null 2>&1; then
    setsid "$PYTHON_BIN" -m uvicorn orchestrator.gateway:app --host 127.0.0.1 --port 17000 > "$LOG_FILE" 2>&1 &
else
    "$PYTHON_BIN" -m uvicorn orchestrator.gateway:app --host 127.0.0.1 --port 17000 > "$LOG_FILE" 2>&1 &
fi
GATEWAY_PID=$!
echo "$GATEWAY_PID" > "$PROJECT_ROOT/scratch/gateway.pid"

echo "[launcher] Gateway launched with PID $GATEWAY_PID. Waiting for port 17000..."

# Wait up to 10 seconds for http://127.0.0.1:17000/session/status
READY=false
for i in {1..20}; do
    if curl -sf http://127.0.0.1:17000/session/status >/dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 0.5
done

if [ "$READY" = false ]; then
    echo "[launcher] Timeout waiting for AudioGen gateway on port 17000." >&2
    exit 1
fi

echo "[launcher] AudioGen gateway ready. Opening browser..."
xdg-open "http://127.0.0.1:17000" >/dev/null 2>&1 || sensible-browser "http://127.0.0.1:17000" >/dev/null 2>&1 || true
