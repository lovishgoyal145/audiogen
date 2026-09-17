#!/usr/bin/env bash
# ==============================================================================
# AudioGen Unified One-Click Launch Script
#
# 1. Loads configuration from .env
# 2. Executes scripts/verify_env_and_auth.py pre-flight validation
# 3. Launches uvicorn gateway service ONLY when all pre-flight checks pass
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Resolve python interpreter (.venv/bin/python preferred)
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

echo "=========================================================="
echo " AudioGen Service Initialization"
echo " Project root: $PROJECT_ROOT"
echo " Python binary: $PYTHON_BIN"
echo "=========================================================="

# 1. Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "[start.sh] Loading environment variables from .env..."
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "[start.sh] Notice: .env file not found at $PROJECT_ROOT/.env"
fi

# 2. Execute automated pre-flight checks
echo "[start.sh] Executing pre-flight environment and authorization check..."
if ! "$PYTHON_BIN" "$PROJECT_ROOT/scripts/verify_env_and_auth.py"; then
    echo "" >&2
    echo "==========================================================" >&2
    echo " [FATAL ERROR] Pre-flight verification failed!" >&2
    echo " Server startup aborted to prevent erroneous state." >&2
    echo " Please resolve the error reported above and retry." >&2
    echo "==========================================================" >&2
    exit 1
fi

echo "=========================================================="
echo " Pre-flight check successful. Launching gateway on port 17000..."
echo "=========================================================="

# 3. Launch uvicorn gateway
exec "$PYTHON_BIN" -m uvicorn orchestrator.gateway:app --host 127.0.0.1 --port 17000
