#!/usr/bin/env bash
set -euo pipefail

PY_DIR='C:/Python310/PySpin'
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_SCRIPT="$REPO_ROOT/third_party/flir-spinnaker-4.2/Examples/Python3/AcquireAndDisplay.py"

if [[ ! -d "$PY_DIR" ]]; then
    echo "[ERROR] Directory not found: $PY_DIR" >&2
    exit 1
fi

if [[ ! -f "$RAW_SCRIPT" ]]; then
    echo "[ERROR] Example not found: $RAW_SCRIPT" >&2
    exit 1
fi

SCRIPT="$RAW_SCRIPT"
if command -v cygpath >/dev/null 2>&1; then
    SCRIPT="$(cygpath -w "$RAW_SCRIPT")"
fi

pushd "$PY_DIR" >/dev/null

echo "Running AcquireAndDisplay from $PY_DIR"
py -3.10 "$SCRIPT" "$@"

popd >/dev/null
