#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DEFAULT_PY_DIR='C:/Python310/PySpin'
PYTHON_CMD='python3.10'
WHEEL_FILE='spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl'
TARGET_PY_DIR="$DEFAULT_PY_DIR"
PUSHED=0

usage() {
    cat <<USAGE
Ensures the Spinnaker/PySpin environment is ready.

Usage: $SCRIPT_NAME [--dir PATH] [--python EXE] [--wheel FILE]

  --dir PATH      Directory that contains python.exe and the wheel (default: $DEFAULT_PY_DIR)
  --python EXE    Python executable to run the pip commands (default: $PYTHON_CMD)
  --wheel FILE    Wheel filename relative to the directory (default: $WHEEL_FILE)
  -h, --help      Show this help message and exit.
USAGE
}

err() {
    echo "[ERROR] $1" >&2
}

cleanup() {
    if [[ $PUSHED -eq 1 ]]; then
        popd >/dev/null || true
    fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            shift || { err "Missing value for --dir"; exit 64; }
            TARGET_PY_DIR="$1"
            ;;
        --python)
            shift || { err "Missing value for --python"; exit 64; }
            PYTHON_CMD="$1"
            ;;
        --wheel)
            shift || { err "Missing value for --wheel"; exit 64; }
            WHEEL_FILE="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown option: $1"
            echo "Run '$SCRIPT_NAME --help' for usage." >&2
            exit 64
            ;;
    esac
    shift
done

if [[ ! -d "$TARGET_PY_DIR" ]]; then
    err "Could not find directory: $TARGET_PY_DIR"
    exit 1
fi

if pushd "$TARGET_PY_DIR" >/dev/null; then
    PUSHED=1
else
    err "Unable to enter directory: $TARGET_PY_DIR"
    exit 1
fi

echo "[INFO] Using python command: $PYTHON_CMD"

run_cmd() {
    echo
    printf '> %s\n' "$*"
    "$@"
}

run_cmd "$PYTHON_CMD" -m ensurepip
run_cmd "$PYTHON_CMD" -m pip install --upgrade pip numpy matplotlib
run_cmd "$PYTHON_CMD" -m pip install Pillow==9.2.0

if [[ ! -f "$WHEEL_FILE" ]]; then
    err "Wheel not found: $TARGET_PY_DIR/$WHEEL_FILE"
    exit 1
fi
run_cmd "$PYTHON_CMD" -m pip install "$WHEEL_FILE"

echo
echo "[SUCCESS] PySpin installation commands completed."
