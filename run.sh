#!/bin/sh
# Launch the ESPI live viewer in its venv. Pass --demo for synthetic video.
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src
exec .venv/bin/python -m espi_live "$@"
