#!/bin/bash
# Run from the repository root after cloning:   ./check_repository.sh          (or  --full)
#   default : checks the web-server predictor (self-check against stored reference predictions + unit tests)
#   --full  : additionally reproduces the final models from the shipped data (verify_reproduction.py, ~2 min)
# Use a Python 3.9/3.10 environment with requirements.txt installed:  PYTHON=/path/to/python ./check_repository.sh
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONNOUSERSITE=1
PYTHON="$PY" ./webserver/check.sh
if [ "$1" = "--full" ]; then
  echo "== reproducing the final models from the shipped data"
  $PY -W ignore verify_reproduction.py
fi
echo; echo "REPOSITORY CHECK PASSED."
