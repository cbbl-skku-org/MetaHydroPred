#!/bin/bash
# Check that the final models work in your environment:   ./check.sh          (seconds)
#                                                         ./check.sh --full   (+ reproduce the final models, ~2 min)
# Needs Python 3.9/3.10 with requirements.txt installed; use PYTHON=/path/to/python if it is not the first `python`.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONNOUSERSITE=1
$PY - <<'PYEOF'
import sys
v = sys.version_info[:2]
assert (3, 9) <= v <= (3, 10), f"Python {v[0]}.{v[1]} found; the saved models need Python 3.9 or 3.10 (see requirements.txt)"
PYEOF
echo "== 1/2 predictor self-check"; $PY -W ignore tests/self_check.py
echo "== 2/2 unit tests";          $PY -W ignore -m unittest discover -s tests
if [ "$1" = "--full" ]; then
  echo "== reproducing the final models from the shipped data"; $PY -W ignore verify_reproduction.py
fi
echo; echo "ALL CHECKS PASSED."
