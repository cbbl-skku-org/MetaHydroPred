# -*- coding: utf-8 -*-
"""
self_check.py - is the predictor in this folder working correctly?

1. loads the six final models and compares the installed library versions with the ones the models were built with;
2. predicts the example CSV for all six tasks and compares the result with examples/expected_predictions.csv
   (a stored reference produced from the validated models; tolerance 1e-4 relative);
3. checks the back-transformation.
Exit code 0 = everything correct. Run it after cloning / copying the folder, before anyone uses the server:

    python tests/self_check.py
    python tests/self_check.py --write-expected      # only the maintainers, after rebuilding the models
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
sys.path.insert(0, WEB)

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
from metahydropred import config                              # noqa: E402
from backtransform import to_original_scale                   # noqa: E402
from metahydropred.predictor import Predictor                 # noqa: E402
from metahydropred.validation import parse_and_validate       # noqa: E402

EXAMPLE = os.path.join(WEB, "examples", "example_input.csv")
EXPECTED = os.path.join(WEB, "examples", "expected_predictions.csv")
failures = []


def check(ok, text):
    print(("  OK    " if ok else "  FAIL  ") + text)
    if not ok:
        failures.append(text)


def main():
    p = Predictor()
    print(f"Loaded {len(p.bundles)} final models from {config.MODELS_DIR}")
    check(len(p.bundles) == 6, "six final models loaded")
    check(not p.env_problems, "library versions match the build" if not p.env_problems
          else "library versions differ: " + "; ".join(p.env_problems))

    X, _ = parse_and_validate(open(EXAMPLE).read())
    out, _ = p.run(X, list(config.TASKS))
    cols = [config.task_column(t) for t in config.TASKS]
    got = out[cols].reset_index(drop=True)

    if "--write-expected" in sys.argv:
        got.to_csv(EXPECTED, index=False)
        print(f"wrote reference predictions to {EXPECTED}")
        return 0

    exp = pd.read_csv(EXPECTED)
    for t, c in zip(config.TASKS, cols):
        ok = (c in exp.columns) and np.allclose(got[c].to_numpy(), exp[c].to_numpy(), rtol=1e-4, atol=1e-6)
        worst = float(np.max(np.abs(got[c].to_numpy() - exp[c].to_numpy()))) if c in exp.columns else float("nan")
        check(ok, f"{config.task_label(t):<38} matches reference (max |diff| = {worst:.2g})")

    z = p.predict_log("CurrentDensity_Acetate", X)
    check(np.allclose(p.predict_original("CurrentDensity_Acetate", X), np.exp(z) - 1) and
          np.allclose(to_original_scale(z), np.exp(z) - 1), "back-transformation is exp(z) - 1")

    print("\nSELF-CHECK:", "PASSED" if not failures else f"FAILED ({len(failures)} problem(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
