# -*- coding: utf-8 -*-
"""
predict.py - predict current density and/or H2 production rate for NEW (unseen) data from the command line,
across any of the six tasks, with the final models in final_models/.

    python predict.py new_conditions.csv                         # all 6 tasks -> predictions.csv
    python predict.py new_conditions.csv -o out.csv --substrates Acetate --targets CurrentDensity
    python predict.py new_conditions.csv --tasks H2Rate_All,H2Rate_Complex

Input: a table with one header row and the six columns Temperature (°C), Substrate concentration, Reactor working
volume (mL), S/V ratio (m2/m3), Applied voltage (V), Cathode projected surface area (cm2); comma, semicolon or tab
separated. See examples/example_input.csv.
Output: the inputs plus one prediction column per task in the ORIGINAL measurement units (already back-transformed),
and an "Input range" column per task saying whether the row lies inside the observed training range.
Predictions are point estimates without prediction intervals; outside the observed ranges they are extrapolations.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from metahydropred import config                                  # noqa: E402
from metahydropred.predictor import Predictor                     # noqa: E402
from metahydropred.validation import InputError, parse_and_validate   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="CSV/TSV file with the six input parameters")
    ap.add_argument("-o", "--output", default="predictions.csv", help="output CSV (default: predictions.csv)")
    ap.add_argument("--tasks", help="comma-separated task names, e.g. CurrentDensity_All,H2Rate_Acetate (default: all)")
    ap.add_argument("--substrates", default="All,Acetate,Complex", help="subset of All,Acetate,Complex")
    ap.add_argument("--targets", default="CurrentDensity,H2Rate", help="subset of CurrentDensity,H2Rate")
    a = ap.parse_args(argv)

    if a.tasks:
        tasks = [t.strip() for t in a.tasks.split(",")]
        bad = [t for t in tasks if t not in config.TASKS]
        if bad:
            ap.error(f"unknown task(s) {bad}; choose from {list(config.TASKS)}")
    else:
        subs = [s.strip() for s in a.substrates.split(",")]
        tgts = [t.strip() for t in a.targets.split(",")]
        tasks = [f"{t}_{s}" for t in tgts for s in subs if f"{t}_{s}" in config.TASKS]
        if not tasks:
            ap.error("no valid task selected")

    try:
        with open(a.input, encoding="utf-8-sig") as f:
            X, notes = parse_and_validate(f.read())
    except FileNotFoundError:
        print(f"ERROR: input file '{a.input}' not found. To try the predictor, run:  "
              "python predict.py examples/example_input.csv", file=sys.stderr)
        return 2
    except (InputError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    results, summaries = Predictor().run(X, tasks)
    results.to_csv(a.output, index=False, encoding="utf-8-sig")

    print(f"{len(X)} row(s), {len(tasks)} task(s) -> {a.output}")
    for n in notes:
        print("NOTE:", n)
    for s in summaries:
        flag = f"  [{s['n_outside_range']} row(s) outside the observed input range]" if s["n_outside_range"] else ""
        print(f"  {s['label']}{flag}")
    print(config.LIMITED_DATA_NOTICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
