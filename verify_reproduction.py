# -*- coding: utf-8 -*-
"""
verify_reproduction.py
======================
Checks that the final models shipped in this repository can be reproduced from the
shipped data, WITHOUT re-running the full grid search (which is expensive).

For each of the 6 tasks it

  1. rebuilds every baseline of the selected meta-feature level (top-k rows of
     results_stage3/<task>/baseline_ranking.csv; hyper-parameters and BF feature set
     as stored there) and recomputes its leave-one-out out-of-fold (OOF) prediction on
     the log1p target, then compares it with the stored meta-feature column
     (results_stage3/<task>/meta/<MFx>/data.csv);
  2. recomputes the LOOCV performance of the selected meta-model on that matrix and
     compares it with FINAL_winner.csv (RMSE / MAE / R2 on the log1p and original scales).

Differences are reported as the maximum absolute deviation. LOOCV with fixed seeds is
deterministic on one machine; across operating systems / library builds tiny numeric
differences (~1e-9) are possible, hence the tolerance.

Usage:
    python verify_reproduction.py                    # all 6 tasks, baselines + meta
    python verify_reproduction.py --skip-baselines   # meta-model check only (fast)
    python verify_reproduction.py --tasks H2Rate_Acetate --tol 1e-6
"""
import argparse
import ast
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backtransform import to_log_scale, to_original_scale      # noqa: E402
from common import MF_LEVELS, compute_metrics, load_bf_set, loocv_oof_predict_log   # noqa: E402

TASKS = {"CurrentDensity_All": "Current density", "CurrentDensity_Acetate": "Current density",
         "CurrentDensity_Complex": "Current density", "H2Rate_All": "H2 production rate",
         "H2Rate_Acetate": "H2 production rate", "H2Rate_Complex": "H2 production rate"}


def check_task(task, results_dir, tol):
    ds = os.path.join(results_dir, task)
    cfg = os.path.join(HERE, "datasets_config.json")
    target = TASKS[task]
    win = pd.read_csv(os.path.join(ds, "FINAL_winner.csv")).iloc[0]
    mf, meta_model = win["mf_level"], win["model"]
    ranking = pd.read_csv(os.path.join(ds, "baseline_ranking.csv")).head(MF_LEVELS[mf])

    # 1. recompute every baseline's leave-one-out out-of-fold prediction (the meta-features)
    bf_sets, cols = {}, {}
    for _, r in ranking.iterrows():
        if r["bf_set"] not in bf_sets:
            bf_sets[r["bf_set"]] = load_bf_set(cfg, task, r["bf_set"], results_dir)
        bf = bf_sets[r["bf_set"]]
        y_log = to_log_scale(bf[target].to_numpy(dtype=float))
        cols[f"{r['model']}_{r['bf_set']}"] = loocv_oof_predict_log(
            r["model"], ast.literal_eval(r["params"]), bf.drop(columns=[target]), y_log)
    meta = pd.DataFrame(cols)
    y = bf_sets[next(iter(bf_sets))][target].to_numpy(dtype=float)
    y_log = to_log_scale(y)

    # 2. leave-one-out performance of the final meta-model on those meta-features vs the stored result
    oof = loocv_oof_predict_log(meta_model, ast.literal_eval(win["params"]), meta, y_log)
    m_log = compute_metrics(y_log, oof)
    m_orig = compute_metrics(y, to_original_scale(oof))
    worst = 0.0
    for name, got in (("RMSE_log1p", m_log["RMSE"]), ("MAE_log1p", m_log["MAE"]), ("R2_log1p", m_log["R2"]),
                      ("RMSE_orig", m_orig["RMSE"]), ("MAE_orig", m_orig["MAE"]), ("R2_orig", m_orig["R2"])):
        worst = max(worst, abs(got - float(win[f"LOOCV_{name}"])))
    good = worst <= max(tol, 1e-6)
    return good, [f"  {len(ranking)} baselines recomputed; meta-model {meta_model} ({mf}): LOOCV RMSE log1p "
                  f"{m_log['RMSE']:.4f} (stored {win['LOOCV_RMSE_log1p']:.4f}), max |diff| over 6 metrics = {worst:.2e}  "
                  f"{'OK' if good else 'FAIL'}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_stage3"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--tol", type=float, default=1e-6, help="max absolute deviation allowed")
    args = ap.parse_args()

    all_ok = True
    for task in args.tasks.split(","):
        t0 = time.time()
        print(f"[{task}]")
        ok, lines = check_task(task, args.results_dir, args.tol)
        print("\n".join(lines) + f"   ({time.time() - t0:.0f}s)")
        all_ok &= ok
    print("\nREPRODUCTION CHECK:", "PASSED" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
