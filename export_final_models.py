# -*- coding: utf-8 -*-
"""
export_final_models.py
======================
Offline build step: turns the FINAL full-dataset LOOCV selections into the model
files the web server loads. Run it once (and again whenever the final models
change); the server never trains anything.

For each of the 6 tasks (Current density / H2 production rate x All / Acetate /
Complex) it reads results_stage3/<task>/ and

  1. takes the final meta-model (FINAL_winner.csv: model, MF level, hyper-parameters),
  2. takes the top-k baseline configurations that define that MF level
     (baseline_ranking.csv; MF1..MF6 = top 5, 10, ..., 30) with their
     hyper-parameters and BF feature sets (BF1..BF4 = top 6..3 FIS-ranked inputs),
  3. refits every one of those baselines on the ENTIRE dataset (log1p target),
  4. fits the meta-model on the stored out-of-fold meta-feature matrix
     (results_stage3/<task>/meta/<MF>/data.csv), exactly as in model selection,
  5. writes final_models/<task>.joblib, final_models/ranges.json (observed input
     ranges per task) and final_models/model_card.json (what is deployed, its
     LOOCV performance, library versions).

At prediction time a new input goes through the refitted baselines to create the
meta-features, then through the meta-model; see metahydropred/predictor.py.

Usage (use the same Python environment as the server -- pickles are version-bound):
    python export_final_models.py
    python export_final_models.py --results-dir results_stage3 --out-dir final_models
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
import sys

import shutil

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = HERE
sys.path.insert(0, REPO_ROOT)                      # models.py, common.py, backtransform.py

import sklearn                                     # noqa: E402
import xgboost, catboost, lightgbm                 # noqa: E402,F401
from backtransform import to_log_scale             # noqa: E402
from common import MF_LEVELS, load_bf_set, loocv_oof_predict_log   # noqa: E402
from models import build_estimator                 # noqa: E402

from metahydropred import config                   # noqa: E402

TARGET_COLUMN = {"CurrentDensity": "Current density", "H2Rate": "H2 production rate"}


def parse_params(text):
    return ast.literal_eval(text) if isinstance(text, str) else dict(text)


def build_task(task, results_dir):
    ds_dir = os.path.join(results_dir, task)
    target_key, substrate_key = config.TASKS[task]
    target_col = TARGET_COLUMN[target_key]

    # ---- final meta-model ------------------------------------------------
    winner = pd.read_csv(os.path.join(ds_dir, "FINAL_winner.csv")).iloc[0]
    meta_model, mf_level = winner["model"], winner["mf_level"]
    meta_params = parse_params(winner["params"])
    k = MF_LEVELS[mf_level]

    # ---- baselines that make up this MF level ---------------------------
    ranking = pd.read_csv(os.path.join(ds_dir, "baseline_ranking.csv")).head(k)
    cfg_path = os.path.join(REPO_ROOT, "datasets_config.json")
    bf_sets = {bf: load_bf_set(cfg_path, task, bf, results_dir) for bf in ("BF1", "BF2", "BF3", "BF4")}
    bf_columns = {bf: [c for c in df.columns if c != target_col] for bf, df in bf_sets.items()}   # training column order
    y = bf_sets["BF1"][target_col].to_numpy(dtype=float)
    y_log = to_log_scale(y)
    full = bf_sets["BF1"].drop(columns=[target_col])
    assert sorted(full.columns) == sorted(config.FEATURES), "unexpected feature columns"

    baselines = []
    for _, r in ranking.iterrows():
        tag = f"{r['model']}_{r['bf_set']}"
        cols = bf_columns[r["bf_set"]]
        est = build_estimator(r["model"], parse_params(r["params"]))
        est.fit(full[cols], y_log)                                    # refit on ENTIRE dataset
        baselines.append({"tag": tag, "model": r["model"], "bf_set": r["bf_set"],
                          "params": parse_params(r["params"]), "features": cols, "estimator": est})

    # ---- meta-features = leave-one-out out-of-fold predictions of the baselines (as in step 3) ----
    meta_data = pd.DataFrame({
        b["tag"]: loocv_oof_predict_log(b["model"], b["params"], bf_sets[b["bf_set"]][b["features"]], y_log)
        for b in baselines})
    meta_cols = list(meta_data.columns)
    meta_est = build_estimator(meta_model, meta_params)
    meta_est.fit(meta_data[meta_cols], y_log)

    # ---- observed input ranges ------------------------------------------
    ranges = {f: {"min": float(full[f].min()), "max": float(full[f].max())} for f in config.FEATURES}

    bundle = {
        "task": task, "target": target_key, "substrate": substrate_key,
        "target_column": target_col, "n_samples": int(len(full)),
        "meta": {"model": meta_model, "mf_level": mf_level, "params": meta_params,
                 "columns": meta_cols, "estimator": meta_est},
        "baselines": baselines, "ranges": ranges,
        "target_range": {"min": float(y.min()), "max": float(y.max())},
    }

    card = {
        "task": task, "label": config.task_label(task), "n_samples": int(len(full)),
        "meta_model": meta_model, "mf_level": mf_level, "n_baselines": k,
        "meta_params": {kk: (vv if isinstance(vv, (int, float, str, bool)) or vv is None else str(vv))
                        for kk, vv in meta_params.items()},
        "baselines": [{"tag": b["tag"], "bf_set": b["bf_set"], "n_inputs": len(b["features"])} for b in baselines],
        "loocv": {m: float(winner[f"LOOCV_{m}"]) for m in
                  ["RMSE_log1p", "MAE_log1p", "R2_log1p", "RMSE_orig", "MAE_orig", "R2_orig"]},
        "target_min": float(y.min()), "target_max": float(y.max()),
        "data_sha256": hashlib.sha256(pd.util.hash_pandas_object(full, index=False).values.tobytes()).hexdigest()[:16],
    }
    return bundle, ranges, card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results_stage3"))
    ap.add_argument("--out-dir", default=config.MODELS_DIR, help="where the final models are saved")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_ranges, cards = {}, {}
    for task in config.TASKS:
        bundle, ranges, card = build_task(task, args.results_dir)
        joblib.dump(bundle, os.path.join(args.out_dir, f"{task}.joblib"), compress=3)
        all_ranges[task], cards[task] = ranges, card
        print(f"[{task}] n={card['n_samples']}  meta={card['meta_model']}/{card['mf_level']}  "
              f"baselines={card['n_baselines']}  LOOCV RMSE(log1p)={card['loocv']['RMSE_log1p']:.4f}")

    # selection table (baseline vs meta-model per task) shipped with the models
    champion = os.path.join(args.results_dir, "champion_comparison.csv")      # written by aggregate_across_datasets.py
    if os.path.exists(champion):
        shutil.copyfile(champion, os.path.join(args.out_dir, "selection_summary.csv"))

    with open(os.path.join(args.out_dir, "ranges.json"), "w") as f:
        json.dump({"units": config.UNITS, "ranges": all_ranges}, f, indent=2)
    meta = {
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "libraries": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                      "xgboost": xgboost.__version__, "catboost": catboost.__version__,
                      "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "tasks": cards,
    }
    with open(os.path.join(args.out_dir, "model_card.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved 6 final models + ranges.json + model_card.json to {args.out_dir}")


if __name__ == "__main__":
    main()
