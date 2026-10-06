# -*- coding: utf-8 -*-
"""
Prediction pipeline for the final, full-dataset LOOCV-selected MetaHydroPred models.

    validated input (6 parameters)
      -> task-specific baseline models (refit on the entire dataset)  => meta-features
      -> task-specific meta-model                                     => z_hat (log1p scale)
      -> backtransform.to_original_scale                              => y_hat (original units)

The back-transformation is defined once, in backtransform.py (repository root), and shared with the training scripts.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from backtransform import to_original_scale                # y_hat = exp(z_hat) - 1 (repository root)
from . import config
from .validation import range_flags


def check_environment(card):
    """Compare installed library versions with those the models were built with."""
    import importlib
    built = card.get("libraries", {})
    problems = []
    for lib, mod in (("scikit-learn", "sklearn"), ("xgboost", "xgboost"), ("catboost", "catboost"),
                     ("lightgbm", "lightgbm"), ("numpy", "numpy"), ("pandas", "pandas")):
        have = importlib.import_module(mod).__version__
        if lib in built and have != built[lib]:
            problems.append(f"{lib}: installed {have}, models built with {built[lib]}")
    return problems


class Predictor:
    def __init__(self, models_dir=None):
        self.dir = models_dir or config.MODELS_DIR
        self.bundles = {}
        for task in config.TASKS:
            path = os.path.join(self.dir, f"{task}.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing model file {path}. Run export_final_models.py first.")
            self.bundles[task] = joblib.load(path)
        with open(os.path.join(self.dir, "ranges.json")) as f:
            self.ranges = json.load(f)["ranges"]
        with open(os.path.join(self.dir, "model_card.json")) as f:
            self.card = json.load(f)
        self.env_problems = check_environment(self.card)

    # -------------------------------------------------------------- core ----
    def predict_log(self, task, X):
        """z_hat on the log1p scale for the rows of X (DataFrame with the 6 parameters)."""
        b = self.bundles[task]
        meta = pd.DataFrame(index=X.index)
        for base in b["baselines"]:
            meta[base["tag"]] = base["estimator"].predict(X[base["features"]])
        return np.asarray(b["meta"]["estimator"].predict(meta[b["meta"]["columns"]]), dtype=float)

    def predict_original(self, task, X):
        """Prediction in the original measurement units (back-transformed)."""
        return to_original_scale(self.predict_log(task, X))

    # ----------------------------------------------------- user-facing run --
    def run(self, X, tasks):
        """
        X: validated DataFrame (config.FEATURES columns). tasks: iterable of task keys.
        Returns (results DataFrame, list of per-task summaries).
        Result columns: row, the 6 inputs, then per task the prediction (original
        units) and an 'inside observed range' status.
        """
        out = X.copy()
        out.insert(0, "Row", np.arange(1, len(X) + 1))
        summaries = []
        for task in tasks:
            y = self.predict_original(task, X)
            flags = range_flags(X, self.ranges[task])
            col = config.task_column(task)
            out[col] = np.round(y, 4)
            out[f"Input range: {config.task_label(task)}"] = [
                "within observed range" if not f else "outside: " + "; ".join(f) for f in flags]
            neg = int((y < 0).sum())
            t_max = self.bundles[task]["target_range"]["max"]
            n_above = int((y > t_max).sum())
            summaries.append({
                "task": task, "label": config.task_label(task), "column": col,
                "n_outside_range": int(sum(1 for f in flags if f)),
                "n_negative": neg, "n_above_target_max": n_above, "target_max": t_max,
                "model": self.card["tasks"][task]["meta_model"],
                "mf_level": self.card["tasks"][task]["mf_level"],
            })
        return out, summaries
