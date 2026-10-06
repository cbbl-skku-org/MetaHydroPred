# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import pandas as pd
from backtransform import to_log_scale, to_original_scale
from sklearn.model_selection import LeaveOneOut, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import SEARCH_SPACE, build_estimator

# 8 model tree-based dung de tinh FIS-score (trung binh feature importance)
FIS_MODELS = ["AB", "CB", "DT", "ET", "GB", "LGBM", "RF", "XGB"]

# 6 muc meta-feature-set: ten -> so luong baseline model duoc lay lam cot
MF_LEVELS = {"MF1": 5, "MF2": 10, "MF3": 15, "MF4": 20, "MF5": 25, "MF6": 30}

_EPS = 1e-8


def get_target_col(config_path, dataset_name):
    with open(config_path) as f:
        config = json.load(f)
    for d in config["datasets"]:
        if d["name"] == dataset_name:
            return d["target"]
    raise ValueError(f"Dataset '{dataset_name}' khong co trong config {config_path}")


def compute_metrics(y_true, y_pred):
    """RMSE, MAE, R2, MAPE, SMAPE. Dung chung cho ca thang log1p va thang goc."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    denom_mape = np.maximum(np.abs(y_true), _EPS)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom_mape)) * 100)

    denom_smape = np.maximum(np.abs(y_true) + np.abs(y_pred), _EPS)
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denom_smape) * 100)

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape, "SMAPE": smape}


def loocv_oof_predict_log(model_name, params, X, y_log):
    """LOOCV tren TOAN BO X/y_log; tra ve mang du doan out-of-fold (thang log1p)."""
    n = len(y_log)
    preds = np.full(n, np.nan, dtype=float)
    loo = LeaveOneOut()
    X_arr = X.reset_index(drop=True)
    for tr_idx, te_idx in loo.split(X_arr):
        est = build_estimator(model_name, params)
        est.fit(X_arr.iloc[tr_idx], y_log[tr_idx])
        preds[te_idx[0]] = est.predict(X_arr.iloc[te_idx])[0]
    return preds


def run_full_grid(model_name, X, y_log, y_orig):
    """
    Chay TOAN BO grid hyperparameter cua model_name bang LOOCV tren TOAN BO
    dataset (khong con train/test). Voi moi combo:
      - LOOCV OOF predictions (log1p)
      - Metrics tren thang log1p (so voi y_log)
      - Metrics tren thang goc (sau expm1, so voi y_orig)

    Tra ve:
      rows: list[dict] - 2 dong/combo (Scale = "log1p scale" / "original scale")
      best: dict cua combo co LOOCV_RMSE (thang log1p) thap nhat - dung lam
            nguon meta-feature (OOF vector) / feature importance.
    """
    X = X.reset_index(drop=True)
    rows, best = [], None

    for params in ParameterGrid(SEARCH_SPACE[model_name]):
        try:
            oof_log = loocv_oof_predict_log(model_name, params, X, y_log)
            m_log = compute_metrics(y_log, oof_log)
            oof_orig = to_original_scale(oof_log)
            m_orig = compute_metrics(y_orig, oof_orig)
        except Exception:
            continue

        row_log = dict(params)
        row_log.update({f"LOOCV_{k}": v for k, v in m_log.items()})
        row_log["Scale"] = "log1p scale"
        rows.append(row_log)

        row_orig = dict(params)
        row_orig.update({f"LOOCV_{k}": v for k, v in m_orig.items()})
        row_orig["Scale"] = "original scale"
        rows.append(row_orig)

        if best is None or m_log["RMSE"] < best["loocv_log"]["RMSE"]:
            best = {
                "params": params, "oof_log": oof_log,
                "loocv_log": m_log, "loocv_orig": m_orig,
            }
    return rows, best


def write_grid_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
