# -*- coding: utf-8 -*-
"""
GIAI DOAN 3 - BUOC 1: KHONG con chia 80/20. Voi TOAN BO dataset:
  1. Tinh FIS-score: trung binh feature importance (chuan hoa tong=1) tu 8
     model tree-based (AB, CB, DT, ET, GB, LGBM, RF, XGB), moi model duoc
     TUNE bang LOOCV grid-search (chon combo LOOCV_RMSE thang log1p thap
     nhat) tren TOAN BO du lieu, roi moi lay feature_importances_.
  2. Xep hang 6 feature -> BF1 (ca 6), BF2 (top5), BF3 (top4), BF4 (top3)
     -> luu thanh file CSV (KHONG con "_dev"/"_test", chi 1 file/BF-set
     chua toan bo mau).

Output (moi dataset):
  <results_dir>/<dataset>/
      feature_ranking.csv
      fis_tuning_log.csv
      BF1.csv / BF2.csv / BF3.csv / BF4.csv

Cach dung:
    python step1_generate_bf_sets.py CONFIG.json [--only DS1,DS2]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from backtransform import to_log_scale, to_original_scale

from models import build_estimator
from common import FIS_MODELS, run_full_grid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_fis_ranking(X, y_log, tuning_log_path):
    scores = np.zeros(X.shape[1])
    tuning_rows = []

    for model_name in FIS_MODELS:
        # Chi dung de tune (LOOCV) + lay feature_importances_; tham so y_orig
        # truyen vao chi de khop chu ky ham run_full_grid, khong dung ket qua
        # thang goc o day.
        _, best = run_full_grid(model_name, X, y_log, to_original_scale(y_log))
        if best is None:
            continue
        est = build_estimator(model_name, best["params"])
        est.fit(X.reset_index(drop=True), y_log)
        importances = np.asarray(est.feature_importances_, dtype=float)
        if importances.sum() > 0:
            importances = importances / importances.sum()
        scores += importances
        tuning_rows.append({
            "fis_model": model_name,
            "best_params": json.dumps(best["params"]),
            "LOOCV_RMSE_log1p": best["loocv_log"]["RMSE"],
            **{f"importance_{c}": v for c, v in zip(X.columns, importances)},
        })

    scores /= len(FIS_MODELS)
    pd.DataFrame(tuning_rows).to_csv(tuning_log_path, index=False)

    ranking = sorted(zip(X.columns, scores), key=lambda x: x[1], reverse=True)
    ranked_cols = [c for c, _ in ranking]
    return ranked_cols, dict(ranking)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", default=None, help="Ten dataset can chay, cach nhau boi dau phay")
    ap.add_argument("--results-dir", default=os.path.join(BASE_DIR, "results_stage3"))
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    only = set(args.only.split(",")) if args.only else None

    for ds in config["datasets"]:
        ds_name = ds["name"]
        if only and ds_name not in only:
            continue
        target_col = ds["target"]
        full_path = os.path.normpath(os.path.join(BASE_DIR, ds["feature_sets"]["Set1"]["full_path"]))
        df = pd.read_csv(full_path)
        y_all = df[target_col].to_numpy(dtype=float)
        X_all = df.drop(columns=[target_col]).reset_index(drop=True)
        y_all_log = to_log_scale(y_all)

        out_dir = os.path.join(args.results_dir, ds_name)
        os.makedirs(out_dir, exist_ok=True)

        ranked_cols, fis_scores = compute_fis_ranking(
            X_all, y_all_log, os.path.join(out_dir, "fis_tuning_log.csv")
        )

        bf_sets = {
            "BF1": ranked_cols[:6],
            "BF2": ranked_cols[:5],
            "BF3": ranked_cols[:4],
            "BF4": ranked_cols[:3],
        }

        pd.DataFrame(
            [{"feature": c, "FIS_score": fis_scores[c]} for c in ranked_cols]
        ).to_csv(os.path.join(out_dir, "feature_ranking.csv"), index=False)

        for bf_name, cols in bf_sets.items():
            bf_df = X_all[cols].copy()
            bf_df[target_col] = y_all
            bf_df.to_csv(os.path.join(out_dir, f"{bf_name}.csv"), index=False)

        print(f"[{ds_name}] N={len(df)} mau | " +
              ", ".join(f"{k}={v}" for k, v in bf_sets.items()))


if __name__ == "__main__":
    main()
