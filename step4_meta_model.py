# -*- coding: utf-8 -*-
"""
GIAI DOAN 3 - BUOC 4: 1 JOB = 1 (dataset, MF-level, model) - CA 19 MODEL
(khong chi EN/ET/XGB nhu hinh goc). Doc data.csv cua tung MF-level (toan
bo mau), chay TOAN BO grid hyperparameter bang LOOCV, ghi CSV 2 dong/combo
(log1p + original scale).

Ten file: <model>_<MF_level>.csv (vd RF_MF3.csv)

Cach dung:
    python step4_meta_model.py CONFIG.json --dataset CurrentDensity_Acetate --mf-level MF3 --model RF
"""
import argparse
import os

import numpy as np
import pandas as pd
from backtransform import to_log_scale, to_original_scale

from common import get_target_col, run_full_grid, write_grid_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mf-level", required=True, choices=["MF1", "MF2", "MF3", "MF4", "MF5", "MF6"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-dir", default=os.path.join(BASE_DIR, "results_stage3"))
    args = ap.parse_args()

    target_col = get_target_col(args.config, args.dataset)
    mf_dir = os.path.join(args.results_dir, args.dataset, "meta", args.mf_level)

    df = pd.read_csv(os.path.join(mf_dir, "data.csv"))
    y = df[target_col].to_numpy(dtype=float)
    X = df.drop(columns=[target_col])
    y_log = to_log_scale(y)

    rows, best = run_full_grid(args.model, X, y_log, y)

    out_dir = os.path.join(mf_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    write_grid_csv(os.path.join(out_dir, f"{args.model}_{args.mf_level}.csv"), rows)

    if best is not None:
        best_dir = os.path.join(out_dir, "_best")
        os.makedirs(best_dir, exist_ok=True)
        pd.DataFrame([{
            "model": args.model, "mf_level": args.mf_level, "params": str(best["params"]),
            "LOOCV_RMSE_log1p": best["loocv_log"]["RMSE"], "LOOCV_MAE_log1p": best["loocv_log"]["MAE"],
            "LOOCV_R2_log1p": best["loocv_log"]["R2"], "LOOCV_MAPE_log1p": best["loocv_log"]["MAPE"],
            "LOOCV_SMAPE_log1p": best["loocv_log"]["SMAPE"],
            "LOOCV_RMSE_orig": best["loocv_orig"]["RMSE"], "LOOCV_MAE_orig": best["loocv_orig"]["MAE"],
            "LOOCV_R2_orig": best["loocv_orig"]["R2"], "LOOCV_MAPE_orig": best["loocv_orig"]["MAPE"],
            "LOOCV_SMAPE_orig": best["loocv_orig"]["SMAPE"],
        }]).to_csv(os.path.join(best_dir, f"{args.model}_{args.mf_level}_summary.csv"), index=False)
        print(f"[{args.dataset} {args.mf_level} {args.model}] {len(rows)//2} combo, "
              f"best LOOCV_RMSE(log1p)={best['loocv_log']['RMSE']:.4f}")
    else:
        print(f"[{args.dataset} {args.mf_level} {args.model}] LOI: khong co combo nao chay thanh cong")


if __name__ == "__main__":
    main()
