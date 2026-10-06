# -*- coding: utf-8 -*-
"""
GIAI DOAN 3 - BUOC 2: 1 JOB = 1 (dataset, BF-set, model). KHONG con --split
(vi khong con chia/lap lai). Doc <BF_set>.csv (toan bo mau) do Buoc 1 sinh,
chay TOAN BO grid hyperparameter bang LOOCV tren toan bo mau, ghi CSV 2 dong/
combo (log1p scale + original scale), luu OOF vector (log1p) cua combo tot
nhat de Buoc 3 dung lam meta-feature.

Ten file: <model>_<BF_set>.csv (vd DT_BF1.csv)

Cach dung:
    python step2_baseline_model.py CONFIG.json --dataset CurrentDensity_Acetate --bf-set BF1 --model DT
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
    ap.add_argument("--bf-set", required=True, choices=["BF1", "BF2", "BF3", "BF4"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-dir", default=os.path.join(BASE_DIR, "results_stage3"))
    args = ap.parse_args()

    target_col = get_target_col(args.config, args.dataset)
    ds_dir = os.path.join(args.results_dir, args.dataset)

    df = pd.read_csv(os.path.join(ds_dir, f"{args.bf_set}.csv"))
    y = df[target_col].to_numpy(dtype=float)
    X = df.drop(columns=[target_col])
    y_log = to_log_scale(y)

    rows, best = run_full_grid(args.model, X, y_log, y)

    out_dir = os.path.join(ds_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    write_grid_csv(os.path.join(out_dir, f"{args.model}_{args.bf_set}.csv"), rows)

    if best is not None:
        best_dir = os.path.join(out_dir, "_best")
        os.makedirs(best_dir, exist_ok=True)
        tag = f"{args.model}_{args.bf_set}"
        np.save(os.path.join(best_dir, f"{tag}_oof.npy"), best["oof_log"])
        pd.DataFrame([{
            "model": args.model, "bf_set": args.bf_set, "params": str(best["params"]),
            "LOOCV_RMSE_log1p": best["loocv_log"]["RMSE"], "LOOCV_MAE_log1p": best["loocv_log"]["MAE"],
            "LOOCV_R2_log1p": best["loocv_log"]["R2"], "LOOCV_MAPE_log1p": best["loocv_log"]["MAPE"],
            "LOOCV_SMAPE_log1p": best["loocv_log"]["SMAPE"],
            "LOOCV_RMSE_orig": best["loocv_orig"]["RMSE"], "LOOCV_MAE_orig": best["loocv_orig"]["MAE"],
            "LOOCV_R2_orig": best["loocv_orig"]["R2"], "LOOCV_MAPE_orig": best["loocv_orig"]["MAPE"],
            "LOOCV_SMAPE_orig": best["loocv_orig"]["SMAPE"],
        }]).to_csv(os.path.join(best_dir, f"{tag}_summary.csv"), index=False)
        print(f"[{args.dataset} {args.bf_set} {args.model}] {len(rows)//2} combo, "
              f"best LOOCV_RMSE(log1p)={best['loocv_log']['RMSE']:.4f}")
    else:
        print(f"[{args.dataset} {args.bf_set} {args.model}] LOI: khong co combo nao chay thanh cong")


if __name__ == "__main__":
    main()
