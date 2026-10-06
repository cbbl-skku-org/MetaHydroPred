# -*- coding: utf-8 -*-
"""
GIAI DOAN 3 - BUOC 2b: chay SAU KHI toan bo 76 job cua Buoc 2 hoan tat.
Gom *_summary.csv -> baseline_ranking.csv (xep hang gop chung theo
LOOCV_RMSE_log1p) + baseline_all19_by_BFset.csv (xep hang rieng tung BF-set).

Cach dung:
    python step2b_aggregate_baseline.py --dataset CurrentDensity_Acetate
"""
import argparse
import glob
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results-dir", default=os.path.join(BASE_DIR, "results_stage3"))
    args = ap.parse_args()

    ds_dir = os.path.join(args.results_dir, args.dataset)
    best_dir = os.path.join(ds_dir, "baseline", "_best")
    files = sorted(glob.glob(os.path.join(best_dir, "*_summary.csv")))
    if not files:
        print(f"[{args.dataset}] Khong tim thay summary nao trong {best_dir}")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df_global = df.sort_values("LOOCV_RMSE_log1p").reset_index(drop=True)
    df_global.insert(0, "rank", range(1, len(df_global) + 1))
    out_path = os.path.join(ds_dir, "baseline_ranking.csv")
    df_global.to_csv(out_path, index=False)

    df_by_bf = df.sort_values(["bf_set", "LOOCV_RMSE_log1p"]).reset_index(drop=True)
    df_by_bf["rank_within_bfset"] = df_by_bf.groupby("bf_set")["LOOCV_RMSE_log1p"].rank(method="first").astype(int)
    by_bf_path = os.path.join(ds_dir, "baseline_all19_by_BFset.csv")
    df_by_bf.to_csv(by_bf_path, index=False)

    print(f"[{args.dataset}] Da tong hop {len(df)}/76 baseline model")
    print(f"  -> {out_path}")
    print(f"  -> {by_bf_path}")
    print(df_global[["rank", "bf_set", "model", "LOOCV_RMSE_log1p", "LOOCV_R2_log1p"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
