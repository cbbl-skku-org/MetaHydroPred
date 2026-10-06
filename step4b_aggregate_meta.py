# -*- coding: utf-8 -*-
"""
GIAI DOAN 3 - BUOC 4b: chay SAU KHI toan bo 114 job (19 model x 6 MF-level)
cua Buoc 4 hoan tat. Gom summary -> meta_ranking.csv (xep hang gop chung) +
meta_all19_by_MFlevel.csv (xep hang rieng tung MF-level). Dong rank=1 cua
meta_ranking.csv la CAU HINH THANG CUOC cuoi cung (MF-level + model) cho
dataset nay - ghi vao FINAL_winner.csv.

Cach dung:
    python step4b_aggregate_meta.py --dataset CurrentDensity_Acetate
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
    files = sorted(glob.glob(os.path.join(ds_dir, "meta", "MF*", "results", "_best", "*_summary.csv")))
    if not files:
        print(f"[{args.dataset}] Khong co summary meta nao.")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df_global = df.sort_values("LOOCV_RMSE_log1p").reset_index(drop=True)
    df_global.insert(0, "rank", range(1, len(df_global) + 1))
    out_path = os.path.join(ds_dir, "meta_ranking.csv")
    df_global.to_csv(out_path, index=False)

    df_by_mf = df.sort_values(["mf_level", "LOOCV_RMSE_log1p"]).reset_index(drop=True)
    df_by_mf["rank_within_mflevel"] = df_by_mf.groupby("mf_level")["LOOCV_RMSE_log1p"].rank(method="first").astype(int)
    by_mf_path = os.path.join(ds_dir, "meta_all19_by_MFlevel.csv")
    df_by_mf.to_csv(by_mf_path, index=False)

    winner = df_global.iloc[0]
    print(f"[{args.dataset}] Da tong hop {len(df)}/114 cau hinh meta")
    print(f"  -> {out_path}")
    print(f"  -> {by_mf_path}")
    print(f"  Cau hinh THANG CUOC: {winner['mf_level']} + {winner['model']} "
          f"| LOOCV_RMSE(log1p)={winner['LOOCV_RMSE_log1p']:.4f} "
          f"| LOOCV_R2(log1p)={winner['LOOCV_R2_log1p']:.4f} "
          f"| LOOCV_RMSE(goc)={winner['LOOCV_RMSE_orig']:.4f}")

    final_path = os.path.join(ds_dir, "FINAL_winner.csv")
    pd.DataFrame([winner]).to_csv(final_path, index=False)


if __name__ == "__main__":
    main()
