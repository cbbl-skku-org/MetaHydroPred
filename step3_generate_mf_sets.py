# -*- coding: utf-8 -*-
"""
GIAI DOAN 3 - BUOC 3: tu baseline_ranking.csv, xay 6 muc meta-feature-set
(MF1=top5 ... MF6=top30). Moi meta-feature = 1 cot OOF prediction (log1p,
tu LOOCV tren TOAN BO mau) cua 1 baseline model. KHONG con phan Test/holdout
rieng - chi 1 file duy nhat cho moi MF-level, chua toan bo mau.

Cach dung:
    python step3_generate_mf_sets.py CONFIG.json --dataset CurrentDensity_Acetate
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import get_target_col, MF_LEVELS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results-dir", default=os.path.join(BASE_DIR, "results_stage3"))
    args = ap.parse_args()

    target_col = get_target_col(args.config, args.dataset)
    ds_dir = os.path.join(args.results_dir, args.dataset)

    ranking = pd.read_csv(os.path.join(ds_dir, "baseline_ranking.csv"))
    best_dir = os.path.join(ds_dir, "baseline", "_best")

    # target (thang goc, se duoc log1p lai o Buoc 4) - lay tu file BF1 (du 6 feature)
    any_df = pd.read_csv(os.path.join(ds_dir, "BF1.csv"))
    y_all = any_df[target_col].to_numpy(dtype=float)

    for mf_name, k in MF_LEVELS.items():
        top = ranking.head(k)
        if len(top) < k:
            print(f"[canh bao] {mf_name}: chi co {len(top)}/{k} baseline model kha dung")

        cols_data, colnames = {}, []
        for _, r in top.iterrows():
            tag = f"{r['model']}_{r['bf_set']}"
            colnames.append(tag)
            cols_data[tag] = np.load(os.path.join(best_dir, f"{tag}_oof.npy"))

        mf_dir = os.path.join(ds_dir, "meta", mf_name)
        os.makedirs(mf_dir, exist_ok=True)

        meta_df = pd.DataFrame(cols_data)[colnames]
        meta_df[target_col] = y_all
        meta_df.to_csv(os.path.join(mf_dir, "data.csv"), index=False)
        print(f"[{args.dataset} {mf_name}] {len(colnames)} meta-feature -> {mf_dir}")


if __name__ == "__main__":
    main()
