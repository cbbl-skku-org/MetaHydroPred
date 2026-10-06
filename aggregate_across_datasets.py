# -*- coding: utf-8 -*-
"""
Gop ket qua cua GIAI DOAN 3 (khong chia 80/20, LOOCV tren toan bo du lieu,
KHONG co "20 lan lap") qua CA 6 DATASET (CurrentDensity_All/Acetate/Complex,
H2Rate_All/Acetate/Complex).

Voi moi dataset, doc 2 file co san:
    <dataset>/baseline_all19_by_BFset.csv   (76 dong: 4 BF-set x 19 model)
    <dataset>/meta_all19_by_MFlevel.csv     (114 dong: 6 MF-level x 19 model)

Xuat ra:
  1. all_baseline_across_datasets.csv - gop 6 file baseline thanh 1 bang dai,
     them cot "dataset" - dung de loc/pivot/so sanh giua cac dataset.
  2. all_meta_across_datasets.csv - tuong tu cho meta.
  3. champion_comparison.csv - BANG SO SANH CHINH: voi moi dataset, lay
     model+BF-set TOT NHAT (baseline) va model+MF-level TOT NHAT (meta),
     dat canh nhau de xem stacking co cai thien so voi baseline khong,
     tren CA thang log1p va thang goc.
  4. best_model_per_bfset_across_datasets.csv - voi moi (dataset, BF-set),
     model nao thang - de xem model nao on dinh manh o BF-set nao xuyen
     suot cac dataset.
  5. best_model_per_mflevel_across_datasets.csv - tuong tu cho MF-level.

Cach dung:
    python aggregate_across_datasets.py --results-dir results_stage3 \
        --datasets CurrentDensity_All,CurrentDensity_Acetate,CurrentDensity_Complex,H2Rate_All,H2Rate_Acetate,H2Rate_Complex
"""
import argparse
import os

import pandas as pd

DEFAULT_DATASETS = [
    "CurrentDensity_All", "CurrentDensity_Acetate", "CurrentDensity_Complex",
    "H2Rate_All", "H2Rate_Acetate", "H2Rate_Complex",
]


def load_all(results_dir, datasets, filename, extra_cols_first=None):
    frames = []
    missing = []
    for ds in datasets:
        path = os.path.join(results_dir, ds, filename)
        if not os.path.exists(path):
            missing.append(ds)
            continue
        df = pd.read_csv(path)
        df.insert(0, "dataset", ds)
        frames.append(df)
    if missing:
        print(f"[CANH BAO] Thieu file '{filename}' cho: {', '.join(missing)} (bo qua)")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_stage3")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--out-dir", default=None, help="Mac dinh: giong --results-dir")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1 & 2: gop long-format ----
    baseline_all = load_all(args.results_dir, datasets, "baseline_all19_by_BFset.csv")
    meta_all = load_all(args.results_dir, datasets, "meta_all19_by_MFlevel.csv")

    if baseline_all.empty and meta_all.empty:
        print("Khong tim thay file nao. Kiem tra lai --results-dir / --datasets.")
        return

    if not baseline_all.empty:
        p = os.path.join(out_dir, "all_baseline_across_datasets.csv")
        baseline_all.to_csv(p, index=False)
        print(f"-> {p} ({len(baseline_all)} dong)")

    if not meta_all.empty:
        p = os.path.join(out_dir, "all_meta_across_datasets.csv")
        meta_all.to_csv(p, index=False)
        print(f"-> {p} ({len(meta_all)} dong)")

    # ---- 3: bang so sanh Champion (baseline tot nhat vs meta tot nhat) ----
    # Day du CA 5 METRIC (RMSE, MAE, R2, MAPE, SMAPE) x 2 THANG DO (log1p, goc)
    METRICS = ["RMSE", "MAE", "R2", "MAPE", "SMAPE"]
    SCALES = ["log1p", "orig"]

    champion_rows = []
    for ds in datasets:
        row = {"dataset": ds}
        if not baseline_all.empty:
            sub = baseline_all[baseline_all["dataset"] == ds]
            if not sub.empty:
                best_b = sub.loc[sub["LOOCV_RMSE_log1p"].idxmin()]
                row["Baseline_best_model"] = best_b["model"]
                row["Baseline_best_BFset"] = best_b["bf_set"]
                for scale in SCALES:
                    for m in METRICS:
                        col = f"LOOCV_{m}_{scale}"
                        row[f"Baseline_{col}"] = best_b[col]
        if not meta_all.empty:
            sub = meta_all[meta_all["dataset"] == ds]
            if not sub.empty:
                best_m = sub.loc[sub["LOOCV_RMSE_log1p"].idxmin()]
                row["Meta_best_model"] = best_m["model"]
                row["Meta_best_MFlevel"] = best_m["mf_level"]
                for scale in SCALES:
                    for m in METRICS:
                        col = f"LOOCV_{m}_{scale}"
                        row[f"Meta_{col}"] = best_m[col]
        if "Baseline_LOOCV_RMSE_log1p" in row and "Meta_LOOCV_RMSE_log1p" in row:
            improve = row["Baseline_LOOCV_RMSE_log1p"] - row["Meta_LOOCV_RMSE_log1p"]
            row["Meta_cai_thien_RMSE_log1p"] = improve
            row["Meta_co_tot_hon_Baseline"] = improve > 0
        champion_rows.append(row)

    champion_df = pd.DataFrame(champion_rows)
    p = os.path.join(out_dir, "champion_comparison.csv")
    champion_df.to_csv(p, index=False)
    print(f"-> {p}")

    # ---- 4 & 5: model tot nhat trong tung (dataset, BF-set) / (dataset, MF-level) ----
    # Day du CA 5 METRIC (RMSE, MAE, R2, MAPE, SMAPE) x 2 THANG DO (log1p, goc)
    metric_cols = [f"LOOCV_{m}_{s}" for s in SCALES for m in METRICS]

    if not baseline_all.empty:
        best_per_bf = baseline_all[baseline_all["rank_within_bfset"] == 1][
            ["dataset", "bf_set", "model", "params"] + metric_cols
        ].sort_values(["dataset", "bf_set"])
        p = os.path.join(out_dir, "best_model_per_bfset_across_datasets.csv")
        best_per_bf.to_csv(p, index=False)
        print(f"-> {p}")

    if not meta_all.empty:
        best_per_mf = meta_all[meta_all["rank_within_mflevel"] == 1][
            ["dataset", "mf_level", "model", "params"] + metric_cols
        ].sort_values(["dataset", "mf_level"])
        p = os.path.join(out_dir, "best_model_per_mflevel_across_datasets.csv")
        best_per_mf.to_csv(p, index=False)
        print(f"-> {p}")

    print("\n=== BANG SO SANH CHAMPION (xem nhanh - day du 5 metric) ===")
    cols_show = ["dataset", "Baseline_best_model", "Baseline_best_BFset"]
    cols_show += [f"Baseline_LOOCV_{m}_log1p" for m in METRICS]
    cols_show += ["Meta_best_model", "Meta_best_MFlevel"]
    cols_show += [f"Meta_LOOCV_{m}_log1p" for m in METRICS]
    cols_show += ["Meta_co_tot_hon_Baseline"]
    cols_show = [c for c in cols_show if c in champion_df.columns]
    print(champion_df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
