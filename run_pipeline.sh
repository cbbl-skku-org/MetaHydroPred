#!/bin/bash
# GIAI DOAN 3: Chay TOAN BO pipeline (baseline + meta-stacking) tren TOAN BO
# du lieu (khong con 80/20, khong con 20 lan lap - LOOCV la danh gia xac
# dinh nen chi can chay 1 lan). Nhanh hon Giai doan 2 rat nhieu (khong con
# nhan voi 20).
#
# Cach dung:
#   ./run_all_datasets.sh                          # chay ca 6 dataset
#   ./run_all_datasets.sh CurrentDensity_Acetate    # chi 1 dataset

set -e

CONFIG="datasets_config.json"
RESULTS_DIR="results_stage3"
CORES="${CORES:-8}"   # so process song song - chinh qua bien moi truong CORES

if [ -n "$1" ]; then
  IFS=',' read -r -a DATASETS <<< "$1"
else
  DATASETS=(
    "CurrentDensity_All" "CurrentDensity_Acetate" "CurrentDensity_Complex"
    "H2Rate_All" "H2Rate_Acetate" "H2Rate_Complex"
  )
fi

BF_SETS=("BF1" "BF2" "BF3" "BF4")
MF_LEVELS=("MF1" "MF2" "MF3" "MF4" "MF5" "MF6")
MODELS=("LR" "Lasso" "Ridge" "Elastic" "DT" "RF" "ET" "XGB" "CB" "LGBM" \
        "GB" "HGB" "AB" "SVR_linear" "SVR_rbf" "SVR_poly" "SVR_sigmoid" "KNN" "MLP")

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo ">>> [Buoc 1] FIS-score + tao BF1-BF4 (tren TOAN BO du lieu) ..."
python3 step1_generate_bf_sets.py "$CONFIG" \
    --only "$(IFS=,; echo "${DATASETS[*]}")" --results-dir "$RESULTS_DIR"

for ds in "${DATASETS[@]}"; do
  echo ""
  echo "=== Dataset: $ds ==="

  echo ">>> [Buoc 2] 76 baseline model (19 model x 4 BF-set) song song ..."
  for bf in "${BF_SETS[@]}"; do
    for model in "${MODELS[@]}"; do
      echo "python3 step2_baseline_model.py $CONFIG --dataset $ds --bf-set $bf --model $model --results-dir $RESULTS_DIR"
    done
  done | xargs -I {} -P "$CORES" sh -c "{}"

  N=$(ls "${RESULTS_DIR}/${ds}/baseline/_best/"*_summary.csv 2>/dev/null | wc -l)
  if [ "$N" -ne 76 ]; then
    echo "  [CANH BAO] $ds: chi co $N/76 baseline summary!"
  fi

  echo ">>> [Buoc 2b] Tong hop baseline_ranking.csv ..."
  python3 step2b_aggregate_baseline.py --dataset "$ds" --results-dir "$RESULTS_DIR"

  echo ">>> [Buoc 3] Xay MF1-MF6 ..."
  python3 step3_generate_mf_sets.py "$CONFIG" --dataset "$ds" --results-dir "$RESULTS_DIR"

  echo ">>> [Buoc 4] 114 meta-model (19 model x 6 MF-level) song song ..."
  for mf in "${MF_LEVELS[@]}"; do
    for model in "${MODELS[@]}"; do
      echo "python3 step4_meta_model.py $CONFIG --dataset $ds --mf-level $mf --model $model --results-dir $RESULTS_DIR"
    done
  done | xargs -I {} -P "$CORES" sh -c "{}"

  N=$(ls "${RESULTS_DIR}/${ds}/meta/"MF*/results/_best/*_summary.csv 2>/dev/null | wc -l)
  if [ "$N" -ne 114 ]; then
    echo "  [CANH BAO] $ds: chi co $N/114 meta summary!"
  fi

  echo ">>> [Buoc 4b] Tong hop meta_ranking.csv + FINAL_winner ..."
  python3 step4b_aggregate_meta.py --dataset "$ds" --results-dir "$RESULTS_DIR"
done

echo ""
echo "=== HOAN TAT ==="
