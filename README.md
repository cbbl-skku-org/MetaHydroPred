# MetaHydroPred

Meta-learning (stacked) models for predicting **current density** and **H₂ production rate** of microbial
electrolysis cells (MECs) from six physical parameters, for three substrate datasets (All-organic, Acetate,
Complex substrate). Web server: <https://balalab-skku.org/MetaHydroPred/>.

This repository contains the **final model pipeline** (leave-one-out cross-validation on the entire dataset), the
final results, the six saved models, a checker that reproduces them, and the web-server code.

## Method in one paragraph
For each of 6 tasks, the six inputs are ranked by a feature-importance score (FIS) and four baseline feature sets
BF1–BF4 (top 6, 5, 4, 3 inputs) are formed. 19 regressors × BF1–BF4 = 76 baseline models are tuned and evaluated by
LOOCV; their out-of-fold predictions on the log1p target are the *meta-features*. The top 5, 10, …, 30 baselines
define MF1–MF6; 19 regressors × MF1–MF6 = 114 meta-models are tuned by LOOCV and the best one (lowest LOOCV RMSE on the
log1p scale) is the final model. Predictions are back-transformed to original units with ŷ = exp(ẑ) − 1
([backtransform.py](backtransform.py), used by every script and by the server).

## Repository layout
| Path | Content |
|---|---|
| `Current_density/`, `H2_Production_Rate/` | the six datasets (original units) |
| `datasets_config.json` | tasks, target columns, data paths |
| `models.py`, `common.py`, `backtransform.py` | 19-model registry and search grids, LOOCV helpers, shared back-transformation |
| `step1_generate_bf_sets.py` | FIS ranking and BF1–BF4 |
| `step2_baseline_model.py`, `step2b_aggregate_baseline.py` | baseline LOOCV grid search; `baseline_ranking.csv` |
| `step3_generate_mf_sets.py` | meta-feature matrices MF1–MF6 |
| `step4_meta_model.py`, `step4b_aggregate_meta.py` | meta-model LOOCV grid search; `meta_ranking.csv`, `FINAL_winner.csv` |
| `aggregate_across_datasets.py` | cross-task tables, incl. `champion_comparison.csv` (baseline vs meta, Table 1) |
| `run_pipeline.sh` | runs steps 1–4b for all tasks (parallel) |
| `results_stage3/` | final results: rankings, `FINAL_winner.csv`, meta-feature matrices, aggregate tables |
| `verify_reproduction.py` | reproduces the final models from the shipped files (≈2 min) |
| `webserver/` | saved final models (`final_models/`), export script, Flask web server, help page, tests, docs |

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.9 recommended
pip install -r requirements.txt
```

## 2. Check the installation and reproduce the final models
```bash
./check_repository.sh                           # predictor self-check + tests (seconds); add --full for the check below
python verify_reproduction.py                   # add --skip-baselines for the meta-model check only (seconds)
```
For all six tasks it recomputes the LOOCV out-of-fold prediction of every baseline that feeds the final meta-model
and compares it with the stored meta-feature matrix, then recomputes the LOOCV metrics of the final meta-model and
compares them with `FINAL_winner.csv`. Expected last line: `REPRODUCTION CHECK: PASSED`
(on the reference machine all deviations were below 1e-14).

## 3. Full reproduction from the raw data (grid search)
```bash
CORES=8 ./run_pipeline.sh                       # all six tasks; or: ./run_pipeline.sh H2Rate_Acetate
python aggregate_across_datasets.py --results-dir results_stage3
```
This runs 76 baseline jobs and 114 meta-model jobs per task (each a full hyper-parameter grid with LOOCV), so it is
computationally heavy; `CORES` sets the number of parallel jobs. It needs bash and `xargs` (Linux/macOS; use WSL on
Windows). Outputs per task in `results_stage3/<task>/`: `feature_ranking.csv`, `BF1–4.csv`, `baseline_ranking.csv`,
`meta/MF*/data.csv`, `meta_ranking.csv`, `FINAL_winner.csv`; across tasks: `champion_comparison.csv` etc.
Random seeds are fixed (`RANDOM_STATE = 42` in `models.py`) and all models run single-threaded; results should agree
with the shipped ones up to tiny numerical differences between library builds.

## 4. Predict on new (unseen) data with the final models — all six tasks
Prepare a CSV with the six input columns (section 5; example: `webserver/examples/example_input.csv`) and run:
```bash
python webserver/predict.py my_new_conditions.csv                 # all 6 tasks -> predictions.csv
python webserver/predict.py my_new_conditions.csv -o out.csv --substrates Acetate --targets CurrentDensity
python webserver/predict.py my_new_conditions.csv --tasks H2Rate_All,H2Rate_Complex
```
The output has one prediction column per task in the **original** measurement units (already back-transformed) and,
per task, whether each row lies inside the observed training range. The six tasks are `CurrentDensity_` / `H2Rate_` +
`All` / `Acetate` / `Complex`. Predictions are point estimates without intervals and are extrapolations outside the
observed ranges. From Python: `from metahydropred.predictor import Predictor` (see `webserver/predict.py`).
Before using the models in a new environment run `./check_repository.sh` (section 2).

## 5. Input parameters
| Column | Short name | Unit |
|---|---|---|
| Temperature | Temp. | °C |
| Substrate concentration | Sub conc. | see source dataset |
| Reactor working volume | Rtx vol. | mL |
| S/V ratio | S/V ratio | m²/m³ (≈ 100 × area[cm²] / volume[mL]) |
| Applied voltage | Eap | V |
| Cathode projected surface area | Cat proj. area | cm² |

Targets: `Current density` and `H2 production rate`, in the units of the source dataset (Yoon et al., 2024).
A temperature of 25 °C denotes a reported room-temperature condition in the source data.

## 6. Rebuild the saved models / run the web server
```bash
cd webserver
python export_final_models.py                   # refits the selected baselines and meta-models -> final_models/
python -m unittest discover -s tests -v
python app.py                                   # http://127.0.0.1:5000
```
The `webserver/` folder is self-contained (copy it alone to a host). Deployment and the manuscript text: `webserver/README.md`, `webserver/docs/`.
The `.joblib` files only load with the library versions in `requirements.txt`; rebuild them after changing versions.

## Scope
This repository covers the final full-dataset LOOCV models and the web server. Evaluation analyses reported in the
paper (repeated-split Monte-Carlo evaluation, SHAP and permutation analyses, temperature-sensitivity scenarios) are
not required to reproduce or use the final models and are not included here.

## Limitations
The models were developed from a small literature-derived dataset (72, 32 and 40 observations) and evaluated by
cross-validation on those observations without an independent external validation set. Predictions for conditions
outside the observed ranges are extrapolations and do not replace experiments.

## Citation, data source and licence
Data: compiled by Yoon et al. (2024) — please cite the source study together with this work.
Citation: *to be added on publication.*   Licence: *to be added by the authors.*
