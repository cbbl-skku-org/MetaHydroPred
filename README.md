# MetaHydroPred
Meta-learning models that predict **current density** and **H₂ production rate** of microbial electrolysis cells (MECs) from six physical parameters, for three substrate datasets (All-organic, Acetate, Complex). This repository contains the final models, a command-line predictor, and the code and data to reproduce them. Web server: <https://balalab-skku.org/MetaHydroPred/>

## Install (Python 3.9)
```bash
conda create -n mhp python=3.9 -y && conda activate mhp
export PYTHONNOUSERSITE=1
pip install -r requirements.txt
./check.sh                       # expect: ALL CHECKS PASSED
```
The saved models only load with the versions pinned in `requirements.txt`.

## Predict new data (all six tasks)
```bash
python predict.py examples/example_input.csv             # try it: predicts all six tasks -> predictions.csv
cp examples/example_input.csv my_conditions.csv          # then edit the rows (keep the header) and run on your data:
python predict.py my_conditions.csv --tasks H2Rate_All,CurrentDensity_Acetate
```
Input: a table with the six columns below. Output: one prediction column per task in original units (already back-transformed, ŷ = exp(ẑ) − 1) and a flag for rows outside the observed training range. Tasks: `CurrentDensity_` /
`H2Rate_` + `All` / `Acetate` / `Complex`.

| Column | Unit | | Column | Unit |
|---|---|---|---|---|
| Temperature | °C | | S/V ratio | m²/m³ |
| Substrate concentration | g/L | | Applied voltage | V |
| Reactor working volume | mL | | Cathode projected surface area | cm² |

## Reproduce
```bash
./check.sh --full                # ~2 min: recomputes the final models' cross-validation from the shipped data
CORES=8 ./run_pipeline.sh        # full rebuild from the raw data (heavy); overwrites results_stage3/ → use a copy
python export_final_models.py    # refits the selected models → final_models/
```
Workflow per task: feature ranking and feature sets BF1–BF4 (`step1`), 76 baseline models by LOOCV (`step2`), meta-feature sets MF1–MF6 from their out-of-fold predictions (`step3`), 114 meta-models and the final selection (`step4`). Seeds are fixed (`models.py`); results match `results_stage3/` up to tiny numerical differences between library builds.

## Contents
`predict.py`, `metahydropred/` (prediction code) · `final_models/` (six models, input ranges, model card) ·
`step1`–`step4b*.py`, `models.py`, `common.py`, `backtransform.py`, `run_pipeline.sh`, `aggregate_across_datasets.py`
(workflow) · `dataset/` (the six datasets), `datasets_config.json` (data paths) · `results_stage3/` (final
selections per task) · `verify_reproduction.py`, `tests/`, `check.sh` (checks)

## Limitations
Small literature-derived dataset (72, 32 and 40 observations), cross-validated but not externally validated; point predictions without intervals; only the six parameters above are used; predictions outside the observed ranges are extrapolations.

## Licence
Code: MIT ([LICENSE](LICENSE)).

##Citation
Citation of this work: *to be added.*