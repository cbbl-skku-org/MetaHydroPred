# -*- coding: utf-8 -*-
"""
Configuration of the MetaHydroPred prediction code: tasks, input parameters, units, header aliases, limits and
the standing notices printed with predictions. Edit UNITS / TARGETS here if a unit needs correcting.
"""
import os

APP_NAME = "MetaHydroPred"          # always written exactly like this (reviewer request)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MHP_MODELS_DIR", os.path.join(ROOT_DIR, "final_models"))   # saved final models

MAX_ROWS = 500                       # rows per input table

# ---------------------------------------------------------------- inputs ---
# Canonical names are exactly the column names used when the models were trained.
FEATURES = [
    "Temperature",
    "Substrate concentration",
    "Reactor working volume",
    "S/V ratio",
    "Applied voltage",
    "Cathode projected surface area",
]

SHORT_NAMES = {
    "Temperature": "Temp.",
    "Substrate concentration": "Sub conc.",
    "Reactor working volume": "Rtx vol.",
    "S/V ratio": "S/V ratio",
    "Applied voltage": "Eap",
    "Cathode projected surface area": "Cat proj. area",
}

# Units. Temperature, volume, area and S/V are verified from the training data
# (S/V = 100 * area[cm2] / volume[mL] holds in every row, i.e. m2/m3).
# Concentration unit and the two target units are taken from the study's data
# description -- CONFIRM before publishing.
UNITS = {
    "Temperature": "°C",
    "Substrate concentration": "g/L",
    "Reactor working volume": "mL",
    "S/V ratio": "m²/m³",
    "Applied voltage": "V",
    "Cathode projected surface area": "cm²",
}

# Hard limits: values outside these are rejected as impossible / unit mistakes.
# (This is NOT the observed training range, which is only used for warnings.)
HARD_LIMITS = {
    "Temperature": (0.0, 100.0),
    "Substrate concentration": (0.0, 1000.0),
    "Reactor working volume": (0.0, 1e6),
    "S/V ratio": (0.0, 1e5),
    "Applied voltage": (0.0, 20.0),
    "Cathode projected surface area": (0.0, 1e6),
}
STRICTLY_POSITIVE = {"Reactor working volume", "S/V ratio", "Applied voltage",
                     "Cathode projected surface area"}

# Header aliases (after lower-casing, removing "(unit)" parts, and collapsing
# punctuation/underscores) -> canonical feature name.
ALIASES = {
    "temperature": "Temperature", "temp": "Temperature",
    "substrate concentration": "Substrate concentration", "sub conc": "Substrate concentration",
    "substrate conc": "Substrate concentration",
    "reactor working volume": "Reactor working volume", "rtx vol": "Reactor working volume",
    "reactor volume": "Reactor working volume", "working volume": "Reactor working volume",
    "s/v ratio": "S/V ratio", "sv ratio": "S/V ratio", "s/v": "S/V ratio",
    "applied voltage": "Applied voltage", "eap": "Applied voltage", "voltage": "Applied voltage",
    "cathode projected surface area": "Cathode projected surface area",
    "cat proj area": "Cathode projected surface area", "cathode area": "Cathode projected surface area",
    "cathode surface area": "Cathode projected surface area",
}

# Soft check: S/V should be close to 100 * area[cm2] / volume[mL]. A larger gap
# usually means a unit mix-up; we warn but do not block.
SV_CONSISTENCY_TOLERANCE = 0.20

# ----------------------------------------------------------------- tasks ---
SUBSTRATES = {
    "All": "All-organic",
    "Acetate": "Acetate",
    "Complex": "Complex substrate",
}
TARGETS = {
    "CurrentDensity": {"label": "Current density", "unit": "A/m³",
                       "column": "Current density"},
    "H2Rate": {"label": "H₂ production rate", "unit": "m³/m³/d",
               "column": "H2 production rate"},
}
TARGET_UNITS = {k: v["unit"] for k, v in TARGETS.items()}

# task key (matches results_stage3/<key>) -> (target key, substrate key)
TASKS = {f"{t}_{s}": (t, s) for t in TARGETS for s in SUBSTRATES}


def task_label(task):
    t, s = TASKS[task]
    return f"{TARGETS[t]['label']} ({SUBSTRATES[s]})"


def task_column(task):
    """Column name used in the results table / CSV, units included."""
    t, s = TASKS[task]
    return f"{TARGETS[t]['column']} ({SUBSTRATES[s]}) [{TARGETS[t]['unit']}]"


# --------------------------------------------------------------- notices ---
LIMITED_DATA_NOTICE = (
    "{app} was trained on a small, literature-compiled dataset (72 observations for "
    "the All-organic set, 32 for Acetate and 40 for Complex substrates). Performance "
    "was estimated by leave-one-out cross-validation on these same observations; no "
    "independent external validation set has been evaluated; validation with larger, independently "
    "collected MEC datasets is needed to establish broader applicability. Predictions for operating "
    "conditions that differ from the training data should be treated as indicative only "
    "and are not a substitute for experiments."
).format(app=APP_NAME)

NO_INTERVAL_NOTICE = (
    "Each result is a point prediction. The models do not provide calibrated prediction "
    "or confidence intervals, so no uncertainty bound is shown."
)

RANGE_NOTICE = (
    "The observed ranges describe the data coverage of the training set; they are not a "
    "guarantee that predictions inside the range are accurate, and predictions outside it "
    "are extrapolations."
)

BACKTRANSFORM_NOTICE = (
    "All displayed predictions are already back-transformed to the original measurement "
    "units (y = exp(z) − 1, where z is the log1p-scale model output); they are not "
    "log1p-scale values."
)
