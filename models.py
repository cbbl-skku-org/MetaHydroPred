"""
Model registry: search space + estimator builder for each of the 19 regressors.

Design notes:
- Scale-sensitive models (linear models, SVR variants, KNN, MLP) are wrapped in a
  Pipeline(StandardScaler -> model) so the scaler is refit on the training fold
  only inside every LOOCV split (no leakage). Tree/boosting based models are
  scale-invariant and are left unscaled.
- All models get a fixed random_state where applicable, single-threaded execution
  (n_jobs=1 / thread_count=1) for reproducibility and because the host has 1 CPU.
- CatBoost / LightGBM / XGBoost are run silent (no per-iteration logging).
"""

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

RANDOM_STATE = 42

SEARCH_SPACE = {
    "LR": {
        "fit_intercept": [True, False],
    },
    "Lasso": {
        "alpha": [0.1, 0.01, 0.001],
        "fit_intercept": [True, False],
    },
    "Ridge": {
        "alpha": [1.0, 0.1, 0.01, 0.001, 10.0],
        "fit_intercept": [True, False],
    },
    "Elastic": {
        "alpha": [0.1, 0.01],
        "l1_ratio": [0.4, 0.7, 0.9],
        "fit_intercept": [True, False],
    },
    "DT": {
        "max_features": [1.0, 0.7, None],
        "min_samples_split": [2, 4, 6, 8, 10],
        "criterion": ["squared_error", "absolute_error"],
    },
    "RF": {
        "n_estimators": [50, 100, 150],
        "max_features": [1.0, 0.7, None],
        "min_samples_split": [2, 4, 6, 8, 10],
        "criterion": ["squared_error", "absolute_error"],
    },
    "ET": {
        "n_estimators": [50, 100, 150],
        "max_features": [1.0, 0.7, None],
        "min_samples_split": [2, 4, 6, 8, 10],
        "criterion": ["squared_error", "absolute_error"],
    },
    "XGB": {
        "n_estimators": [25, 50, 100],
        "learning_rate": [0.1, 0.05],
        "max_depth": [2, 3, 4],
    },
    "CB": {
        "iterations": [25, 50, 100],
        "learning_rate": [0.1, 0.05],
        "depth": [2, 3, 4],
    },
    "LGBM": {
        "n_estimators": [25, 50, 100],
        "learning_rate": [0.1, 0.05],
        "max_depth": [2, 3, 4],
        "min_child_samples": [5, 10, 20, 50],
    },
    "GB": {
        "n_estimators": [25, 50, 100],
        "learning_rate": [0.1, 0.05],
        "max_depth": [2, 3, 4],
    },
    "HGB": {
        "max_iter": [50, 100],
        "learning_rate": [0.1, 0.05],
        "max_leaf_nodes": [7, 15],
        "min_samples_leaf": [5, 10, 20, 50],
    },
    "AB": {
        "n_estimators": [25, 50, 100],
        "learning_rate": [0.1, 0.05],
    },
    "SVR_linear": {
        "C": [0.03125, 0.25, 1.0, 4.0, 32.0],
    },
    "SVR_rbf": {
        "C": [0.03125, 0.25, 1.0, 4.0, 32.0],
        "gamma": [0.0009765625, 0.03125, 0.25, 1.0],
    },
    "SVR_poly": {
        "C": [0.25, 1.0, 4.0],
        "gamma": [0.03125, 0.25],
        "degree": [2, 3],
    },
    "SVR_sigmoid": {
        "C": [0.25, 1.0, 4.0, 32.0],
        "gamma": [0.03125, 0.25],
    },
    "KNN": {
        "n_neighbors": [1, 2, 3, 4, 5, 7, 9],
        "weights": ["uniform", "distance"],
    },
    "MLP": {
        "hidden_layer_sizes": [(8,), (16,), (16, 8)],
        "activation": ["relu", "tanh"],
        "alpha": [0.01, 0.1],
        "learning_rate_init": [0.001],
    },
}

# Models that need feature scaling (wrapped in a Pipeline with StandardScaler)
SCALE_MODELS = {"LR", "Lasso", "Ridge", "Elastic",
    "SVR_linear", "SVR_rbf", "SVR_poly", "SVR_sigmoid",
    "KNN", "MLP",}


def _base_estimator(name, params):
    if name == "LR":
        return LinearRegression(**params)
    if name == "Lasso":
        return Lasso(**params, random_state=RANDOM_STATE, max_iter=20000)
    if name == "Ridge":
        return Ridge(**params, random_state=RANDOM_STATE)
    if name == "Elastic":
        return ElasticNet(**params, random_state=RANDOM_STATE, max_iter=20000)
    if name == "DT":
        return DecisionTreeRegressor(**params, random_state=RANDOM_STATE)
    if name == "RF":
        return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=1)
    if name == "ET":
        return ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=1)
    if name == "XGB":
        return XGBRegressor(
            **params, random_state=RANDOM_STATE, verbosity=0, n_jobs=1,
            objective="reg:squarederror",)
    if name == "CB":
        return CatBoostRegressor(
            **params, random_state=RANDOM_STATE, verbose=0, thread_count=1,
            allow_writing_files=False, loss_function="RMSE",)
    if name == "LGBM":
        return LGBMRegressor(**params, random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)
    if name == "GB":
        return GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    if name == "HGB":
        return HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    if name == "AB":
        return AdaBoostRegressor(**params, random_state=RANDOM_STATE)
    if name == "SVR_linear":
        return SVR(kernel="linear", **params)
    if name == "SVR_rbf":
        return SVR(kernel="rbf", **params)
    if name == "SVR_poly":
        return SVR(kernel="poly", **params)
    if name == "SVR_sigmoid":
        return SVR(kernel="sigmoid", **params)
    if name == "KNN":
        return KNeighborsRegressor(**params)
    if name == "MLP":
        return MLPRegressor(**params, random_state=RANDOM_STATE, max_iter=3000)
    raise ValueError(f"Unknown model name: {name}")


def build_estimator(name, params):
    """Build a (possibly scaled) unfit estimator for `name` with hyperparameters `params`."""
    est = _base_estimator(name, params)
    if name in SCALE_MODELS:
        return Pipeline([("scaler", StandardScaler()), ("model", est)])
    return est
