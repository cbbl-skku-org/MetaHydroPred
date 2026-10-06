# -*- coding: utf-8 -*-
"""
backtransform.py
================
Single source of truth for the target transformation used by MetaHydroPred.

Both prediction targets (current density and H2 production rate) are modelled
on the log1p scale:

    z    = log(1 + y)          (to_log_scale,       used when training)
    y_hat = exp(z_hat) - 1     (to_original_scale,  used for every reported value)

This module is imported by the training / evaluation scripts (common.py,
step*.py, nested_loocv_full_rigor.py) AND by the web server
(webserver/metahydropred/predictor.py). Do not re-implement the formula
anywhere else: webserver/tests/test_pipeline.py fails if np.expm1 / np.log1p
is used outside this file.
"""
import numpy as np


def to_log_scale(y):
    """Original measurement units -> log1p scale (training target)."""
    return np.log1p(np.asarray(y, dtype=float))


def to_original_scale(z):
    """log1p-scale prediction -> original measurement units: y_hat = exp(z_hat) - 1."""
    return np.expm1(np.asarray(z, dtype=float))
