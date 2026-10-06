# -*- coding: utf-8 -*-
"""
Input validation for the MetaHydroPred web server.

parse_and_validate(text) -> (DataFrame with the 6 canonical columns, notes)
    Raises InputError with a user-readable message when the input cannot be used.
    `notes` are non-blocking remarks (ignored columns, unit-consistency warnings).

range_flags(df, ranges) -> per-row list of out-of-range messages for one task
    Observed-range checks are warnings about data coverage, never errors.
"""
import io
import re

import numpy as np
import pandas as pd

from . import config


class InputError(ValueError):
    """The submitted data cannot be processed; the message is shown to the user."""


def _norm_header(h):
    h = str(h).strip().lower()
    h = re.sub(r"\(.*?\)|\[.*?\]", " ", h)           # drop "(unit)" / "[unit]"
    h = h.replace("_", " ").replace(".", " ")
    h = re.sub(r"[^a-z0-9/ ]+", " ", h)
    return re.sub(r"\s+", " ", h).strip()


def _read_table(text):
    text = (text or "").strip()
    if not text:
        raise InputError("No data was submitted. Paste a table or upload a CSV file.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")   # , ; or tab
    except Exception:
        raise InputError("The data could not be read as a table. Use comma-, semicolon- or "
                         "tab-separated values with one header row (see the help page).")
    if df.shape[1] < len(config.FEATURES):
        raise InputError(f"Expected {len(config.FEATURES)} columns, found {df.shape[1]}. "
                         "Check that the separator is a comma, semicolon or tab.")
    return df


def parse_and_validate(text):
    df = _read_table(text)
    if len(df) == 0:
        raise InputError("The table has a header but no data rows.")
    if len(df) > config.MAX_ROWS:
        raise InputError(f"Too many rows ({len(df)}). The limit is {config.MAX_ROWS} per submission.")

    # ---- map headers to canonical names ------------------------------------
    mapping, ignored = {}, []
    for col in df.columns:
        canon = config.ALIASES.get(_norm_header(col))
        if canon is None:
            ignored.append(str(col))
        elif canon in mapping.values():
            raise InputError(f"Column '{col}' duplicates the parameter '{canon}'.")
        else:
            mapping[col] = canon
    missing = [f for f in config.FEATURES if f not in mapping.values()]
    if missing:
        raise InputError("Missing required column(s): " + ", ".join(missing) +
                         ". All six parameters are required: " + ", ".join(config.FEATURES) + ".")
    df = df.rename(columns=mapping)[config.FEATURES]

    notes = []
    if ignored:
        notes.append("Ignored column(s) not used by the models: " + ", ".join(ignored) + ".")

    # ---- numeric, finite, within hard limits -------------------------------
    problems = []
    for feat in config.FEATURES:
        num = pd.to_numeric(df[feat], errors="coerce")
        bad_missing = num.isna()
        for i in np.flatnonzero(bad_missing.to_numpy())[:5]:
            problems.append(f"Row {i + 1}: '{feat}' is missing or not a number.")
        ok = ~bad_missing
        lo, hi = config.HARD_LIMITS[feat]
        for i in np.flatnonzero((ok & ~np.isfinite(num)).to_numpy())[:5]:
            problems.append(f"Row {i + 1}: '{feat}' is not finite.")
        if feat in config.STRICTLY_POSITIVE:
            out = ok & ((num <= lo) | (num > hi))
        else:
            out = ok & ((num < lo) | (num > hi))
        for i in np.flatnonzero(out.to_numpy())[:5]:
            problems.append(f"Row {i + 1}: '{feat}' = {num.iloc[i]:g} is outside the physically "
                            f"plausible limits ({lo:g}–{hi:g} {config.UNITS[feat]}). Check the units.")
        df[feat] = num
    if problems:
        raise InputError("Please correct the input:\n" + "\n".join(problems[:15]))

    # ---- soft check: S/V ~ 100 * area[cm2] / volume[mL] ---------------------
    expected = 100.0 * df["Cathode projected surface area"] / df["Reactor working volume"]
    rel = (df["S/V ratio"] - expected).abs() / expected
    off = np.flatnonzero((rel > config.SV_CONSISTENCY_TOLERANCE).to_numpy())
    if len(off):
        rows = ", ".join(str(i + 1) for i in off[:10]) + (" ..." if len(off) > 10 else "")
        notes.append(f"Row(s) {rows}: S/V ratio differs by more than "
                     f"{int(config.SV_CONSISTENCY_TOLERANCE * 100)}% from 100 × cathode area (cm²) / "
                     "working volume (mL). Please check the units of these three inputs.")

    return df.reset_index(drop=True), notes


def range_flags(df, task_ranges):
    """
    For one task: list (one entry per row) of out-of-observed-range messages.
    An empty list means the row lies inside the observed range for every parameter.
    """
    flags = [[] for _ in range(len(df))]
    for feat in config.FEATURES:
        lo, hi = task_ranges[feat]["min"], task_ranges[feat]["max"]
        below = (df[feat] < lo).to_numpy()
        above = (df[feat] > hi).to_numpy()
        for i in np.flatnonzero(below):
            flags[i].append(f"{config.SHORT_NAMES[feat]} {df[feat].iloc[i]:g} < {lo:g}")
        for i in np.flatnonzero(above):
            flags[i].append(f"{config.SHORT_NAMES[feat]} {df[feat].iloc[i]:g} > {hi:g}")
    return flags
