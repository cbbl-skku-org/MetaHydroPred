# -*- coding: utf-8 -*-
"""
Tests for the MetaHydroPred prediction code and the shipped final models.
Run from the repository root, in the environment of requirements.txt:
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from backtransform import to_log_scale, to_original_scale        # noqa: E402
from metahydropred import config                                 # noqa: E402
from metahydropred.predictor import Predictor                    # noqa: E402
from metahydropred.validation import InputError, parse_and_validate, range_flags   # noqa: E402

with open(os.path.join(ROOT, "examples", "example_input.csv")) as _f:
    EXAMPLE = _f.read()


class BackTransform(unittest.TestCase):
    def test_roundtrip_and_formula(self):
        y = np.array([0.0, 0.5, 7.0, 460.0])
        self.assertTrue(np.allclose(to_original_scale(to_log_scale(y)), y))
        self.assertAlmostEqual(float(to_original_scale(1.0)), np.e - 1)

    def test_formula_defined_only_once(self):
        """np.expm1 / np.log1p may only appear in backtransform.py (prevents silent drift)."""
        offenders = []
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "tests", ".git", "results_stage3")]
            for f in files:
                if f.endswith(".py") and f != "backtransform.py":
                    with open(os.path.join(d, f), encoding="utf8", errors="ignore") as fh:
                        code = "\n".join(l for l in fh.read().splitlines() if not l.strip().startswith("#"))
                    if re.search(r"np\.(expm1|log1p)\(", code):
                        offenders.append(os.path.relpath(os.path.join(d, f), ROOT))
        self.assertEqual(offenders, [], f"formula duplicated in: {offenders}")


class Validation(unittest.TestCase):
    def test_example_ok(self):
        df, notes = parse_and_validate(EXAMPLE)
        self.assertEqual(list(df.columns), config.FEATURES)
        self.assertEqual((len(df), notes), (5, []))

    def test_aliases_units_and_separator(self):
        txt = "Temp (°C);Sub conc.;Rtx vol.;S/V ratio;Eap;Cat proj. area\n30;1;120;10;0.8;12\n"
        df, _ = parse_and_validate(txt)
        self.assertEqual(df.iloc[0]["Applied voltage"], 0.8)

    def test_errors(self):
        with self.assertRaises(InputError):
            parse_and_validate("")
        with self.assertRaises(InputError):                               # missing columns
            parse_and_validate("Temperature,Applied voltage\n30,0.8\n")
        with self.assertRaises(InputError):                               # non-numeric
            parse_and_validate(EXAMPLE.replace("30,1.0,120", "abc,1.0,120", 1))
        with self.assertRaises(InputError):                               # impossible voltage
            parse_and_validate(EXAMPLE.replace(",0.8,12\n", ",-0.8,12\n", 1))

    def test_unit_consistency_warning(self):
        _, notes = parse_and_validate(EXAMPLE.replace("30,1.0,120,10,0.8,12", "30,1.0,120,500,0.8,12"))
        self.assertTrue(any("S/V" in n for n in notes))


class FinalModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Predictor()

    def test_all_tasks_predict_finite(self):
        X, _ = parse_and_validate(EXAMPLE)
        out, summ = self.p.run(X, list(config.TASKS))
        self.assertEqual(len(summ), 6)
        for s in summ:
            self.assertTrue(np.isfinite(out[s["column"]]).all(), s["task"])

    def test_back_transform_is_applied(self):
        X, _ = parse_and_validate(EXAMPLE)
        task = "CurrentDensity_All"
        z = self.p.predict_log(task, X)
        self.assertTrue(np.allclose(self.p.predict_original(task, X), np.expm1(z)))
        out, _ = self.p.run(X, [task])
        self.assertTrue(np.allclose(out[config.task_column(task)], np.round(np.expm1(z), 4)))

    def test_training_rows_track_observed_targets(self):
        """Predictions on the training inputs should follow the observed targets (sanity check, not performance)."""
        with open(os.path.join(ROOT, "datasets_config.json")) as f:
            paths = {d["name"]: d["feature_sets"]["Set1"]["full_path"] for d in json.load(f)["datasets"]}
        for task in config.TASKS:
            d = pd.read_csv(os.path.join(ROOT, paths[task]))
            y_hat = self.p.predict_original(task, d[config.FEATURES])
            r = np.corrcoef(np.log1p(d.iloc[:, -1].to_numpy()), np.log1p(np.clip(y_hat, 0, None)))[0, 1]
            self.assertGreater(r, 0.5, f"{task}: corr {r:.2f}")

    def test_range_flags(self):
        X, _ = parse_and_validate(EXAMPLE)
        f = range_flags(X, self.p.ranges["CurrentDensity_Complex"])
        self.assertTrue(any(f[2]))          # 150 cm2 is far above the Complex range (7-20)
        self.assertFalse(any(f[0]))

    def test_deployed_models_equal_selection_table(self):
        """The models must be exactly those selected in the results table (champion_comparison.csv)."""
        champ = pd.read_csv(os.path.join(ROOT, "final_models", "selection_summary.csv")).set_index("dataset")
        label = {"CurrentDensity": "Current Density", "H2Rate": "H2 Production Rate"}
        n_baselines = {"MF1": 5, "MF2": 10, "MF3": 15, "MF4": 20, "MF5": 25, "MF6": 30}
        n_obs = {"All": 72, "Acetate": 32, "Complex": 40}
        for task, (t, s) in config.TASKS.items():
            row = champ.loc[f"{label[t]} ({s})"]
            card = self.p.card["tasks"][task]
            self.assertEqual(card["meta_model"], row["Meta_best_model"], task)
            self.assertEqual(card["mf_level"], row["Meta_best_MFlevel"], task)
            self.assertAlmostEqual(card["loocv"]["RMSE_log1p"], row["Meta_LOOCV_RMSE_log1p"], places=6, msg=task)
            self.assertEqual(len(self.p.bundles[task]["baselines"]), n_baselines[row["Meta_best_MFlevel"]], task)
            self.assertEqual(self.p.bundles[task]["n_samples"], n_obs[s], task)


class CommandLine(unittest.TestCase):
    def test_predict_py_on_unseen_data_all_tasks(self):
        import predict
        d = tempfile.mkdtemp()
        new, out = os.path.join(d, "unseen.csv"), os.path.join(d, "pred.csv")
        with open(new, "w") as f:                                          # values not in the example file
            f.write("Temperature,Substrate concentration,Reactor working volume,S/V ratio,"
                    "Applied voltage,Cathode projected surface area\n32,1.4,150,20,0.9,30\n26,2.0,60,50,0.7,30\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(predict.main([new, "-o", out]), 0)
        df = pd.read_csv(out, encoding="utf-8-sig")
        self.assertEqual(len(df), 2)
        for t in config.TASKS:
            self.assertTrue(np.isfinite(df[config.task_column(t)]).all(), t)
        with contextlib.redirect_stderr(io.StringIO()):                    # expected error message
            self.assertEqual(predict.main([os.path.join(d, "missing.csv"), "-o", out]), 2)


if __name__ == "__main__":
    unittest.main()
