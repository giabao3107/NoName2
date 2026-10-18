"""Regression tests for aggregate profile counts with all-null columns."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


EDA_PATH = Path(__file__).resolve().parents[1] / "Code" / "2. EDA.py"


def load_count_helper():
    tree = ast.parse(EDA_PATH.read_text(encoding="utf-8"))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "safe_aggregate_count"),
        None,
    )
    if function is None:
        raise AssertionError("EDA must define safe_aggregate_count")
    namespace = {"pd": pd, "np": np}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(EDA_PATH), "exec"), namespace)
    return namespace["safe_aggregate_count"]


def load_plot_function(name: str):
    tree = ast.parse(EDA_PATH.read_text(encoding="utf-8"))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function is None:
        raise AssertionError(f"EDA must define {name}")
    saved = {}

    def save_fig(fig, path, source=None, **kwargs):
        saved["left"] = kwargs.get("left", 0.07)
        saved["right"] = kwargs.get("right", 0.98)
        saved["xlim"] = fig.axes[0].get_xlim()
        plt.close(fig)

    namespace = {
        "pd": pd, "np": np, "plt": plt, "Path": Path,
        "PAL": {"navy": "#0F172A", "grey": "#64748B", "red": "#EF4444"},
        "EDA_FIGURES": Path("."), "save_fig": save_fig,
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(EDA_PATH), "exec"), namespace)
    return namespace[name], saved

class EdaProfileTests(unittest.TestCase):
    def test_all_null_aggregate_counts_are_zero(self):
        safe_aggregate_count = load_count_helper()

        self.assertEqual(safe_aggregate_count(float("nan")), 0)
        self.assertEqual(safe_aggregate_count(None), 0)
        self.assertEqual(safe_aggregate_count(7), 7)


    def test_product_outcome_plot_preserves_headroom_for_sample_labels(self):
        plot, saved = load_plot_function("plot_product_outcomes")
        data = pd.DataFrame(
            {
                "product": ["A", "B"],
                "mature_signed_contracts": [12, 18],
                "observed_bad_rate": [0.0497, 0.0477],
            }
        )

        plot(data)

        self.assertGreaterEqual(saved["left"], 0.15)
        self.assertGreaterEqual(saved["right"], 0.85)
        self.assertGreaterEqual(saved["xlim"][1], 0.0497 * 100 * 1.5)

    def test_long_eda_category_labels_receive_extra_left_margin(self):
        cases = [
            (
                "plot_source_coverage",
                pd.DataFrame({"dataset": ["Customers", "App behavior"], "customer_coverage_pct": [100, 47]}),
                0.16,
            ),
            (
                "plot_raw_missingness",
                pd.DataFrame({"dataset": ["customer"], "column": ["CNT_PERSON_DEPENDENT"], "null_pct": [98.5]}),
                0.30,
            ),
            (
                "plot_behavior_channels",
                pd.DataFrame({"channel": ["PARTNER", "BRANCH"], "events": [396829, 761]}),
                0.16,
            ),
        ]
        for name, data, expected_left in cases:
            with self.subTest(plot=name):
                plot, saved = load_plot_function(name)
                plot(data)
                self.assertGreaterEqual(saved["left"], expected_left)


if __name__ == "__main__":
    unittest.main()
