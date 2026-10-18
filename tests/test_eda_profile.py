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


class EdaProfileTests(unittest.TestCase):
    def test_all_null_aggregate_counts_are_zero(self):
        safe_aggregate_count = load_count_helper()

        self.assertEqual(safe_aggregate_count(float("nan")), 0)
        self.assertEqual(safe_aggregate_count(None), 0)
        self.assertEqual(safe_aggregate_count(7), 7)


    def test_product_outcome_plot_reads_the_product_column(self):
        tree = ast.parse(EDA_PATH.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "plot_product_outcomes"
        )

        def save_fig(fig, path, source):
            plt.close(fig)

        namespace = {
            "pd": pd,
            "plt": plt,
            "PAL": {"navy": "#0F172A"},
            "save_fig": save_fig,
            "EDA_FIGURES": Path("."),
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(EDA_PATH), "exec"), namespace)
        data = pd.DataFrame(
            {"product": ["A", "B"], "mature_signed_contracts": [12, 18], "observed_bad_rate": [0.1, 0.2]}
        )

        namespace["plot_product_outcomes"](data)


if __name__ == "__main__":
    unittest.main()
