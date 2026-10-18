"""Behavioral tests for the customer-level Task 3 output."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK3_PATH = ROOT / "Code" / "5. Task 3.py"


def load_task3_function(name: str):
    tree = ast.parse(TASK3_PATH.read_text(encoding="utf-8"))
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function is None:
        raise AssertionError(f"Task 3 must define {name}")
    namespace = {"np": np, "pd": pd}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(TASK3_PATH), "exec"), namespace)
    return namespace[name]


def load_builder():
    return load_task3_function("build_customer_intelligence")


class CustomerIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.financial = pd.DataFrame(
            {
                "SKP_CLIENT": [101, 202],
                "Financial_Health_Score": [88.0, 42.0],
                "fh_thin_file": [False, True],
                "fh_policy_cap": ["", ""],
                "fh_drivers": ["3 clean loans", "no repayment history"],
                "pillar_Repayment History": [90.0, 50.0],
                "pts_dti": [85.0, 60.0],
            }
        )
        self.engagement = pd.DataFrame(
            {
                "SKP_CLIENT": [101, 202],
                "Engagement_Score": [84.0, 25.0],
                "E1_digital": [90.0, 10.0],
                "E2_recency": [80.0, 20.0],
                "days_since_last_activity": [14.0, np.nan],
            }
        )

    def test_builds_one_explainable_customer_row_and_distinguishes_thin_file(self):
        builder = load_builder()
        result = builder(self.financial, self.engagement)

        self.assertEqual(result["SKP_CLIENT"].tolist(), [101, 202])
        self.assertTrue(result["SKP_CLIENT"].is_unique)
        self.assertEqual(result.loc[0, "Matrix_Segment"], "High health × High engagement")
        self.assertEqual(result.loc[1, "Customer_Profile"], "Thin file — no signed loan history")
        self.assertIn("financial education", result.loc[1, "Next_Best_Action"].lower())
        self.assertNotIn("repayment support", result.loc[1, "Next_Best_Action"].lower())
        self.assertIn("Financial_Health_Score", result.columns)
        self.assertIn("E1_digital", result.columns)
        self.assertIn("42", result.loc[1, "Action_Rationale"])

    def test_thin_file_quality_check_accepts_explicit_non_stress_rationale(self):
        builder = load_builder()
        checks_function = load_task3_function("qa_checks")
        output = builder(self.financial, self.engagement)

        failed = [check["check"] for check in checks_function(output, len(output)) if check["status"] != "PASS"]
        self.assertEqual(failed, [])

    def test_rejects_duplicate_customer_keys_before_joining(self):
        builder = load_builder()
        duplicate_financial = pd.concat([self.financial, self.financial.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "unique customer keys"):
            builder(duplicate_financial, self.engagement)

    def test_rejects_mismatched_customer_populations(self):
        builder = load_builder()
        missing_customer = self.engagement.iloc[[0]].copy()

        with self.assertRaisesRegex(ValueError, "same customer population"):
            builder(self.financial, missing_customer)


    def test_thin_file_engagement_does_not_enter_repayment_support_queue(self):
        write_report = load_task3_function("write_report")
        output = pd.DataFrame(
            {
                "Matrix_Segment": ["Low health × Medium engagement", "Low health × Medium engagement", "Low health × Low engagement"],
                "Next_Best_Action": ["Education", "Repayment support", "Self-service"],
                "Financial_Health_Tier": ["Financial Stress / Low Health"] * 3,
                "Engagement_Tier": ["Medium Engagement", "Medium Engagement", "Low / Dormant Engagement"],
                "fh_thin_file": [True, False, False],
            }
        )

        report = write_report(output, [])

        self.assertIn("1 of these 2", report)
        self.assertIn("established loan history", report)


if __name__ == "__main__":
    unittest.main()
