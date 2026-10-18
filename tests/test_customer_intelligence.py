"""Behavioral tests for the customer-level Task 3 output."""

from __future__ import annotations

import ast
import unittest
from unittest.mock import patch
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


    def test_kmeans_comparison_reports_holdout_metrics_and_preserves_input(self):
        compare = load_task3_function("compare_segmentations")
        rng = np.random.default_rng(42)
        groups = [
            (88, 86, False, "High health × High engagement"),
            (76, 28, False, "Medium health × Low engagement"),
            (42, 23, True, "Low health × Low engagement"),
        ]
        frames = []
        for financial_mean, engagement_mean, thin_file, segment in groups:
            frames.append(
                pd.DataFrame(
                    {
                        "Financial_Health_Score": np.clip(rng.normal(financial_mean, 2, 100), 0, 100),
                        "Engagement_Score": np.clip(rng.normal(engagement_mean, 3, 100), 0, 100),
                        "fh_thin_file": thin_file,
                        "Matrix_Segment": segment,
                    }
                )
            )
        output = pd.concat(frames, ignore_index=True)
        original_columns = output.columns.tolist()

        comparison = compare(output, seed=42)

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["population_n"], len(output))
        self.assertGreaterEqual(comparison["selected_k"], 2)
        self.assertTrue(np.isfinite(comparison["validation"]["kmeans_silhouette"]))
        self.assertTrue(np.isfinite(comparison["validation"]["matrix_silhouette"]))
        self.assertEqual(comparison["validation"]["equal_segments_k"], output["Matrix_Segment"].nunique())
        self.assertTrue(np.isfinite(comparison["validation"]["equal_segments_kmeans_silhouette"]))
        self.assertEqual(sum(cluster["customers"] for cluster in comparison["clusters"]), len(output))
        self.assertEqual(output.columns.tolist(), original_columns)


    def test_report_handles_tied_cluster_metrics_without_claiming_a_winner(self):
        write_report = load_task3_function("write_report")
        output = load_builder()(self.financial, self.engagement)
        comparison = {
            "status": "ok",
            "selected_k": 3,
            "population_n": 2,
            "train_n": 100,
            "validation_n": 20,
            "validation_metric_n": 20,
            "seed": 42,
            "validation": {
                "kmeans_silhouette": 0.5,
                "matrix_silhouette": 0.5,
                "equal_segments_k": 9,
                "equal_segments_kmeans_silhouette": 0.5,
                "kmeans_to_matrix_dominant_cell_pct": 66.0,
            },
            "k_selection": [{"k": 3, "train_silhouette": 0.4}],
            "clusters": [{"cluster": 1, "customers": 2, "thin_file_pct": 50.0}],
        }

        report = write_report(output, [], comparison)

        self.assertIn("Model comparison", report)
        self.assertIn("similar", report)
        self.assertNotIn("has the higher", report)
        comparison_section = report.split("## Model comparison", 1)[1].split("## Quality checks", 1)[0]
        self.assertNotIn("| nan", comparison_section.lower())
        self.assertIn("no observed post-action outcome label", report)
        self.assertIn("thin-file safeguard", report)

        comparison["validation"]["equal_segments_kmeans_silhouette"] = np.nan
        unavailable_report = write_report(output, [], comparison)
        self.assertIn("No valid equal-count K-means result", unavailable_report)
        self.assertNotIn("score matrix has the higher", unavailable_report)

    def test_comparison_marks_an_undersized_population_unavailable(self):
        compare = load_task3_function("compare_segmentations")
        output = pd.DataFrame(
            {
                "Financial_Health_Score": [50.0] * 12,
                "Engagement_Score": [50.0] * 12,
                "fh_thin_file": [False] * 12,
                "Matrix_Segment": ["Cell A"] * 6 + ["Cell B"] * 6,
            }
        )

        comparison = compare(output, seed=42)

        self.assertEqual(comparison["status"], "unavailable")
        self.assertIn("20 customers", comparison["reason"])


    def test_comparison_handles_degenerate_score_space(self):
        compare = load_task3_function("compare_segmentations")
        output = pd.DataFrame(
            {
                "Financial_Health_Score": [50.0] * 100,
                "Engagement_Score": [50.0] * 100,
                "fh_thin_file": [False] * 100,
                "Matrix_Segment": ["Cell A"] * 50 + ["Cell B"] * 50,
            }
        )

        comparison = compare(output, seed=42)

        self.assertEqual(comparison["status"], "unavailable")
        self.assertIn("distinct score pairs", comparison["reason"])




    def test_comparison_reports_scipy_cluster_failure_as_unavailable(self):
        from scipy.cluster.vq import ClusterError

        compare = load_task3_function("compare_segmentations")
        base = load_builder()(self.financial, self.engagement)
        output = pd.concat([base] * 50, ignore_index=True)
        with patch("scipy.cluster.vq.kmeans2", side_effect=ClusterError("empty cluster")):
            comparison = compare(output, seed=42)

        self.assertEqual(comparison["status"], "unavailable")
        self.assertIn("valid clusters", comparison["reason"])


    def test_low_health_profile_counts_separate_thin_file_from_established_customers(self):
        profile_counts = load_task3_function("low_health_profile_counts")
        output = pd.DataFrame(
            {
                "Financial_Health_Tier": [
                    "Financial Stress / Low Health",
                    "Financial Stress / Low Health",
                    "Financial Stress / Low Health",
                    "Moderate Health",
                ],
                "Engagement_Tier": [
                    "High Engagement",
                    "Medium Engagement",
                    "Medium Engagement",
                    "Low / Dormant Engagement",
                ],
                "fh_thin_file": [True, False, False, True],
            }
        )

        counts = profile_counts(output)

        self.assertEqual(counts.loc["Thin file", "High Engagement"], 1)
        self.assertEqual(counts.loc["Established history", "Medium Engagement"], 2)
        self.assertEqual(counts.loc["Thin file", "Low / Dormant Engagement"], 0)


if __name__ == "__main__":
    unittest.main()
