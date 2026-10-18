# %% [markdown]
# # 5. Customer segmentation and next best action
#
# Join the financial-health and engagement score files at customer level. The 3 × 3 matrix is descriptive; the actions are service recommendations, not credit decisions or causal claims. Thin-file customers are separated from customers with observed repayment stress.

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_project_root() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for directory in (start, *start.parents):
        has_data = any((directory / name).is_dir() for name in ("Data", "Data_raw", "Data_clean"))
        if has_data and (directory / "Code").is_dir():
            return directory
    raise RuntimeError(f"Could not find project root from {start}; expected Data_clean/ (or Data_raw/) and Code/.")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_CLEAN = ROOT / "Data_clean"
TASK1_SCORES = ROOT / "output" / "Task1" / "scores"
TASK2_SCORES = ROOT / "output" / "Task2" / "scores"
TASK3_SCORES = ROOT / "output" / "Task3" / "scores"
TASK3_FIGURES = ROOT / "output" / "Task3" / "figures"
TASK3_REPORT = ROOT / "output" / "Task3" / "report" / "report.md"
OUTPUT_CSV = TASK3_SCORES / "final_customer_intelligence_output.csv"

from Code.config import config as C
from Code.config.utils import PAL, ensure_dirs, load_json, navy_cmap, save_fig, save_json, set_style

# %% [markdown]
# ## Build the customer-level file
#
# Require a one-to-one match between the two score populations. Keep the component scores and available drivers in the final file so each recommendation can be checked without joining extra CSVs.

# %%
def build_customer_intelligence(financial: pd.DataFrame, engagement: pd.DataFrame,
                                snapshot_date: str = "2025-06-05") -> pd.DataFrame:
    """Join the two score populations and assign an explainable matrix action."""
    health_label = {"High": "High", "Medium": "Moderate", "Low": "Low"}      # same wording as Task 1's tiers
    sales_cells = {("High", "High"), ("High", "Medium"), ("Medium", "High"), ("Medium", "Medium")}
    tet_window = ("2024-12-01", "2025-02-01")      # the Tết application wave that defines the customer file
    required_financial = {"SKP_CLIENT", "Financial_Health_Score", "fh_thin_file"}
    required_engagement = {"SKP_CLIENT", "Engagement_Score"}
    missing_financial = required_financial - set(financial.columns)
    missing_engagement = required_engagement - set(engagement.columns)
    if missing_financial or missing_engagement:
        raise ValueError(
            "Missing score fields: "
            f"financial={sorted(missing_financial)}, engagement={sorted(missing_engagement)}"
        )

    for name, frame in (("financial", financial), ("engagement", engagement)):
        if frame["SKP_CLIENT"].isna().any() or not frame["SKP_CLIENT"].is_unique:
            raise ValueError(f"{name} scores must have unique customer keys")

    financial_ids = pd.Index(financial["SKP_CLIENT"])
    engagement_ids = pd.Index(engagement["SKP_CLIENT"])
    if len(financial_ids.difference(engagement_ids)) or len(engagement_ids.difference(financial_ids)):
        raise ValueError("Financial-health and engagement scores must cover the same customer population")

    left = financial.drop(columns=["fh_short"], errors="ignore")
    right = engagement.drop(columns=["eng_short"], errors="ignore")
    result = left.merge(right, on="SKP_CLIENT", how="inner", validate="one_to_one", sort=False)

    for column in ("Financial_Health_Score", "Engagement_Score"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not result[column].between(0, 100).all():
            raise ValueError(f"{column} must contain scores from 0 to 100")

    result["Financial_Health_Tier"] = np.select(
        [result.Financial_Health_Score >= 80, result.Financial_Health_Score >= 50],
        ["High Financial Health", "Moderate Health"],
        default="Financial Stress / Low Health",
    )
    result["Engagement_Tier"] = np.select(
        [result.Engagement_Score >= 80, result.Engagement_Score >= 50],
        ["High Engagement", "Medium Engagement"],
        default="Low / Dormant Engagement",
    )
    result["fh_short"] = pd.cut(
        result.Financial_Health_Score,
        bins=[-0.01, 50, 80, 100.01],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype(str)
    result["eng_short"] = pd.cut(
        result.Engagement_Score,
        bins=[-0.01, 50, 80, 100.01],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype(str)

    actions = {
        ("High", "High"): ("Recognize the relationship and offer a needs-based product review after standard eligibility checks.", "Relationship service", "Retention"),
        ("High", "Medium"): ("Offer a relevant product review with suitability and affordability checks.", "Customer service", "Relationship growth"),
        ("High", "Low"): ("Use an opt-in digital check-in; do not assume low activity means dissatisfaction.", "Digital", "Re-engagement"),
        ("Medium", "High"): ("Provide a tailored service review and check affordability before any credit offer.", "Customer service", "Retention"),
        ("Medium", "Medium"): ("Share concise account and product guidance; contact only when there is a clear customer need.", "Digital", "Relationship growth"),
        ("Medium", "Low"): ("Send a low-frequency, opt-in reminder focused on account utility.", "Digital", "Re-engagement"),
        ("Low", "High"): ("Offer a repayment-support review and suppress new-credit promotion.", "Customer support", "Support"),
        ("Low", "Medium"): ("Share repayment-support options and offer a service contact; suppress new-credit promotion.", "Customer support", "Support"),
        ("Low", "Low"): ("Keep self-service repayment resources available and avoid sales outreach.", "Self-service", "Support"),
    }
    matrix_segments = []
    profiles = []
    next_actions = []
    channels = []
    opportunities = []
    rationales = []

    # Call evidence is used only as per-customer counts: call dates are unreliable (Task 1, chart 1).
    def column(name: str, default=0) -> pd.Series:
        return result[name] if name in result.columns else pd.Series(default, index=result.index, dtype=float)

    ignored_calls = ((column("out_attempts") >= 5) & (column("out_connected") == 0)).to_numpy()
    collection_on_record = (column("collection_calls") > 0).to_numpy()
    last_active = pd.Timestamp(snapshot_date) - pd.to_timedelta(column("days_since_last_activity", np.nan), unit="D")
    one_time_tet = ((last_active >= tet_window[0]) & (last_active < tet_window[1])
                    & (column("active_months_12m", np.nan) <= 1)).to_numpy()

    for i, row in enumerate(result.itertuples(index=False)):
        health = row.fh_short
        engagement_level = row.eng_short
        thin_file = bool(getattr(row, "fh_thin_file"))
        matrix_segments.append(f"{health_label[health]} health × {engagement_level} engagement")
        contact_note = ""

        if health == "Low" and thin_file:
            profile = "Thin file — no signed loan history"
            action = "Offer neutral financial education and explain that affordability checks precede any credit discussion."
            channel = "Digital"
            opportunity = "Credit-history education"
            evidence_note = "The low health score is a thin-file signal, not evidence of missed payments."
        elif health == "Low" and engagement_level == "Low":
            profile = "Financial-stress signal — review repayment evidence"
            action = ("Send proactive repayment reminders and a self-service repayment-plan link by SMS/app; "
                      "no phone calls and no sales outreach.")
            channel = "Digital (SMS/app)"
            opportunity = "Support"
            evidence_note = ("Review the repayment indicators before contacting the customer; the score is not a diagnosis. "
                             "Few customers in this group answer the phone, so reminders go through SMS/app.")
        elif health == "Low":
            profile = "Financial-stress signal — review repayment evidence"
            action, channel, opportunity = actions[(health, engagement_level)]
            evidence_note = "Review the repayment indicators before contacting the customer; the score is not a diagnosis."
        elif thin_file:
            profile = "Thin file above Low — no signed loan yet"
            action, channel, opportunity = actions[(health, engagement_level)]
            evidence_note = ("No repayment history yet: the score rests on application and pre-approval evidence, "
                             "so confirm affordability before any credit offer.")
        elif engagement_level == "Low" and one_time_tet[i]:
            profile = "One-time pre-Tết borrower"
            action = ("Schedule one opt-in reminder for early December, ahead of the next Tết borrowing season; "
                      "no earlier contact unless the customer reaches out.")
            channel = "Digital"
            opportunity = "Seasonal re-engagement"
            evidence_note = "Last active only in the Dec-2024/Jan-2025 Tết window: a seasonal borrower, not a lost customer."
        else:
            profile = "Established credit history"
            action, channel, opportunity = actions[(health, engagement_level)]
            evidence_note = "Use the score as a prioritization cue, not as proof of a customer's needs or response."

        if (health, engagement_level) in sales_cells:
            if ignored_calls[i]:
                channel = "Digital (SMS/app) — phone suppressed"
                contact_note += " 5+ outbound calls went unanswered: contact by SMS/app only, no phone calls."
            if collection_on_record[i]:
                contact_note += " Collection-coded calls are on record: confirm there are no current arrears before any sales contact."

        profiles.append(profile)
        next_actions.append(action)
        channels.append(channel)
        opportunities.append(opportunity)
        driver = getattr(row, "fh_drivers", "")
        driver_text = str(driver).strip() if pd.notna(driver) and str(driver).strip() else "no additional driver summary"
        rationales.append(
            f"Financial health: {row.Financial_Health_Score:.0f}/100 ({row.Financial_Health_Tier}); "
            f"engagement: {row.Engagement_Score:.0f}/100 ({row.Engagement_Tier}). "
            f"Profile: {profile}. Financial-health note: {driver_text}. {evidence_note}{contact_note} "
            f"Recommended action: {action}"
        )

    result["Customer_Profile"] = profiles
    result["Matrix_Segment"] = matrix_segments
    result["Opportunity_Type"] = opportunities
    result["Next_Best_Action"] = next_actions
    result["NBA_Channel"] = channels
    result["Action_Rationale"] = rationales
    return result.drop(columns=["fh_short", "eng_short"])


# %% [markdown]
# ## Quality checks and reporting
#
# Fail the export if either score range, customer key, segment label, or rationale is incomplete. The matrix need not contain all nine cells; an empty cell is a valid portfolio result.

# %%
def qa_checks(output: pd.DataFrame, expected_customers: int) -> list[dict]:
    checks = []

    def add(name: str, condition: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})

    add(
        "One row per customer",
        output.SKP_CLIENT.is_unique and len(output) == expected_customers,
        f"{len(output):,} rows; {expected_customers:,} customers expected",
    )
    required = [
        "Financial_Health_Score", "Financial_Health_Tier", "Engagement_Score", "Engagement_Tier",
        "Matrix_Segment", "Next_Best_Action", "Action_Rationale",
    ]
    add("Required outputs are populated", output[required].notna().all().all(), "Scores, segment, action, and rationale")
    valid_scores = output.Financial_Health_Score.between(0, 100).all() and output.Engagement_Score.between(0, 100).all()
    add("Scores stay within 0–100", valid_scores, "Both scores are bounded to the published scale")
    expected_fh = np.select(
        [output.Financial_Health_Score >= 80, output.Financial_Health_Score >= 50],
        ["High Financial Health", "Moderate Health"],
        default="Financial Stress / Low Health",
    )
    expected_engagement = np.select(
        [output.Engagement_Score >= 80, output.Engagement_Score >= 50],
        ["High Engagement", "Medium Engagement"],
        default="Low / Dormant Engagement",
    )
    tiers_match = (output.Financial_Health_Tier.to_numpy() == expected_fh).all() and (
        output.Engagement_Tier.to_numpy() == expected_engagement
    ).all()
    add("Tier labels match score cut-offs", tiers_match, "80 and 50 point boundaries")
    add(
        "Action and rationale are present",
        output.Next_Best_Action.str.len().gt(0).all() and output.Action_Rationale.str.len().gt(40).all(),
        "Every row has a recommended action and an evidence note",
    )
    thin_rows = output.loc[output.Customer_Profile.eq("Thin file — no signed loan history")]
    thin_file_is_distinct = (
        thin_rows.Next_Best_Action.str.contains("financial education", case=False).all()
        and not thin_rows.Next_Best_Action.str.contains("repayment support", case=False).any()
        and thin_rows.Action_Rationale.str.contains("thin-file signal", case=False).all()
    )
    add("Thin-file cases remain distinct from repayment stress", thin_file_is_distinct, "No repayment failure is inferred from missing history")
    demographic_fields = {"NAME_GENDER", "DATE_BIRTH", "ADDRESS", "NAME_EDUCATION_TYPE"}
    add("Demographics are absent from the output", not demographic_fields.intersection(output.columns), "No demographic fields are used for scoring or actions")
    return checks


def compare_segmentations(output: pd.DataFrame, seed: int = 42) -> dict:
    from scipy.cluster.vq import ClusterError, kmeans2, vq
    from scipy.spatial.distance import cdist

    def unavailable(reason: str) -> dict:
        return {"status": "unavailable", "population_n": int(len(output)), "reason": reason}

    feature_columns = ["Financial_Health_Score", "Engagement_Score"]
    required = {*feature_columns, "Matrix_Segment", "fh_thin_file"}
    missing = required - set(output.columns)
    if missing:
        raise ValueError(f"Missing segmentation comparison fields: {sorted(missing)}")
    if output["Matrix_Segment"].isna().any() or output["fh_thin_file"].isna().any():
        raise ValueError("Matrix segments and thin-file flags must be complete")
    if len(output) < 20:
        return unavailable("At least 20 customers are required for the comparison.")

    matrix_segments = output["Matrix_Segment"].astype(str)
    matrix_cell_count = int(matrix_segments.nunique())
    if matrix_cell_count < 2:
        return unavailable("At least two matrix cells are required for the comparison.")

    features = output[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Segmentation comparison scores must be finite")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(output))
    train_n = min(50_000, int(len(output) * 0.8))
    validation_n = min(10_000, len(output) - train_n)
    train_indices = order[:train_n]
    validation_indices = order[train_n : train_n + validation_n]
    validation_matrix = matrix_segments.iloc[validation_indices]
    matrix_counts = validation_matrix.value_counts()
    if len(matrix_counts) != matrix_cell_count or matrix_counts.min() < 2:
        return unavailable("The validation sample needs at least two customers in every populated matrix cell.")

    means = features[train_indices].mean(axis=0)
    scales = features[train_indices].std(axis=0)
    scales[scales == 0] = 1
    train = (features[train_indices] - means) / scales
    validation = (features[validation_indices] - means) / scales

    def silhouette(points: np.ndarray, labels: np.ndarray) -> float:
        _, group_index, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if len(counts) < 2:
            return float("nan")
        distances = cdist(points, points)
        mean_distances = np.column_stack(
            [distances[:, group_index == group].mean(axis=1) for group in range(len(counts))]
        )
        rows = np.arange(len(points))
        own_counts = counts[group_index]
        own = np.zeros(len(points), dtype=float)
        nonsingleton = own_counts > 1
        own[nonsingleton] = (
            mean_distances[rows[nonsingleton], group_index[nonsingleton]]
            * own_counts[nonsingleton]
            / (own_counts[nonsingleton] - 1)
        )
        mean_distances[rows, group_index] = np.inf
        nearest_other = mean_distances.min(axis=1)
        denominator = np.maximum(own, nearest_other)
        values = np.zeros_like(denominator)
        valid = nonsingleton & (denominator > 0)
        values[valid] = (nearest_other[valid] - own[valid]) / denominator[valid]
        return float(values.mean())

    train_sample = rng.choice(train_n, size=min(3_000, train_n), replace=False)
    candidates = []
    unique_score_pairs = len(np.unique(train, axis=0))
    max_k = min(10, train_n - 1, unique_score_pairs)
    if max_k < 2:
        return unavailable("Fewer than two distinct score pairs occur in the training sample.")
    for k in range(2, max_k + 1):
        best_centers = None
        best_inertia = float("inf")
        for restart in range(3):
            try:
                centers, _ = kmeans2(
                    train,
                    k,
                    iter=50,
                    minit="++",
                    missing="raise",
                    rng=np.random.default_rng(seed + k * 100 + restart),
                )
            except (ValueError, ClusterError):
                continue
            labels, _ = vq(train, centers)
            inertia = float(np.square(train - centers[labels]).sum())
            if inertia < best_inertia:
                best_centers, best_inertia = centers, inertia
        if best_centers is None:
            continue
        labels, _ = vq(train[train_sample], best_centers)
        if len(np.unique(labels)) != k:
            continue
        train_silhouette = silhouette(train[train_sample], labels)
        if np.isfinite(train_silhouette):
            candidates.append(
                {"k": k, "train_silhouette": train_silhouette, "centers": best_centers}
            )
    if not candidates:
        return unavailable("No K-means candidate produced valid clusters.")

    selected = max(candidates, key=lambda candidate: (candidate["train_silhouette"], -candidate["k"]))
    centers = selected["centers"]
    centers = centers[np.lexsort((centers[:, 1], centers[:, 0]))]
    validation_sample = rng.choice(validation_n, size=min(3_000, validation_n), replace=False)
    validation_points = validation[validation_sample]
    matrix_labels = validation_matrix.iloc[validation_sample].astype(str).to_numpy()
    sample_cell_counts = pd.Series(matrix_labels).value_counts()
    if len(sample_cell_counts) != matrix_cell_count or sample_cell_counts.min() < 2:
        return unavailable("The silhouette sample needs at least two customers in every populated matrix cell.")
    kmeans_labels, _ = vq(validation_points, centers)
    if len(np.unique(kmeans_labels)) != selected["k"]:
        return unavailable("The validation sample did not receive customers in every selected K-means cluster.")
    kmeans_silhouette = silhouette(validation_points, kmeans_labels)
    matrix_silhouette = silhouette(validation_points, matrix_labels)
    if not np.isfinite(kmeans_silhouette) or not np.isfinite(matrix_silhouette):
        return unavailable("A finite validation silhouette could not be calculated.")

    equal_candidate = next(
        (candidate for candidate in candidates if candidate["k"] == matrix_cell_count),
        None,
    )
    equal_k_silhouette = None
    if equal_candidate is not None:
        equal_centers = equal_candidate["centers"]
        equal_centers = equal_centers[np.lexsort((equal_centers[:, 1], equal_centers[:, 0]))]
        equal_labels, _ = vq(validation_points, equal_centers)
        if len(np.unique(equal_labels)) == matrix_cell_count:
            equal_k_silhouette = silhouette(validation_points, equal_labels)
            if not np.isfinite(equal_k_silhouette):
                equal_k_silhouette = None

    cell_table = pd.crosstab(pd.Series(kmeans_labels), pd.Series(matrix_labels))
    grid_cell_share = float(cell_table.max(axis=1).sum() / len(matrix_labels))
    all_scaled = (features - means) / scales
    all_clusters, _ = vq(all_scaled, centers)
    profile = output[[*feature_columns, "Matrix_Segment", "fh_thin_file"]].copy()
    profile["cluster"] = all_clusters
    cluster_profiles = []
    for cluster, group in profile.groupby("cluster", sort=True):
        dominant_cell = group["Matrix_Segment"].value_counts()
        cluster_profiles.append(
            {
                "cluster": int(cluster + 1),
                "customers": int(len(group)),
                "population_pct": round(100 * len(group) / len(output), 2),
                "mean_financial_health": round(float(group["Financial_Health_Score"].mean()), 1),
                "mean_engagement": round(float(group["Engagement_Score"].mean()), 1),
                "thin_file_pct": round(100 * float(group["fh_thin_file"].mean()), 2),
                "dominant_matrix_segment": str(dominant_cell.index[0]),
                "dominant_matrix_share_pct": round(100 * float(dominant_cell.iloc[0]) / len(group), 2),
            }
        )

    return {
        "status": "ok",
        "population_n": int(len(output)),
        "train_n": int(train_n),
        "validation_n": int(validation_n),
        "validation_metric_n": int(len(validation_sample)),
        "features": feature_columns,
        "seed": int(seed),
        "selected_k": int(selected["k"]),
        "training_silhouette": float(selected["train_silhouette"]),
        "equal_segments_k": matrix_cell_count,
        "equal_segments_kmeans_silhouette": equal_k_silhouette,
        "k_selection": [
            {"k": int(candidate["k"]), "train_silhouette": float(candidate["train_silhouette"])}
            for candidate in candidates
        ],
        "validation": {
            "kmeans_silhouette": kmeans_silhouette,
            "matrix_silhouette": matrix_silhouette,
            "equal_segments_k": matrix_cell_count,
            "equal_segments_kmeans_silhouette": equal_k_silhouette,
            "kmeans_to_matrix_dominant_cell_pct": round(100 * grid_cell_share, 2),
        },
        "clusters": cluster_profiles,
    }




def write_report(
    output: pd.DataFrame,
    checks: list[dict],
    segmentation_comparison: dict | None = None,
    metrics: dict | None = None,
    snapshot_date: str = "2025-06-05",
) -> str:
    metrics = metrics or {}
    tet_window = ("2024-12-01", "2025-02-01")
    total = len(output)
    segments = output.groupby("Matrix_Segment", observed=True).size().sort_values(ascending=False)
    actions = output.groupby("Next_Best_Action", observed=True).size().sort_values(ascending=False)
    checks_table = pd.DataFrame(checks).to_markdown(index=False)
    segment_table = segments.map(lambda count: f"{count:,}").rename("customers").to_frame().to_markdown()
    action_table = actions.map(lambda count: f"{count:,}").rename("customers").to_frame().to_markdown()

    low_health = output[output.Financial_Health_Tier == "Financial Stress / Low Health"]
    thin_file_low = int(low_health.fh_thin_file.fillna(False).sum())
    established_low = len(low_health) - thin_file_low
    healthy_low_engagement = output[
        output.Financial_Health_Tier.isin(["High Financial Health", "Moderate Health"])
        & (output.Engagement_Tier == "Low / Dormant Engagement")
    ]
    healthy_high_engagement = output[
        output.Financial_Health_Tier.isin(["High Financial Health", "Moderate Health"])
        & (output.Engagement_Tier == "High Engagement")
    ]
    support_reachable = low_health[
        ~low_health.fh_thin_file.fillna(False)
        & (low_health.Engagement_Tier != "Low / Dormant Engagement")
    ]

    def column(name: str, default=np.nan) -> pd.Series:
        return output[name] if name in output.columns else pd.Series(default, index=output.index, dtype=float)

    last_active = pd.Timestamp(snapshot_date) - pd.to_timedelta(column("days_since_last_activity"), unit="D")
    in_tet = (last_active >= tet_window[0]) & (last_active < tet_window[1])
    dormant = output.index.isin(healthy_low_engagement.index)
    tet_share = float(in_tet[dormant].mean()) if dormant.any() else float("nan")
    one_month_share = float((column("active_months_12m")[dormant] <= 1).mean()) if dormant.any() else float("nan")
    text = lambda name: output[name].astype(str) if name in output.columns else pd.Series("", index=output.index)
    one_time = int(text("Customer_Profile").eq("One-time pre-Tết borrower").sum())
    low_low = low_health[~low_health.fh_thin_file.fillna(False) & (low_health.Engagement_Tier == "Low / Dormant Engagement")]
    low_low_answered = float((column("out_connected", 0).loc[low_low.index] > 0).mean()) if len(low_low) else float("nan")
    phone_suppressed = int(text("NBA_Channel").str.contains("phone suppressed", regex=False).sum())
    collection_note = int(text("Action_Rationale").str.contains("Collection-coded calls are on record", regex=False).sum())
    collection_any = column("collection_calls", 0) > 0
    collection_by_tier = {tier: float(collection_any[output.Financial_Health_Tier == tier].mean())
                          for tier in ("High Financial Health", "Financial Stress / Low Health")}
    thin_above_low = int(text("Customer_Profile").eq("Thin file above Low — no signed loan yet").sum())

    def read_metric(name: str) -> float:
        value = metrics.get(name)
        return float(value) if value is not None else float("nan")

    fh_gini = read_metric("fh_gini")
    never_late = read_metric("collection_called_never_late_share")
    eng_auc = read_metric("eng_auc_apply")
    fmt = lambda value, spec: format(value, spec) if np.isfinite(value) else "n/a"
    comparison_section = ""
    if segmentation_comparison is not None:
        if segmentation_comparison.get("status") != "ok":
            comparison_section = "\n".join(
                [
                    "## Model comparison",
                    "",
                    f"Unsupervised K-means comparison unavailable: {segmentation_comparison['reason']}",
                    "The 3 × 3 score matrix remains the operational segmentation; no customer actions were changed.",
                    "",
                ]
            )
        else:
            validation = segmentation_comparison["validation"]
            matrix_silhouette = validation["matrix_silhouette"]
            selected_silhouette = validation["kmeans_silhouette"]
            equal_k = validation["equal_segments_k"]
            equal_silhouette = validation["equal_segments_kmeans_silhouette"]

            def format_silhouette(value: float | None) -> str:
                return f"{value:.3f}" if value is not None and np.isfinite(value) else "n/a"

            method_rows = [
                {
                    "method": "3 × 3 score matrix",
                    "segments": equal_k,
                    "holdout_silhouette": format_silhouette(matrix_silhouette),
                },
                {
                    "method": f"K-means selected k={segmentation_comparison['selected_k']}",
                    "segments": segmentation_comparison["selected_k"],
                    "holdout_silhouette": format_silhouette(selected_silhouette),
                },
            ]
            if segmentation_comparison["selected_k"] != equal_k:
                method_rows.append(
                    {
                        "method": f"K-means matched k={equal_k}",
                        "segments": equal_k,
                        "holdout_silhouette": format_silhouette(equal_silhouette),
                    }
                )
            method_table = pd.DataFrame(method_rows).to_markdown(index=False)
            selection_table = pd.DataFrame(segmentation_comparison["k_selection"]).to_markdown(index=False)
            cluster_table = pd.DataFrame(segmentation_comparison["clusters"]).to_markdown(index=False)
            if (
                equal_silhouette is None
                or not np.isfinite(equal_silhouette)
                or not np.isfinite(matrix_silhouette)
            ):
                equal_k_note = f"No valid equal-count K-means result at k={equal_k} was available; the selected-k silhouette is not directly comparable."
            elif np.isclose(equal_silhouette, matrix_silhouette, atol=0.01):
                equal_k_note = f"At the matched count of {equal_k} segments, the two silhouettes are similar."
            elif equal_silhouette > matrix_silhouette:
                equal_k_note = f"At the matched count of {equal_k} segments, K-means has the higher silhouette."
            else:
                equal_k_note = f"At the matched count of {equal_k} segments, the score matrix has the higher silhouette."
            comparison_section = "\n".join(
                [
                    "## Model comparison",
                    "",
                    f"K-means was fit on the two score axes after standardization using the {segmentation_comparison['train_n']:,}-customer training sample. K was selected from 2–10 by silhouette on a 3,000-row training subsample, then checked on a separate random {segmentation_comparison['validation_n']:,}-customer sample. Validation silhouette uses {segmentation_comparison['validation_metric_n']:,} held-out rows. Seed: {segmentation_comparison['seed']}; means and scales came from training only.",
                    "",
                    method_table,
                    "",
                    f"K-means selected k={segmentation_comparison['selected_k']}. Its silhouette row is not directly comparable with the {equal_k}-cell matrix when the number of segments differs. {equal_k_note}",
                    "",
                    "### Training silhouette by K",
                    "",
                    selection_table,
                    "",
                    f"For selected k={segmentation_comparison['selected_k']}, the dominant 3 × 3 matrix cell covers {validation['kmeans_to_matrix_dominant_cell_pct']:.1f}% of the held-out comparison sample. Cluster profiles below apply the fitted model to all {segmentation_comparison['population_n']:,} customers.",
                    "",
                    cluster_table,
                    "",
                    f"Silhouette measures geometric separation, not repayment, applications, customer response, or action success. Task 3 has no observed post-action outcome label for supervised model selection. Task 1's out-of-time Gini ({fmt(fh_gini, '.3f')}) and Task 2's application AUC ({fmt(eng_auc, '.3f')}) validate score ordering, not the success of Task 3 actions. Keep the 3 × 3 matrix as the operational segmentation: it maps those score tiers to actions and preserves the thin-file safeguard. Use K-means only for exploratory profiles; it does not replace the action rules.",
                    "",
                ]
            )
    return "\n".join(
        [
            "# Task 3 — Match the action to the need",
            "",
            "## Business question",
            "",
            "How should the company act when financial-health evidence and engagement evidence point to different customer needs?",
            "",
            "## Integrated analysis",
            "",
            f"Task 1 ranks observed repayment evidence; Task 2 estimates recent engagement and validates it against later applications (forward) and, cross-sectionally, against call responses. This stage joins both scores one-to-one, assigns the 3 × 3 matrix, distinguishes thin-file customers from established borrowers with stress signals, and attaches an action and row-level rationale. The output contains {total:,} customers and {output.Matrix_Segment.nunique()} populated matrix cells. Health tiers use Task 1's wording (High / Moderate / Low).",
            "",
            "**Call data (team-wide rule, Task 1 chart 1).** CALL timestamps are unreliable for every call type, so call evidence enters this stage only as per-customer counts whose meaning does not depend on dates: unanswered outbound calls and collection-coded calls.",
            "",
            "## Findings and business insight",
            "",
            f"- **Most low-engagement healthy customers are one-time pre-Tết borrowers, not lost customers:** {len(healthy_low_engagement):,} customers ({len(healthy_low_engagement) / max(total, 1):.1%}) have High/Moderate health and Low engagement, but {fmt(tet_share, '.1%')} were last active in Dec-2024/Jan-2025 and {fmt(one_month_share, '.1%')} were active in only one month of the year: the low score reflects the Tết-cohort sampling, not churn. {one_time:,} of them get one opt-in reminder in early December, ahead of the next Tết season, instead of an immediate campaign; do not infer dissatisfaction from inactivity.\n"
            f"- **The Low-health tier is mostly thin-file:** {thin_file_low:,} of {len(low_health):,} Low-health customers ({thin_file_low / max(len(low_health), 1):.1%}) have no signed loan history. Offer neutral education and assess affordability before any credit discussion.\n"
            f"- **Observed repayment stress is a distinct queue:** {established_low:,} Low-health customers have established loan history; "
            f"{len(support_reachable):,} of these {established_low:,} show Medium/High engagement and are reachable for a repayment-support review without mixing in thin-file cases. "
            f"The other {len(low_low):,} have Low engagement and are the riskiest, hardest-to-reach group (only {fmt(low_low_answered, '.1%')} have ever answered an outbound call), so they receive proactive SMS/app repayment reminders rather than a passive self-service note.\n"
            f"- **Contact guards from call counts:** {phone_suppressed:,} customers routed to a sales-type action have 5+ unanswered outbound calls, so their channel is switched to SMS/app. "
            f"{collection_note:,} have collection-coded calls on record; they are not blocked because the code is not a distress marker ({fmt(never_late, '.0%')} of collection-called customers were never late, and the code appears for {fmt(collection_by_tier['High Financial Health'], '.1%')} of High-health vs {fmt(collection_by_tier['Financial Stress / Low Health'], '.1%')} of Low-health customers), but their rationale asks staff to confirm there are no current arrears before any sales contact.\n"
            f"- **Thin files above the Low tier:** {thin_above_low:,} customers with no signed loan score Moderate or High (mostly approved applications the customer cancelled; Task 1 shows their later stress is close to customers with history). They keep the matrix action, labelled as thin file, with an affordability check before any credit offer.\n"
            f"- **Protect engaged relationships:** {len(healthy_high_engagement):,} customers ({len(healthy_high_engagement) / max(total, 1):.1%}) combine High/Moderate health with High engagement; prioritize retention and service quality rather than assuming they need a new credit offer.",
            "",
            "## Matrix distribution",
            "",
            segment_table,
            "",
            "## Proposed actions and workload",
            "",
            action_table,
            "",
            comparison_section,
            "",
            "## Quality checks",
            "",
            checks_table,
            "",
            "## Decision limits",
            "",
            "These are rule-based service-routing suggestions, not estimated treatment effects or credit decisions. Review consent, suitability, affordability, capacity, and policy before activation. The score associations do not establish causes, and the team should verify action wording against its operating procedures.",
            "",
        ]
    )


# %% [markdown]
# ## Run the customer-level join
#
# Run Stages 1–4 first. This stage reads their Parquet outputs, writes one comprehensive customer CSV, and records aggregate counts and QA results as JSON/Markdown rather than several sidecar CSVs.

# %%
def upstream_metrics() -> dict:
    """Validation figures quoted in the report, read from the Task 1 and Task 2 metric files (never typed in)."""
    def read(file_name: str, *keys):
        try:
            value = load_json(DATA_CLEAN / file_name)
            for key in keys:
                value = value[key]
            return value
        except (OSError, KeyError, TypeError, ValueError):
            return None

    return {"fh_gini": read("metrics_phase2.json", "validation", "gini_report"),
            "collection_called_never_late_share": read("metrics_phase2.json", "data_foundations", "collection_called_never_late_share"),
            "eng_auc_apply": read("metrics_phase3.json", "validation", "auc_apply")}


def run() -> dict:
    ensure_dirs(TASK3_SCORES, TASK3_FIGURES, TASK3_REPORT.parent, DATA_CLEAN)
    financial_path = TASK1_SCORES / "fh_scores.parquet"
    engagement_path = TASK2_SCORES / "eng_scores.parquet"
    missing = [str(path) for path in (financial_path, engagement_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run Stages 1, 3, and 4 before Task 3; missing score files: {missing}")

    financial = pd.read_parquet(financial_path)
    engagement = pd.read_parquet(engagement_path)
    customer_intelligence = build_customer_intelligence(financial, engagement, C.SNAPSHOT_DATE)
    customer_intelligence = customer_intelligence.sort_values("SKP_CLIENT").reset_index(drop=True)
    checks = qa_checks(customer_intelligence, len(financial))
    failed = [check for check in checks if check["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"Task 3 QA failed: {failed}")
    customer_intelligence.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "customers": len(customer_intelligence),
        "segment_counts": customer_intelligence["Matrix_Segment"].value_counts().to_dict(),
        "action_counts": customer_intelligence["Next_Best_Action"].value_counts().to_dict(),
        "thin_file_customers": int(customer_intelligence["fh_thin_file"].fillna(False).sum()),
        "qa_checks": checks,
        "customer_output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "output_columns": customer_intelligence.columns.tolist(),
    }
    save_json(summary, DATA_CLEAN / "metrics_phase4.json")
    try:
        segmentation_comparison = compare_segmentations(customer_intelligence)
    except Exception as error:
        segmentation_comparison = {
            "status": "unavailable",
            "reason": f"K-means comparison failed ({type(error).__name__}); the rule-based output was preserved.",
        }
    TASK3_REPORT.write_text(
        write_report(customer_intelligence, checks, segmentation_comparison, upstream_metrics(), C.SNAPSHOT_DATE),
        encoding="utf-8",
    )
    globals()["CUSTOMER_INTELLIGENCE"] = customer_intelligence
    globals()["TASK3_SUMMARY"] = summary
    return summary


# %% [markdown]
# ## Customer opportunity matrix
#
# Each cell combines financial-health and engagement tiers. Counts show the size of each service opportunity, not expected response or profit.

# %%
def plot_segment_matrix(output: pd.DataFrame) -> None:
    health_order = ["High", "Medium", "Low"]
    engagement_order = ["High", "Medium", "Low"]
    health = output.Financial_Health_Tier.map({
        "High Financial Health": "High", "Moderate Health": "Medium", "Financial Stress / Low Health": "Low"
    })
    engagement = output.Engagement_Tier.map({
        "High Engagement": "High", "Medium Engagement": "Medium", "Low / Dormant Engagement": "Low"
    })
    counts = pd.crosstab(health, engagement).reindex(index=health_order, columns=engagement_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    image = ax.imshow(counts.to_numpy(), cmap=navy_cmap())
    maximum = max(int(counts.to_numpy().max()), 1)
    for row in range(len(health_order)):
        for column in range(len(engagement_order)):
            value = int(counts.iloc[row, column])
            text_color = "white" if value > maximum * 0.45 else PAL["navy"]
            ax.text(column, row, f"{value:,}\n{value / max(len(output), 1):.1%}", ha="center", va="center", color=text_color)
    ax.set_xticks(range(len(engagement_order)), engagement_order)
    ax.set_yticks(range(len(health_order)), ["High", "Moderate", "Low"])
    ax.set_xlabel("Engagement tier")
    ax.set_ylabel("Financial-health tier")
    fig.colorbar(image, ax=ax, label="Customers")
    save_fig(fig, TASK3_FIGURES / "01_customer_matrix.png", None)



def low_health_profile_counts(output: pd.DataFrame) -> pd.DataFrame:
    """Count thin-file and established-history customers across engagement tiers."""
    low_health = output.loc[output["Financial_Health_Tier"].eq("Financial Stress / Low Health")].copy()
    low_health["health_profile"] = np.where(
        low_health["fh_thin_file"].fillna(False), "Thin file", "Established history"
    )
    profiles = ["Thin file", "Established history"]
    engagement_tiers = ["High Engagement", "Medium Engagement", "Low / Dormant Engagement"]
    return pd.crosstab(low_health["health_profile"], low_health["Engagement_Tier"]).reindex(
        index=profiles, columns=engagement_tiers, fill_value=0
    )


def plot_score_relationship(output: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    image = ax.hexbin(
        output["Financial_Health_Score"], output["Engagement_Score"],
        gridsize=35, mincnt=1, bins="log", cmap=navy_cmap(), linewidths=0.15, extent=(0, 100, 0, 100),
    )
    for threshold in (50, 80):
        ax.axvline(threshold, color=PAL["grey"], linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axhline(threshold, color=PAL["grey"], linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Financial-health score", ylabel="Engagement score")
    fig.colorbar(image, ax=ax, label="Customers per hexagon (log scale)")
    save_fig(fig, TASK3_FIGURES / "03_score_relationship.png", None)


def plot_low_health_profile(output: pd.DataFrame) -> None:
    counts = low_health_profile_counts(output)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_yscale("symlog", linthresh=1000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{int(value):,}"))
    x = np.arange(len(counts.columns))
    width = 0.36
    thin_bars = ax.bar(x - width / 2, counts.loc["Thin file"], width, label="Thin file", color=PAL["navy"])
    established_bars = ax.bar(x + width / 2, counts.loc["Established history"], width, label="Established history", color=PAL["red"])
    for bars in (thin_bars, established_bars):
        ax.bar_label(bars, labels=[f"{int(value):,}" for value in bars.datavalues], padding=3, fontsize=8)
    ax.set_xticks(x, ["High", "Medium", "Low / dormant"])
    ax.set_ylabel("Customers (symlog scale)")
    ax.set_xlabel("Engagement tier among low-health customers")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save_fig(fig, TASK3_FIGURES / "04_low_health_profile.png", None, left=0.13, bottom=0.14)


# %%
def plot_action_volume(output: pd.DataFrame) -> None:
    from textwrap import fill

    # chart-only short names (keyword + health x engagement cell); the CSV keeps the full action text
    short = {
        "Send a low-frequency, opt-in reminder": "Opt-in reminder\nModerate health · Low engagement",
        "Use an opt-in digital check-in": "Digital check-in\nHigh health · Low engagement",
        "Offer neutral financial education": "Financial education\nLow health · thin file",
        "Share concise account and product guidance": "Light product guidance\nModerate health · Medium engagement",
        "Offer a relevant product review": "Product review\nHigh health · Medium engagement",
        "Send proactive repayment reminders": "Proactive SMS/app repayment reminder\nLow health · Low engagement",
        "Schedule one opt-in reminder for early December": "Early-December seasonal reminder\nOne-time pre-Tết borrowers",
        "Provide a tailored service review": "Tailored service review\nModerate health · High engagement",
        "Recognize the relationship": "Relationship review\nHigh health · High engagement",
        "Share repayment-support options": "Repayment options\nLow health · Medium engagement",
        "Offer a repayment-support review": "Repayment-support review\nLow health · High engagement",
    }
    counts = output.Next_Best_Action.value_counts().sort_values()
    labels = [next((v for k, v in short.items() if str(action).startswith(k)),
                   fill(str(action), width=42, break_long_words=False, break_on_hyphens=False)) for action in counts.index]
    fig, ax = plt.subplots(figsize=(11.5, max(7, 0.7 * len(counts))))
    bars = ax.barh(labels, counts.values, color=PAL["navy"])
    ax.bar_label(bars, labels=[f"{v / 1e3:,.0f}K" if v >= 1000 else f"{v:,}" for v in counts.values], padding=4, fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1e3:,.0f}K"))
    ax.set_xlim(0, counts.max() * 1.1)
    ax.set_xlabel("Customers")
    ax.grid(axis="y", visible=False)
    save_fig(fig, TASK3_FIGURES / "02_action_volume.png", None, left=0.27, right=0.97)


def display_saved_figure(path: Path) -> None:
    if "get_ipython" in globals():
        from IPython.display import Image, display

        display(Image(filename=str(path)))



# %% [markdown]
# ## Export the customer-level file
#
# The report and CSV are generated after the one-to-one join and all QA checks pass.

# %%
if __name__ == "__main__":
    task3_summary = run()

# %% [markdown]
# ## Customer opportunity matrix
#
# Read the matrix as an allocation view. Cell counts describe customers in each health × engagement combination.

# %%
if __name__ == "__main__":
    set_style()
    plot_segment_matrix(CUSTOMER_INTELLIGENCE)
    display_saved_figure(TASK3_FIGURES / "01_customer_matrix.png")

# %% [markdown]
# **Interpretation.** Start with the Low-health row for repayment support. Within High/Moderate health, low engagement mostly marks one-time pre-Tết borrowers, so the re-engagement prompt is timed for early December rather than sent now. Thin-file customers in the Low row receive education rather than a stress label.

# %% [markdown]
# ## Next-best-action workload
#
# Counts show how many customers each rule routes to an action. They do not forecast acceptance, conversion, or financial return.

# %%
if __name__ == "__main__":
    plot_action_volume(CUSTOMER_INTELLIGENCE)
    display_saved_figure(TASK3_FIGURES / "02_action_volume.png")

# %% [markdown]
# ## Score relationship and low-health profile
#
# The score-density view shows joint positioning; the low-health view separates thin-file customers from established borrowers across engagement tiers.

# %%
if __name__ == "__main__":
    plot_score_relationship(CUSTOMER_INTELLIGENCE)
    display_saved_figure(TASK3_FIGURES / "03_score_relationship.png")
    plot_low_health_profile(CUSTOMER_INTELLIGENCE)
    display_saved_figure(TASK3_FIGURES / "04_low_health_profile.png")

# %% [markdown]
# **Interpretation.** Validate capacity and contact policy before activating these routes. The rationale in the customer file explains each assignment; the rules are not a causal uplift model.
