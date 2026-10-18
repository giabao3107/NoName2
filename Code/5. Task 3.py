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
        if ((directory / "Data").is_dir() or (directory / "Data_raw").is_dir()) and (directory / "Code").is_dir():
            return directory
    raise RuntimeError(f"Could not find project root from {start}; expected Data/ or Data_raw/ and Code/.")


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

from Code.config.utils import PAL, ensure_dirs, navy_cmap, save_fig, save_json, set_style

# %% [markdown]
# ## Build the customer-level file
#
# Require a one-to-one match between the two score populations. Keep the component scores and available drivers in the final file so each recommendation can be checked without joining extra CSVs.

# %%
def build_customer_intelligence(financial: pd.DataFrame, engagement: pd.DataFrame) -> pd.DataFrame:
    """Join the two score populations and assign an explainable matrix action."""
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

    for row in result.itertuples(index=False):
        health = row.fh_short
        engagement_level = row.eng_short
        thin_file = bool(getattr(row, "fh_thin_file"))
        matrix_segments.append(f"{health} health × {engagement_level} engagement")

        if health == "Low" and thin_file:
            profile = "Thin file — no signed loan history"
            action = "Offer neutral financial education and explain that affordability checks precede any credit discussion."
            channel = "Digital"
            opportunity = "Credit-history education"
            evidence_note = "The low health score is a thin-file signal, not evidence of missed payments."
        elif health == "Low":
            profile = "Financial-stress signal — review repayment evidence"
            action, channel, opportunity = actions[(health, engagement_level)]
            evidence_note = "Review the repayment indicators before contacting the customer; the score is not a diagnosis."
        else:
            profile = "Established credit history"
            action, channel, opportunity = actions[(health, engagement_level)]
            evidence_note = "Use the score as a prioritization cue, not as proof of a customer's needs or response."

        profiles.append(profile)
        next_actions.append(action)
        channels.append(channel)
        opportunities.append(opportunity)
        driver = getattr(row, "fh_drivers", "")
        driver_text = str(driver).strip() if pd.notna(driver) and str(driver).strip() else "no additional driver summary"
        rationales.append(
            f"Financial health: {row.Financial_Health_Score:.0f}/100 ({row.Financial_Health_Tier}); "
            f"engagement: {row.Engagement_Score:.0f}/100 ({row.Engagement_Tier}). "
            f"Profile: {profile}. Financial-health note: {driver_text}. {evidence_note} "
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


def write_report(output: pd.DataFrame, checks: list[dict]) -> str:
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
            f"Task 1 ranks observed repayment evidence; Task 2 estimates recent engagement and validates it against later applications and call responses. This stage joins both scores one-to-one, assigns the 3 × 3 matrix, distinguishes thin-file customers from established borrowers with stress signals, and attaches an action and row-level rationale. The output contains {total:,} customers and {output.Matrix_Segment.nunique()} populated matrix cells.",
            "",
            "## Findings and business insight",
            "",
            f"- **Re-engagement is the largest service opportunity:** {len(healthy_low_engagement):,} customers ({len(healthy_low_engagement) / max(total, 1):.1%}) have High/Moderate health and Low engagement. Favor low-cost, opt-in service prompts; do not infer dissatisfaction from inactivity.\n"
            f"- **The Low-health tier is mostly thin-file:** {thin_file_low:,} of {len(low_health):,} Low-health customers ({thin_file_low / max(len(low_health), 1):.1%}) have no signed loan history. Offer neutral education and assess affordability before any credit discussion.\n"
            f"- **Observed repayment stress is a distinct queue:** {established_low:,} Low-health customers have established loan history; "
            f"{len(support_reachable):,} of these {established_low:,} show Medium/High engagement and are reachable for a repayment-support review without mixing in thin-file cases.\n"
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
def run() -> dict:
    ensure_dirs(TASK3_SCORES, TASK3_FIGURES, TASK3_REPORT.parent, DATA_CLEAN)
    financial_path = TASK1_SCORES / "fh_scores.parquet"
    engagement_path = TASK2_SCORES / "eng_scores.parquet"
    missing = [str(path) for path in (financial_path, engagement_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Run Stages 1, 3, and 4 before Task 3; missing score files: {missing}")

    financial = pd.read_parquet(financial_path)
    engagement = pd.read_parquet(engagement_path)
    customer_intelligence = build_customer_intelligence(financial, engagement)
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
    TASK3_REPORT.write_text(write_report(customer_intelligence, checks), encoding="utf-8")
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
            text_color = "white" if value > maximum * 0.55 else PAL["navy"]
            ax.text(column, row, f"{value:,}\n{value / max(len(output), 1):.1%}", ha="center", va="center", color=text_color)
    ax.set_xticks(range(len(engagement_order)), engagement_order)
    ax.set_yticks(range(len(health_order)), health_order)
    ax.set_xlabel("Engagement tier")
    ax.set_ylabel("Financial-health tier")
    fig.colorbar(image, ax=ax, label="Customers")
    save_fig(fig, TASK3_FIGURES / "01_customer_matrix.png", None)


# %%
def plot_action_volume(output: pd.DataFrame) -> None:
    counts = output.Next_Best_Action.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.48 * len(counts))))
    ax.barh(counts.index, counts.values, color=PAL["navy"])
    ax.set_xlabel("Customers")
    ax.grid(axis="y", visible=False)
    save_fig(fig, TASK3_FIGURES / "02_action_volume.png", None)


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
# **Interpretation.** Start with the Low-health row for repayment support. Within High/Medium health, low engagement suggests a lower-cost re-engagement path. Thin-file customers in the Low row receive education rather than a stress label.

# %% [markdown]
# ## Next-best-action workload
#
# Counts show how many customers each rule routes to an action. They do not forecast acceptance, conversion, or financial return.

# %%
if __name__ == "__main__":
    plot_action_volume(CUSTOMER_INTELLIGENCE)
    display_saved_figure(TASK3_FIGURES / "02_action_volume.png")

# %% [markdown]
# **Interpretation.** Validate capacity and contact policy before activating these routes. The rationale in the customer file explains each assignment; the rules are not a causal uplift model.
