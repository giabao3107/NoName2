# %% [markdown]
# # 2. Exploratory data analysis
#
# Summarize the cleaned tables before scoring. Check customer coverage, time windows, raw missingness, app-session distribution, and observed delinquency by product. These are descriptive patterns, not causal evidence.

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
EDA_FIGURES = ROOT / "output" / "EDA" / "figures"
EDA_REPORT = ROOT / "output" / "EDA" / "eda_report.md"
TABLES = {
    "Customers": ("customer_clean", None),
    "Contracts": ("contract_clean", "cdate"),
    "Leads": ("leads_clean", "lead_time"),
    "Digital behavior": ("behavior_clean", "event_time"),
    "Calls": ("call_clean", "call_start"),
}

from Code.config import config as C
from Code.config.utils import PAL, ensure_dirs, get_con, load_json, save_fig, save_json, set_style


# %% [markdown]
# ## Profile clean tables
#
# Counts and time ranges show what each source can support. Raw missingness stays separate from cleaning decisions; a zero is not assumed to mean missing.

# %%
def pq(name: str) -> str:
    path = DATA_CLEAN / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run Stage 1 first; missing {path.name}.")
    return "'" + path.as_posix().replace("'", "''") + "'"


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def safe_aggregate_count(value) -> int:
    return 0 if pd.isna(value) else int(value)


def profile_tables(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_rows = []
    variable_rows = []
    total_customers = int(con.execute(f"SELECT count(*) FROM {pq('customer_clean')}").fetchone()[0])

    for label, (table_name, date_column) in TABLES.items():
        source = pq(table_name)
        description = con.execute(f"DESCRIBE SELECT * FROM {source}").df()
        expressions = ["count(*) AS row_count", 'count(DISTINCT "SKP_CLIENT") AS customer_count']
        if date_column:
            quoted_date = quote_identifier(date_column)
            expressions.extend([f"min({quoted_date})::VARCHAR AS date_start", f"max({quoted_date})::VARCHAR AS date_end"])
        stats = con.execute(f"SELECT {', '.join(expressions)} FROM {source}").df().iloc[0]
        count = int(stats.row_count)
        distinct_customers = int(stats.customer_count)
        coverage_rows.append(
            {
                "dataset": label,
                "rows": count,
                "distinct_customers": distinct_customers,
                "customer_coverage_pct": round(100 * distinct_customers / max(total_customers, 1), 2),
                "date_start": stats.get("date_start"),
                "date_end": stats.get("date_end"),
            }
        )

        audit_expressions = ["count(*) AS row_count"]
        field_aliases = []
        for index, column in enumerate(description.column_name):
            quoted = quote_identifier(column)
            null_alias = f"null_{index}"
            audit_expressions.append(f"sum(({quoted} IS NULL)::INT) AS {null_alias}")
            placeholder_alias = None
            zero_alias = None
            dtype = str(description.loc[index, "column_type"])
            if dtype == "VARCHAR":
                placeholder_alias = f"placeholder_{index}"
                audit_expressions.append(
                    f"sum((upper(trim({quoted})) IN ('XNA', '', 'NONE', '#NAME?'))::INT) AS {placeholder_alias}"
                )
            elif dtype in ("BIGINT", "INTEGER", "DOUBLE", "FLOAT"):
                zero_alias = f"zero_{index}"
                audit_expressions.append(f"sum(({quoted} = 0)::INT) AS {zero_alias}")
            field_aliases.append((column, dtype, null_alias, placeholder_alias, zero_alias))
        audit = con.execute(f"SELECT {', '.join(audit_expressions)} FROM {source}").df().iloc[0]
        for column, dtype, null_alias, placeholder_alias, zero_alias in field_aliases:
            variable_rows.append(
                {
                    "dataset": label,
                    "column": column,
                    "type": dtype,
                    "rows": int(audit.row_count),
                    "null_count": safe_aggregate_count(audit[null_alias]),
                    "null_pct": round(100 * float(audit[null_alias] or 0) / max(int(audit.row_count), 1), 3),
                    "placeholder_count": safe_aggregate_count(audit[placeholder_alias]) if placeholder_alias else None,
                    "zero_count": safe_aggregate_count(audit[zero_alias]) if zero_alias else None,
                }
            )

    return pd.DataFrame(coverage_rows), pd.DataFrame(variable_rows)


def read_raw_missingness() -> pd.DataFrame:
    metrics_path = DATA_CLEAN / "metrics_phase1.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Run Stage 1 before EDA; metrics_phase1.json is missing.")
    raw_metrics = load_json(metrics_path)
    return pd.DataFrame(raw_metrics.get("missingness", []))




def product_outcomes(con) -> pd.DataFrame:
    return con.execute(
        f"""SELECT PRODUCT AS product, count(*) AS mature_signed_contracts,
                   avg(bad::INT) AS observed_bad_rate
            FROM {pq('contract_clean')}
            WHERE is_signed AND cdate < TIMESTAMP '{C.OBS_COMPLETE_BEFORE}'
            GROUP BY PRODUCT ORDER BY observed_bad_rate DESC"""
    ).df()


def app_session_values(con) -> pd.Series:
    frame = con.execute(f"SELECT app_sessions_wk FROM {pq('customer_clean')}").df()
    return pd.to_numeric(frame.app_sessions_wk, errors="coerce").dropna()


# %% [markdown]
# ## Product outcomes
#
# Include mature signed contracts only; the rates are descriptive.

# %%
def plot_product_outcomes(outcomes: pd.DataFrame) -> None:
    ordered = outcomes.sort_values("observed_bad_rate")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(ordered["product"], ordered.observed_bad_rate * 100, color=PAL["navy"])
    for position, row in enumerate(ordered.itertuples(index=False)):
        ax.text(row.observed_bad_rate * 100 + 0.05, position,
                f"{row.observed_bad_rate:.1%} · n={row.mature_signed_contracts:,}", va="center", fontsize=9)
    ax.set_xlabel("Observed overdue / write-off / sold (%)")
    ax.set_ylabel("Product")
    ax.grid(axis="y", visible=False)
    save_fig(fig, EDA_FIGURES / "02_product_outcomes.png", None)


# %% [markdown]
# ## App-session distribution
#
# Cap the display at p99; keep the full series for summary statistics.

# %%
def plot_app_sessions(app_values: pd.Series) -> None:
    if app_values.empty:
        return
    p99 = float(app_values.quantile(0.99))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(app_values.clip(upper=p99), bins=40, color=PAL["navy"], edgecolor="white")
    ax.axvline(app_values.median(), color=PAL["orange"], linestyle="--", label=f"Median {app_values.median():.2f}")
    ax.set_xlabel("Sessions / week · display capped at p99")
    ax.set_ylabel("Customers")
    ax.legend()
    save_fig(fig, EDA_FIGURES / "03_app_session_distribution.png", None)


# %% [markdown]
# ## Channel mix

# %%
def behavior_channel_counts(con) -> pd.DataFrame:
    return con.execute(
        f"""SELECT channel, count(*) AS events, count(DISTINCT SKP_CLIENT) AS customers
            FROM {pq('behavior_clean')}
            GROUP BY channel ORDER BY events DESC"""
    ).df()


def monthly_activity(con) -> pd.DataFrame:
    return con.execute(
        f"""SELECT month, source, sum(events) AS events FROM (
                SELECT date_trunc('month', cdate) AS month, 'Contracts' AS source, count(*) AS events
                  FROM {pq('contract_clean')} WHERE cdate IS NOT NULL GROUP BY 1
                UNION ALL
                SELECT date_trunc('month', event_time), 'Digital behavior', count(*)
                  FROM {pq('behavior_clean')} GROUP BY 1
                UNION ALL
                SELECT date_trunc('month', lead_time), 'Leads', count(*)
                  FROM {pq('leads_clean')} GROUP BY 1
                UNION ALL
                SELECT date_trunc('month', call_start), 'Calls', count(*)
                  FROM {pq('call_clean')} GROUP BY 1
            ) GROUP BY month, source ORDER BY month, source"""
    ).df()


def write_eda_report(coverage: pd.DataFrame, raw_missing: pd.DataFrame, outcomes: pd.DataFrame,
                     app_values: pd.Series, channels: pd.DataFrame, monthly: pd.DataFrame) -> str:
    missing_top = raw_missing.sort_values("null_pct", ascending=False).head(15) if not raw_missing.empty else raw_missing
    app_p99 = float(app_values.quantile(0.99)) if len(app_values) else np.nan
    above_p99 = int((app_values > app_p99).sum()) if len(app_values) and pd.notna(app_p99) else 0
    monthly_ranges = monthly.groupby("source").agg(first_month=("month", "min"), last_month=("month", "max"), total_events=("events", "sum"))
    return "\n".join(
        [
            "# Exploratory data analysis",
            "",
            "## What the data can support",
            "",
            coverage.to_markdown(index=False),
            "",
            "## Raw missingness: fields with the largest gaps",
            "",
            missing_top.to_markdown(index=False) if not missing_top.empty else "Stage 1 recorded no raw missing values.",
            "",
            "## Mature signed-contract outcomes by product",
            "",
            outcomes.assign(observed_bad_rate_pct=lambda frame: (100 * frame.observed_bad_rate).round(2))
                     .drop(columns="observed_bad_rate").to_markdown(index=False),
            "",
            "## Digital behavior channels",
            "",
            channels.to_markdown(index=False),
            "",
            "## Observation windows",
            "",
            monthly_ranges.to_markdown(),
            "",
            f"## App-session profile\n\nMedian: {app_values.median():.2f} sessions/week; p99: {app_p99:.2f}; above p99: {above_p99:,}.",
            "",
            "## Reading these patterns",
            "",
            "- Coverage is the share of the customer file appearing in each source, not a measure of data quality.\n"
            "- Stage 1 coerces missing app-session values to zero. Treat zero and missing as combined unless source semantics are confirmed.\n"
            "- Product differences and channel counts are descriptive; they do not establish causal effects.\n"
            "- Raw column profiles and missingness are in `Data_clean/metrics_phase1.json`; cleaned profiles are in `Data_clean/metrics_eda.json`.",
            "",
        ]
    )


# %% [markdown]
# ## Figure 1 — customer coverage
#
# The denominator is the distinct customer count in the cleaned customer table.

# %%
def plot_source_coverage(coverage: pd.DataFrame) -> None:
    ordered = coverage.sort_values("customer_coverage_pct")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(ordered.dataset, ordered.customer_coverage_pct, color=PAL["navy"])
    for position, value in enumerate(ordered.customer_coverage_pct):
        ax.text(value + 1, position, f"{value:.1f}%", va="center", color=PAL["navy"])
    ax.set_xlim(0, min(110, max(100, ordered.customer_coverage_pct.max() * 1.12)))
    ax.set_xlabel("Customers represented (% of customer file)")
    ax.grid(axis="y", visible=False)
    save_fig(fig, EDA_FIGURES / "01_source_coverage.png", None)


# %% [markdown]
# ## Figure 2 — observation windows
#
# Each bar shows the date range available for one source. Gaps are extract coverage limits, not customer events.

# %%
def plot_observation_windows(coverage: pd.DataFrame) -> None:
    data = coverage.dropna(subset=["date_start", "date_end"]).copy()
    data["date_start"] = pd.to_datetime(data.date_start)
    data["date_end"] = pd.to_datetime(data.date_end)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    duration = (data.date_end - data.date_start).dt.days
    ax.barh(data.dataset, duration, left=data.date_start, color=PAL["navy"], height=0.58)
    ax.set_xlabel("Observed date range")
    ax.grid(axis="y", visible=False)
    fig.autofmt_xdate()
    save_fig(fig, EDA_FIGURES / "04_observation_windows.png", None)


# %% [markdown]
# ## Figure 3 — raw missingness
#
# Show the fifteen fields with the largest raw missing-value percentages. Missing values remain distinct from recorded zeros.

# %%
def plot_raw_missingness(raw_missing: pd.DataFrame) -> None:
    data = raw_missing.nlargest(15, "null_pct").copy()
    data["field"] = data.dataset + " · " + data.column
    data = data.sort_values("null_pct")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(data.field, data.null_pct, color=PAL["navy"])
    ax.set_xlabel("Raw values missing (%)")
    ax.set_ylabel("Dataset · field")
    ax.grid(axis="y", visible=False)
    save_fig(fig, EDA_FIGURES / "05_raw_missingness.png", None)


# %% [markdown]
# ## Figure 6 — behavior events by channel
#
# Counts show where behavior was recorded; they do not measure satisfaction or intent.

# %%
def plot_behavior_channels(channels: pd.DataFrame) -> None:
    data = channels.sort_values("events")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(data.channel, data.events, color=PAL["navy"])
    ax.set_xlabel("Recorded behavior events")
    ax.set_ylabel("Channel")
    ax.grid(axis="y", visible=False)
    save_fig(fig, EDA_FIGURES / "06_behavior_channels.png", None)


# %% [markdown]
# ## Figure 7 — monthly record volume
#
# The source windows differ, so compare each line within its own observation period.

# %%
def plot_monthly_activity(monthly: pd.DataFrame) -> None:
    palette = {"Contracts": PAL["navy"], "Digital behavior": PAL["orange"], "Leads": PAL["amber"], "Calls": PAL["grey"]}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for source, group in monthly.groupby("source", sort=False):
        ax.plot(group.month, group.events, color=palette[source], label=source, linewidth=1.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Records")
    ax.legend(ncol=2)
    ax.grid(axis="x", visible=False)
    fig.autofmt_xdate()
    save_fig(fig, EDA_FIGURES / "07_monthly_activity.png", None)


def display_saved_figure(path: Path) -> None:
    if "get_ipython" in globals():
        from IPython.display import Image, display

        display(Image(filename=str(path)))


# %% [markdown]
# # EDA — establish the operating picture
#
# Before scoring, check source coverage, observation windows, missingness, and recorded behavior. These limits shape the story in Tasks 1–3.

# %%
if __name__ == "__main__":
    ensure_dirs(EDA_FIGURES, DATA_CLEAN)
    set_style()
    con = get_con()
    coverage, variable_profile = profile_tables(con)
    raw_missing = read_raw_missingness()
    plot_source_coverage(coverage)
    display_saved_figure(EDA_FIGURES / "01_source_coverage.png")

# %% [markdown]
# **Interpretation.** Absence from a table means that source has no recorded row; it does not prove inactivity.

# %% [markdown]
# ## Figure 2 — loan outcomes by product
#
# Include signed contracts only when the early repayment flags have matured.

# %%
if __name__ == "__main__":
    outcomes = product_outcomes(con)
    plot_product_outcomes(outcomes)
    display_saved_figure(EDA_FIGURES / "02_product_outcomes.png")

# %% [markdown]
# **Interpretation.** Product differences are descriptive evidence for Task 1's score design. They do not show that product type causes repayment outcomes.

# %% [markdown]
# ## Figure 3 — app-session distribution
#
# Cap the display at p99; retain the full series for summary statistics.

# %%
if __name__ == "__main__":
    app_values = app_session_values(con)
    plot_app_sessions(app_values)
    display_saved_figure(EDA_FIGURES / "03_app_session_distribution.png")

# %% [markdown]
# **Interpretation.** Session volume is zero-inflated, and Stage 1 maps missing sessions to zero. Treat recorded zero and missing as combined until the source definition is confirmed.

# %% [markdown]
# ## Figure 4 — observation windows
#
# Compare the extract periods before interpreting activity or response rates.

# %%
if __name__ == "__main__":
    plot_observation_windows(coverage)
    display_saved_figure(EDA_FIGURES / "04_observation_windows.png")

# %% [markdown]
# **Interpretation.** Call data extend beyond the scoring and behavior windows. Task 2 therefore treats call responsiveness separately from recency and reserves post-snapshot outcomes for validation.

# %% [markdown]
# ## Figure 5 — raw missingness
#
# Missing values remain separate from recorded zeros.

# %%
if __name__ == "__main__":
    plot_raw_missingness(raw_missing)
    display_saved_figure(EDA_FIGURES / "05_raw_missingness.png")

# %% [markdown]
# **Interpretation.** Task 1 uses explicit missing buckets and a separate thin-file flag rather than treating missing history as repayment failure.

# %% [markdown]
# ## Figure 6 — behavior by channel
#
# Event counts describe where behavior was recorded; they do not measure satisfaction.

# %%
if __name__ == "__main__":
    channels = behavior_channel_counts(con)
    plot_behavior_channels(channels)
    display_saved_figure(EDA_FIGURES / "06_behavior_channels.png")

# %% [markdown]
# **Interpretation.** App and web activity establish an observable service footprint. Task 2 adds recency and response evidence rather than equating volume with intent.

# %% [markdown]
# ## Figure 7 — monthly record volume
#
# The source windows differ, so compare each line within its own observation period.

# %%
if __name__ == "__main__":
    monthly = monthly_activity(con)
    plot_monthly_activity(monthly)
    display_saved_figure(EDA_FIGURES / "07_monthly_activity.png")

# %% [markdown]
# **Interpretation.** Avoid comparing raw counts across systems without accounting for observation-window length; the trend is not a customer-growth rate.

# %% [markdown]
# ## Save the EDA record
#
# Store aggregate profiles in JSON and the interpretation in Markdown. No EDA CSV tables are needed.

# %%
if __name__ == "__main__":
    eda_summary = {
        "coverage": coverage.to_dict("records"),
        "clean_variable_profile": variable_profile.to_dict("records"),
        "raw_missingness_top_15": raw_missing.sort_values("null_pct", ascending=False).head(15).to_dict("records"),
        "mature_product_outcomes": outcomes.to_dict("records"),
        "behavior_channels": channels.to_dict("records"),
        "monthly_activity": monthly.to_dict("records"),
        "app_sessions": {
            "n": len(app_values),
            "median": float(app_values.median()),
            "p99": float(app_values.quantile(0.99)),
            "above_p99": int((app_values > app_values.quantile(0.99)).sum()),
        },
    }
    save_json(eda_summary, DATA_CLEAN / "metrics_eda.json")
    EDA_REPORT.write_text(
        write_eda_report(coverage, raw_missing, outcomes, app_values, channels, monthly),
        encoding="utf-8",
    )
    con.close()
