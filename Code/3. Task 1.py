# %% [markdown]
# # 3. Customer financial health (Challenge 1)
#
# **Business question.** Which customers can comfortably carry credit, which are under repayment pressure, and what
# evidence supports each verdict?
#
# **Answer in one line.** An explainable 0–100 points scorecard built on five pillars (repayment history, credit
# standing, debt burden, down-payment buffer, pre-approved capacity). It is fitted only on 2023–2024 outcomes and
# validated **out of time** on the Tet-2025 lending wave. The scale is anchored to risk: **50 points = 1.5x the
# average stress rate, 80 points = half of it**, so the High / Moderate / Low tiers have an economic meaning.
#
# **Story of this notebook.**
# 1. *Know the data before scoring it*: who is in the file, where outcomes are censored, which tables cannot be timed.
# 2. *Contract-level evidence*: product risk, roll rates, debt burden, down payment, insurance, credit seeking.
# 3. *Scorecard*: point-in-time features, monotone evidence-based points, published weights.
# 4. *Validation*: out-of-time ranking, benchmarks, stability, thin files, fairness.
# 5. *Population scores*: one row per customer with tier, band, segment and a plain-language reason.
#
# Every chart is followed by a printed **story** computed from the data and a short interpretation. The same stories
# are written to `output/Task1/report/report.md`, so the numbers in the charts, the notebook and the report agree.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:                                   # optional: only used for the gradient-boosting benchmark
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:                    # pragma: no cover
    HistGradientBoostingClassifier = None


def find_project_root():
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    for directory in (start, *start.parents):
        has_data = any((directory / name).is_dir() for name in ("Data", "Data_raw", "Data_clean"))
        if has_data and (directory / "Code").is_dir():
            return directory
    raise RuntimeError(f"Could not find project root from {start}; expected Data_clean/ (or Data_raw/) and Code/.")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_CLEAN = ROOT / "Data_clean"
TASK_FIGURES = ROOT / "output" / "Task1" / "figures"
TASK_SCORES = ROOT / "output" / "Task1" / "scores"
TASK_REPORT = ROOT / "output" / "Task1" / "report" / "report.md"


def pq(name: str) -> str:
    return f"'{(DATA_CLEAN / f'{name}.parquet').as_posix()}'"


from Code.config import config as C
from Code.config.utils import (PAL, TIER_COLORS, apply_card, assign_tier, auc, calibrate_bins, ensure_dirs, get_con, gini,
                               save_fig, save_json, set_style, tint_rgba, two_prop_ztest, wilson_ci)

SOURCE = "BI10 Round 2 data (CONTRACT, CUSTOMER, LEADS, CALL); team analysis"
INST = ", ".join(f"'{p}'" for p in C.INSTALMENT_PRODUCTS)
REV = ", ".join(f"'{p}'" for p in C.REVOLVING_PRODUCTS)
PRODUCTS = ("CD", "TW", "CL", "CW", "BNPL", "CC")

# Out-of-time design: three training windows (outcomes Oct-2023 .. Sep-2024), one later test window (Tet 2025).
TRAIN_WINDOWS = [("2023-10-01", "2024-02-01"), ("2024-02-01", "2024-06-01"), ("2024-06-01", "2024-10-01")]
OOT_START = "2024-10-01"
NEVER_LATE = 9_999            # months_since_delinq for customers with history and no late loan

# Pillar design ----------------------------------------------------------------------------------
FACTORS = {
    # factor                 pillar                    bucket edges                                    risk direction
    "delinq_sev": ("Repayment History", [0, 1, 2, 3, 99], "increasing"),
    "clean_track": ("Repayment History", [0, 1, 2, 3, 5, 8, 10_000], "decreasing"),
    "months_since_delinq": ("Repayment History", [0, 6, 12, 24, 48, 9_000, 10_000], "decreasing"),
    "reject_rate": ("Credit Standing", [0, .0001, .20, .40, .60, 1.01], "increasing"),
    "history_months": ("Credit Standing", [0, 6, 12, 24, 48, 96, 10_000], "decreasing"),
    "dti": ("Debt Burden (DTI)", [0, .10, .20, .25, .30, .35, .40, .50, .60, 1_000], "increasing"),
    "dp_ratio": ("Down-payment Buffer", [0, .10, .30, .50, .80, 1.20, 1_000], "decreasing"),
    "max_offer": ("Pre-approved Capacity", [0, 5e6, 10e6, 20e6, 40e6, 80e6, 1e12], "decreasing"),
}
FACTOR_LABELS = {
    "delinq_sev": "# past late loans (3 = 3+ or write-off)",
    "clean_track": "# loans completed with no overdue",
    "months_since_delinq": "Months since a late loan was signed",
    "reject_rate": "Share of applications rejected",
    "history_months": "Months since first signed loan",
    "dti": "Debt-to-income (instalments / income)",
    "dp_ratio": "Down-payment ratio (CD/TW loans)",
    "max_offer": "Largest pre-approved offer (LEADS)",
}
PILLAR_WEIGHTS = {"Repayment History": 0.35, "Credit Standing": 0.26, "Debt Burden (DTI)": 0.13,
                  "Down-payment Buffer": 0.13, "Pre-approved Capacity": 0.13}
# v1 design kept as a benchmark (six factors, no offer pillar, no delinquency recency, no thin-file split)
V1_FACTORS = ["delinq_sev", "clean_track", "dti", "dp_ratio", "reject_rate", "history_months"]
V1_WEIGHTS = {"Repayment History": 0.40, "Debt Burden (DTI)": 0.15, "Down-payment Buffer": 0.15, "Credit Standing": 0.30}
# Customers without a signed loan have no repayment or tenure evidence; these factors use the thin-file type instead.
THIN_FACTORS = ("delinq_sev", "clean_track", "months_since_delinq", "reject_rate", "history_months")
THIN_LABELS = {"rejected_before": "Thin file: rejected before", "cancelled_only": "Thin file: cancelled only",
               "first_application": "Thin file: first application", "history": "Has loan history"}
BANDS = [(80, 101, "A Thriving"), (65, 80, "B Healthy"), (50, 65, "C Stable"), (35, 50, "D Fragile"), (0, 35, "E Stressed")]
POLICY_CAPS = [
    # (rule name, condition, max FHS) - transparent knock-out rules applied after the points total
    ("Past write-off / debt sold", lambda d: d.any_default == 1, 40.0),
    ("Overdue on a loan still open", lambda d: d.n_open_delinquent >= 1, 49.0),
    ("2+ loans with overdue history", lambda d: d.n_delinquent >= 2, 49.0),
]

STORIES: dict[str, list[str]] = {}


def story(key: str, lines: list[str]) -> None:
    """Print the data-driven story under a chart and keep it for the report."""
    STORIES[key] = lines
    text = "\n".join(f"- {line}" for line in lines)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def display_saved_figure(path: Path) -> None:
    if "get_ipython" in globals():
        from IPython.display import Image, display

        display(Image(filename=str(path)))


def ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def ranged(card: pd.DataFrame) -> pd.DataFrame:
    """Scorecard rows with a value range (the Missing bucket removed)."""
    return card[card.bucket != "Missing"].reset_index(drop=True)


# %% [markdown]
# ## A. Know the data before scoring it
#
# Three facts decide how a score can be built and validated honestly:
# 1. **Who is in the file.** If every customer was selected because they applied at the same moment, the file is a
#    cohort, not the whole book, and "applied again" cannot be an outcome.
# 2. **When outcomes are complete.** Missed-payment flags exist only for instalments 1–4 and only after each
#    instalment falls due; recent loans look clean simply because they are young.
# 3. **Which tables can be timed.** A table whose timestamps are scrambled cannot be used for "before / after" tests.

# %%
def label_complete_before(con) -> tuple[str, pd.DataFrame]:
    """First signing week whose FOURTH_DUE/THIRD_DUE ratio collapses: later loans cannot have all four flags yet."""
    weekly = con.execute(f"""
        SELECT CAST(date_trunc('week', cdate) AS TIMESTAMP) AS week, count(*) n, avg(THIRD_DUE) tpd, avg(FOURTH_DUE) f4
        FROM {pq('contract_clean')} WHERE is_signed AND cdate >= TIMESTAMP '2024-09-30' AND cdate < TIMESTAMP '2025-03-03'
        GROUP BY 1 ORDER BY 1""").df()
    weekly["ratio"] = weekly.f4 / weekly.tpd
    base = weekly[weekly.week < pd.Timestamp("2025-01-01")].ratio.median()
    collapsed = weekly[(weekly.week >= pd.Timestamp("2025-01-01")) & (weekly.ratio < 0.85 * base)]
    weekly.attrs["baseline_ratio"] = float(base)
    return pd.Timestamp(collapsed.week.min()).strftime("%Y-%m-%d"), weekly


def data_foundations(con, label_cut: str) -> dict:
    monthly = con.execute(f"""
        SELECT CAST(date_trunc('month', cdate) AS TIMESTAMP) AS month, count(*) FILTER (WHERE is_signed) signed, count(*) apps
        FROM {pq('contract_clean')} WHERE cdate >= TIMESTAMP '2023-01-01' AND cdate < TIMESTAMP '2025-06-01'
        GROUP BY 1 ORDER BY 1""").df()
    cohort = con.execute(f"""
        WITH f AS (SELECT SKP_CLIENT, min(cdate) first_app,
                          max((cdate >= TIMESTAMP '2024-12-01' AND cdate < TIMESTAMP '2025-02-01')::INT) tet
                   FROM {pq('contract_clean')} GROUP BY 1)
        SELECT count(*) customers, sum(tet) tet_applicants, sum((first_app >= TIMESTAMP '2024-12-01')::INT) first_time
        FROM f""").fetchone()
    base = monthly[(monthly.month >= pd.Timestamp("2024-06-01")) & (monthly.month < pd.Timestamp("2024-12-01"))].apps.mean()
    peak = monthly[monthly.month.isin(pd.to_datetime(["2024-12-01", "2025-01-01"]))].apps.mean()
    # CALL timestamps: if they were real, a collection call could not be dated before the customer's first loan
    calls = con.execute(f"""
        WITH fl AS (SELECT SKP_CLIENT, min(cdate) first_signed FROM {pq('contract_clean')} WHERE is_signed GROUP BY 1),
             cc AS (SELECT SKP_CLIENT, call_start FROM {pq('call_clean')} WHERE disposition_group = 'COLLECTION')
        SELECT CAST(date_trunc('month', first_signed) AS TIMESTAMP) AS first_loan_month, count(*) collection_calls,
               avg((call_start < first_signed)::INT) observed_before_loan,
               date_diff('day', DATE '2025-01-01', CAST(any_value(date_trunc('month', first_signed)) AS DATE)) / 608.0
                   AS uniform_expectation
        FROM cc JOIN fl USING (SKP_CLIENT) WHERE first_signed >= TIMESTAMP '2025-02-01'
        GROUP BY 1 ORDER BY 1""").df()
    never = con.execute(f"""
        SELECT count(DISTINCT SKP_CLIENT), count(*) FROM {pq('call_clean')}
        WHERE disposition_group = 'COLLECTION'
          AND SKP_CLIENT NOT IN (SELECT SKP_CLIENT FROM {pq('contract_clean')} WHERE is_signed)""").fetchone()
    per_month = con.execute(f"""
        SELECT date_trunc('month', call_start) m, avg((disposition_group = 'COLLECTION')::INT) collection_share
        FROM {pq('call_clean')} GROUP BY 1""").df()
    never_late = con.execute(f"""
        WITH cu AS (SELECT SKP_CLIENT, max(any_due) FILTER (WHERE is_signed) due, max(is_default) dflt
                    FROM {pq('contract_clean')} GROUP BY 1)
        SELECT avg((coalesce(due, 0) = 0 AND coalesce(dflt, 0) = 0)::INT) FROM cu
        WHERE SKP_CLIENT IN (SELECT SKP_CLIENT FROM {pq('call_clean')} WHERE disposition_group = 'COLLECTION')""").fetchone()[0]
    # the same test for sales calls: a 'sign successful' call should sit next to an actual signing
    sales_near_signing = con.execute(f"""
        WITH sc AS (SELECT SKP_CLIENT, call_start FROM {pq('call_clean')} WHERE disposition_group = 'SALES_CONVERSION'),
             sg AS (SELECT SKP_CLIENT, cdate FROM {pq('contract_clean')} WHERE is_signed AND cdate >= TIMESTAMP '2024-12-01'),
             j AS (SELECT sc.SKP_CLIENT, sc.call_start, min(abs(date_diff('day', sc.call_start, sg.cdate))) gap
                   FROM sc LEFT JOIN sg USING (SKP_CLIENT) GROUP BY 1, 2)
        SELECT avg((gap <= 14)::INT) FROM j""").fetchone()[0]
    same_day_redial = con.execute(f"""
        WITH c AS (SELECT date_diff('hour', lag(call_start) OVER (PARTITION BY SKP_CLIENT ORDER BY call_start), call_start) gap_h
                   FROM {pq('call_clean')})
        SELECT avg((gap_h < 24)::INT) FROM c WHERE gap_h IS NOT NULL""").fetchone()[0]
    return {"monthly": monthly, "customers": cohort[0], "tet_applicant_share": cohort[1] / cohort[0],
            "sales_conversion_calls_within_14d_of_signing": float(sales_near_signing),
            "same_customer_calls_within_24h": float(same_day_redial),
            "first_time_share": cohort[2] / cohort[0], "peak_multiple": peak / base, "label_cut": label_cut,
            "calls_before_first_loan": calls, "never_signed_collection_customers": never[0],
            "never_signed_collection_calls": never[1],
            "collection_share_range": (float(per_month.collection_share.min()), float(per_month.collection_share.max())),
            "collection_called_never_late_share": float(never_late)}


def chart_foundations(found: dict, weekly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.5, 1.2, 1]})
    m = found["monthly"]
    tet = m.month.isin(pd.to_datetime(["2024-12-01", "2025-01-01"]))
    axes[0].bar(m.month, m.apps / 1e3, width=25, color=np.where(tet, PAL["navy"], PAL["amber"]))
    axes[0].set_ylabel("Applications per month (K)")
    axes[0].set_xlabel("Application month")
    axes[0].text(0.02, 0.95, f"Dec-24/Jan-25 = {found['peak_multiple']:.0f}x a normal month\n"
                             f"{found['tet_applicant_share']:.2%} of customers applied then",
                 transform=axes[0].transAxes, va="top", fontsize=9.5, color=PAL["ink"])
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].grid(axis="x", visible=False)

    w = weekly[weekly.week >= pd.Timestamp("2024-11-25")]
    axes[1].plot(w.week, w.ratio, "o-", color=PAL["navy"])
    cut = pd.Timestamp(found["label_cut"])
    axes[1].axvline(cut, color=PAL["red"], lw=1.5)
    axes[1].axvspan(cut, w.week.max() + pd.Timedelta(days=7), color=PAL["red"], alpha=0.07)
    axes[1].axhline(weekly.attrs["baseline_ratio"], color=PAL["grey"], linestyle=":")
    axes[1].text(cut + pd.Timedelta(days=2), 0.6, f"censored\nfrom {cut:%d %b %Y}", color=PAL["red"], fontsize=9)
    axes[1].set_ylabel("4th-instalment / 3rd-instalment late rate")
    axes[1].set_xlabel("Signing week")
    axes[1].tick_params(axis="x", labelrotation=30)

    cb = found["calls_before_first_loan"]
    x = np.arange(len(cb))
    axes[2].bar(x - 0.2, cb.observed_before_loan * 100, 0.4, color=PAL["navy"], label="Observed")
    axes[2].bar(x + 0.2, cb.uniform_expectation * 100, 0.4, color=PAL["amber"], label="If call dates were random")
    axes[2].set_xticks(x, [f"{d:%b-%y}" for d in pd.to_datetime(cb.first_loan_month)])
    axes[2].set_xlabel("Month of customer's first-ever loan")
    axes[2].set_ylabel("Collection calls dated BEFORE that loan (%)")
    axes[2].legend(fontsize=8.5, loc="upper left")
    axes[2].grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "01_data_foundations.png", None, left=0.05, wspace=0.3, bottom=0.17)


# %% [markdown]
# ## B. Contract-level evidence
#
# Before any customer score, test which repayment patterns the portfolio actually shows. All comparisons use
# signed contracts whose four instalment flags are complete (signed before the censoring cut found above).

# %%
def signed_obs(label_cut: str) -> str:
    return f"(SELECT * FROM {pq('contract_clean')} WHERE is_signed AND cdate < TIMESTAMP '{label_cut}')"


def q1_product_risk(con, obs: str, label_cut: str) -> dict:
    prod = con.execute(f"""
        SELECT PRODUCT, count(*) n_contracts, avg(any_due) delinquency_rate, avg(bad) bad_rate, avg(is_default) default_rate,
               avg(FIRST_DUE) fpd, sum(FIRST_DUE) / nullif(sum(any_due), 0) fpd_share_of_late,
               avg(SECOND_DUE) FILTER (WHERE FIRST_DUE = 1) roll_1_to_2
        FROM {obs} WHERE PRODUCT IN {PRODUCTS} GROUP BY 1 ORDER BY bad_rate DESC""").df()
    tot = con.execute(f"SELECT avg(bad), avg(any_due), count(*) FROM {obs}").fetchone()
    # a write-off needs time to happen: use contracts at least 12 months old at the snapshot
    wo = con.execute(f"""
        SELECT PRODUCT, avg(is_default) FILTER (WHERE FIRST_DUE = 1) wo_after_fpd,
               avg(is_default) FILTER (WHERE any_due = 1 AND FIRST_DUE = 0) wo_after_later_late,
               avg(is_default) FILTER (WHERE any_due = 0) wo_if_never_late,
               count(*) FILTER (WHERE FIRST_DUE = 1) n_fpd
        FROM {pq('contract_clean')}
        WHERE is_signed AND cdate < TIMESTAMP '{C.SNAPSHOT_DATE}' - INTERVAL 365 DAY AND PRODUCT IN {PRODUCTS}
        GROUP BY 1""").df()
    prod = prod.merge(wo, on="PRODUCT")
    # first loan vs repeat loan in the Tet wave and in the five months before it
    fr = con.execute(f"""
        WITH s AS (SELECT *, row_number() OVER (PARTITION BY SKP_CLIENT ORDER BY cdate, SKP_CREDIT_CASE) loan_no
                   FROM {pq('contract_clean')} WHERE is_signed)
        SELECT CASE WHEN cdate >= TIMESTAMP '2024-12-01' THEN 'Tet wave' ELSE 'Jul-Nov 2024' END period, PRODUCT,
               CASE WHEN loan_no = 1 THEN 'First loan' ELSE 'Repeat loan' END borrower, count(*) n, avg(bad) bad_rate
        FROM s WHERE cdate >= TIMESTAMP '2024-07-01' AND cdate < TIMESTAMP '{label_cut}' AND PRODUCT IN {PRODUCTS}
        GROUP BY ALL ORDER BY ALL""").df()
    fr["bad"] = fr.n * fr.bad_rate
    tet = fr[fr.period == "Tet wave"].groupby("borrower")[["bad", "n"]].sum()
    tw = prod.set_index("PRODUCT").loc["TW"]
    cd = prod.set_index("PRODUCT").loc["CD"]
    z, p = two_prop_ztest(tw.bad_rate * tw.n_contracts, tw.n_contracts, cd.bad_rate * cd.n_contracts, cd.n_contracts)
    return {"table": prod, "portfolio_bad_rate": tot[0], "portfolio_delinq_rate": tot[1], "n": tot[2],
            "first_vs_repeat": fr, "tet_first_loan_bad": float(tet.loc["First loan", "bad"] / tet.loc["First loan", "n"]),
            "tet_repeat_loan_bad": float(tet.loc["Repeat loan", "bad"] / tet.loc["Repeat loan", "n"]),
            "tw_vs_cd": {"z": z, "p": p}}


def chart_product(pr: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    tet = pr["first_vs_repeat"][pr["first_vs_repeat"].period == "Tet wave"]
    n = tet.pivot(index="PRODUCT", columns="borrower", values="n").fillna(0)
    fr = tet.pivot(index="PRODUCT", columns="borrower", values="bad_rate").where(n >= 1000)   # hide cells with < 1,000 loans
    fr = fr.loc[fr.max(axis=1).sort_values().index]
    n = n.loc[fr.index]
    y = np.arange(len(fr))
    axes[0].barh(y + 0.2, fr["First loan"].fillna(0) * 100, 0.4, color=PAL["navy"], label="First loan with us")
    axes[0].barh(y - 0.2, fr["Repeat loan"].fillna(0) * 100, 0.4, color=PAL["amber"], label="Repeat loan")
    for i, prod in enumerate(fr.index):
        for col, dy in (("First loan", 0.2), ("Repeat loan", -0.2)):
            v = fr.loc[prod, col]
            text = f"{v:.1%}" if pd.notna(v) else f"too few loans (n={int(n.loc[prod, col]):,})"
            axes[0].text((0 if pd.isna(v) else v * 100) + 0.1, i + dy, text, va="center", fontsize=8.5,
                         color=PAL["ink"] if pd.notna(v) else PAL["grey"])
    axes[0].set_yticks(y, [C.PRODUCT_LABELS[p] for p in fr.index])
    axes[0].set_xlabel("Early stress rate, Tet-wave loans (%)\n(late on instalment 1-4 or written off)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(axis="y", visible=False)

    t = pr["table"].set_index("PRODUCT").loc[list(fr.index)]
    axes[1].barh(y + 0.27, t.wo_after_fpd * 100, 0.27, color=PAL["red"], label="Missed the FIRST instalment")
    axes[1].barh(y, t.wo_after_later_late * 100, 0.27, color=PAL["grey"], label="Late only on a later instalment")
    axes[1].barh(y - 0.27, t.wo_if_never_late * 100, 0.27, color=PAL["green"], label="Never late on instalments 1-4")
    for i, v in enumerate(t.wo_after_fpd):
        axes[1].text(v * 100 + 0.5, i + 0.27, f"{v:.0%}", va="center", fontsize=8.5)
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].set_xlabel("Later written off / debt sold (%)\n(contracts signed 12+ months before the snapshot)")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(axis="y", visible=False)
    save_fig(fig, TASK_FIGURES / "02_product_risk_and_roll_rates.png", None, left=0.17, wspace=0.08, bottom=0.14)


# %% [markdown]
# ### Debt burden and income

# %%
def q2_income_resilience(con, obs: str) -> dict:
    d = con.execute(f"SELECT bad, income_main, income_household, dti_main, dti_household FROM {obs}").df()
    m, h = d.income_main.notna(), d.income_household.notna()
    res = {
        "n_contracts": len(d), "coverage_main": m.mean(), "coverage_household": h.mean(),
        "auc_main_low_income": auc(d.bad[m], -d.income_main[m]),
        "auc_household_low_income": auc(d.bad[h], -d.income_household[h]),
        "auc_dti_main": auc(d.bad[d.dti_main.notna()], d.dti_main[d.dti_main.notna()]),
        "auc_dti_household": auc(d.bad[d.dti_household.notna()], d.dti_household[d.dti_household.notna()]),
    }
    q = pd.qcut(d.income_main[m], 5, duplicates="drop")
    tab = d[m].groupby(q, observed=True).bad.agg(["mean", "size"]).reset_index()
    tab.columns = ["income_main_quintile", "bad_rate", "n"]
    res["quintiles"] = tab.assign(income_main_quintile=tab.income_main_quintile.astype(str)).to_dict("records")
    return res


def q4_dti_tipping(con, obs: str) -> dict:
    step = 0.025
    cur = con.execute(f"""
        SELECT least(floor(dti_main / {step}) * {step}, 0.70) AS dti_bin, count(*) n, avg(bad) bad_rate
        FROM {obs} WHERE dti_main IS NOT NULL GROUP BY 1 ORDER BY 1""").df()
    cur["mid"] = np.where(cur.dti_bin >= 0.70, 0.725, cur.dti_bin + step / 2)
    fit = cur[cur.n >= 300].copy()
    best = None
    for bp in np.arange(0.10, 0.551, 0.025):
        X = np.column_stack([np.ones(len(fit)), fit.mid, np.clip(fit.mid - bp, 0, None)])
        W = np.sqrt(fit.n.values)
        coef, *_ = np.linalg.lstsq(X * W[:, None], fit.bad_rate.values * W, rcond=None)
        sse = float((((X @ coef) - fit.bad_rate.values) ** 2 * fit.n.values).sum())
        if best is None or sse < best[1]:
            best = (bp, sse, coef)
    bp, _, coef = best
    below = con.execute(f"SELECT avg(bad), count(*) FROM {obs} WHERE dti_main < {bp}").fetchone()
    above = con.execute(f"SELECT avg(bad), count(*) FROM {obs} WHERE dti_main >= {bp}").fetchone()
    above40 = con.execute(f"SELECT avg(bad), count(*) FROM {obs} WHERE dti_main >= 0.40").fetchone()
    return {"curve": cur, "breakpoint": float(round(bp, 3)), "coef": coef.tolist(),
            "slope_before_per_10pp": float(coef[1] * 0.1), "slope_after_per_10pp": float((coef[1] + coef[2]) * 0.1),
            "bad_below": below[0], "n_below": below[1], "bad_above": above[0], "n_above": above[1],
            "bad_above_40": above40[0], "n_above_40": above40[1]}


def chart_dti(dti: dict, inc: dict) -> None:
    curve = dti["curve"]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    observed = curve.n >= 300
    ax.plot(curve.mid[observed] * 100, curve.bad_rate[observed] * 100, "o-", color=PAL["navy"], lw=2.2, ms=5,
            label="Observed early stress")
    x = np.linspace(0.0125, 0.7, 100)
    c0, c1, c2 = dti["coef"]
    ax.plot(x * 100, (c0 + c1 * x + c2 * np.clip(x - dti["breakpoint"], 0, None)) * 100,
            color=PAL["grey"], lw=2, linestyle="--", label="Piecewise fit")
    bpx = dti["breakpoint"] * 100
    ax.axvline(bpx, color=PAL["red"], lw=1.5)
    ax.axvspan(bpx, 75, color=PAL["red"], alpha=0.06)
    steeper = ratio(dti["slope_after_per_10pp"], dti["slope_before_per_10pp"])
    ax.text(bpx + 0.8, ax.get_ylim()[1] * 0.93, f"Breakpoint {bpx:.1f}%\nslope x{steeper:.1f} after it", color=PAL["red"], va="top")
    ax.set_xlabel("Debt-to-income ratio of the contract (%)")
    ax.set_ylabel("Early stress rate (%)")
    ax.set_xlim(0, 75)
    ax.legend(loc="upper left")

    labels = ["Low\npersonal\nincome", "Low\nhousehold\nincome", "DTI\n(personal)", "DTI\n(household)"]
    axes[1].tick_params(axis="x", labelsize=8.5)
    values = [inc["auc_main_low_income"], inc["auc_household_low_income"], inc["auc_dti_main"], inc["auc_dti_household"]]
    coverage = [inc["coverage_main"], inc["coverage_household"], inc["coverage_main"], inc["coverage_household"]]
    axes[1].bar(labels, [v - 0.5 for v in values], bottom=0.5, color=PAL["navy"], width=0.6)
    for i, (v, cov) in enumerate(zip(values, coverage)):
        axes[1].text(i, max(v, 0.5) + 0.003, f"AUC {v:.3f}\ncoverage {cov:.1%}", ha="center", fontsize=8.5)
    axes[1].axhline(0.5, color=PAL["grey"], linewidth=1)
    axes[1].set_ylim(0.44, 0.58)
    axes[1].set_ylabel("AUC on mature contracts · 0.5 = no signal")
    axes[1].grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "03_dti_vs_delinquency_tipping_point.png", None, wspace=0.3)


# %% [markdown]
# ### Down payment and insurance

# %%
def q3_mitigation(con, obs: str, label_cut: str) -> dict:
    ins = con.execute(f"""SELECT PRODUCT, flag_ins, count(*) n, avg(bad) bad_rate FROM {obs}
                          WHERE PRODUCT IN ('CD','CL','TW','CC','CW') GROUP BY ALL""").df()
    piv = ins.pivot(index="PRODUCT", columns="flag_ins", values=["bad_rate", "n"])
    rows = []
    for prod in piv.index:
        b0, b1 = piv.loc[prod, ("bad_rate", 0)], piv.loc[prod, ("bad_rate", 1)]
        n0, n1 = piv.loc[prod, ("n", 0)], piv.loc[prod, ("n", 1)]
        z, p = two_prop_ztest(b1 * n1, n1, b0 * n0, n0)
        rows.append({"product": prod, "bad_rate_uninsured": b0, "bad_rate_insured": b1, "n_uninsured": n0,
                     "n_insured": n1, "insured_share": n1 / (n0 + n1), "relative_diff": b1 / b0 - 1, "z": z, "p_value": p})
    conf = con.execute(f"""
        SELECT PRODUCT, CASE WHEN dti_main < .1 THEN 'a' WHEN dti_main < .2 THEN 'b' WHEN dti_main < .3 THEN 'c' ELSE 'd' END band,
               flag_ins, avg(bad) bad_rate
        FROM {obs} WHERE PRODUCT IN ('CD','CL','TW') AND dti_main IS NOT NULL GROUP BY ALL""").df()
    cp = conf.pivot_table(index=["PRODUCT", "band"], columns="flag_ins", values="bad_rate")
    dp = con.execute(f"""
        WITH s AS (SELECT *, row_number() OVER (PARTITION BY SKP_CLIENT ORDER BY cdate, SKP_CREDIT_CASE) loan_no
                   FROM {pq('contract_clean')} WHERE is_signed),
        b AS (SELECT PRODUCT, bad, loan_no,
            CASE WHEN dp_ratio < .1 THEN 0 WHEN dp_ratio < .3 THEN 1 WHEN dp_ratio < .5 THEN 2 WHEN dp_ratio < .8 THEN 3
                 WHEN dp_ratio < 1.2 THEN 4 ELSE 5 END AS bin
            FROM s WHERE cdate < TIMESTAMP '{label_cut}' AND PRODUCT IN ('CD','TW') AND dp_ratio IS NOT NULL)
        SELECT PRODUCT, bin, count(*) n, avg(bad) bad_rate, avg((loan_no > 1)::INT) repeat_share FROM b GROUP BY ALL ORDER BY 1, 2""").df()
    dp["label"] = dp["bin"].map({0: "<10%", 1: "10-30%", 2: "30-50%", 3: "50-80%", 4: "80-120%", 5: "120%+"})
    dp["bad"] = dp.bad_rate * dp.n
    low = dp[dp.bin == 1].set_index("PRODUCT")
    high = dp[dp.bin >= 3].groupby("PRODUCT")[["bad", "n"]].sum()
    return {"insurance": pd.DataFrame(rows), "ins_higher_in_share_of_cells": float((cp[1] > cp[0]).mean()),
            "n_cells": len(cp), "dp": dp,
            "dp_10_30_vs_50plus": {p: float(low.loc[p, "bad_rate"] / (high.loc[p, "bad"] / high.loc[p, "n"])) for p in ("CD", "TW")}}


def chart_mitigation(mit: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7))
    ins = mit["insurance"].set_index("product").loc[["CD", "CL", "CC", "TW", "CW"]]
    x = np.arange(len(ins))
    axes[0].bar(x - 0.19, ins.bad_rate_uninsured * 100, 0.38, color=PAL["navy"], label="Not insured")
    axes[0].bar(x + 0.19, ins.bad_rate_insured * 100, 0.38, color=PAL["red"], label="Insured")
    for i, r in enumerate(ins.itertuples()):
        axes[0].text(i + 0.19, r.bad_rate_insured * 100 + 0.08, f"{r.relative_diff:+.0%}", ha="center", fontsize=8.5)
    axes[0].set_xticks(x, ins.index)
    axes[0].set_xlabel("Product")
    axes[0].set_ylabel("Early stress rate (%)")
    axes[0].legend()
    axes[0].grid(axis="x", visible=False)
    dp = mit["dp"]
    for product, color in (("TW", PAL["navy"]), ("CD", PAL["grey"])):
        g = dp[dp.PRODUCT == product]
        axes[1].plot(g.label, g.bad_rate * 100, "o-", color=color, label=C.PRODUCT_LABELS[product])
    cd0 = dp[(dp.PRODUCT == "CD") & (dp.bin == 0)].iloc[0]
    axes[1].annotate(f"0% down-payment CD:\n{cd0.repeat_share:.0%} repeat customers", (0, cd0.bad_rate * 100),
                     xytext=(0.6, cd0.bad_rate * 100 + 2.2), arrowprops={"arrowstyle": "->", "color": PAL["grey"]}, fontsize=9)
    axes[1].set_xlabel("Down payment / credit amount")
    axes[1].set_ylabel("Early stress rate (%)")
    axes[1].legend()
    save_fig(fig, TASK_FIGURES / "04_insurance_and_downpayment_impact.png", None)


# %% [markdown]
# ## C. Point-in-time customer features and the out-of-time design
#
# Features are frozen *as of* a date and use only what was observable then (a missed instalment counts only after
# 120 days, a write-off only after 12 months, no contract status from the future). The label is stress on the loans
# a customer signs after that date. Three earlier windows train the scorecard; the Tet window is never used for
# fitting the points. It is split at random into a **selection half** (used once, to anchor the 0–100 scale) and a
# **report half** (untouched until the final numbers).

# %%
def features(con, asof: str, production: bool = False) -> pd.DataFrame:
    a = f"TIMESTAMP '{asof}'"
    due_days, status_days = (0, 0) if production else (C.DUE_FLAG_MATURITY_DAYS, C.STATUS_MATURITY_DAYS)
    # in production the real contract status is known; in history only the contractual end date is (no look-ahead)
    open_ = "is_open" if production else f"((PRODUCT IN ({INST}) AND end_date > {a}) OR (PRODUCT IN ({REV}) AND is_open))"
    return con.execute(f"""
    WITH c AS (
      SELECT *, CASE WHEN cdate < {a} - INTERVAL {due_days} DAY THEN any_due ELSE 0 END AS due_k,
                CASE WHEN cdate < {a} - INTERVAL {status_days} DAY THEN is_default ELSE 0 END AS def_k
      FROM {pq('contract_clean')} WHERE cdate < {a}),
    agg AS (SELECT SKP_CLIENT,
      count(*) FILTER (WHERE is_signed) AS n_signed,
      coalesce(sum(due_k) FILTER (WHERE is_signed), 0) AS n_delinquent,
      coalesce(max(def_k) FILTER (WHERE is_signed), 0) AS any_default,
      count(*) FILTER (WHERE is_signed AND end_date < {a} AND due_k = 0 AND def_k = 0) AS n_clean_completed,
      count(*) FILTER (WHERE is_rejected) AS n_rejected,
      count(*) FILTER (WHERE NAME_CREDIT_STATUS = 'Cancelled') AS n_cancelled,
      count(*) AS n_applications,
      count(*) FILTER (WHERE cdate >= {a} - INTERVAL 90 DAY) AS apps_90d,
      max_by(income_main, (cdate, SKP_CREDIT_CASE)) FILTER (WHERE income_main IS NOT NULL) AS income,
      round(coalesce(sum(monthly_obligation) FILTER (WHERE is_signed AND {open_}), 0), 0) AS monthly_obligations,
      count(*) FILTER (WHERE is_signed AND {open_}) AS n_open_contracts,
      count(*) FILTER (WHERE is_signed AND due_k = 1 AND {open_}) AS n_open_delinquent,
      round(avg(dp_ratio) FILTER (WHERE is_signed), 6) AS avg_dp_ratio,
      round(avg(flag_ins) FILTER (WHERE is_signed), 6) AS insurance_share,
      date_diff('month', min(cdate) FILTER (WHERE is_signed), {a}) AS history_months_raw,
      date_diff('month', max(cdate) FILTER (WHERE is_signed AND due_k = 1), {a}) AS months_since_delinq_raw,
      round(sum(amt_credit) FILTER (WHERE is_signed), 0) AS total_credit,
      round(sum(annuity) FILTER (WHERE is_signed AND PRODUCT IN ({INST}) AND {open_}), 0) AS open_annuity
      FROM c GROUP BY 1),
    ld AS (SELECT SKP_CLIENT, max(max_offer) AS max_offer, count(*) AS n_leads
           FROM {pq('leads_clean')} WHERE lead_time < {a} GROUP BY 1)
    SELECT agg.*, ld.max_offer, coalesce(ld.n_leads, 0) AS n_leads FROM agg LEFT JOIN ld USING (SKP_CLIENT)""").df()


def outcome(con, start: str, end: str) -> pd.DataFrame:
    """Stress on the loans signed in [start, end): late on instalment 1-4 or written off. At application the lender
    knows the new instalment and down payment, so they enter the debt-burden and down-payment factors."""
    return con.execute(f"""
        SELECT SKP_CLIENT, max(bad) AS y, round(sum(monthly_obligation), 0) AS new_obligation,
               max_by(income_main, (cdate, SKP_CREDIT_CASE)) AS new_income, round(avg(dp_ratio), 6) AS new_dp,
               arg_min(PRODUCT, cdate) AS new_product
        FROM {pq('contract_clean')}
        WHERE is_signed AND cdate >= TIMESTAMP '{start}' AND cdate < TIMESTAMP '{end}' GROUP BY 1""").df()


def derive(d: pd.DataFrame) -> pd.DataFrame:
    zero = ["n_signed", "n_delinquent", "any_default", "n_clean_completed", "n_rejected", "n_cancelled", "n_applications",
            "apps_90d", "monthly_obligations", "n_open_contracts", "n_open_delinquent", "n_leads"]
    d[zero] = d[zero].fillna(0)
    has = d.n_signed > 0
    d["delinq_sev"] = np.where(has, np.where(d.any_default == 1, 3, d.n_delinquent.clip(upper=3)), np.nan)
    d["clean_track"] = np.where(has, d.n_clean_completed, np.nan)
    d["months_since_delinq"] = np.where(has, d.months_since_delinq_raw.fillna(NEVER_LATE), np.nan)
    d["reject_rate"] = np.where(d.n_applications > 0, d.n_rejected / d.n_applications.replace(0, np.nan), np.nan)
    d["history_months"] = np.where(has, d.history_months_raw, np.nan)
    income = d.income.fillna(d.new_income) if "new_income" in d else d.income
    obligation = d.monthly_obligations + (d.new_obligation.fillna(0) if "new_obligation" in d else 0)
    d["dti"] = np.where(income > 0, obligation / income, np.nan)
    d["dp_ratio"] = d.avg_dp_ratio.fillna(d.new_dp) if "new_dp" in d else d.avg_dp_ratio
    d["thin_type"] = np.select([has, d.n_rejected > 0, d.n_cancelled > 0],
                               ["history", "rejected_before", "cancelled_only"], "first_application")
    return d


def sample(con, asof: str, end: str) -> pd.DataFrame:
    d = outcome(con, asof, end).merge(features(con, asof), on="SKP_CLIENT", how="left")
    return derive(d).assign(window=asof)


# %% [markdown]
# ### Scorecard mechanics
#
# Each factor is cut into business-readable buckets. A bucket's points are linear in the log-odds of its training
# stress rate after a monotone (PAV) smoothing in the economically expected direction: lowest-risk bucket = 100,
# highest-risk bucket = 0. Pillar score = average of its factors; composite = published pillar weights. Customers
# with no signed loan get repayment and standing points from the stress rate of their *thin-file type*
# (rejected before / cancelled only / first application) on the same log-odds scale, so "no history" is not
# confused with "bad history".

# %%
def fit_cards(con, train: pd.DataFrame, factors) -> dict:
    cards = {}
    for f in factors:
        _, edges, direction = FACTORS[f]
        if f == "dti":
            # Customer-level DTI is confounded with track record (experienced borrowers carry more loans), so DTI points
            # come from the contract-level DTI curve of loans signed before the test window.
            cl = con.execute(f"""SELECT dti_main x, bad y FROM {pq('contract_clean')}
                                 WHERE is_signed AND dti_main IS NOT NULL
                                   AND cdate < TIMESTAMP '{OOT_START}' - INTERVAL {C.DUE_FLAG_MATURITY_DAYS} DAY""").df()
            cards[f] = calibrate_bins(cl.x, cl.y, edges, direction)
        else:
            cards[f] = calibrate_bins(train[f], train.y, edges, direction)
    return cards


def thin_points(train: pd.DataFrame, cards: dict) -> dict:
    br = train[train.thin_type != "history"].groupby("thin_type").y.mean().clip(1e-4, 1 - 1e-4)
    lo = np.log(br / (1 - br))
    out = {}
    for f in THIN_FACTORS:
        if f in cards:
            lmax, lmin = cards[f].log_odds.max(), cards[f].log_odds.min()
            out[f] = (100 * (lmax - lo) / (lmax - lmin)).clip(0, 100).round(1)
    return out


def factor_points(d: pd.DataFrame, cards: dict, thin: dict | None) -> pd.DataFrame:
    pts = pd.DataFrame({f: apply_card(d[f], card) for f, card in cards.items()}, index=d.index)
    if thin:
        m = d.thin_type != "history"
        for f, tp in thin.items():
            if f in pts:
                pts.loc[m, f] = d.loc[m, "thin_type"].map(tp).fillna(pts.loc[m, f])
    return pts


def pillar_scores(pts: pd.DataFrame, weights: dict) -> pd.DataFrame:
    return pd.DataFrame({p: pts[[f for f in pts if FACTORS[f][0] == p]].mean(axis=1) for p in weights}, index=pts.index)


def composite(pts: pd.DataFrame, weights: dict) -> pd.Series:
    pil = pillar_scores(pts, weights)
    return sum(w * pil[p] for p, w in weights.items()) / sum(weights.values())


def fit_logit(x, y, iters=50):
    """Logistic regression with intercept by Newton-Raphson (x: one column or a matrix)."""
    X = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def fit_logit_1d(x, y):
    """Two-parameter logistic fit that translates the composite into a stress probability."""
    return fit_logit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def risk_scale(comp: pd.Series, y: pd.Series) -> dict:
    """Anchor the 0-100 scale to risk: 50 points = 1.5x the average stress rate, 80 points = 0.5x."""
    b = fit_logit_1d(comp.values / 100, y.values)
    p_bar = float(y.mean())
    logit = lambda p: np.log(p / (1 - p))
    l50, l80 = logit(1.5 * p_bar), logit(0.5 * p_bar)
    slope = 30 / (l50 - l80)                     # FHS points per unit of log-odds
    return {"b0": float(b[0]), "b1": float(b[1]), "p_bar": p_bar, "l50": float(l50), "l80": float(l80),
            "points_per_logit": float(slope), "points_to_double_odds": float(slope * np.log(2))}


def to_fhs(comp: pd.Series, scale: dict) -> pd.Series:
    logodds = scale["b0"] + scale["b1"] * comp / 100
    return (50 + scale["points_per_logit"] * (scale["l50"] - logodds)).clip(0, 100).round(1)


def fhs_to_pd(fhs: pd.Series, scale: dict) -> pd.Series:
    logodds = scale["l50"] - (fhs - 50) / scale["points_per_logit"]
    return 1 / (1 + np.exp(-logodds))


def apply_policy_caps(d: pd.DataFrame, score_col: str) -> pd.DataFrame:
    d["fh_policy_cap"] = ""
    for name, cond, cap in POLICY_CAPS:
        m = cond(d).fillna(False).astype(bool) & (d[score_col] > cap)
        d.loc[m, score_col] = cap
        d.loc[m & (d["fh_policy_cap"] == ""), "fh_policy_cap"] = name
    return d


def score(d: pd.DataFrame, cards: dict, thin: dict | None, scale: dict | None, weights=PILLAR_WEIGHTS) -> pd.DataFrame:
    pts = factor_points(d, cards, thin)
    out = d.copy()
    for f in pts:
        out[f"pts_{f}"] = pts[f]
    pil = pillar_scores(pts, weights)
    for p in weights:
        out[f"pillar_{p}"] = pil[p].round(1)
    out["composite"] = composite(pts, weights)
    if scale is not None:
        out["Financial_Health_Score"] = to_fhs(out.composite, scale)
        apply_policy_caps(out, "Financial_Health_Score")
    return out


def band(s: pd.Series) -> pd.Series:
    return assign_tier(s, BANDS)


# %% [markdown]
# ## D. Validation helpers
#
# Ranking is measured with Gini (= 2·AUC − 1) on the report half of the Tet window. Benchmarks answer "is the simple
# scorecard leaving accuracy on the table?": the v1 design, repayment history alone, a logistic regression and a
# gradient-boosting model on the same inputs, equal weights, a ±25% weight perturbation and a leave-one-pillar-out
# ablation.

# %%
def validation(train: pd.DataFrame, oot: pd.DataFrame, cards: dict, thin: dict, scale: dict, label_cut: str) -> dict:
    rep = oot[oot.half == "report"].copy()
    hist = rep.n_signed > 0
    res = {"n_train": len(train), "bad_rate_train": float(train.y.mean()), "n_oot": len(oot),
           "n_report": len(rep), "bad_rate_report": float(rep.y.mean()),
           "train_windows": TRAIN_WINDOWS, "oot_window": [OOT_START, label_cut]}
    g = lambda frame, s: float(gini(frame.y, -s))
    res["gini_train_in_time"] = g(train, train.composite)
    res["gini_report"] = g(rep, rep.Financial_Health_Score)
    res["gini_report_history"] = g(rep[hist], rep.Financial_Health_Score[hist])
    res["gini_report_thin"] = g(rep[~hist], rep.Financial_Health_Score[~hist])
    res["auc_report"] = float(auc(rep.y, -rep.Financial_Health_Score))
    s = rep.sort_values("Financial_Health_Score")
    res["ks_report"] = float(np.max(np.abs((s.y.cumsum() / s.y.sum()).values - ((1 - s.y).cumsum() / (1 - s.y).sum()).values)))

    # Benchmarks: all trained on the same three windows and scored on the same report half
    pil = rep[[f"pillar_{p}" for p in PILLAR_WEIGHTS]].set_axis(list(PILLAR_WEIGHTS), axis=1)
    bench = {}
    v1 = score(rep, {f: cards[f] for f in V1_FACTORS}, None, None, V1_WEIGHTS).composite
    bench["v1 design (6 factors)"] = (g(rep, v1), g(rep[hist], v1[hist]))
    rp = pil["Repayment History"]
    bench["Repayment history only"] = (g(rep, rp), g(rep[hist], rp[hist]))
    eq = pil.mean(axis=1)
    bench["Equal pillar weights"] = (g(rep, eq), g(rep[hist], eq[hist]))
    beta = fit_logit(factor_points(train, cards, thin).to_numpy() / 100, train.y.to_numpy())
    p = pd.Series(np.column_stack([np.ones(len(rep)), factor_points(rep, cards, thin).to_numpy() / 100]) @ beta, index=rep.index)
    bench["Logistic regression, same buckets"] = (g(rep, -p), g(rep[hist], -p[hist]))
    if HistGradientBoostingClassifier is not None:
        cats = train.thin_type.astype("category").cat.categories
        raw = list(FACTORS) + ["thin_code"]
        tr_raw = train.assign(thin_code=pd.Categorical(train.thin_type, categories=cats).codes)
        rp_raw = rep.assign(thin_code=pd.Categorical(rep.thin_type, categories=cats).codes)
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=C.RANDOM_SEED).fit(tr_raw[raw], train.y)
        p = pd.Series(gbm.predict_proba(rp_raw[raw])[:, 1], index=rep.index)
        bench["Gradient boosting, same raw inputs"] = (g(rep, -p), g(rep[hist], -p[hist]))
    bench["Final scorecard"] = (res["gini_report"], res["gini_report_history"])
    res["benchmarks"] = {k: {"gini_all": v[0], "gini_history": v[1]} for k, v in bench.items()}

    res["ablation_gini_without_pillar"] = {
        k: g(rep, sum(w * pil[q] for q, w in PILLAR_WEIGHTS.items() if q != k)) for k in PILLAR_WEIGHTS}
    rng = np.random.default_rng(C.RANDOM_SEED)
    base = sum(w * pil[q] for q, w in PILLAR_WEIGHTS.items())
    ginis, rhos = [], []
    for _ in range(200):
        s2 = sum(w * rng.uniform(0.75, 1.25) * pil[q] for q, w in PILLAR_WEIGHTS.items())
        ginis.append(g(rep, s2))
        rhos.append(stats.spearmanr(base, s2).statistic)
    res["weight_perturbation"] = {"gini_min": min(ginis), "gini_max": max(ginis), "rank_corr_min": float(min(rhos))}

    # Calibration, tiers and bands on the report half
    rep["pd_hat"] = fhs_to_pd(rep.Financial_Health_Score, scale)
    rep["tier"] = assign_tier(rep.Financial_Health_Score, C.FH_TIERS).map(C.FH_SHORT)
    rep["band"] = band(rep.Financial_Health_Score)
    rep["decile"] = pd.qcut(rep.Financial_Health_Score.rank(method="first"), 10, labels=False) + 1
    res["decile_table"] = rep.groupby("decile").agg(fhs_min=("Financial_Health_Score", "min"),
        fhs_max=("Financial_Health_Score", "max"), n=("y", "size"), bad_rate=("y", "mean"),
        expected=("pd_hat", "mean")).reset_index().to_dict("records")
    tt = rep.groupby("tier").agg(n=("y", "size"), bad_rate=("y", "mean")).reindex(["High", "Medium", "Low"])
    tt["x_average"] = tt.bad_rate / rep.y.mean()
    res["tier_table"] = tt.reset_index().to_dict("records")
    bt = rep.groupby("band").agg(n=("y", "size"), bad_rate=("y", "mean")).reindex([b for *_, b in BANDS])
    bt["x_average"] = bt.bad_rate / rep.y.mean()
    res["band_table"] = bt.reset_index().to_dict("records")
    capped = rep.fh_policy_cap != ""
    res["policy_capped_share"] = float(capped.mean())
    res["policy_capped_bad_rate"] = float(rep.y[capped].mean())
    res["thin_type_table"] = rep.groupby("thin_type").agg(n=("y", "size"), bad_rate=("y", "mean"),
                                                          median_fhs=("Financial_Health_Score", "median")).reset_index().to_dict("records")
    res["report_frame"] = rep
    return res


def psi(expected: pd.Series, actual: pd.Series, edges=np.linspace(0, 100, 11)) -> float:
    e = np.histogram(expected.clip(0, 100), edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual.clip(0, 100), edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def behaviour_signals(con, oot: pd.DataFrame, label_cut: str) -> dict:
    stack = con.execute(f"""
        WITH s AS (SELECT SKP_CLIENT, SKP_CREDIT_CASE, cdate, bad FROM {pq('contract_clean')}
                   WHERE is_signed AND cdate >= TIMESTAMP '2024-12-01' AND cdate < TIMESTAMP '{label_cut}')
        SELECT least(cnt, 3) AS prior_apps_30d, count(*) n, avg(bad) bad_rate FROM (
            SELECT s.SKP_CREDIT_CASE, any_value(s.bad) bad, count(c.SKP_CREDIT_CASE) cnt FROM s
            LEFT JOIN {pq('contract_clean')} c ON c.SKP_CLIENT = s.SKP_CLIENT AND c.cdate < s.cdate
                 AND c.cdate >= s.cdate - INTERVAL 30 DAY
            GROUP BY 1) GROUP BY 1 ORDER BY 1""").df()
    h = oot[oot.n_signed > 0]
    paradox = h.groupby(h.n_open_contracts.clip(upper=3)).y.agg(["mean", "size"]).reset_index()
    offer_band = pd.cut(h.max_offer, [0, 10e6, 40e6, 80e6, 1e12], right=False,
                        labels=["<10M", "10-40M", "40-80M", "80M+"]).cat.add_categories("No offer").fillna("No offer")
    offer = h.groupby(offer_band, observed=True).y.agg(["mean", "size"]).reindex(["No offer", "<10M", "10-40M", "40-80M", "80M+"]).reset_index()
    offer.columns = ["offer_band", "mean", "size"]
    return {"stacking": stack, "paradox": paradox, "offer": offer, "history_bad_rate": float(h.y.mean())}


def chart_behaviour(beh: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    st = beh["stacking"]
    axes[0].bar(["0", "1", "2", "3+"], st.bad_rate * 100, color=[PAL["navy"], PAL["navy"], PAL["red"], PAL["red"]])
    for i, r in enumerate(st.itertuples()):
        axes[0].text(i, r.bad_rate * 100 + 0.1, f"{r.bad_rate:.1%}\nn={r.n / 1e3:,.1f}K", ha="center", fontsize=8.5)
    axes[0].set_xlabel("Other applications in the 30 days before\nsigning a Tet-wave loan")
    axes[0].set_ylabel("Early stress rate (%)")
    axes[0].set_ylim(0, st.bad_rate.max() * 125)
    axes[0].grid(axis="x", visible=False)

    px = beh["paradox"]
    axes[1].bar(["0", "1", "2", "3+"], px["mean"] * 100, color=PAL["grey"])
    for i, r in enumerate(px.itertuples()):
        axes[1].text(i, r.mean * 100 + 0.05, f"{r.mean:.2%}\nn={r.size / 1e3:,.0f}K", ha="center", fontsize=8.5)
    axes[1].set_xlabel("Loans already open when applying")
    axes[1].set_ylabel("Early stress on the new loan (%)")
    axes[1].set_ylim(0, px["mean"].max() * 130)
    axes[1].grid(axis="x", visible=False)

    of = beh["offer"]
    axes[2].bar(of.offer_band, of["mean"] * 100, color=[PAL["amber"]] + [PAL["navy"]] * 4)
    for i, r in enumerate(of.itertuples()):
        axes[2].text(i, r.mean * 100 + 0.05, f"{r.mean:.2%}", ha="center", fontsize=8.5)
    axes[2].axhline(beh["history_bad_rate"] * 100, color=PAL["red"], linestyle="--", lw=1.2)
    axes[2].set_xlabel("Largest pre-approved offer before the window (VND)")
    axes[2].set_ylabel("Early stress on the new loan (%)")
    axes[2].grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "05_behaviour_signals.png", None, wspace=0.3, bottom=0.17)


BUCKET_LABELS = {
    "delinq_sev": ["0", "1", "2", "3+/WO"], "clean_track": ["0", "1", "2", "3-4", "5-7", "8+"],
    "months_since_delinq": ["<6", "6-12", "12-24", "24-48", "48+", "never"],
    "reject_rate": ["0%", "<20%", "20-40", "40-60", "60%+"], "history_months": ["<6", "6-12", "12-24", "24-48", "48-96", "96+"],
    "dti": ["<10", "10-20", "20-25", "25-30", "30-35", "35-40", "40-50", "50-60", "60+"],
    "dp_ratio": ["<10%", "10-30", "30-50", "50-80", "80-120", "120+"],
    "max_offer": ["<5M", "5-10M", "10-20M", "20-40M", "40-80M", "80M+"],
}
MISSING_LABELS = {"dti": "n/a", "dp_ratio": "no\nCD/TW", "max_offer": "no\noffer"}


def chart_factors(cards: dict) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, f in zip(axes.ravel(), FACTORS):
        card = cards[f]
        rows = ranged(card)
        edges = FACTORS[f][1]
        labels = [BUCKET_LABELS[f][edges.index(lo)] for lo in rows.lo]
        values, pts = list(rows.bad_rate * 100), list(rows.points)
        if f in MISSING_LABELS and (card.bucket == "Missing").any():
            miss = card[card.bucket == "Missing"].iloc[0]
            labels.append(MISSING_LABELS[f]); values.append(miss.bad_rate * 100); pts.append(miss.points)
        colors = [PAL["green"] if p >= 80 else PAL["amber"] if p >= 50 else PAL["red"] for p in pts]
        ax.bar(range(len(values)), values, color=colors, edgecolor="white")
        for i, (v, p) in enumerate(zip(values, pts)):
            ax.text(i, v, f"{p:.0f}", ha="center", va="bottom", fontsize=8, color=PAL["grey"])
        ax.set_xticks(range(len(values)), labels, fontsize=7.5)
        ax.set_ylabel("Training stress (%)", fontsize=8)
        ax.set_title(f"{FACTOR_LABELS[f]}\n[{FACTORS[f][0]}]", fontsize=9.5, loc="left")
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="x", visible=False)
    ax = axes.ravel()[-1]
    w = pd.Series(PILLAR_WEIGHTS).sort_values()
    ax.barh(w.index, w.values * 100, color=PAL["navy"])
    for i, v in enumerate(w.values):
        ax.text(v * 100 + 0.5, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_title("Published pillar weights", fontsize=9.5, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(hspace=0.75, wspace=0.32)
    save_fig(fig, TASK_FIGURES / "06_key_factor_impacts.png", None, top=0.95, bottom=0.07, left=0.06)


def chart_validation(val: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1.2, 1]})
    dec = pd.DataFrame(val["decile_table"])
    axes[0].bar(dec.decile, dec.bad_rate * 100, color=PAL["navy"], label="Observed (report half, never used)")
    axes[0].plot(dec.decile, dec.expected * 100, "o--", color=PAL["red"], label="Expected from the score")
    for r in dec.itertuples():
        axes[0].text(r.decile, r.bad_rate * 100 + 0.12, f"{r.bad_rate:.1%}", ha="center", fontsize=8)
    axes[0].set_xticks(dec.decile, [f"D{int(d)}\n{lo:.0f}-{hi:.0f}" for d, lo, hi in zip(dec.decile, dec.fhs_min, dec.fhs_max)], fontsize=8)
    axes[0].set_xlabel("Score decile (D1 = lowest FHS) and score range")
    axes[0].set_ylabel("Early stress rate, Tet-2025 loans (%)")
    axes[0].legend()
    axes[0].grid(axis="x", visible=False)

    b = pd.DataFrame(val["benchmarks"]).T.sort_values("gini_all")
    y = np.arange(len(b))
    final = b.index == "Final scorecard"
    axes[1].barh(y + 0.2, b.gini_all, 0.4, color=PAL["navy"], label="All customers")
    axes[1].barh(y - 0.2, b.gini_history, 0.4, color=PAL["amber"], label="Customers with loan history")
    for i, (a_, h_) in enumerate(zip(b.gini_all, b.gini_history)):
        axes[1].text(a_ + 0.004, i + 0.2, f"{a_:.3f}", va="center", fontsize=8.5)
        axes[1].text(h_ + 0.004, i - 0.2, f"{h_:.3f}", va="center", fontsize=8.5)
    axes[1].set_yticks(y, b.index, fontsize=9)
    for label in axes[1].get_yticklabels():
        label.set_fontweight("bold" if label.get_text() == "Final scorecard" else "normal")
    axes[1].set_xlim(0, max(b.gini_all.max(), b.gini_history.max()) * 1.18)
    axes[1].set_xlabel("Out-of-time Gini (higher = better ranking)")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=8.5)
    axes[1].grid(axis="y", visible=False)
    save_fig(fig, TASK_FIGURES / "07_out_of_time_validation.png", None, left=0.06, wspace=0.55, bottom=0.14)


def chart_distribution(pop: pd.DataFrame, val: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1.4, 1]})
    bins = np.arange(0, 101, 2)
    for tier, (lo, hi) in {"Low": (0, 50), "Medium": (50, 80), "High": (80, 101)}.items():
        v = pop.Financial_Health_Score[(pop.Financial_Health_Score >= lo) & (pop.Financial_Health_Score < hi)]
        axes[0].hist(v, bins=bins, color=TIER_COLORS[tier], alpha=0.9)
    axes[0].hist(pop.Financial_Health_Score[pop.fh_thin_file], bins=bins, histtype="step", color=PAL["ink"], lw=1.2,
                 label="of which thin file")
    rejected = pop.Financial_Health_Score[pop.fh_thin_type == "rejected_before"]
    axes[0].annotate(f"{len(rejected) / 1e3:,.0f}K thin files rejected before:\nno repayment evidence, median {rejected.median():.0f}",
                     (rejected.median(), axes[0].get_ylim()[1] * 0.8), xytext=(3, axes[0].get_ylim()[1] * 0.62),
                     arrowprops={"arrowstyle": "->", "color": PAL["grey"]}, fontsize=9)
    shares = pop.fh_short.value_counts(normalize=True)
    for tier, x in [("Low", 25), ("Medium", 65), ("High", 90)]:
        axes[0].text(x, axes[0].get_ylim()[1] * 0.92, f"{tier} · {shares.get(tier, 0):.1%}", ha="center",
                     color=PAL["grey"] if tier == "Medium" else TIER_COLORS[tier], fontweight="bold")
    for x in (50, 80):
        axes[0].axvline(x, color=PAL["ink"], linestyle=":")
    axes[0].set_xlabel("Financial Health Score at the snapshot (0–100)")
    axes[0].set_ylabel("Customers")
    axes[0].legend(loc="center left")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}K"))

    bt = pd.DataFrame(val["band_table"]).set_index("band")
    colors = [PAL["navy"], tint_rgba(PAL["navy"], 0.55), PAL["amber"], tint_rgba(PAL["red"], 0.55), PAL["red"]]
    axes[1].bar(range(len(bt)), bt.bad_rate * 100, color=colors, width=0.65)
    for i, r in enumerate(bt.itertuples()):
        axes[1].text(i, r.bad_rate * 100 + 0.15, f"{r.bad_rate:.1%}\n{r.x_average:.1f}x avg", ha="center", fontsize=8.5)
    axes[1].axhline(val["bad_rate_report"] * 100, color=PAL["grey"], linestyle="--", lw=1.2)
    axes[1].set_xticks(range(len(bt)), [b.replace(" ", "\n", 1) for b in bt.index], fontsize=8.5)
    axes[1].set_ylabel("Out-of-time early stress rate (%)")
    axes[1].set_ylim(0, bt.bad_rate.max() * 125)
    axes[1].grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "08_financial_health_score_distribution.png", None, wspace=0.25)


def chart_thin(pop: pd.DataFrame, val: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.5))
    tt = pd.DataFrame(val["thin_type_table"]).set_index("thin_type")
    order = [o for o in ["history", "cancelled_only", "first_application", "rejected_before"] if o in tt.index]
    tt = tt.loc[order]
    palette = {"history": PAL["grey"], "cancelled_only": PAL["navy"], "first_application": PAL["amber"], "rejected_before": PAL["red"]}
    axes[0].bar([THIN_LABELS[i].replace(": ", ":\n") for i in tt.index], tt.bad_rate * 100, color=[palette[i] for i in tt.index])
    for i, r in enumerate(tt.itertuples()):
        lo, hi = wilson_ci(r.bad_rate * r.n, r.n)
        axes[0].text(i, r.bad_rate * 100 + 0.2, f"{r.bad_rate:.1%}\n[{lo:.1%}-{hi:.1%}]", ha="center", fontsize=8.5)
    axes[0].set_ylabel("Out-of-time early stress rate (%)")
    axes[0].set_ylim(0, tt.bad_rate.max() * 130)
    axes[0].tick_params(axis="x", labelsize=8.5)
    axes[0].grid(axis="x", visible=False)
    p = pop[pop.fh_thin_file]
    for t in ("cancelled_only", "rejected_before"):
        axes[1].hist(p.Financial_Health_Score[p.fh_thin_type == t], bins=np.arange(0, 101, 2), color=palette[t], alpha=0.75,
                     label=f"{THIN_LABELS[t]} (n={(p.fh_thin_type == t).sum() / 1e3:,.0f}K)")
    for x in (50, 80):
        axes[1].axvline(x, color=PAL["ink"], linestyle=":")
    axes[1].set_xlabel("Financial Health Score at the snapshot")
    axes[1].set_ylabel("Thin-file customers")
    axes[1].legend(fontsize=9)
    save_fig(fig, TASK_FIGURES / "09_thin_file_decomposition.png", None, wspace=0.25, bottom=0.14)


def fairness(con, rep: pd.DataFrame, pop: pd.DataFrame) -> dict:
    cust = con.execute(f"SELECT SKP_CLIENT, NAME_GENDER, DATE_BIRTH FROM {pq('customer_clean')}").df()
    r = rep[["SKP_CLIENT", "y", "Financial_Health_Score", "tier"]].merge(cust, on="SKP_CLIENT")
    age = (pd.Timestamp(OOT_START) - pd.to_datetime(r.DATE_BIRTH)).dt.days / 365.25
    r["age_band"] = pd.cut(age, [0, 25, 35, 45, 55, 120], labels=["<25", "25-34", "35-44", "45-54", "55+"], right=False)
    out = {}
    for dim in ("NAME_GENDER", "age_band"):
        sub = r[r[dim].notna()]
        within = sub.pivot_table(index="tier", columns=dim, values="y", aggfunc="mean", observed=True).reindex(["High", "Medium", "Low"])
        groups = {str(k): {"n": len(x), "mean_fhs": float(x.Financial_Health_Score.mean()), "bad_rate": float(x.y.mean()),
                           "gini_within": float(gini(x.y, -x.Financial_Health_Score))}
                  for k, x in sub.groupby(dim, observed=True)}
        out[dim] = {"bad_rate_within_tier": within, "groups": groups}
    pm = pop[["SKP_CLIENT", "fh_short"]].merge(cust[["SKP_CLIENT", "NAME_GENDER"]], on="SKP_CLIENT")
    out["population_low_share_by_gender"] = pm.groupby("NAME_GENDER").fh_short.apply(lambda s: float((s == "Low").mean())).to_dict()
    return out


def chart_fairness(fair: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), gridspec_kw={"width_ratios": [1, 1.6]})
    for ax, dim, palette in ((axes[0], "NAME_GENDER", [PAL["navy"], PAL["grey"]]),
                             (axes[1], "age_band", [tint_rgba(PAL["navy"], 0.35), tint_rgba(PAL["navy"], 0.65), PAL["navy"], PAL["grey"], PAL["amber"]])):
        t = fair[dim]["bad_rate_within_tier"]
        cols = list(t.columns)
        width = 0.8 / len(cols)
        x = np.arange(len(t))
        for j, c in enumerate(cols):
            ax.bar(x + (j - (len(cols) - 1) / 2) * width, t[c] * 100, width, color=palette[j % len(palette)], label=str(c))
        ax.set_xticks(x, [f"{i} tier" for i in t.index])
        ax.set_ylabel("Out-of-time early stress rate (%)")
        ax.legend(fontsize=8.5, ncol=2 if len(cols) > 2 else 1)
        ax.grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "10_fairness_calibration.png", None, wspace=0.2)


# %% [markdown]
# ## E. Reasons per customer
#
# Every customer gets the factor that pulls the score down the most and the one that lifts it the most, both measured
# against the portfolio-average customer and expressed in FHS points.

# %%
def describe(f: str, v: float, thin_type: str) -> str:
    if thin_type != "history" and f in THIN_FACTORS:
        return {"rejected_before": "no signed loan yet, earlier application(s) rejected",
                "cancelled_only": "no signed loan yet, earlier application(s) cancelled by the customer",
                "first_application": "first application, no credit history"}[thin_type]
    if f == "delinq_sev":
        return ["no late loans", "1 late loan", "2 late loans", "3+ late loans or a write-off"][int(v)]
    if f == "clean_track":
        return f"{int(v)} loan(s) completed with no overdue"
    if f == "months_since_delinq":
        return "never late" if v >= NEVER_LATE else f"a late loan signed {int(v)} months ago"
    if f == "reject_rate":
        return f"{v:.0%} of applications rejected"
    if f == "history_months":
        return f"{int(v)} months of credit history"
    if f == "dti":
        return "income not declared" if np.isnan(v) else f"instalments = {min(v, 9.99):.0%} of income"
    if f == "dp_ratio":
        return "no goods/bike loan" if np.isnan(v) else f"average down payment {v:.0%}"
    if f == "max_offer":
        return "no pre-approved offer" if np.isnan(v) else f"pre-approved offer {v / 1e6:,.0f}M VND"
    return f


def reasons(pop: pd.DataFrame, scale: dict) -> pd.DataFrame:
    n_in = pd.Series([FACTORS[f][0] for f in FACTORS]).value_counts()
    w = np.array([PILLAR_WEIGHTS[FACTORS[f][0]] / n_in[FACTORS[f][0]] for f in FACTORS])
    contrib = pop[[f"pts_{f}" for f in FACTORS]].to_numpy() * w           # composite points per factor
    delta = (contrib - contrib.mean(axis=0)) * scale["points_per_logit"] * -scale["b1"] / 100   # in FHS points
    names = list(FACTORS)
    neg, pos = delta.argmin(axis=1), delta.argmax(axis=1)
    vals = pop[names].to_numpy(dtype=float)
    tt = pop.thin_type.to_numpy()
    rows = np.arange(len(pop))
    drag = [f"{describe(names[j], vals[i, j], tt[i])} ({delta[i, j]:+.0f} pts)" if delta[i, j] <= -1 else ""
            for i, j in zip(rows, neg)]
    strength = [f"{describe(names[j], vals[i, j], tt[i])} ({delta[i, j]:+.0f} pts)" if delta[i, j] >= 1 else ""
                for i, j in zip(rows, pos)]
    return pd.DataFrame({"fh_main_drag": drag, "fh_main_strength": strength}, index=pop.index)


def drivers_text(pop: pd.DataFrame) -> pd.Series:
    text = "Band " + pop.fh_band.str[0] + " (" + pop.fh_segment + ")"
    text = np.where(pop.fh_policy_cap != "", text + "; policy cap: " + pop.fh_policy_cap, text)
    text = np.where(pop.fh_main_drag != "", text + "; main drag: " + pop.fh_main_drag, text)
    text = np.where(pop.fh_main_strength != "", text + "; main strength: " + pop.fh_main_strength, text)
    return pd.Series(text, index=pop.index)


# %% [markdown]
# # Run — Challenge 1
#
# ## 1. Know the data before scoring it

# %%
if __name__ == "__main__":
    ensure_dirs(DATA_CLEAN, TASK_FIGURES, TASK_SCORES, TASK_REPORT.parent)
    set_style()
    con = get_con()
    LABEL_CUT, weekly = label_complete_before(con)
    found = data_foundations(con, LABEL_CUT)
    chart_foundations(found, weekly)
    display_saved_figure(TASK_FIGURES / "01_data_foundations.png")
    cb = found["calls_before_first_loan"]
    story("01_data_foundations", [
        f"**The file is a cohort, not the whole book.** {found['tet_applicant_share']:.2%} of the {found['customers']:,} customers "
        f"applied in Dec-2024/Jan-2025, when monthly applications ran at {found['peak_multiple']:.0f}x a normal month; "
        f"{found['first_time_share']:.0%} applied for the first time. Older vintages belong to customers who came back, so their "
        "stress rates are a survivor view; the score is therefore validated on the Tet wave itself.",
        f"**Outcome flags are complete only for loans signed before {LABEL_CUT}.** The 4th/3rd-instalment late-rate ratio is "
        f"{weekly.attrs['baseline_ratio']:.2f} in normal weeks and collapses for loans signed from {LABEL_CUT}: those loans cannot "
        "have missed instalment 4 yet. Every label and contract comparison stops at this date (v1 used 2025-02-01).",
        f"**CALL dates cannot be used for before/after tests.** For customers whose first-ever loan came in "
        f"{pd.Timestamp(cb.first_loan_month.iloc[-2]):%b-%Y}, {cb.observed_before_loan.iloc[-2]:.1%} of their collection calls are dated "
        f"before that loan existed, against {cb.uniform_expectation.iloc[-2]:.1%} expected if dates were random; "
        f"{found['never_signed_collection_customers']:,} customers who never signed a loan received "
        f"{found['never_signed_collection_calls']:,} collection calls; the collection share is "
        f"{found['collection_share_range'][0]:.1%}-{found['collection_share_range'][1]:.1%} in every one of 20 months; and "
        f"{found['collection_called_never_late_share']:.0%} of collection-called customers were never late. "
        "Collection calls are therefore neither a score input nor a 'future' validation.",
        f"**This holds for every call type, not only collection.** Only {found['sales_conversion_calls_within_14d_of_signing']:.1%} of "
        f"'sign successful' calls fall within 14 days of an actual signing, and only {found['same_customer_calls_within_24h']:.1%} of "
        "consecutive calls to the same customer are less than a day apart, where a real dialler retries within hours. **Team-wide rule:** "
        "call dates are not used for any before/after test in Tasks 1-3; call counts and outcomes per customer, whose meaning does not "
        "depend on dates, remain usable (Task 2 responsiveness, Task 3 contact guards).",
    ])

# %% [markdown]
# **How to read it.** Left: the Tet spike defines who is in the data. Middle: to the right of the red line a loan simply
# has not lived long enough to show a 4th missed instalment, so it would look "healthy" by construction. Right: if call
# timestamps were real, almost no collection call could predate the customer's first loan; the observed share tracks the
# random-date line instead.
#
# **Business meaning.** The score is built only on evidence whose timing can be trusted (contracts and leads). That is
# what makes the out-of-time validation below credible — and it is a data-governance finding in itself: the call log
# needs its true timestamps before it can feed early-warning models.

# %% [markdown]
# ## 2. Product risk and roll rates — not every missed payment means the same thing

# %%
if __name__ == "__main__":
    OBS = signed_obs(LABEL_CUT)
    product_risk = q1_product_risk(con, OBS, LABEL_CUT)
    chart_product(product_risk)
    display_saved_figure(TASK_FIGURES / "02_product_risk_and_roll_rates.png")
    pt = product_risk["table"].set_index("PRODUCT")
    fvr = product_risk["first_vs_repeat"]
    tw_first = fvr[(fvr.period == "Tet wave") & (fvr.PRODUCT == "TW") & (fvr.borrower == "First loan")].bad_rate.iloc[0]
    worst_wo, best_wo = pt.wo_after_fpd.idxmax(), pt.wo_after_fpd.idxmin()
    story("02_product_risk", [
        f"**First loans in the Tet wave are {ratio(product_risk['tet_first_loan_bad'], product_risk['tet_repeat_loan_bad']):.1f}x as risky "
        f"as repeat loans** ({product_risk['tet_first_loan_bad']:.2%} vs {product_risk['tet_repeat_loan_bad']:.2%}); a first-time "
        f"two-wheeler loan reaches {tw_first:.1%}. Across all mature loans TW is the riskiest product ({pt.loc['TW', 'bad_rate']:.2%}) and "
        f"CD the safest ({pt.loc['CD', 'bad_rate']:.2%}; z = {product_risk['tw_vs_cd']['z']:.0f}).",
        f"**A missed first instalment is a different event per product.** It ends in write-off for {pt.loc[worst_wo, 'wo_after_fpd']:.0%} "
        f"of {worst_wo} loans but only {pt.loc[best_wo, 'wo_after_fpd']:.0%} of {best_wo} loans. BNPL starts "
        f"{pt.loc['BNPL', 'fpd_share_of_late']:.0%} of its late cases at instalment 1, yet only {pt.loc['BNPL', 'roll_1_to_2']:.0%} roll to "
        f"instalment 2 (CD: {pt.loc['CD', 'roll_1_to_2']:.0%}): BNPL first-payment misses look like payment friction, instalment-loan "
        "misses like a loss of affordability.",
        f"Never-late loans are written off at only {pt.wo_if_never_late.min():.1%}-{pt.wo_if_never_late.max():.1%}: the first four "
        "instalments carry most of the repayment signal, which is why the Repayment pillar has the largest weight.",
    ])

# %% [markdown]
# **How to read it.** Left: orange bars (first loans) sit above navy bars (repeat loans) wherever both exist; cells with
# fewer than 1,000 loans are not drawn (cash loans go almost only to repeat customers, walk-in cash almost only to new
# ones — itself a sign of how the lender already gates its riskiest products). Right: red =
# share later written off after missing instalment 1, amber = after a later miss, green = never late.
#
# **Business meaning.** (1) Tet-wave underwriting for first-time borrowers, especially two-wheelers, carries the most
# risk. (2) Collections should treat a BNPL first-payment miss as a reminder problem (SMS, auto-debit) and an
# instalment-loan first-payment miss as an early-hardship case (restructuring conversation). Product differences are
# associations, not causal product effects.

# %% [markdown]
# ## 3. Debt burden — where affordability starts to bite

# %%
if __name__ == "__main__":
    income_resilience = q2_income_resilience(con, OBS)
    dti_evidence = q4_dti_tipping(con, OBS)
    chart_dti(dti_evidence, income_resilience)
    display_saved_figure(TASK_FIGURES / "03_dti_vs_delinquency_tipping_point.png")
    story("03_dti", [
        f"**Stress is almost flat until DTI ≈ {dti_evidence['breakpoint']:.1%}, then climbs "
        f"{ratio(dti_evidence['slope_after_per_10pp'], dti_evidence['slope_before_per_10pp']):.1f}x faster** "
        f"(+{dti_evidence['slope_before_per_10pp']:.2%} vs +{dti_evidence['slope_after_per_10pp']:.2%} per 10 DTI points). Above the "
        f"breakpoint the rate is {dti_evidence['bad_above']:.2%} vs {dti_evidence['bad_below']:.2%} below it "
        f"({dti_evidence['n_above']:,} vs {dti_evidence['n_below']:,} contracts); above 40% it is {dti_evidence['bad_above_40']:.2%}.",
        f"**Personal income is the right denominator.** It covers {income_resilience['coverage_main']:.1%} of contracts vs "
        f"{income_resilience['coverage_household']:.1%} for household income, and income level alone ranks weakly "
        f"(AUC {income_resilience['auc_main_low_income']:.3f}): affordability is about the instalment relative to income.",
    ])

# %% [markdown]
# **How to read it.** The navy line is observed stress per 2.5-point DTI bin; the dashed line is the best two-segment
# fit; the red zone is where each extra point of DTI costs more. The right panel shows that income level alone barely
# ranks risk.
#
# **Business meaning.** Use the breakpoint as a monitoring line for new offers: an instalment that pushes DTI past it
# should trigger an affordability conversation. It is a portfolio pattern, not a universal legal cutoff.

# %% [markdown]
# ## 4. Down payment and insurance — skin in the game vs. a risk marker

# %%
if __name__ == "__main__":
    mitigation = q3_mitigation(con, OBS, LABEL_CUT)
    chart_mitigation(mitigation)
    display_saved_figure(TASK_FIGURES / "04_insurance_and_downpayment_impact.png")
    ins = mitigation["insurance"].set_index("product")
    dp = mitigation["dp"]
    cd0 = dp[(dp.PRODUCT == "CD") & (dp.bin == 0)].iloc[0]
    cd1 = dp[(dp.PRODUCT == "CD") & (dp.bin == 1)].iloc[0]
    story("04_mitigation", [
        f"**Down payment protects most where the collateral is a vehicle.** TW loans with a 10-30% down payment are "
        f"{mitigation['dp_10_30_vs_50plus']['TW']:.1f}x as stressed as those with 50%+ (CD: {mitigation['dp_10_30_vs_50plus']['CD']:.1f}x).",
        f"**Zero down payment is not reckless on CD.** CD loans with <10% down payment run at {cd0.bad_rate:.2%}, about half the "
        f"10-30% band ({cd1.bad_rate:.2%}); {cd0.repeat_share:.0%} of them go to repeat customers, consistent with 0% promotions "
        "reserved for proven clients. Down payment is evidence, not a linear rule.",
        f"**Insurance marks risk, it does not remove it.** Insured contracts are more stressed for every product "
        f"(CW {ins.loc['CW', 'bad_rate_insured']:.2%} vs {ins.loc['CW', 'bad_rate_uninsured']:.2%}; CL {ins.loc['CL', 'relative_diff']:+.0%}) "
        f"and in {mitigation['ins_higher_in_share_of_cells']:.0%} of {mitigation['n_cells']} product × DTI cells. "
        "It is not a score input: penalising insurance would discourage protection.",
    ])

# %% [markdown]
# **How to read it.** Left: navy = not insured, orange = insured, labels = relative difference. Right: stress by
# down-payment band for two-wheelers (navy) and consumer durables (orange).
#
# **Business meaning.** (1) For first-time TW borrowers a higher down payment is the most actionable risk lever.
# (2) Insured customers in hardship should be routed to the insurance-claim process before collections, and insurance
# attach rates by channel are worth monitoring as a sales-practice signal. Both comparisons are observational.

# %% [markdown]
# ## 5. Build the samples, fit the scorecard, and check behaviour signals

# %%
if __name__ == "__main__":
    train = pd.concat([sample(con, start, end) for start, end in TRAIN_WINDOWS], ignore_index=True)
    oot = sample(con, OOT_START, LABEL_CUT).sort_values("SKP_CLIENT").reset_index(drop=True)
    oot["half"] = np.where(np.random.default_rng(C.RANDOM_SEED).random(len(oot)) < 0.5, "select", "report")
    cards = fit_cards(con, train, FACTORS)
    thin = thin_points(train, cards)
    train = score(train, cards, thin, None)
    scale = risk_scale(score(oot[oot.half == "select"], cards, thin, None).composite, oot.loc[oot.half == "select", "y"])
    train["Financial_Health_Score"] = to_fhs(train.composite, scale)
    oot = score(oot, cards, thin, scale)
    behaviour = behaviour_signals(con, oot, LABEL_CUT)
    chart_behaviour(behaviour)
    display_saved_figure(TASK_FIGURES / "05_behaviour_signals.png")
    st = behaviour["stacking"]
    px = behaviour["paradox"].set_index("n_open_contracts")
    of = behaviour["offer"].set_index("offer_band")
    story("05_behaviour", [
        f"**Loan stacking is a stress signal.** A Tet-wave loan signed after 2 other applications in the previous 30 days runs at "
        f"{st.bad_rate.iloc[2]:.2%}, and after 3+ at {st.bad_rate.iloc[3]:.2%}, vs {st.bad_rate.iloc[0]:.2%} with none "
        f"({ratio(st.bad_rate.iloc[3], st.bad_rate.iloc[0]):.1f}x).",
        f"**The lender-selection paradox.** Customers with 3+ loans already open look *safer* on the new loan "
        f"({px.loc[3, 'mean']:.2%} vs {px.loc[1, 'mean']:.2%} with one): the lender only tops up clients it trusts. A score fitted on "
        "this label would reward leverage, so the number of open loans is deliberately not a score input; leverage enters only "
        "through DTI, which is calibrated on single-contract affordability.",
        f"**The lender's own pre-approved offer carries information the contract table does not.** Customers whose largest "
        f"pre-approved offer was 80M+ VND run at {of.loc['80M+', 'mean']:.2%}, vs {of.loc['<10M', 'mean']:.2%} under 10M and "
        f"{of.loc['No offer', 'mean']:.2%} with no offer. It becomes the 'Pre-approved Capacity' pillar, held at 13% so the score "
        "does not simply echo past lender decisions.",
    ])

# %% [markdown]
# **How to read it.** Left: early stress on Tet-wave loans by how many other applications the customer had made in
# the previous 30 days. Middle: stress on a new loan by how many loans were already open (the paradox). Right: stress by
# the largest offer the customer had been pre-approved for (dashed = average for customers with history).
#
# **Business meaning.** Several applications in a short time is a cheap, real-time trigger for an affordability check
# at origination. The pre-approval limit is the bank's own capacity view, which is why it improves ranking; it is kept
# at a modest weight and monitored for feedback loops.

# %% [markdown]
# ## 6. The scorecard — what raises and lowers financial health

# %%
if __name__ == "__main__":
    chart_factors(cards)
    display_saved_figure(TASK_FIGURES / "06_key_factor_impacts.png")
    ds, ct, ms = (ranged(cards[f]) for f in ("delinq_sev", "clean_track", "months_since_delinq"))
    rj, hm, mo = (ranged(cards[f]) for f in ("reject_rate", "history_months", "max_offer"))
    story("06_scorecard", [
        f"**Negative drivers.** Late loans: training stress rises from {ds.bad_rate.iloc[0]:.2%} (none) to {ds.bad_rate.iloc[1]:.2%} (one) "
        f"and {ds.bad_rate.iloc[-1]:.1%} (3+ or write-off). A high rejection share ({rj.bad_rate.iloc[-1]:.2%} when 60%+ of applications "
        f"were rejected vs {rj.bad_rate.iloc[0]:.2%} at 0%) and a recent late loan ({ms.bad_rate.iloc[0]:.2%} if signed <6 months ago vs "
        f"{ms.bad_rate.iloc[-1]:.2%} if never late) pull the score down.",
        f"**Positive drivers.** A clean completion record ({ct.bad_rate.iloc[-1]:.2%} at 8+ clean loans vs {ct.bad_rate.iloc[0]:.2%} at none), "
        f"long history ({hm.bad_rate.iloc[-1]:.2%} at 8+ years vs {hm.bad_rate.iloc[0]:.2%} under 6 months) and a large pre-approved offer "
        f"({mo.bad_rate.iloc[-1]:.2%} at 80M+ vs {mo.bad_rate.iloc[0]:.2%} under 5M).",
        f"**Scale.** Every {scale['points_to_double_odds']:.0f} FHS points halve the odds of stress; 50 points = {1.5 * scale['p_bar']:.2%} "
        f"expected stress (1.5x the Tet-window average of {scale['p_bar']:.2%}), 80 points = {0.5 * scale['p_bar']:.2%}.",
        "**Thin files.** Customers with no signed loan take repayment and standing points from their type ("
        + ", ".join(f"{THIN_LABELS[k].split(': ')[1]}: {v:.0f} pts" for k, v in thin["delinq_sev"].items())
        + " on the late-loan factor), instead of one 'no history' value.",
    ])

# %% [markdown]
# **How to read it.** Each panel is one factor: bar height = training stress rate per bucket, colour and the number on
# top = score points (green ≥ 80, amber 50–79, red < 50). Buckets are monotone by construction, so every point can be
# explained as "this bucket repays better than that one". The last panel shows the published pillar weights.
#
# **Business meaning.** A customer improves the score by completing loans cleanly, staying current, keeping instalments
# below the DTI breakpoint and putting more money down on goods/vehicle loans — the levers are behaviours, not
# demographics.

# %% [markdown]
# ## 7. Out-of-time validation — does the score rank the Tet wave it never saw?

# %%
if __name__ == "__main__":
    validation_result = validation(train, oot, cards, thin, scale, LABEL_CUT)
    rep = validation_result.pop("report_frame")
    chart_validation(validation_result)
    display_saved_figure(TASK_FIGURES / "07_out_of_time_validation.png")
    bm = validation_result["benchmarks"]
    dec = pd.DataFrame(validation_result["decile_table"])
    ab = validation_result["ablation_gini_without_pillar"]
    wp = validation_result["weight_perturbation"]
    story("07_validation", [
        f"**The score ranks a period it never saw.** On {validation_result['n_report']:,} Tet-wave customers (report half), the lowest "
        f"decile is stressed at {dec.bad_rate.iloc[0]:.2%} and the highest at {dec.bad_rate.iloc[-1]:.2%} "
        f"({ratio(dec.bad_rate.iloc[0], dec.bad_rate.iloc[-1]):.1f}x); Gini {validation_result['gini_report']:.3f}, KS "
        f"{validation_result['ks_report']:.3f} — although the average stress rate jumped from {validation_result['bad_rate_train']:.2%} "
        f"in training to {validation_result['bad_rate_report']:.2%}.",
        f"**Better than v1, as good as complex models.** Gini {bm['Final scorecard']['gini_all']:.3f} vs {bm['v1 design (6 factors)']['gini_all']:.3f} "
        f"for the v1 design (customers with history: {bm['Final scorecard']['gini_history']:.3f} vs {bm['v1 design (6 factors)']['gini_history']:.3f}). "
        f"A logistic regression that re-estimates every weight on the same buckets reaches "
        f"{bm['Logistic regression, same buckets']['gini_all']:.3f}"
        + (f" and gradient boosting on the same raw inputs {bm['Gradient boosting, same raw inputs']['gini_all']:.3f}"
           if "Gradient boosting, same raw inputs" in bm else "")
        + ": complexity buys little or nothing.",
        f"**The weights are not fragile.** Dropping any pillar changes Gini to {min(ab.values()):.3f}-{max(ab.values()):.3f}; equal weights "
        f"give {bm['Equal pillar weights']['gini_all']:.3f}; 200 random ±25% weight changes keep Gini within {wp['gini_min']:.3f}-"
        f"{wp['gini_max']:.3f} and the customer ranking correlation ≥ {wp['rank_corr_min']:.3f}.",
        "**Expected vs observed** stress per decile (orange line) tracks the bars, so the 0-100 scale doubles as an expected stress rate.",
    ])

# %% [markdown]
# **How to read it.** Left: each bar is 10% of the Tet-wave report sample ordered by score; a good score shows a
# staircase falling to the right. Right: Gini of each candidate on exactly the same customers; orange = final scorecard.
#
# **Business meaning.** The bank gets an explainable points table that ranks as well as a black-box model, survives a
# change of regime (normal months → Tet surge) and has stable weights — the properties a credit committee needs before
# using a score to set review priorities.

# %% [markdown]
# ## 8. Score the customer base and set the tiers

# %%
if __name__ == "__main__":
    pop = score(derive(features(con, C.SNAPSHOT_DATE, production=True)), cards, thin, scale)
    pop["expected_bad_rate"] = fhs_to_pd(pop.Financial_Health_Score, scale).round(4)
    pop["Financial_Health_Tier"] = assign_tier(pop.Financial_Health_Score, C.FH_TIERS)
    pop["fh_short"] = pop.Financial_Health_Tier.map(C.FH_SHORT)
    pop["fh_band"] = band(pop.Financial_Health_Score)
    pop["fh_thin_file"] = pop.n_signed == 0
    pop["fh_thin_type"] = pop.thin_type
    pop["fh_segment"] = np.select(
        [pop.fh_policy_cap != "", pop.thin_type == "rejected_before", pop.thin_type == "cancelled_only",
         pop.thin_type == "first_application"],
        ["Distressed (" + pop.fh_policy_cap + ")", THIN_LABELS["rejected_before"], THIN_LABELS["cancelled_only"],
         THIN_LABELS["first_application"]], default=pop.fh_band.str[2:])
    pop = pd.concat([pop, reasons(pop, scale)], axis=1)
    pop["fh_drivers"] = drivers_text(pop)
    chart_distribution(pop, validation_result)
    display_saved_figure(TASK_FIGURES / "08_financial_health_score_distribution.png")
    tiers = pop.fh_short.value_counts().reindex(["High", "Medium", "Low"])
    tt = pd.DataFrame(validation_result["tier_table"]).set_index("tier")
    bt = pd.DataFrame(validation_result["band_table"]).set_index("band")
    low = pop.fh_short == "Low"
    psi_values = {"train_vs_oot": psi(train.Financial_Health_Score, oot.Financial_Health_Score),
                  "oot_vs_population_with_history": psi(oot.Financial_Health_Score[oot.n_signed > 0],
                                                        pop.Financial_Health_Score[~pop.fh_thin_file])}
    story("08_tiers", [
        f"**Three tiers with an economic meaning.** High (≥80): {tiers['High']:,} customers ({tiers['High'] / len(pop):.1%}), out-of-time "
        f"stress {tt.loc['High', 'bad_rate']:.2%} ({tt.loc['High', 'x_average']:.1f}x average). Moderate (50-79): {tiers['Medium']:,} "
        f"({tiers['Medium'] / len(pop):.1%}), {tt.loc['Medium', 'bad_rate']:.2%}. Low (<50): {tiers['Low']:,} ({tiers['Low'] / len(pop):.1%}), "
        f"{tt.loc['Low', 'bad_rate']:.2%} ({tt.loc['Low', 'x_average']:.1f}x).",
        f"**Five bands for operations.** Stress climbs from band A ({bt.bad_rate.iloc[0]:.2%}) to band E ({bt.bad_rate.iloc[-1]:.2%}), "
        f"a {ratio(bt.bad_rate.iloc[-1], bt.bad_rate.iloc[0]):.1f}x spread; band E holds {(pop.fh_band == 'E Stressed').sum():,} customers today.",
        f"**Policy caps catch the worst cases.** {validation_result['policy_capped_share']:.1%} of the test sample hit a knock-out rule "
        f"(write-off, overdue on an open loan, 2+ late loans) and were stressed at {validation_result['policy_capped_bad_rate']:.1%}.",
        f"**Low no longer simply means 'no history'.** Thin-file customers are {pop.fh_thin_file[low].mean():.0%} of the Low tier "
        f"(v1: 81%), and {(pop.fh_thin_file & ~low).sum():,} thin-file customers sit above it. Score stability: PSI train→test "
        f"{psi_values['train_vs_oot']:.3f}, test→today for customers with history {psi_values['oot_vs_population_with_history']:.3f}.",
    ])

# %% [markdown]
# **How to read it.** Left: today's score distribution coloured by tier, with the thin-file population outlined.
# Right: out-of-time stress per band, with the multiple of the average (dashed line).
#
# **Business meaning.** Tiers answer "how should we treat this customer?" (High: growth offers after standard checks;
# Moderate: needs-based service; Low: protect and support). Bands give risk and collections teams a finer queue. The
# expected stress rate travels with every customer in `fh_scores.parquet`.

# %% [markdown]
# ## 9. Thin files — different customers behind "no history"

# %%
if __name__ == "__main__":
    chart_thin(pop, validation_result)
    display_saved_figure(TASK_FIGURES / "09_thin_file_decomposition.png")
    thin_tab = pd.DataFrame(validation_result["thin_type_table"]).set_index("thin_type")
    tc = pop.loc[pop.fh_thin_file, "thin_type"].value_counts()
    rej, can = thin_tab.loc["rejected_before"], thin_tab.loc["cancelled_only"]
    z, p = two_prop_ztest(rej.bad_rate * rej.n, rej.n, can.bad_rate * can.n, can.n)
    story("09_thin_file", [
        f"**'No history' hides two opposite profiles.** When they do get credit, customers previously *rejected* are stressed at "
        f"{rej.bad_rate:.2%}, while customers who only *cancelled* an application run at {can.bad_rate:.2%}, close to customers with "
        f"history ({thin_tab.loc['history', 'bad_rate']:.2%}); z = {z:.1f}, p = {p:.1g}.",
        f"**Today's thin-file base:** {tc.get('rejected_before', 0):,} rejected-before and {tc.get('cancelled_only', 0):,} cancelled-only "
        f"customers. Median FHS: rejected {pop.loc[pop.thin_type == 'rejected_before', 'Financial_Health_Score'].median():.0f}, "
        f"cancelled {pop.loc[pop.thin_type == 'cancelled_only', 'Financial_Health_Score'].median():.0f}.",
        "**Action split.** Rejected-before: education and a documented affordability check before any offer. Cancelled-only: "
        "they were approved and walked away — a service follow-up on why (price, product fit), not a stress treatment.",
    ])

# %% [markdown]
# **How to read it.** Left: out-of-time stress by thin-file type with 95% confidence intervals. Right: today's score
# distribution of the two thin-file groups.
#
# **Business meaning.** Treating every "no history" customer as financially stressed would waste a re-activation
# opportunity on the cancelled-only group and under-protect the rejected group.

# %% [markdown]
# ## 10. Fairness — does the same score mean the same risk for every group?

# %%
if __name__ == "__main__":
    fair = fairness(con, rep, pop)
    chart_fairness(fair)
    display_saved_figure(TASK_FIGURES / "10_fairness_calibration.png")
    gw = fair["NAME_GENDER"]["bad_rate_within_tier"]
    gg, ag = fair["NAME_GENDER"]["groups"], fair["age_band"]["groups"]
    aw = fair["age_band"]["bad_rate_within_tier"]
    story("10_fairness", [
        f"**Average scores are close; risk at the same score is not.** Mean FHS is {gg['Female']['mean_fhs']:.1f} for women and "
        f"{gg['Male']['mean_fhs']:.1f} for men, but inside the Low tier men are stressed at {gw.loc['Low', 'Male']:.2%} vs "
        f"{gw.loc['Low', 'Female']:.2%} for women ({ratio(gw.loc['Low', 'Male'], gw.loc['Low', 'Female']):.1f}x), and men are higher in every tier.",
        f"**The score ranks inside every group** (Gini women {gg['Female']['gini_within']:.3f}, men {gg['Male']['gini_within']:.3f}; "
        f"age bands {min(v['gini_within'] for v in ag.values()):.3f}-{max(v['gini_within'] for v in ag.values()):.3f}). Under-25s are the "
        f"weakest-ranked group, mostly because they are new to credit (Low-tier stress {aw.loc['Low', '<25']:.2%} vs "
        f"{aw.loc['Low', '25-34']:.2%} for 25-34).",
        "**Decision.** Gender and age stay out of the score (brief note 6): using them would price people by who they are. The gap is "
        "published as a monitoring item: women in the Low tier are, on evidence, closer to Moderate risk, so Low-tier actions must stay "
        "supportive (no automatic declines).",
    ])

# %% [markdown]
# **How to read it.** Bars show out-of-time stress per tier for each group. A perfectly calibrated score would show
# equal bars inside each tier.
#
# **Business meaning.** The score is fit for prioritising reviews, not for automatic adverse decisions. Monitoring
# within-tier outcomes by group belongs in the score's governance.

# %% [markdown]
# ## 11. Save scores, metrics and the report

# %%
def scorecard_table(cards: dict) -> list[dict]:
    return pd.concat([c.assign(factor=f, pillar=FACTORS[f][0], factor_label=FACTOR_LABELS[f],
                               pillar_weight=PILLAR_WEIGHTS[FACTORS[f][0]]) for f, c in cards.items()],
                     ignore_index=True).to_dict("records")


def md_table(df: pd.DataFrame) -> str:
    """Plain Markdown table (no optional dependency)."""
    cells = [[str(c) for c in df.columns]] + [[f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in row]
                                            for row in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(r) + " |" for r in cells[1:]])


def financial_health_report(m: dict) -> str:
    val = m["validation"]
    bm = pd.DataFrame(val["benchmarks"]).T.reset_index().rename(columns={"index": "model"})
    bm[["gini_all", "gini_history"]] = bm[["gini_all", "gini_history"]].round(3)
    bt = pd.DataFrame(val["band_table"])
    bt["customers_today"] = bt.band.map(m["band_counts"]).fillna(0).astype(int).map("{:,}".format)
    bt["bad_rate"] = bt.bad_rate.map("{:.2%}".format)
    bt["x_average"] = bt.x_average.map("{:.2f}x".format)
    weights = pd.DataFrame({"pillar": list(PILLAR_WEIGHTS), "weight": [f"{w:.0%}" for w in PILLAR_WEIGHTS.values()],
                            "factors": [", ".join(FACTOR_LABELS[f] for f in FACTORS if FACTORS[f][0] == p) for p in PILLAR_WEIGHTS]})
    figs = {"01_data_foundations": ("01_data_foundations.png", "Data foundations: cohort, censoring, call timestamps"),
            "02_product_risk": ("02_product_risk_and_roll_rates.png", "Product risk and roll rates"),
            "03_dti": ("03_dti_vs_delinquency_tipping_point.png", "Debt-burden tipping point"),
            "04_mitigation": ("04_insurance_and_downpayment_impact.png", "Down payment and insurance"),
            "05_behaviour": ("05_behaviour_signals.png", "Behaviour signals: stacking, selection paradox, pre-approved offer"),
            "06_scorecard": ("06_key_factor_impacts.png", "Scorecard points per factor"),
            "07_validation": ("07_out_of_time_validation.png", "Out-of-time validation and benchmarks"),
            "08_tiers": ("08_financial_health_score_distribution.png", "Score distribution, tiers and bands"),
            "09_thin_file": ("09_thin_file_decomposition.png", "Thin-file decomposition"),
            "10_fairness": ("10_fairness_calibration.png", "Fairness: calibration within tiers")}
    lines = ["# Task 1 — Customer Financial Health Score", "", "## Executive summary", ""]
    lines += [f"- {STORIES['07_validation'][0]}", f"- {STORIES['07_validation'][1]}", f"- {STORIES['08_tiers'][0]}",
              f"- {STORIES['02_product_risk'][0]}", f"- {STORIES['09_thin_file'][0]}", f"- {STORIES['05_behaviour'][0]}",
              f"- {STORIES['01_data_foundations'][2]}", f"- {STORIES['01_data_foundations'][3]}", ""]
    lines += ["## Method", "",
              f"- **Features** frozen as of each window start (late flags only after {C.DUE_FLAG_MATURITY_DAYS} days, write-offs after "
              f"{C.STATUS_MATURITY_DAYS} days, no future contract status). **Label**: late on instalment 1-4 or written off on loans "
              "signed in the window.",
              f"- **Training windows** {', '.join(f'{a} to {b}' for a, b in TRAIN_WINDOWS)} ({val['n_train']:,} customer-windows, stress "
              f"{val['bad_rate_train']:.2%}). **Out-of-time test** {OOT_START} to {m['label_cut']} (Tet wave, {val['n_oot']:,} customers), "
              f"split into a selection half used only to anchor the 0-100 scale and a report half of {val['n_report']:,}.",
              "- **Points**: monotone buckets, points linear in log-odds (100 = safest bucket, 0 = riskiest); pillar = mean of its "
              "factors; composite = published weights; FHS = composite on a risk-anchored scale "
              f"(50 = 1.5x average stress, 80 = 0.5x, {m['scale']['points_to_double_odds']:.0f} points halve the odds); "
              "transparent policy caps for write-offs and current arrears.", "",
              md_table(weights), "",
              "## Evidence, chart by chart", ""]
    for key, (fname, caption) in figs.items():
        lines += [f"### {caption}", "", f"![{caption}](../figures/{fname})", "",
                  f"*Figure: {caption}. Source: {SOURCE}.*", ""]
        lines += [f"- {s}" for s in STORIES.get(key, [])] + [""]
        if key == "07_validation":
            lines += [md_table(bm), ""]
        if key == "08_tiers":
            lines += [md_table(bt[["band", "n", "bad_rate", "x_average", "customers_today"]]), ""]
    lines += ["## Changes from v1 (audit trail)", "",
              f"1. Labels and contract comparisons stop at the censoring cut {m['label_cut']} (was 2025-02-01).",
              "2. Validation is genuinely out of time (earlier windows train, Tet wave tests) instead of a random split of one window.",
              "3. New pillar *Pre-approved Capacity* (LEADS `MAX_OFFER`) and the factor *months since a late loan was signed*.",
              "4. Thin files are split into rejected-before / cancelled-only / first application.",
              "5. The 0-100 scale is anchored to risk, so the fixed 50/80 tier cut-offs mean 1.5x and 0.5x the average stress rate.",
              "6. Production debt burden uses the real contract status (closed loans no longer count as open obligations).",
              "7. The collection-call 'forward validation' was removed: CALL timestamps fail a timing test (chart 1).",
              "8. Fairness is tested on calibration within tiers, not only on average scores.", "",
              "## Limits", "",
              "- The label sees instalments 1-4 only; later arrears are not observable in the contract table.",
              "- The file is the Tet-2025 applicant cohort; older vintages are a survivor view.",
              "- Product, insurance, down-payment and offer comparisons are associations, not causal effects.",
              "- The pre-approved offer reflects the lender's own past judgement; its weight is held at 13% and should be "
              "monitored for feedback loops.",
              "- Demographics are never inputs; they are used only for the fairness check.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    keep = [
        "SKP_CLIENT", "Financial_Health_Score", "Financial_Health_Tier", "fh_short", "fh_band", "fh_segment", "fh_thin_file",
        "fh_thin_type", "fh_policy_cap", "expected_bad_rate", "fh_drivers", "fh_main_drag", "fh_main_strength",
        "n_open_delinquent", "n_signed", "n_delinquent", "any_default", "n_clean_completed", "n_rejected", "n_cancelled",
        "n_applications", "apps_90d", "income", "monthly_obligations", "n_open_contracts", "dti", "dp_ratio", "reject_rate",
        "history_months", "months_since_delinq", "max_offer", "total_credit", "open_annuity", "insurance_share",
    ] + [c for c in pop.columns if c.startswith("pillar_") or c.startswith("pts_")]
    pop[keep].to_parquet(TASK_SCORES / "fh_scores.parquet", index=False)
    metrics = {
        "label_cut": LABEL_CUT,
        "data_foundations": {k: v for k, v in found.items() if k not in ("monthly", "calls_before_first_loan")}
                            | {"calls_before_first_loan": found["calls_before_first_loan"].to_dict("records")},
        "product_risk": product_risk["table"].to_dict("records"),
        "first_vs_repeat": product_risk["first_vs_repeat"].to_dict("records"),
        "portfolio_bad_rate": product_risk["portfolio_bad_rate"],
        "tw_vs_cd": product_risk["tw_vs_cd"],
        "income_resilience": income_resilience,
        "dti": {k: v for k, v in dti_evidence.items() if k != "curve"} | {"curve": dti_evidence["curve"].to_dict("records")},
        "mitigation": {"insurance": mitigation["insurance"].to_dict("records"), "downpayment": mitigation["dp"].to_dict("records"),
                       "ins_higher_in_share_of_cells": mitigation["ins_higher_in_share_of_cells"],
                       "dp_10_30_vs_50plus": mitigation["dp_10_30_vs_50plus"]},
        "behaviour": {k: v.to_dict("records") for k, v in behaviour.items() if isinstance(v, pd.DataFrame)},
        "scorecard": scorecard_table(cards),
        "thin_points": {f: v.to_dict() for f, v in thin.items()},
        "pillar_weights": PILLAR_WEIGHTS,
        "scale": scale,
        "validation": validation_result,
        "psi": psi_values,
        "tier_counts": tiers.to_dict(),
        "band_counts": pop.fh_band.value_counts().to_dict(),
        "segment_counts": pop.fh_segment.value_counts().to_dict(),
        "thin_file_customers": int(pop.fh_thin_file.sum()),
        "low_tier_thin_file_share": float(pop.fh_thin_file[low].mean()),
        "policy_caps_population": pop.fh_policy_cap.replace("", "none").value_counts().to_dict(),
        "fairness": {dim: {"bad_rate_within_tier": fair[dim]["bad_rate_within_tier"].to_dict(), "groups": fair[dim]["groups"]}
                     for dim in ("NAME_GENDER", "age_band")}
                    | {"population_low_share_by_gender": fair["population_low_share_by_gender"]},
        "stories": STORIES,
    }
    save_json(metrics, DATA_CLEAN / "metrics_phase2.json")
    TASK_REPORT.write_text(financial_health_report(metrics), encoding="utf-8")
    con.close()
    print(f"Saved {len(pop):,} customer scores, 10 figures, metrics_phase2.json and the Task 1 report.")
