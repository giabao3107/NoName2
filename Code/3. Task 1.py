# %% [markdown]
# # 3. Customer financial health
#
# Build an explainable 0–100 score from repayment history, credit standing, debt burden, and down-payment evidence. The scorecard is fitted on a development sample and checked on a held-out, later outcome window.

# %%
"""
PHASE 2 - Challenge 1: Customer Financial Health Score (0-100).

Method (fully explainable, no black box)
  1. Contract-level deep dives: product risk, income resilience, mitigation (insurance / down-payment),
     DTI tipping point.
  2. Customer features frozen "as of" a past date (FH_ASOF) -> out-of-time target = delinquency on the
     contracts the customer signs AFTER that date. Features never see the outcome window.
  3. Each factor is bucketed with business-readable edges; each bucket earns 0-100 points linear in the
     log-odds of its observed bad rate (PAV-monotone, Weight-of-Evidence logic).
     Lowest-risk bucket = 100 points, highest-risk bucket = 0 points.
  4. FHS = weighted average of four pillar scores with fixed, published weights
     (Repayment 40% | Credit standing 30% | DTI 15% | Down-payment 15%), justified by out-of-time Gini
     and a leave-one-pillar-out ablation. FHS is therefore a plain points total - no hidden transform.
  5. A one-variable logistic fit translates FHS into an expected bad rate (used for ROI only).
  6. The same cards/weights are then applied to every customer with today's data.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from matplotlib.patches import Patch
from scipy import stats


def find_project_root():
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    for directory in (start, *start.parents):
        if ((directory / "Data").is_dir() or (directory / "Data_raw").is_dir()) and (directory / "Code").is_dir():
            return directory
    raise RuntimeError(f"Could not find project root from {start}; expected Data/ or Data_raw/ and Code/.")


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
                               pct, save_fig, save_json, set_style, two_prop_ztest)

SIGNED_OBS = f"(SELECT * FROM {pq('contract_clean')} WHERE is_signed AND cdate < TIMESTAMP '{C.OBS_COMPLETE_BEFORE}')"

# Pillar design ----------------------------------------------------------------------------------
FACTORS = {
    # factor            pillar                 bucket edges                               risk direction
    "delinq_sev": ("Repayment History", [0, 1, 2, 3, 99], "increasing"),
    "clean_track": ("Repayment History", [0, 1, 2, 3, 5, 8, 10_000], "decreasing"),
    "dti": ("Debt Burden (DTI)", [0, .10, .20, .25, .30, .35, .40, .50, .60, 1_000], "increasing"),
    "dp_ratio": ("Down-payment Buffer", [0, .10, .30, .50, .80, 1.20, 1_000], "decreasing"),
    "reject_rate": ("Credit Standing & Buffer", [0, .0001, .20, .40, .60, 1.01], "increasing"),
    "history_months": ("Credit Standing & Buffer", [0, 6, 12, 24, 48, 96, 10_000], "decreasing"),
}
FACTOR_LABELS = {
    "delinq_sev": "# past delinquent contracts (default = 3+)",
    "clean_track": "# loans completed with no overdue",
    "dti": "Debt-to-income (open instalments / income)",
    "dp_ratio": "Down-payment ratio (CD/TW)",
    "reject_rate": "Share of applications rejected",
    "history_months": "Months since first signed loan",
}
PILLAR_WEIGHTS = {"Repayment History": 0.40, "Debt Burden (DTI)": 0.15,
                  "Down-payment Buffer": 0.15, "Credit Standing & Buffer": 0.30}


# %% [markdown]
# ## Contract-level evidence
#
# Compare mature signed contracts by product and test income, down-payment, insurance, and debt-to-income patterns before defining customer score points.

# %%
# =================================================================================================
# A. Deep-dive analytics (contract level)
# =================================================================================================
def q1_product_risk(con) -> pd.DataFrame:
    df = con.execute(f"""
        SELECT PRODUCT, count(*) n_contracts, count(DISTINCT SKP_CLIENT) n_customers,
               avg(any_due) delinquency_rate, avg(n_due) avg_overdue_count,
               avg(n_due) FILTER (WHERE any_due = 1) avg_overdue_if_delinquent,
               avg(is_default) default_rate, avg(bad) bad_rate,
               avg(FIRST_DUE) fpd, avg(SECOND_DUE) spd, avg(THIRD_DUE) tpd, avg(FOURTH_DUE) fourth_pd,
               median(amt_credit) median_credit, median(dti_main) median_dti
        FROM {SIGNED_OBS} WHERE PRODUCT IN ('CD','TW','BNPL','CC','CL','CW')
        GROUP BY 1 ORDER BY bad_rate DESC""").df()
    tot = con.execute(f"SELECT avg(bad), avg(any_due), count(*) FROM {SIGNED_OBS}").fetchone()
    df.attrs["portfolio_bad_rate"], df.attrs["portfolio_delinq_rate"], df.attrs["n"] = tot
    # significance: TW vs CD
    tw = df.set_index("PRODUCT").loc["TW"]
    cd = df.set_index("PRODUCT").loc["CD"]
    z, p = two_prop_ztest(tw.bad_rate * tw.n_contracts, tw.n_contracts, cd.bad_rate * cd.n_contracts, cd.n_contracts)
    df.attrs["tw_vs_cd_z"], df.attrs["tw_vs_cd_p"] = z, p
    return df


def q2_income_resilience(con) -> dict:
    d = con.execute(f"SELECT bad, income_main, income_household, dti_main, dti_household FROM {SIGNED_OBS}").df()
    m, h = d.income_main.notna(), d.income_household.notna()
    both = m & h
    res = {
        "n_contracts": len(d),
        "coverage_main": m.mean(), "coverage_household": h.mean(),
        "auc_main_low_income": auc(d.bad[m], -d.income_main[m]),
        "auc_household_low_income": auc(d.bad[h], -d.income_household[h]),
        "auc_main_on_household_subset": auc(d.bad[both], -d.income_main[both]),
        "auc_dti_main": auc(d.bad[d.dti_main.notna()], d.dti_main[d.dti_main.notna()]),
        "auc_dti_household": auc(d.bad[d.dti_household.notna()], d.dti_household[d.dti_household.notna()]),
        "household_gt_main_share": float((d.income_household[both] > d.income_main[both]).mean()),
    }
    q = pd.qcut(d.income_main[m], 5, duplicates="drop")
    tab = d[m].groupby(q, observed=True).bad.agg(["mean", "size"]).reset_index()
    tab.columns = ["income_main_quintile", "bad_rate", "n"]
    res["quintiles"] = tab.assign(income_main_quintile=tab.income_main_quintile.astype(str)).to_dict("records")
    return res


def q3_mitigation(con) -> dict:
    ins = con.execute(f"""
        SELECT PRODUCT, flag_ins, count(*) n, avg(bad) bad_rate
        FROM {SIGNED_OBS} WHERE PRODUCT IN ('CD','CL','TW','CC','CW') GROUP BY ALL ORDER BY 1, 2""").df()
    piv = ins.pivot(index="PRODUCT", columns="flag_ins", values=["bad_rate", "n"])
    rows = []
    for prod in piv.index:
        b0, b1 = piv.loc[prod, ("bad_rate", 0)], piv.loc[prod, ("bad_rate", 1)]
        n0, n1 = piv.loc[prod, ("n", 0)], piv.loc[prod, ("n", 1)]
        z, p = two_prop_ztest(b1 * n1, n1, b0 * n0, n0)
        rows.append({"product": prod, "bad_rate_uninsured": b0, "bad_rate_insured": b1, "n_uninsured": n0,
                     "n_insured": n1, "relative_diff": b1 / b0 - 1, "z": z, "p_value": p})
    ins_tab = pd.DataFrame(rows)
    # insurance within DTI bands (confounding check)
    conf = con.execute(f"""
        SELECT PRODUCT, CASE WHEN dti_main < .1 THEN 'a <10%' WHEN dti_main < .2 THEN 'b 10-20%'
                             WHEN dti_main < .3 THEN 'c 20-30%' ELSE 'd 30%+' END dti_band,
               flag_ins, count(*) n, avg(bad) bad_rate
        FROM {SIGNED_OBS} WHERE PRODUCT IN ('CD','CL','TW') AND dti_main IS NOT NULL GROUP BY ALL""").df()
    conf_p = conf.pivot_table(index=["PRODUCT", "dti_band"], columns="flag_ins", values="bad_rate")
    share_higher = float((conf_p[1] > conf_p[0]).mean())
    # point-biserial correlations
    corr = con.execute(f"""
        SELECT corr(flag_ins, bad) r_ins,
               corr(least(dp_ratio, 2.0), bad) FILTER (WHERE dp_ratio IS NOT NULL) r_dp,
               corr(least(dp_ratio, 2.0), bad) FILTER (WHERE dp_ratio IS NOT NULL AND PRODUCT='TW') r_dp_tw,
               corr(least(dp_ratio, 2.0), bad) FILTER (WHERE dp_ratio IS NOT NULL AND PRODUCT='CD') r_dp_cd
        FROM {SIGNED_OBS}""").fetchone()
    dp = con.execute(f"""
        WITH b AS (SELECT PRODUCT, bad,
            CASE WHEN dp_ratio < .1 THEN 0 WHEN dp_ratio < .3 THEN 1 WHEN dp_ratio < .5 THEN 2 WHEN dp_ratio < .8 THEN 3
                 WHEN dp_ratio < 1.2 THEN 4 ELSE 5 END AS bin
            FROM {SIGNED_OBS} WHERE PRODUCT IN ('CD','TW') AND dp_ratio IS NOT NULL)
        SELECT PRODUCT, bin, count(*) n, avg(bad) bad_rate FROM b GROUP BY ALL ORDER BY 1, 2""").df()
    dp["label"] = dp["bin"].map({0: "<10%", 1: "10-30%", 2: "30-50%", 3: "50-80%", 4: "80-120%", 5: "120%+"})
    dp["bad"] = dp.bad_rate * dp.n
    lo_g = dp[dp.bin <= 1].groupby("PRODUCT")[["bad", "n"]].sum()
    hi_g = dp[dp.bin >= 3].groupby("PRODUCT")[["bad", "n"]].sum()
    lo, hi = lo_g.bad / lo_g.n, hi_g.bad / hi_g.n
    return {"insurance": ins_tab, "ins_confounding_share_higher": share_higher, "ins_confounding": conf,
            "r_ins_bad": corr[0], "r_dp_bad": corr[1], "r_dp_bad_tw": corr[2], "r_dp_bad_cd": corr[3],
            "dp": dp, "dp_low_vs_high": {p: {"low_dp_<30%": float(lo[p]), "high_dp_>=50%": float(hi[p]),
                                              "ratio": float(lo[p] / hi[p])} for p in lo.index}}


def q4_dti_tipping(con) -> dict:
    step = 0.025
    cur = con.execute(f"""
        SELECT least(floor(dti_main / {step}) * {step}, 0.70) AS dti_bin, count(*) n, avg(bad) bad_rate
        FROM {SIGNED_OBS} WHERE dti_main IS NOT NULL GROUP BY 1 ORDER BY 1""").df()
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
    slope_before, slope_after = coef[1], coef[1] + coef[2]
    below = con.execute(f"SELECT avg(bad), count(*) FROM {SIGNED_OBS} WHERE dti_main < {bp}").fetchone()
    above = con.execute(f"SELECT avg(bad), count(*) FROM {SIGNED_OBS} WHERE dti_main >= {bp}").fetchone()
    above40 = con.execute(f"SELECT avg(bad), count(*) FROM {SIGNED_OBS} WHERE dti_main >= 0.40").fetchone()
    # Affordability ceiling for NEW offers: first DTI level whose bad rate reaches 1.5x the portfolio average
    port = con.execute(f"SELECT avg(bad) FROM {SIGNED_OBS} WHERE dti_main IS NOT NULL").fetchone()[0]
    over = fit[(fit.bad_rate >= 1.5 * port) & (fit.dti_bin >= bp)]
    ceiling = float(over.dti_bin.min()) if len(over) else 0.40
    return {"curve": cur, "breakpoint": float(round(bp, 3)), "coef": coef.tolist(),
            "slope_before_per_10pp": float(slope_before * 0.1), "slope_after_per_10pp": float(slope_after * 0.1),
            "bad_below": below[0], "n_below": below[1], "bad_above": above[0], "n_above": above[1],
            "bad_above_40": above40[0], "n_above_40": above40[1], "dti_ceiling": round(ceiling, 3),
            "portfolio_bad_with_dti": port,
            "spearman": stats.spearmanr(fit.mid, fit.bad_rate).correlation}


# =================================================================================================
# B. Customer features (as-of) and scorecard
# =================================================================================================
# %% [markdown]
# ## Fit the scorecard without look-ahead
#
# Freeze customer features before the target window. Estimate factor points on the fit fold, choose weights from the stated design, then report performance on the holdout fold.

# %%
def build_features(con, asof: str, due_maturity_days: int, status_maturity_days: int) -> pd.DataFrame:
    inst = ", ".join(f"'{p}'" for p in C.INSTALMENT_PRODUCTS)
    rev = ", ".join(f"'{p}'" for p in C.REVOLVING_PRODUCTS)
    a = f"TIMESTAMP '{asof}'"
    df = con.execute(f"""
    WITH c AS (
      SELECT *,
        CASE WHEN cdate < {a} - INTERVAL {due_maturity_days} DAY THEN any_due ELSE 0 END AS due_k,
        CASE WHEN cdate < {a} - INTERVAL {status_maturity_days} DAY THEN is_default ELSE 0 END AS def_k
      FROM {pq('contract_clean')} WHERE cdate < {a})
    SELECT SKP_CLIENT,
      count(*) FILTER (WHERE is_signed) AS n_signed,
      coalesce(sum(due_k) FILTER (WHERE is_signed), 0) AS n_delinquent,
      coalesce(max(def_k) FILTER (WHERE is_signed), 0) AS any_default,
      count(*) FILTER (WHERE is_signed AND end_date < {a} AND due_k = 0 AND def_k = 0) AS n_clean_completed,
      count(*) FILTER (WHERE is_rejected) AS n_rejected,
      count(*) AS n_applications,
      max_by(income_main, (cdate, SKP_CREDIT_CASE)) FILTER (WHERE income_main IS NOT NULL) AS income,
      round(coalesce(sum(monthly_obligation) FILTER (WHERE is_signed AND (
            (PRODUCT IN ({inst}) AND end_date > {a}) OR (PRODUCT IN ({rev}) AND is_open))), 0), 0) AS monthly_obligations,
      count(*) FILTER (WHERE is_signed AND ((PRODUCT IN ({inst}) AND end_date > {a}) OR (PRODUCT IN ({rev}) AND is_open)))
            AS n_open_contracts,
      count(*) FILTER (WHERE is_signed AND due_k = 1 AND ((PRODUCT IN ({inst}) AND end_date > {a}) OR (PRODUCT IN ({rev}) AND is_open)))
            AS n_open_delinquent,
      round(avg(dp_ratio) FILTER (WHERE is_signed), 6) AS avg_dp_ratio,
      round(avg(flag_ins) FILTER (WHERE is_signed), 6) AS insurance_share,
      date_diff('month', min(cdate) FILTER (WHERE is_signed), {a}) AS history_months,
      round(sum(amt_credit) FILTER (WHERE is_signed), 0) AS total_credit,
      round(sum(annuity) FILTER (WHERE is_signed AND PRODUCT IN ({inst}) AND end_date > {a}), 0) AS open_annuity
    FROM c GROUP BY 1""").df()
    has = df.n_signed > 0
    df["delinq_sev"] = np.where(has, np.where(df.any_default == 1, 3, df.n_delinquent.clip(upper=3)), np.nan)
    df["clean_track"] = np.where(has, df.n_clean_completed, np.nan)
    df["dti"] = np.where(df.income > 0, df.monthly_obligations / df.income, np.nan)
    df["dp_ratio"] = df.avg_dp_ratio
    df["reject_rate"] = df.n_rejected / df.n_applications
    df["history_months"] = np.where(has, df.history_months, np.nan)
    return df


def dev_sample(con) -> pd.DataFrame:
    feat = build_features(con, C.FH_ASOF, C.DUE_FLAG_MATURITY_DAYS, C.STATUS_MATURITY_DAYS)
    tgt = con.execute(f"""
        SELECT SKP_CLIENT, max(bad) AS y, round(sum(monthly_obligation), 0) AS new_obligation,
               max_by(income_main, (cdate, SKP_CREDIT_CASE)) AS new_income, round(avg(dp_ratio), 6) AS new_dp
        FROM {pq('contract_clean')}
        WHERE is_signed AND cdate >= TIMESTAMP '{C.FH_TARGET_START}' AND cdate < TIMESTAMP '{C.FH_TARGET_END}'
        GROUP BY 1""").df()
    d = tgt.merge(feat, on="SKP_CLIENT", how="left")
    # brand-new customers have no as-of row: fill structural zeros
    for c_ in ["n_signed", "n_delinquent", "any_default", "n_clean_completed", "n_rejected", "n_applications",
               "monthly_obligations", "n_open_contracts", "n_open_delinquent"]:
        d[c_] = d[c_].fillna(0)
    has = d.n_signed > 0
    d["delinq_sev"] = np.where(has, np.where(d.any_default == 1, 3, d.n_delinquent.clip(upper=3)), np.nan)
    d["clean_track"] = np.where(has, d.n_clean_completed, np.nan)
    d["reject_rate"] = np.where(d.n_applications > 0, d.n_rejected / d.n_applications.replace(0, np.nan), np.nan)
    d["history_months"] = np.where(has, d.history_months, np.nan)
    # At application time the lender knows the new instalment + down-payment: include them (as production does)
    inc = d.income.fillna(d.new_income)
    d["dti"] = np.where(inc > 0, (d.monthly_obligations + d.new_obligation.fillna(0)) / inc, np.nan)
    d["dp_ratio"] = d.avg_dp_ratio.fillna(d.new_dp)
    d = d.sort_values("SKP_CLIENT").reset_index(drop=True)          # deterministic split
    rng = np.random.default_rng(C.RANDOM_SEED)
    d["fold"] = np.where(rng.random(len(d)) < 0.5, "fit", "holdout")
    return d


def fit_logit_1d(x, y, iters=50):
    """Two-parameter logistic regression by Newton-Raphson (used only to scale the composite to PD)."""
    X = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def composite(points: pd.DataFrame) -> pd.Series:
    pil = pillar_scores(points)
    return sum(PILLAR_WEIGHTS[k] * pil[k] for k in PILLAR_WEIGHTS)


def pillar_scores(points: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for pillar in PILLAR_WEIGHTS:
        fs = [f for f, (p, _, _) in FACTORS.items() if p == pillar]
        out[pillar] = points[[f"pts_{f}" for f in fs]].mean(axis=1)
    return pd.DataFrame(out)


def fit_scorecard(con, dev: pd.DataFrame) -> dict:
    fit = dev[dev.fold == "fit"]
    cards = {}
    for f, (pillar, edges, direction) in FACTORS.items():
        if f == "dti":
            # Customer-level DTI is confounded by track record (indebted customers are also experienced repeat
            # borrowers), so DTI points are calibrated on the 2.0M+ contract-level DTI -> delinquency curve
            # observed before the as-of date (no look-ahead).
            cl = con.execute(f"""SELECT dti_main x, bad y FROM {pq('contract_clean')}
                                 WHERE is_signed AND dti_main IS NOT NULL
                                   AND cdate < TIMESTAMP '{C.FH_ASOF}' - INTERVAL {C.DUE_FLAG_MATURITY_DAYS} DAY""").df()
            cards[f] = calibrate_bins(cl.x, cl.y, edges, direction)
        else:
            cards[f] = calibrate_bins(fit[f], fit["y"], edges, direction)
    for f in FACTORS:
        dev[f"pts_{f}"] = apply_card(dev[f], cards[f])
    dev["composite"] = composite(dev)
    # 1-variable logistic fit: translates any score into an expected bad rate (used for ROI, not for ranking)
    b = fit_logit_1d(dev.loc[dev.fold == "fit", "composite"].values / 100, dev.loc[dev.fold == "fit", "y"].values)
    p_bar = float(dev.y.mean())
    calib = {"b0": b[0], "b1": b[1], "p_bar": p_bar,
             "pd_at_80": float(1 / (1 + np.exp(-(b[0] + b[1] * 0.8)))),
             "pd_at_50": float(1 / (1 + np.exp(-(b[0] + b[1] * 0.5)))),
             "points_to_double_odds": float(-100 * np.log(2) / b[1]) if b[1] != 0 else None}
    dev["FHS"] = to_fhs(dev["composite"], calib)
    dev["FHS_no_caps"] = dev["FHS"].copy()
    apply_policy_caps(dev, "FHS")
    return {"cards": cards, "calib": calib}


def to_fhs(comp: pd.Series, calib: dict | None = None) -> pd.Series:
    """FHS is the weighted points total itself (0-100) - no hidden transformation."""
    return comp.clip(0, 100).round(1)


POLICY_CAPS = [
    # (rule name, condition, max FHS) - transparent knock-out rules applied after the points total
    ("Past write-off / debt sold", lambda d: d.any_default == 1, 40.0),
    ("Overdue on a loan still open", lambda d: d.n_open_delinquent >= 1, 49.0),
    ("2+ loans with overdue history", lambda d: d.n_delinquent >= 2, 49.0),
]


def apply_policy_caps(d: pd.DataFrame, score_col: str) -> pd.DataFrame:
    d["fh_policy_cap"] = ""
    for name, cond, cap in POLICY_CAPS:
        m = cond(d).fillna(False).astype(bool) & (d[score_col] > cap)
        d.loc[m, score_col] = cap
        d.loc[m & (d["fh_policy_cap"] == ""), "fh_policy_cap"] = name
    return d


def validate(dev: pd.DataFrame, calib: dict) -> dict:
    h = dev[dev.fold == "holdout"].copy()
    res = {"n_dev": len(dev), "n_holdout": len(h), "bad_rate_dev": float(dev.y.mean())}
    res["gini_holdout"] = gini(h.y, -h.FHS)
    res["gini_holdout_without_policy_caps"] = gini(h.y, -h.FHS_no_caps)
    capped = h.fh_policy_cap != ""
    res["policy_capped_share_holdout"] = float(capped.mean())
    res["policy_capped_bad_rate_holdout"] = float(h.y[capped].mean()) if capped.any() else None
    res["auc_holdout"] = auc(h.y, -h.FHS)
    # KS
    s = h.sort_values("FHS")
    cb = (s.y.cumsum() / s.y.sum()).values
    cg = ((1 - s.y).cumsum() / (1 - s.y).sum()).values
    res["ks_holdout"] = float(np.max(np.abs(cb - cg)))
    h["tier"] = assign_tier(h.FHS, C.FH_TIERS).map(C.FH_SHORT)
    tt = h.groupby("tier").agg(n=("y", "size"), bad_rate=("y", "mean")).reindex(["High", "Medium", "Low"])
    tt["share"] = tt.n / tt.n.sum()
    res["tier_table"] = tt.reset_index().to_dict("records")
    h["decile"] = pd.qcut(h.FHS.rank(method="first"), 10, labels=False) + 1
    dec = h.groupby("decile").agg(fhs_min=("FHS", "min"), fhs_max=("FHS", "max"), n=("y", "size"),
                                  bad_rate=("y", "mean"))
    logodds = calib["b0"] + calib["b1"] * h.composite / 100
    h["pd_hat"] = 1 / (1 + np.exp(-logodds))
    dec["pd_predicted"] = h.groupby("decile").pd_hat.mean()
    res["decile_table"] = dec.reset_index().to_dict("records")
    # factor + pillar ginis (holdout) and ablation
    fg = {}
    for f in FACTORS:
        fg[f] = gini(h.y, -h[f"pts_{f}"])
    res["factor_gini"] = fg
    pil = pillar_scores(h)
    res["pillar_gini"] = {k: gini(h.y, -pil[k]) for k in PILLAR_WEIGHTS}
    ab = {}
    for k in PILLAR_WEIGHTS:
        w = {kk: (0 if kk == k else v) for kk, v in PILLAR_WEIGHTS.items()}
        tot = sum(w.values())
        comp = sum(w[kk] / tot * pil[kk] for kk in w)
        ab[k] = gini(h.y, -comp)
    res["ablation_gini_without_pillar"] = ab
    res["equal_weight_gini"] = gini(h.y, -pil.mean(axis=1))
    # DTI conditional on track record (Simpson check)
    cond = {}
    for lab, g in h.groupby(pd.cut(h.clean_track, [-1, 0, 2, 5, 10_000],
                                   labels=["0 clean loans", "1-2", "3-5", "6+"]), observed=True):
        cond[str(lab)] = {"n": len(g), "gini_dti": gini(g.y, g.dti.fillna(g.dti.median()))}
    res["dti_gini_within_track_strata"] = cond
    res["dti_gini_raw_customer_level"] = gini(h.y, h.dti.fillna(h.dti.median()))
    return res


# %% [markdown]
# ## Score the current customer population
#
# Apply the same learned scorecard to every customer. Policy caps remain visible, and customers without signed-loan history are tagged as thin-file rather than assumed to be financially stressed.

# %%
def score_population(con, cards, calib) -> pd.DataFrame:
    feat = build_features(con, C.SNAPSHOT_DATE, 0, 0)
    for f in FACTORS:
        feat[f"pts_{f}"] = apply_card(feat[f], cards[f])
    pil = pillar_scores(feat)
    for k in PILLAR_WEIGHTS:
        feat[f"pillar_{k}"] = pil[k].round(1)
    feat["composite"] = composite(feat)
    feat["Financial_Health_Score"] = to_fhs(feat["composite"], calib)
    apply_policy_caps(feat, "Financial_Health_Score")
    logodds = calib["b0"] + calib["b1"] * feat["Financial_Health_Score"] / 100
    feat["expected_bad_rate"] = 1 / (1 + np.exp(-logodds))
    feat["Financial_Health_Tier"] = assign_tier(feat["Financial_Health_Score"], C.FH_TIERS)
    feat["fh_short"] = feat["Financial_Health_Tier"].map(C.FH_SHORT)
    # Thin-file = no signed loan ever (only rejected / cancelled applications): risky, but "credit-invisible"
    # rather than "stressed" - Challenge 3 treats them differently.
    feat["fh_thin_file"] = feat["n_signed"] == 0
    return feat


def explain_fh(df: pd.DataFrame) -> pd.Series:
    """Short human-readable driver summary per customer (used in the NBA rationale)."""
    parts = []
    sev = df.delinq_sev
    rep = np.select([sev.isna(), df.any_default == 1, sev >= 1],
                    ["no repayment history", "past write-off", df.n_delinquent.fillna(0).astype(int).astype(str) + " late loan(s)"],
                    default=df.n_clean_completed.fillna(0).astype(int).astype(str) + " clean loan(s)")
    parts.append(pd.Series(rep, index=df.index))
    dti = np.where(df.dti.isna(), "DTI n/a", "DTI " + (100 * df.dti.clip(upper=9.99)).round(0).astype("Int64").astype(str) + "%")
    parts.append(pd.Series(dti, index=df.index))
    rej = np.where(df.reject_rate >= 0.4, (100 * df.reject_rate.fillna(0)).round(0).astype(int).astype(str) + "% apps rejected", "")
    parts.append(pd.Series(rej, index=df.index))
    out = parts[0] + ", " + parts[1]
    out = np.where(parts[2] != "", out + ", " + parts[2], out)
    return pd.Series(out, index=df.index)


# =================================================================================================
# C. Charts
# =================================================================================================
# %% [markdown]
# ## Product repayment outcomes
#
# Three panels keep delinquency rate, installment severity, and write-off/sold rates on separate, interpretable scales.

# %%
def chart_product(prod: pd.DataFrame) -> None:
    ordered = prod.sort_values("bad_rate").copy()
    ordered["label"] = ordered.PRODUCT.map(C.PRODUCT_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
    average = prod.attrs["portfolio_bad_rate"]

    axes[0].barh(ordered.label, ordered.bad_rate * 100, color=PAL["navy"], height=0.62)
    axes[0].axvline(average * 100, color=PAL["orange"], linestyle="--", linewidth=1.5, label="Portfolio average")
    for row, (rate, count) in enumerate(zip(ordered.bad_rate, ordered.n_contracts)):
        axes[0].text(rate * 100 + 0.08, row, f"{rate:.2%} · n={count / 1e3:,.0f}K", va="center", fontsize=8.5)
    axes[0].set_xlabel("Observed delinquency (%)")
    axes[0].set_ylabel("Product")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(axis="y", visible=False)

    axes[1].barh(ordered.label, ordered.avg_overdue_if_delinquent, color=PAL["navy"], height=0.62)
    axes[1].set_xlabel("Mean late installments\n(among delinquent contracts)")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].grid(axis="y", visible=False)

    axes[2].barh(ordered.label, ordered.default_rate * 100, color=PAL["orange"], height=0.62)
    axes[2].set_xlabel("Write-off / sold rate (%)")
    axes[2].tick_params(axis="y", labelleft=False)
    axes[2].grid(axis="y", visible=False)
    save_fig(fig, TASK_FIGURES / "01_delinquency_by_product_type.png", None, left=0.18, wspace=0.28)


# %% [markdown]
# ## Debt burden and income coverage
#
# Compare delinquency across DTI bins and show how personal and household income measures rank later repayment outcomes.

# %%
def chart_dti(dti: dict, inc: dict) -> None:
    curve = dti["curve"]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    observed = curve.n >= 300
    ax.plot(curve.mid[observed] * 100, curve.bad_rate[observed] * 100, "o-", color=PAL["navy"], lw=2.2, ms=5,
            label="Observed delinquency")
    x = np.linspace(0.0125, 0.7, 100)
    c0, c1, c2 = dti["coef"]
    ax.plot(x * 100, (c0 + c1 * x + c2 * np.clip(x - dti["breakpoint"], 0, None)) * 100,
            color=PAL["orange"], lw=2, linestyle="--", label="Piecewise fit")
    breakpoint = dti["breakpoint"] * 100
    ax.axvline(breakpoint, color=PAL["red"], lw=1.5)
    ax.axvspan(breakpoint, 75, color=PAL["red"], alpha=0.06)
    ax.text(breakpoint + 0.8, ax.get_ylim()[1] * 0.93, f"Observed breakpoint\n{breakpoint:.1f}%", color=PAL["red"], va="top")
    ax.set_xlabel("Debt-to-income ratio (%)")
    ax.set_ylabel("Observed delinquency (%)")
    ax.set_xlim(0, 75)
    ax.legend(loc="upper left")

    labels = ["Personal income", "Household income", "DTI · personal", "DTI · household"]
    values = [inc["auc_main_low_income"], inc["auc_household_low_income"], inc["auc_dti_main"], inc["auc_dti_household"]]
    coverage = [inc["coverage_main"], inc["coverage_household"], inc["coverage_main"], inc["coverage_household"]]
    axes[1].bar(labels, [value - 0.5 for value in values], bottom=0.5, color=PAL["navy"], width=0.6)
    for i, (value, available) in enumerate(zip(values, coverage)):
        axes[1].text(i, max(value, 0.5) + 0.003, f"AUC {value:.3f}\ncoverage {available:.1%}", ha="center", fontsize=8.5)
    axes[1].axhline(0.5, color=PAL["orange"], linewidth=1)
    axes[1].set_ylim(0.44, 0.58)
    axes[1].set_ylabel("Holdout AUC · 0.5 = no ranking signal")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "02_dti_vs_delinquency_tipping_point.png", None, wspace=0.38)


# %% [markdown]
# ## Insurance and down-payment evidence
#
# The panels compare two observed risk markers. Neither insurance nor down-payment is presented as a causal effect.

# %%
def chart_mitigation(mit: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    insurance = mit["insurance"].set_index("product").loc[["CD", "CL", "CC", "TW", "CW"]]
    x = np.arange(len(insurance))
    width = 0.38
    axes[0].bar(x - width / 2, insurance.bad_rate_uninsured * 100, width, color=PAL["navy"], label="Not insured")
    axes[0].bar(x + width / 2, insurance.bad_rate_insured * 100, width, color=PAL["orange"], label="Insured")
    axes[0].set_xticks(x, insurance.index)
    axes[0].set_xlabel("Product")
    axes[0].set_ylabel("Observed delinquency (%)")
    axes[0].legend()
    axes[0].grid(axis="x", visible=False)

    downpayment = mit["dp"]
    for product in ("TW", "CD"):
        group = downpayment[downpayment.PRODUCT == product]
        axes[1].plot(group.label, group.bad_rate * 100, "o-", color=PAL["navy"] if product == "TW" else PAL["orange"], label=product)
    axes[1].set_xlabel("Down-payment / credit amount")
    axes[1].set_ylabel("Observed delinquency (%)")
    axes[1].legend()
    save_fig(fig, TASK_FIGURES / "03_insurance_and_downpayment_impact.png", None)


# %% [markdown]
# ## Score distribution and holdout check
#
# The left panel shows score coverage; the right panel tests whether the later cohort separates by score tier.

# %%
def chart_distribution(pop: pd.DataFrame, val: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})
    bins = np.arange(0, 101, 2)
    tiers = {"Low": (0, 50), "Medium": (50, 80), "High": (80, 101)}
    for tier, (lower, upper) in tiers.items():
        values = pop.Financial_Health_Score[(pop.Financial_Health_Score >= lower) & (pop.Financial_Health_Score < upper)]
        axes[0].hist(values, bins=bins, color=TIER_COLORS[tier], alpha=0.9)
    shares = pop.fh_short.value_counts(normalize=True)
    for tier, position in [("Low", 25), ("Medium", 65), ("High", 90)]:
        axes[0].text(position, axes[0].get_ylim()[1] * 0.92, f"{tier} · {shares.get(tier, 0):.1%}", ha="center", color=TIER_COLORS[tier])
    axes[0].axvline(50, color=PAL["ink"], linestyle=":")
    axes[0].axvline(80, color=PAL["ink"], linestyle=":")
    axes[0].set_xlabel("Financial-health score (0–100)")
    axes[0].set_ylabel("Customers")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value / 1e3:.0f}K"))

    tier_table = pd.DataFrame(val["tier_table"]).set_index("tier")
    axes[1].bar(tier_table.index, tier_table.bad_rate * 100,
                color=[TIER_COLORS[tier] for tier in tier_table.index], width=0.6)
    for i, (_, row) in enumerate(tier_table.iterrows()):
        axes[1].text(i, row.bad_rate * 100 + 0.12, f"{row.bad_rate:.2%}\nn={row.n / 1e3:,.0f}K", ha="center", fontsize=9)
    axes[1].axhline(val["bad_rate_dev"] * 100, color=PAL["navy"], linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Financial-health tier")
    axes[1].set_ylabel("Later-window delinquency (%)")
    axes[1].grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "04_financial_health_score_distribution.png", None)


# %% [markdown]
# ## Factor-level score points
#
# Each panel maps a feature bucket to its development-sample delinquency rate; labels above bars are the score points assigned to that bucket.

# %%
BUCKET_LABELS = {
    "delinq_sev": ["0", "1", "2", "3+ or\nwrite-off"], "clean_track": ["0", "1", "2", "3-4", "5-7", "8+"],
    "dti": ["<10%", "10-20", "20-25", "25-30", "30-35", "35-40", "40-50", "50-60", "60%+"],
    "dp_ratio": ["<10%", "10-30", "30-50", "50-80", "80-120", "120%+"],
    "reject_rate": ["0%", "<20%", "20-40", "40-60", "60%+"],
    "history_months": ["<6", "6-12", "12-24", "24-48", "48-96", "96+"],
}
MISSING_LABELS = {"delinq_sev": "no\nhistory", "clean_track": "no\nhistory", "dti": "n/a", "dp_ratio": "no\nCD/TW",
                  "reject_rate": "n/a", "history_months": "no\nhistory"}


def chart_factors(cards: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.33, 7.5))
    for ax, factor in zip(axes.ravel(), FACTORS):
        card = cards[factor]
        labels = [BUCKET_LABELS[factor][int(i)] for i in range(len(card[card.bucket != "Missing"]))]
        if (card.bucket == "Missing").any():
            labels.append(MISSING_LABELS[factor])
        colors = [PAL["green"] if points >= 80 else PAL["amber"] if points >= 50 else PAL["red"] for points in card.points]
        ax.bar(range(len(card)), card.bad_rate * 100, color=colors, edgecolor="white")
        for i, points in enumerate(card.points):
            ax.text(i, card.bad_rate.iloc[i] * 100, f"{points:.0f}", ha="center", va="bottom", fontsize=8, color=PAL["grey"])
        ax.set_xticks(range(len(card)), labels, fontsize=8)
        ax.set_ylabel(f"{FACTOR_LABELS[factor]}\nBad rate (%)", fontsize=8)
        ax.set_ylim(0, card.bad_rate.max() * 100 * 1.18)
        ax.grid(axis="x", visible=False)
    fig.subplots_adjust(hspace=0.7, wspace=0.3)
    save_fig(fig, TASK_FIGURES / "05_key_factor_impacts.png", None, top=0.96, bottom=0.11)


# =================================================================================================
def collection_validation(con, pop: pd.DataFrame) -> dict:
    """Independent forward check: collection calls AFTER the snapshot date, by FHS tier."""
    cc = con.execute(f"""
        SELECT SKP_CLIENT, max((disposition_group = 'COLLECTION')::INT) AS had_collection_call
        FROM {pq('call_clean')} WHERE call_start >= TIMESTAMP '{C.SNAPSHOT_DATE}' GROUP BY 1""").df()
    m = pop[["SKP_CLIENT", "fh_short", "Financial_Health_Score", "n_open_contracts"]].merge(cc, on="SKP_CLIENT", how="inner")
    t = m.groupby("fh_short").had_collection_call.agg(["mean", "size"]).reindex(["High", "Medium", "Low"])
    o = m[m.n_open_contracts > 0]
    to = o.groupby("fh_short").had_collection_call.agg(["mean", "size"]).reindex(["High", "Medium", "Low"])
    return {"n_called_after_snapshot": len(m), "collection_rate_by_tier": t.reset_index().to_dict("records"),
            "auc": auc(m.had_collection_call, -m.Financial_Health_Score),
            "n_open_called": len(o), "collection_rate_by_tier_open_debt": to.reset_index().to_dict("records"),
            "auc_open_debt": auc(o.had_collection_call, -o.Financial_Health_Score)}


def fairness_check(con, pop: pd.DataFrame) -> dict:
    cust = con.execute(f"SELECT SKP_CLIENT, NAME_GENDER, DATE_BIRTH FROM {pq('customer_clean')}").df()
    m = pop[["SKP_CLIENT", "Financial_Health_Score", "fh_short", "fh_thin_file"]].merge(cust, on="SKP_CLIENT")
    age = (pd.Timestamp(C.SNAPSHOT_DATE) - pd.to_datetime(m.DATE_BIRTH)).dt.days / 365.25
    m["age_band"] = pd.cut(age, [0, 25, 35, 45, 55, 120], labels=["<25", "25-34", "35-44", "45-54", "55+"])
    g = m.groupby("NAME_GENDER").agg(mean_fhs=("Financial_Health_Score", "mean"),
                                     low_share=("fh_short", lambda s: (s == "Low").mean()), n=("fh_short", "size"))
    a = m.groupby("age_band", observed=True).agg(mean_fhs=("Financial_Health_Score", "mean"),
                                                 low_share=("fh_short", lambda s: (s == "Low").mean()), n=("fh_short", "size"))
    nt = m[~m.fh_thin_file]
    g2 = nt.groupby("NAME_GENDER").agg(mean_fhs=("Financial_Health_Score", "mean"),
                                       low_share=("fh_short", lambda s: (s == "Low").mean()), n=("fh_short", "size"))
    a2 = nt.groupby("age_band", observed=True).agg(mean_fhs=("Financial_Health_Score", "mean"),
                                                   low_share=("fh_short", lambda s: (s == "Low").mean()), n=("fh_short", "size"))
    thin_age = m.groupby("age_band", observed=True).fh_thin_file.mean()
    return {"by_gender": g.reset_index().to_dict("records"), "by_age": a.reset_index().astype({"age_band": str}).to_dict("records"),
            "by_gender_excl_thin_file": g2.reset_index().to_dict("records"),
            "by_age_excl_thin_file": a2.reset_index().astype({"age_band": str}).to_dict("records"),
            "thin_file_share_by_age": {str(k): float(v) for k, v in thin_age.items()},
            "thin_file_share_by_gender": m.groupby("NAME_GENDER").fh_thin_file.mean().to_dict()}


# %% [markdown]
# # Challenge 1 — protect the customer base
#
# **Story 1 of 3:** identify repayment pressure from observed credit behavior. Keep thin-file customers distinct from customers with repayment evidence. The score ranks observed risk; it does not determine eligibility.

# %%
def financial_health_report(metrics: dict) -> str:
    validation = metrics["validation"]
    tier_table = pd.DataFrame(validation["tier_table"])
    tier_rates = dict(zip(tier_table.tier, tier_table.bad_rate))
    tier_table["bad_rate"] = tier_table.bad_rate.map(lambda value: f"{value:.2%}")
    products = pd.DataFrame(metrics["product_risk"]).sort_values("bad_rate", ascending=False)
    weights = pd.Series(metrics["pillar_weights"], name="weight").to_frame()
    low_customers = int(metrics["tier_counts"].get("Low", 0))
    thin_file_low = int(round(low_customers * metrics["low_tier_thin_file_share"]))
    observed_stress_low = low_customers - thin_file_low
    dti = metrics["dti"]
    risk_lift = tier_rates["Low"] / tier_rates["High"] if tier_rates.get("High", 0) else float("nan")
    strongest_factor, strongest_gini = max(validation["factor_gini"].items(), key=lambda item: item[1])
    return "\n".join(
        [
            "# Task 1 — Protect the base",
            "",
            "## Business question",
            "",
            "Which customers show observable repayment pressure, and which signals provide a defensible early-review score?",
            "",
            "## Analysis",
            "",
            f"The score uses mature contracts and features frozen before the **{C.FH_TARGET_START}–{C.FH_TARGET_END}** outcome window. It is a 0–100 weighted scorecard, not a black-box approval model. Pillar weights:\n\n{weights.to_markdown()}\n\nThe held-out cohort contains {validation['n_holdout']:,} records and is separate from score fitting.",
            "",
            "## Findings and business insight",
            "",
            f"- **The tiers order later repayment outcomes:** holdout delinquency is {tier_rates['Low']:.2%} for Low versus {tier_rates['High']:.2%} for High, a {risk_lift:.1f}× difference (Gini {validation['gini_holdout']:.3f}). Use this as a review-priority signal, not as an individual guarantee.\n"
            f"- **Low score does not always mean repayment stress:** {thin_file_low:,} of {low_customers:,} Low-tier customers ({metrics['low_tier_thin_file_share']:.1%}) have no signed-loan history. Only the remainder ({observed_stress_low:,}) have an established loan history that can be checked for repayment evidence. Keep these paths separate.\n"
            f"- **Debt burden is a useful monitoring boundary:** above the observed {dti['breakpoint']:.1%} DTI breakpoint, the mature-cohort delinquency rate is {dti['bad_above']:.2%}, versus {dti['bad_below']:.2%} below it ({dti['n_above']:,} and {dti['n_below']:,} contracts). This is a portfolio pattern, not a universal affordability cutoff.\n"
            f"- **Product differences deserve a review, not a causal story:** {products.iloc[0]['PRODUCT']} is highest at {products.iloc[0]['bad_rate']:.2%}; {products.iloc[-1]['PRODUCT']} is lowest at {products.iloc[-1]['bad_rate']:.2%}. The strongest single-factor holdout Gini is `{strongest_factor}` ({strongest_gini:.3f}).",
            "",
            "## Business implication and handoff",
            "",
            "Use verified repayment signals to prioritize support and affordability review. Route thin-file customers to a separate evidence-building or education path; do not label them delinquent. Task 2 adds engagement evidence to decide how to reach each group, and Task 3 combines both scores into a service action.",
            "",
            "## Limits",
            "",
            "The outcome is early delinquency on a historical cohort. Product, income, insurance, and down-payment comparisons are observational. Demographics are used only in a post-hoc fairness check, not in scoring or segmentation.",
            "",
        ]
    )


def display_saved_figure(path: Path) -> None:
    if "get_ipython" in globals():
        from IPython.display import Image, display

        display(Image(filename=str(path)))


# %% [markdown]
# ## 1. Compare repayment outcomes by product
#
# Start with the mature, signed-contract sample. The chart compares delinquency rate and loss severity; it does not imply that the product itself causes the difference.

# %%
if __name__ == "__main__":
    ensure_dirs(DATA_CLEAN, TASK_FIGURES, TASK_SCORES, TASK_REPORT.parent)
    set_style()
    con = get_con()
    product_risk = q1_product_risk(con)
    chart_product(product_risk)
    display_saved_figure(TASK_FIGURES / "01_delinquency_by_product_type.png")

# %% [markdown]
# **Interpretation.** The portfolio line is a reference, not a target. Use the rate and sample size together; a product-level difference is an association, not a causal product effect.

# %% [markdown]
# ## 2. Check affordability and income coverage
#
# Compare delinquency across DTI bins. The adjacent panel asks whether personal or household income is both available and informative enough to use as the denominator.

# %%
if __name__ == "__main__":
    income_resilience = q2_income_resilience(con)
    dti_evidence = q4_dti_tipping(con)
    chart_dti(dti_evidence, income_resilience)
    display_saved_figure(TASK_FIGURES / "02_dti_vs_delinquency_tipping_point.png")

# %% [markdown]
# **Interpretation.** The breakpoint is an empirical feature of this portfolio and observation window. It should guide review and monitoring, not be presented as a universal affordability rule.

# %% [markdown]
# ## 3. Separate risk markers from protective evidence
#
# Compare insured with uninsured contracts, then assess how down-payment bands relate to delinquency within products.

# %%
if __name__ == "__main__":
    mitigation_evidence = q3_mitigation(con)
    chart_mitigation(mitigation_evidence)
    display_saved_figure(TASK_FIGURES / "03_insurance_and_downpayment_impact.png")

# %% [markdown]
# **Interpretation.** Insurance may be selected more often for already-risky loans, so a higher observed rate does not show that insurance increases risk. Down-payment comparisons are also observational.

# %% [markdown]
# ## 4. Fit, validate, and apply the scorecard
#
# Freeze features before the target period, fit on the development split, check the later holdout, then apply the same scorecard to the customer population.

# %%
if __name__ == "__main__":
    development = dev_sample(con)
    scorecard = fit_scorecard(con, development)
    validation = validate(development, scorecard["calib"])
    scorecard_table = pd.concat(
        [card.assign(factor=factor, pillar=FACTORS[factor][0], factor_label=FACTOR_LABELS[factor],
                     pillar_weight=PILLAR_WEIGHTS[FACTORS[factor][0]])
         for factor, card in scorecard["cards"].items()],
        ignore_index=True,
    )
    population = score_population(con, scorecard["cards"], scorecard["calib"])
    population["fh_drivers"] = explain_fh(population)
    collection = collection_validation(con, population)
    fairness = fairness_check(con, population)
    keep = [
        "SKP_CLIENT", "Financial_Health_Score", "Financial_Health_Tier", "fh_short", "fh_thin_file", "fh_policy_cap",
        "n_open_delinquent", "expected_bad_rate", "fh_drivers", "n_signed", "n_delinquent", "any_default",
        "n_clean_completed", "n_rejected", "n_applications", "income", "monthly_obligations", "n_open_contracts",
        "dti", "dp_ratio", "reject_rate", "history_months", "total_credit", "open_annuity", "insurance_share",
    ] + [column for column in population.columns if column.startswith("pillar_") or column.startswith("pts_")]
    population[keep].to_parquet(TASK_SCORES / "fh_scores.parquet", index=False)
    tiers = population.fh_short.value_counts().reindex(["High", "Medium", "Low"])
    task1_metrics = {
        "product_risk": product_risk.to_dict("records"),
        "portfolio_bad_rate": product_risk.attrs["portfolio_bad_rate"],
        "portfolio_delinq_rate": product_risk.attrs["portfolio_delinq_rate"],
        "n_signed_obs": product_risk.attrs["n"],
        "tw_vs_cd": {"z": product_risk.attrs["tw_vs_cd_z"], "p": product_risk.attrs["tw_vs_cd_p"]},
        "income_resilience": income_resilience,
        "mitigation": {key: value for key, value in mitigation_evidence.items() if key not in ("insurance", "dp", "ins_confounding")}
                      | {"insurance_table": mitigation_evidence["insurance"].to_dict("records"),
                         "downpayment_table": mitigation_evidence["dp"].to_dict("records")},
        "dti": {key: value for key, value in dti_evidence.items() if key != "curve"}
               | {"curve": dti_evidence["curve"].to_dict("records")},
        "scorecard": scorecard_table.to_dict("records"),
        "calibration": scorecard["calib"],
        "validation": validation,
        "pillar_weights": PILLAR_WEIGHTS,
        "tier_counts": tiers.to_dict(),
        "tier_share": (tiers / tiers.sum()).to_dict(),
        "score_mean": population.Financial_Health_Score.mean(),
        "score_median": population.Financial_Health_Score.median(),
        "thin_file_customers": int((population.n_signed == 0).sum()),
        "low_tier_thin_file_share": float(population.loc[population.fh_short == "Low", "fh_thin_file"].mean()),
        "policy_caps_population": population.fh_policy_cap.replace("", "none").value_counts().to_dict(),
        "thin_file_tier_mix": population.loc[population.fh_thin_file, "fh_short"].value_counts(normalize=True).to_dict(),
        "collection_validation": collection,
        "fairness": fairness,
    }
    save_json(task1_metrics, DATA_CLEAN / "metrics_phase2.json")
    TASK_REPORT.parent.mkdir(parents=True, exist_ok=True)
    TASK_REPORT.write_text(financial_health_report(task1_metrics), encoding="utf-8")
    con.close()

# %% [markdown]
# ## 5. Inspect score coverage and holdout separation
#
# Use the tier boundaries defined by the score. The second panel is a later-outcome check, not a promise about any individual customer.

# %%
if __name__ == "__main__":
    chart_distribution(population, validation)
    display_saved_figure(TASK_FIGURES / "04_financial_health_score_distribution.png")

# %% [markdown]
# **Interpretation.** Compare the tier shares with later delinquency rates. Thin-file customers sit in the low-score group because evidence is limited; review the profile before treating the score as repayment stress.

# %% [markdown]
# ## 6. Review the point assignments
#
# The final figure shows how each feature bucket maps to points. The scorecard rows and factor Gini values are recorded in `Data_clean/metrics_phase2.json`.

# %%
if __name__ == "__main__":
    chart_factors(scorecard["cards"])
    display_saved_figure(TASK_FIGURES / "05_key_factor_impacts.png")

# %% [markdown]
# **Interpretation.** Points reflect observed development-sample delinquency after monotonic calibration. Check factor coverage and holdout performance before using the score for customer-level decisions.
