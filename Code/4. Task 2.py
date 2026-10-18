# %% [markdown]
# # 4. Customer engagement
#
# Score recent activity, frequency, product breadth, and responsiveness with five transparent components. Validate the score against later applications and call outcomes; associations do not establish cause.

# %%
"""
PHASE 3 - Challenge 2: Customer Engagement Score (0-100).

Five explainable components (each 0-100, simple published transforms - no model):
  E1 Digital Adoption   20%  best of: app sessions/week (2 per week = full marks, log scale) or digital
                             self-serve applications (APP / WEB / SOB) in the last 12 months (3 = full marks)
  E2 Recency            20%  100 x 0.5^(days since last self-initiated activity / 120)   (half-life 120 days)
  E3 Frequency          25%  active months in the last 12 months (6+ = full marks)
  E4 Responsiveness     20%  reach rate on outbound calls, positive reaction to sales calls, reply rate to leads
                             (+25 points per customer-initiated inbound call, max +50). Missing if the
                             customer was never contacted -> weight re-distributed (not penalised).
  E5 Product Breadth    15%  distinct product families touched in 24 months (3+ = full marks)

Self-initiated activity = scoring/application events (PAST_BEHAVIOR), contract applications (CONTRACT) and
customer-originated leads (GMA app / landing page). Calls are scored over their own window (2025-01..2026-08).

Validation: features freeze at `ENG_ASOF` (2025-02-01); the target window ends at `ENG_TARGET_END` (2025-06-05).
The validation uses new applications and signings over that four-month window.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path


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
TASK_FIGURES = ROOT / "output" / "Task2" / "figures"
TASK_SCORES = ROOT / "output" / "Task2" / "scores"
TASK_REPORT = ROOT / "output" / "Task2" / "report" / "report.md"


def pq(name: str) -> str:
    return f"'{(DATA_CLEAN / f'{name}.parquet').as_posix()}'"


from Code.config import config as C
from Code.config.utils import (PAL, TIER_COLORS, assign_tier, auc, ensure_dirs, get_con, save_fig, save_json,
                               set_style, two_prop_ztest, wilson_ci)

# %% [markdown]
# ## Define the score components
#
# Fixed weights and transforms live in this file and `Code/config/config.py`. A component with no available call evidence is not treated as a zero response.

# %%
# Weights chosen among simple candidate sets as the best balance of two independent validations
# (re-application AUC and forward sales-call conversion AUC) - see Executive Summary, Challenge 2.
ENG_WEIGHTS = {"E1_digital": 0.20, "E2_recency": 0.20, "E3_frequency": 0.25, "E4_responsiveness": 0.20,
               "E5_breadth": 0.15}
CUSTOMER_LEAD_SOURCES = ("GMA", "LANDING PAGE")
MIN_SOURCE_LEADS = 30    # rates on fewer leads are too noisy to chart (hidden, or n/a for follow-up measures)


# =================================================================================================
# Features
# =================================================================================================
# %% [markdown]
# ## Build features at a defined snapshot
#
# Activity is measured only before the score date. Calls have their own extract window; later outcomes are reserved for validation.

# %%
def build_features(con, asof: str, call_from: str | None, call_to: str | None, use_app: bool) -> pd.DataFrame:
    a = f"TIMESTAMP '{asof}'"
    srcs = ", ".join(f"'{s}'" for s in CUSTOMER_LEAD_SOURCES)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE act AS
      SELECT SKP_CLIENT, event_time AS ts, product_family AS fam, is_digital_self_serve AS digital, 'behavior' AS kind
        FROM {pq('behavior_clean')} WHERE event_time < {a}
      UNION ALL
      SELECT SKP_CLIENT, apply_time, CASE WHEN PRODUCT IN ('CL','CW') THEN 'CL' ELSE PRODUCT END, FALSE, 'application'
        FROM {pq('contract_clean')} WHERE apply_time < {a}
      UNION ALL
      SELECT SKP_CLIENT, lead_time, NULL, TRUE, 'own_lead'
        FROM {pq('leads_clean')} WHERE lead_time < {a} AND LEAD_SOURCE IN ({srcs})""")
    f = con.execute(f"""
      SELECT SKP_CLIENT,
        max(ts) AS last_activity,
        count(DISTINCT date_trunc('month', ts)) FILTER (WHERE ts >= {a} - INTERVAL 365 DAY) AS active_months_12m,
        count(*) FILTER (WHERE ts >= {a} - INTERVAL 365 DAY) AS events_12m,
        count(*) FILTER (WHERE ts >= {a} - INTERVAL 365 DAY AND digital) AS digital_events_12m,
        count(DISTINCT fam) FILTER (WHERE ts >= {a} - INTERVAL 730 DAY AND fam IS NOT NULL) AS product_families_24m,
        count(*) FILTER (WHERE kind = 'application' AND ts >= {a} - INTERVAL 365 DAY) AS applications_12m
      FROM act GROUP BY 1""").df()
    leads = con.execute(f"""
      SELECT SKP_CLIENT, count(*) AS leads_24m, sum(is_deal::INT) AS lead_deals_24m,
             sum((NOT is_deal AND NOT is_expired AND fin_reason NOT IN ('Not Applicable','Unknown'))::INT) AS lead_replies_24m,
             sum(is_expired::INT) AS lead_expired_24m,
             max_by(max_offer, (lead_time, coalesce(max_offer, 0), lead_product)) AS latest_max_offer, max(max_offer) AS max_offer_24m,
             max_by(lead_product, (lead_time, coalesce(max_offer, 0), lead_product)) AS latest_lead_product
      FROM {pq('leads_clean')} WHERE lead_time < {a} AND lead_time >= {a} - INTERVAL 730 DAY GROUP BY 1""").df()
    cust = con.execute(f"SELECT SKP_CLIENT, app_sessions_wk FROM {pq('customer_clean')}").df()
    df = cust.merge(f, on="SKP_CLIENT", how="left").merge(leads, on="SKP_CLIENT", how="left")
    if call_from:
        calls = con.execute(f"""
          SELECT SKP_CLIENT,
            count(*) FILTER (WHERE is_outbound) AS out_attempts,
            count(*) FILTER (WHERE is_outbound AND is_connected) AS out_connected,
            count(*) FILTER (WHERE is_inbound) AS inbound_calls,
            count(*) FILTER (WHERE is_sales_outcome) AS sales_outcomes,
            count(*) FILTER (WHERE is_positive_reaction) AS positive_reactions,
            count(*) FILTER (WHERE is_conversion) AS conversions,
            max((disposition_group = 'OPT_OUT')::INT) AS opted_out,
            count(*) FILTER (WHERE disposition_group = 'COLLECTION') AS collection_calls,
            round(avg(duration_s) FILTER (WHERE is_connected AND is_outbound), 3) AS avg_connected_duration
          FROM {pq('call_clean')}
          WHERE call_start >= TIMESTAMP '{call_from}' AND call_start < TIMESTAMP '{call_to}' GROUP BY 1""").df()
        df = df.merge(calls, on="SKP_CLIENT", how="left")
    else:
        for c_ in ["out_attempts", "out_connected", "inbound_calls", "sales_outcomes", "positive_reactions",
                   "conversions", "opted_out", "collection_calls"]:
            df[c_] = 0
        df["avg_connected_duration"] = np.nan
    zero_cols = ["active_months_12m", "events_12m", "digital_events_12m", "product_families_24m", "applications_12m",
                 "leads_24m", "lead_deals_24m", "lead_replies_24m", "lead_expired_24m", "out_attempts", "out_connected",
                 "inbound_calls", "sales_outcomes", "positive_reactions", "conversions", "opted_out", "collection_calls"]
    df[zero_cols] = df[zero_cols].fillna(0)
    df["days_since_last_activity"] = (pd.Timestamp(asof) - pd.to_datetime(df.last_activity)).dt.days
    if not use_app:
        df["app_sessions_wk"] = np.nan
    return df


def score(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=df.index)
    digital = 100 * np.minimum(df.digital_events_12m / 3, 1)
    if df.app_sessions_wk.notna().any():
        app = 100 * np.minimum(np.log1p(df.app_sessions_wk.fillna(0)) / np.log1p(2.0), 1)
        s["E1_digital"] = np.maximum(app, digital)
    else:
        s["E1_digital"] = digital
    s["E2_recency"] = np.where(df.days_since_last_activity.isna(), 0,
                               100 * 0.5 ** (df.days_since_last_activity.clip(lower=0) / C.RECENCY_HALF_LIFE_DAYS))
    s["E3_frequency"] = 100 * np.minimum(df.active_months_12m / C.FREQ_FULL_MARKS_MONTHS, 1)
    # reach: picking up 30%+ of dial attempts = full marks; reaction: 50%+ positive sales outcomes = full marks
    reach = np.where(df.out_attempts > 0, 100 * np.minimum(df.out_connected / df.out_attempts.replace(0, np.nan) / 0.3, 1), np.nan)
    react = np.where(df.sales_outcomes > 0, 100 * np.minimum(df.positive_reactions / df.sales_outcomes.replace(0, np.nan) / 0.5, 1), np.nan)
    leadr = np.where(df.leads_24m > 0, 100 * np.minimum((df.lead_deals_24m + 0.4 * df.lead_replies_24m) /
                                                        df.leads_24m.replace(0, np.nan), 1), np.nan)
    comp = np.vstack([reach, react, leadr])
    has = ~np.isnan(comp)
    base = np.where(has.any(axis=0), np.nansum(comp, axis=0) / np.maximum(has.sum(axis=0), 1), np.nan)
    bonus = 25 * np.minimum(df.inbound_calls, C.INBOUND_BONUS_MAX_CALLS)
    e4 = np.where(np.isnan(base) & (bonus > 0), bonus, base + bonus)
    s["E4_responsiveness"] = np.minimum(e4, 100)
    s["E5_breadth"] = 100 * np.minimum(df.product_families_24m / C.BREADTH_FULL_MARKS, 1)
    w = pd.DataFrame({k: np.where(s[k].isna(), 0, v) for k, v in ENG_WEIGHTS.items()}, index=s.index)
    s["Engagement_Score"] = (s.fillna(0)[list(ENG_WEIGHTS)] * w).sum(axis=1) / w.sum(axis=1)
    s["Engagement_Score"] = s["Engagement_Score"].clip(0, 100).round(1)
    s["e4_available"] = s["E4_responsiveness"].notna()
    return s


# =================================================================================================
# Validation (out-of-time)
# =================================================================================================
# %% [markdown]
# ## Validate against later behavior
#
# Check whether earlier engagement ranks later applications and responses. These holdout associations are evidence for score usefulness, not causal effects.

# %%
def validate(con) -> dict:
    feat = build_features(con, C.ENG_ASOF, None, None, use_app=False)
    sc = score(feat)
    out = con.execute(f"""
        SELECT SKP_CLIENT, max(1) AS applied, max(is_signed::INT) AS signed
        FROM {pq('contract_clean')}
        WHERE apply_time >= TIMESTAMP '{C.ENG_ASOF}' AND apply_time < TIMESTAMP '{C.ENG_TARGET_END}' GROUP BY 1""").df()
    lead = con.execute(f"""
        SELECT SKP_CLIENT, max(is_deal::INT) AS lead_deal FROM {pq('leads_clean')}
        WHERE lead_time >= TIMESTAMP '{C.ENG_ASOF}' AND lead_time < TIMESTAMP '{C.ENG_TARGET_END}' GROUP BY 1""").df()
    d = pd.concat([feat[["SKP_CLIENT"]], sc], axis=1).merge(out, on="SKP_CLIENT", how="left").merge(lead, on="SKP_CLIENT", how="left")
    d[["applied", "signed"]] = d[["applied", "signed"]].fillna(0)
    d["tier"] = assign_tier(d.Engagement_Score, C.ENG_TIERS).map(C.ENG_SHORT)
    res = {"asof": C.ENG_ASOF, "n": len(d), "apply_rate": d.applied.mean(), "sign_rate": d.signed.mean(),
           "auc_apply": auc(d.applied, d.Engagement_Score), "auc_sign": auc(d.signed, d.Engagement_Score)}
    ld = d[d.lead_deal.notna()]
    res["auc_lead_deal"] = auc(ld.lead_deal, ld.Engagement_Score)
    res["n_with_new_lead"] = len(ld)
    res["component_auc_apply"] = {k: auc(d.applied, d[k].fillna(d[k].median() if d[k].notna().any() else 0))
                                  for k in ENG_WEIGHTS}
    t = d.groupby("tier").agg(n=("applied", "size"), apply_rate=("applied", "mean"), sign_rate=("signed", "mean"))
    res["tier_table"] = t.reindex(["High", "Medium", "Low"]).reset_index().to_dict("records")
    d["decile"] = pd.qcut(d.Engagement_Score.rank(method="first"), 10, labels=False) + 1
    res["decile_table"] = d.groupby("decile").agg(score_min=("Engagement_Score", "min"),
                                                  score_max=("Engagement_Score", "max"),
                                                  apply_rate=("applied", "mean"),
                                                  sign_rate=("signed", "mean")).reset_index().to_dict("records")
    # equal-weight benchmark
    eq = d[list(ENG_WEIGHTS)].fillna(0).mean(axis=1)
    res["auc_apply_equal_weights"] = auc(d.applied, eq)
    return res


def forward_call_validation(con) -> dict:
    """Score built only from data before the snapshot -> conversion on sales calls AFTER the snapshot."""
    feat = build_features(con, C.SNAPSHOT_DATE, "2025-01-01", C.SNAPSHOT_DATE, use_app=True)
    sc = score(feat)
    post = con.execute(f"""
        SELECT SKP_CLIENT, max(is_conversion::INT) conv, max(is_positive_reaction::INT) pos
        FROM {pq('call_clean')} WHERE call_start >= TIMESTAMP '{C.SNAPSHOT_DATE}' AND is_sales_outcome GROUP BY 1""").df()
    d = pd.concat([feat[["SKP_CLIENT"]], sc[["Engagement_Score"]]], axis=1).merge(post, on="SKP_CLIENT")
    d["tier"] = assign_tier(d.Engagement_Score, C.ENG_TIERS).map(C.ENG_SHORT)
    t = d.groupby("tier").agg(n=("conv", "size"), conv=("conv", "mean"), pos=("pos", "mean")).reindex(["High", "Medium", "Low"])
    nc = sc.drop(columns=["E4_responsiveness"])
    w = {k: v for k, v in ENG_WEIGHTS.items() if k != "E4_responsiveness"}
    no_calls = sum(nc[k].fillna(0) * v for k, v in w.items()) / sum(w.values())
    d = d.merge(pd.DataFrame({"SKP_CLIENT": feat.SKP_CLIENT, "score_no_calls": no_calls}), on="SKP_CLIENT")
    return {"n": len(d), "auc_conversion": auc(d.conv, d.Engagement_Score), "auc_positive": auc(d.pos, d.Engagement_Score),
            "auc_conversion_without_call_component": auc(d.conv, d.score_no_calls),
            "conversion_rate": d.conv.mean(), "tier_table": t.reset_index().to_dict("records")}


# =================================================================================================
# Deep-dive questions
# =================================================================================================
# %% [markdown]
# ## Analyze engagement drivers
#
# Compare app activity, call behavior, lead sources, and early-warning patterns using bounded cohorts and explicit time windows.

# %%
def q1_active_vs_passive(con, pop) -> dict:
    d = pop[(pop.sales_outcomes > 0) & (pop.collection_calls == 0)].copy()
    d["converted"] = (d.conversions > 0).astype(int)
    d["positive"] = (d.positive_reactions > 0).astype(int)
    d["app_band"] = pd.cut(d.app_sessions_wk, [-1, 0, 0.5, 2, 1e9], labels=["No app use", "<0.5 / wk", "0.5-2 / wk", "2+ / wk"])
    t = d.groupby("app_band", observed=True).agg(n=("converted", "size"), conv=("converted", "mean"),
                                                 pos=("positive", "mean")).reset_index()
    t["ci_lo"], t["ci_hi"] = zip(*[wilson_ci(c * n, n) for c, n in zip(t.conv, t.n)])
    users = d[d.app_sessions_wk > 0]
    non = d[d.app_sessions_wk == 0]
    active = d[d.app_sessions_wk >= 0.5]
    z, p = two_prop_ztest(active.converted.sum(), len(active), non.converted.sum(), len(non))
    leads = con.execute(f"""
        SELECT LEAD_SOURCE, count(*) n, avg(is_deal::INT) deal_rate, avg(is_expired::INT) expired_rate,
               avg(is_not_interest::INT) not_interest_rate
        FROM {pq('leads_clean')} GROUP BY 1 ORDER BY deal_rate DESC""").df()
    return {"by_app_band": t, "conv_app_users": users.converted.mean(), "conv_non_users": non.converted.mean(),
            "conv_active_0.5plus": active.converted.mean(), "z_active_vs_non": z, "p_active_vs_non": p,
            "n_called": len(d), "n_app_users_called": len(users), "lead_source_deal": leads}


def q2_call_dynamics(con) -> dict:
    d = con.execute(f"""
      WITH c AS (SELECT * FROM {pq('call_clean')}
                 WHERE SKP_CLIENT NOT IN (SELECT SKP_CLIENT FROM {pq('call_clean')} WHERE disposition_group='COLLECTION')),
      first_conn AS (
        SELECT SKP_CLIENT, arg_min(duration_s, call_start) first_dur, arg_min(is_conversion::INT, call_start) first_conv
        FROM c WHERE is_outbound AND is_connected GROUP BY 1)
      SELECT c.SKP_CLIENT,
             count(*) FILTER (WHERE is_outbound) attempts,
             count(*) FILTER (WHERE is_outbound AND is_connected) connected,
             avg(duration_s) FILTER (WHERE is_outbound AND is_connected) avg_dur,
             max(is_conversion::INT) converted,
             any_value(f.first_dur) first_dur, any_value(f.first_conv) first_conv
      FROM c LEFT JOIN first_conn f USING (SKP_CLIENT)
      GROUP BY 1 HAVING count(*) FILTER (WHERE is_outbound) > 0""").df()
    res = {"n_customers": len(d), "conv_rate": d.converted.mean()}
    res["auc_attempts"] = auc(d.converted, d.attempts)
    res["auc_connected"] = auc(d.converted, d.connected)
    m = d.avg_dur.notna()
    res["auc_avg_duration"] = auc(d.converted[m], d.avg_dur[m])
    rob = d[(d.first_dur.notna()) & (d.first_conv == 0)]
    res["auc_first_call_duration_excl_first_call_conversions"] = auc(rob.converted, rob.first_dur)
    res["n_robustness"] = len(rob)
    res["spearman_attempts"] = float(d[["attempts", "converted"]].corr("spearman").iloc[0, 1])
    res["spearman_duration"] = float(d.loc[m, ["avg_dur", "converted"]].corr("spearman").iloc[0, 1])
    d["att_band"] = pd.cut(d.attempts, [0, 1, 2, 3, 5, 10, 20, 50, 1e9], labels=["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "50+"])
    d["dur_band"] = pd.cut(d.avg_dur, [-1, 10, 30, 60, 120, 300, 600, 1e9],
                           labels=["<10s", "10-30s", "30-60s", "1-2m", "2-5m", "5-10m", "10m+"])
    res["by_attempts"] = d.groupby("att_band", observed=True).agg(n=("converted", "size"), conv=("converted", "mean")).reset_index()
    res["by_duration"] = d[m].groupby("dur_band", observed=True).agg(n=("converted", "size"), conv=("converted", "mean")).reset_index()
    res["by_first_duration_rob"] = rob.assign(b=pd.cut(rob.first_dur, [-1, 10, 30, 60, 120, 300, 600, 1e9],
                                                       labels=["<10s", "10-30s", "30-60s", "1-2m", "2-5m", "5-10m", "10m+"])) \
        .groupby("b", observed=True).agg(n=("converted", "size"), conv=("converted", "mean")).reset_index()
    return res


def q3_lead_source(con, pop) -> pd.DataFrame:
    df = con.execute(f"""
      WITH l AS (SELECT *, CASE WHEN lead_product = 'BNP' THEN 'BNPL' WHEN lead_product IN ('CLX','ACL') THEN 'CL'
                                WHEN lead_product = 'SCW' THEN 'CW' WHEN lead_product IN ('CCX','VCC','CCW') THEN 'CC'
                                WHEN lead_product = 'EPOS' THEN 'CD' ELSE lead_product END AS lead_prod_mapped
                 FROM {pq('leads_clean')} WHERE lead_time < TIMESTAMP '{C.SNAPSHOT_DATE}' - INTERVAL 90 DAY
                   AND lead_time >= TIMESTAMP '2022-12-01'),   -- PAST_BEHAVIOR coverage starts 2022-12
      conv AS (   -- the offered product must actually be signed within 90 days of the lead
        SELECT l.SKP_CLIENT, l.lead_time, l.LEAD_SOURCE, max((c.SKP_CLIENT IS NOT NULL)::INT) signed_90d
        FROM l LEFT JOIN {pq('contract_clean')} c ON c.SKP_CLIENT = l.SKP_CLIENT AND c.is_signed
             AND c.PRODUCT = l.lead_prod_mapped AND c.cdate >= l.lead_time AND c.cdate <= l.lead_time + INTERVAL 90 DAY
        GROUP BY ALL),
      act AS (
        SELECT l.SKP_CLIENT, l.lead_time, l.LEAD_SOURCE, count(b.SKP_CLIENT) events_180d
        FROM l LEFT JOIN {pq('behavior_clean')} b ON b.SKP_CLIENT = l.SKP_CLIENT
             AND b.event_time > l.lead_time AND b.event_time <= l.lead_time + INTERVAL 180 DAY
        GROUP BY ALL)
      SELECT l.LEAD_SOURCE, count(*) n_leads, count(DISTINCT l.SKP_CLIENT) n_customers,
             avg(l.is_deal::INT) deal_rate, avg(l.is_not_interest::INT) not_interest_rate, avg(l.is_expired::INT) expired_rate,
             avg(((l.accepted_time IS NOT NULL) OR l.sob_direct_support)::INT) handled_rate,
             median(l.max_offer) FILTER (WHERE l.max_offer > 1000) median_max_offer,
             avg(conv.signed_90d) signed_within_90d,
             avg(act.events_180d) FILTER (WHERE l.lead_time < TIMESTAMP '{C.SNAPSHOT_DATE}' - INTERVAL 180 DAY) avg_events_180d_after,
             avg((act.events_180d > 0)::INT) FILTER (WHERE l.lead_time < TIMESTAMP '{C.SNAPSHOT_DATE}' - INTERVAL 180 DAY) active_180d_after
      FROM l JOIN conv USING (SKP_CLIENT, lead_time, LEAD_SOURCE) JOIN act USING (SKP_CLIENT, lead_time, LEAD_SOURCE)
      GROUP BY 1 ORDER BY n_leads DESC""").df()
    src = con.execute(f"""SELECT SKP_CLIENT, max_by(LEAD_SOURCE, (lead_time, LEAD_SOURCE)) last_source FROM {pq('leads_clean')} GROUP BY 1""").df()
    m = src.merge(pop[["SKP_CLIENT", "Engagement_Score", "days_since_last_activity"]], on="SKP_CLIENT")
    g = m.groupby("last_source").agg(mean_engagement_score=("Engagement_Score", "mean"),
                                     still_active_6m=("days_since_last_activity", lambda s: (s <= 182).mean()))
    deal = con.execute(f"""
      SELECT LEAD_SOURCE, count(*) n_leads_deal, avg(is_deal::INT) deal_rate_all
      FROM {pq('leads_clean')} WHERE lead_time >= TIMESTAMP '2022-12-01' GROUP BY 1""").df()
    return deal.merge(df, on="LEAD_SOURCE", how="left").merge(g, left_on="LEAD_SOURCE", right_index=True, how="left")


def q4_early_warning(con, pop) -> dict:
    # (a) cross-sectional: 2025 app sessions vs delinquency on loans active in 2025
    a = con.execute(f"""
      SELECT c.SKP_CLIENT, max(c.bad) bad, any_value(cu.app_sessions_wk) app
      FROM {pq('contract_clean')} c JOIN {pq('customer_clean')} cu USING (SKP_CLIENT)
      WHERE c.is_signed AND c.cdate >= TIMESTAMP '2024-07-01' AND c.cdate < TIMESTAMP '{C.OBS_COMPLETE_BEFORE}'
      GROUP BY 1""").df()
    a["band"] = pd.cut(a.app, [-1, 0, 0.5, 2, 1e9], labels=["No app use", "<0.5 / wk", "0.5-2 / wk", "2+ / wk"])
    ta = a.groupby("band", observed=True).agg(n=("bad", "size"), bad_rate=("bad", "mean")).reset_index()
    # (b) temporal: digital-activity trajectory BEFORE signing -> delinquency on that loan
    b = con.execute(f"""
      WITH k AS (SELECT SKP_CREDIT_CASE, SKP_CLIENT, cdate, bad FROM {pq('contract_clean')}
                 WHERE is_signed AND cdate >= TIMESTAMP '2023-12-01' AND cdate < TIMESTAMP '{C.OBS_COMPLETE_BEFORE}')
      SELECT k.SKP_CREDIT_CASE, any_value(k.bad) bad,
             count(e.SKP_CLIENT) FILTER (WHERE e.event_time >= k.cdate - INTERVAL 365 DAY AND e.event_time < k.cdate - INTERVAL 180 DAY) early,
             count(e.SKP_CLIENT) FILTER (WHERE e.event_time >= k.cdate - INTERVAL 180 DAY AND e.event_time < k.cdate - INTERVAL 30 DAY) late
      FROM k LEFT JOIN {pq('behavior_clean')} e ON e.SKP_CLIENT = k.SKP_CLIENT
           AND e.event_time >= k.cdate - INTERVAL 365 DAY AND e.event_time < k.cdate - INTERVAL 30 DAY
      GROUP BY 1""").df()
    b["trend"] = np.select([(b.early == 0) & (b.late == 0), b.late < b.early, b.late == b.early, b.late > b.early],
                           ["No prior activity", "Declining", "Stable", "Rising"], default="Stable")
    tb = b.groupby("trend").agg(n=("bad", "size"), bad_rate=("bad", "mean")).reindex(
        ["Rising", "Stable", "Declining", "No prior activity"]).reset_index()
    dec, ris = b[b.trend == "Declining"], b[b.trend == "Rising"]
    z, p = two_prop_ztest(dec.bad.sum(), len(dec), ris.bad.sum(), len(ris))
    return {"cross_sectional": ta, "temporal": tb, "z_declining_vs_rising": z, "p_declining_vs_rising": p,
            "auc_app_sessions": auc(a.bad, -a.app), "n_cross": len(a), "n_temporal": len(b)}


def q5_engagement_reducers(con, pop) -> pd.DataFrame:
    """Factors observed before the snapshot -> share of outbound calls the customer answers AFTER the snapshot.
    Later pickup is not a score input, so the comparison is not circular. Associations only."""
    snap = f"TIMESTAMP '{C.SNAPSHOT_DATE}'"
    pre = con.execute(f"""
      SELECT SKP_CLIENT,
        count(*) FILTER (WHERE is_outbound) AS att, count(*) FILTER (WHERE is_outbound AND is_connected) AS conn,
        count(*) FILTER (WHERE disposition_group = 'COLLECTION') AS coll,
        count(*) FILTER (WHERE disposition_group = 'SALES_REFUSAL') AS refus,
        count(*) FILTER (WHERE disposition_group = 'OPT_OUT') AS optout
      FROM {pq('call_clean')} WHERE call_start >= TIMESTAMP '2025-01-01' AND call_start < {snap} GROUP BY 1""").df()
    post = con.execute(f"""
      SELECT SKP_CLIENT, count(*) AS p_att, count(*) FILTER (WHERE is_connected) AS p_conn
      FROM {pq('call_clean')} WHERE is_outbound AND call_start >= {snap} GROUP BY 1""").df()
    leads = con.execute(f"""
      SELECT SKP_CLIENT, count(*) AS n_leads, sum((is_expired OR is_not_interest)::INT) AS dead
      FROM {pq('leads_clean')} WHERE lead_time < {snap} AND lead_time >= {snap} - INTERVAL 730 DAY GROUP BY 1""").df()
    d = (pop[["SKP_CLIENT", "app_sessions_wk", "days_since_last_activity"]].merge(post, on="SKP_CLIENT")
         .merge(pre, on="SKP_CLIENT", how="left").merge(leads, on="SKP_CLIENT", how="left")
         .fillna({"att": 0, "conn": 0, "coll": 0, "refus": 0, "optout": 0, "n_leads": 0, "dead": 0}))
    d["answer_rate"] = d.p_conn / d.p_att
    factors = {
        "Ignored 5+ calls": (d.att >= 5) & (d.conn == 0),
        "Collection calls": d.coll > 0,
        "Opted out": d.optout > 0,
        "No app": d.app_sessions_wk.fillna(0) == 0,
        "Dormant 180d+": d.days_since_last_activity.isna() | (d.days_since_last_activity > 180),
        "All leads lost": (d.n_leads > 0) & (d.dead == d.n_leads),
        "Refused offer": d.refus > 0,
    }
    rows = [{"factor": name, "n_with": int(f.sum()), "answer_with": d.answer_rate[f].mean(),
             "answer_without": d.answer_rate[~f].mean()} for name, f in factors.items()]
    out = pd.DataFrame(rows)
    out["diff_pp"] = (out.answer_with - out.answer_without) * 100
    out["n_population"] = len(d)
    return out


# =================================================================================================
# Charts
# =================================================================================================
# %% [markdown]
# ## App activity and sales-call conversion
#
# Compare conversion among customers with a sales outcome and lead deal rates by source. Call conversion is conditional on sales-call outcome; these comparisons are descriptive and do not establish an effect of app use. Confidence intervals show uncertainty in the call comparison.

# %%
def chart_active_passive(q1, q3: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), gridspec_kw={"width_ratios": [1.1, 1]})
    bands = q1["by_app_band"]
    colors = [PAL["grey"], PAL["grey"], PAL["orange"], PAL["orange"]]
    axes[0].bar(bands.app_band.astype(str), bands.conv * 100, color=colors, width=0.62,
                yerr=[(bands.conv - bands.ci_lo) * 100, (bands.ci_hi - bands.conv) * 100], capsize=4, ecolor=PAL["grey"])
    for i, row in bands.iterrows():
        axes[0].text(i, row.ci_hi * 100 + 0.12, f"{row.conv:.1%}\nn={row.n:,}", ha="center", fontsize=9)
    axes[0].set_ylabel("Sales-call conversion (%)")
    axes[0].set_xlabel("App sessions / week")
    axes[0].grid(axis="x", visible=False)

    leads = q3[q3.n_leads_deal >= MIN_SOURCE_LEADS].sort_values("deal_rate_all")
    # barh draws bottom-up, so the last three rows are the top three deal rates
    colors = [PAL["orange"] if i >= len(leads) - 3 else PAL["grey"] for i in range(len(leads))]
    axes[1].barh(leads.LEAD_SOURCE, leads.deal_rate_all * 100, color=colors, height=0.6)
    for i, (rate, count) in enumerate(zip(leads.deal_rate_all, leads.n_leads_deal)):
        n_label = f"{count / 1e3:,.0f}K" if count >= 1000 else f"{count:,}"
        axes[1].text(rate * 100 + 0.6, i, f"{rate:.1%} · n={n_label}", va="center", fontsize=9)
    axes[1].set_xlim(0, leads.deal_rate_all.max() * 100 * 1.35)
    axes[1].set_xlabel("Lead deal rate (%)")
    axes[1].set_ylabel("Lead source")
    axes[1].grid(axis="y", visible=False)
    save_fig(fig, TASK_FIGURES / "01_app_users_vs_telesales_conversion.png", None, wspace=0.3)


# %% [markdown]
# ## Call frequency and conversation depth
#
# Both series show conversion rate: bars use average connected-call duration; the line shows later conversion by the first connected-call duration among customers not converted on that call.

# %%
def chart_call_dynamics(q2):
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    attempts = q2["by_attempts"]
    colors = [PAL["orange"] if band == "21-50" else PAL["grey"] for band in attempts.att_band.astype(str)]
    axes[0].bar(attempts.att_band.astype(str), attempts.conv * 100, color=colors, width=0.65)
    for i, row in attempts.iterrows():
        axes[0].text(i, row.conv * 100 + attempts.conv.max() * 1.5, f"{row.conv:.1%}", ha="center", fontsize=9)
    axes[0].set_xlabel("Outbound attempts / customer")
    axes[0].set_ylabel("Customers converted (%)")
    axes[0].grid(axis="x", visible=False)

    durations = q2["by_duration"]
    axes[1].bar(durations.dur_band.astype(str), durations.conv * 100, color=PAL["grey"], width=0.65,
                label="All connected calls, grouped by average duration")
    robust = q2["by_first_duration_rob"]
    position = {band: i for i, band in enumerate(durations.dur_band.astype(str))}
    axes[1].plot(robust.b.astype(str).map(position), robust.conv * 100, "o--", color=PAL["orange"], lw=2,
                 label="Later conversion, grouped by first connected-call duration")
    for i, row in durations.iterrows():
        axes[1].text(i, row.conv * 100 + 1.2, f"{row.conv:.0%}", ha="center", fontsize=9)
    axes[1].set_xlabel("Connected-call duration band")
    axes[1].set_ylabel("Customers converted (%)")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "02_call_duration_vs_lead_conversion.png", None, wspace=0.25)


# %% [markdown]
# ## Lead-source performance
#
# Show deal rate, signing of the offered product within 90 days, and later activity. Sources with small samples remain visible with their counts.

# %%
LEAD_SOURCE_FIGURES = {
    "deal_rate_all": "03a_lead_source_deal_rate.png",
    "signed_within_90d": "03b_lead_source_signed_within_90d.png",
    "active_180d_after": "03c_lead_source_active_after_180d.png",
}


def chart_lead_source(q3: pd.DataFrame):
    # one figure per measure; all three share the same source order so they can sit side by side on a slide
    data = q3[q3.n_leads_deal >= MIN_SOURCE_LEADS].copy()
    too_new = data.n_leads.fillna(0) < MIN_SOURCE_LEADS   # e.g. RTDM, relaunched Mar-2025: no lead has 90/180 days of follow-up yet
    data.loc[too_new, ["signed_within_90d", "active_180d_after"]] = np.nan
    data = data.sort_values("signed_within_90d", na_position="first").reset_index(drop=True)
    measures = [
        ("deal_rate_all", "Deal rate (%)",
         [PAL["orange"] if source in ("SOB", "RTDM", "TLS") else PAL["grey"] for source in data.LEAD_SOURCE]),
        # sources that actually sign the offered product
        ("signed_within_90d", "Offered product signed within 90 days (%)",
         [PAL["orange"] if source in ("SOB", "TLS") else PAL["grey"] for source in data.LEAD_SOURCE]),
        # sources that sign less but keep customers active
        ("active_180d_after", "Active after 180 days (%)",
         [PAL["orange"] if source in ("GMA", "LANDING PAGE") else PAL["grey"] for source in data.LEAD_SOURCE]),
    ]
    for column, label, colors in measures:
        fig, ax = plt.subplots(figsize=(6.67, 7.5))
        ax.barh(data.LEAD_SOURCE, data[column].fillna(0) * 100, color=colors, height=0.6)
        for i, (value, n) in enumerate(zip(data[column], data.n_leads_deal)):
            text = "n/a (too recent)" if pd.isna(value) else f"{value:.1%}"
            if column == "deal_rate_all":
                text += f" · n={n / 1e3:,.0f}K" if n >= 1000 else f" · n={n:,}"
            ax.text((0 if pd.isna(value) else value * 100) + 0.8, i, text, va="center", fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel("Lead source")
        ax.set_xlim(0, max(data[column].max() * 100 * 1.25, 5))
        ax.grid(axis="y", visible=False)
        save_fig(fig, TASK_FIGURES / LEAD_SOURCE_FIGURES[column], None, left=0.25)


# %% [markdown]
# ## Engagement patterns and later delinquency
#
# Compare a cross-sectional app-use measure with activity trends observed before contract signing. Neither panel is a credit rule.

# %%
def chart_early_warning(q4):
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    cross = q4["cross_sectional"]
    axes[0].bar(cross.band.astype(str), cross.bad_rate * 100, color=PAL["grey"], width=0.6)
    for i, row in cross.iterrows():
        axes[0].text(i, row.bad_rate * 100 + 0.05, f"{row.bad_rate:.2%}\nn={row.n / 1e3:,.0f}K", ha="center", fontsize=9)
    axes[0].set_ylabel("Delinquency (%)")
    axes[0].set_xlabel("App sessions / week")
    axes[0].grid(axis="x", visible=False)

    temporal = q4["temporal"]
    axes[1].bar(temporal.trend, temporal.bad_rate * 100, color=PAL["grey"], width=0.6)
    for i, row in temporal.iterrows():
        axes[1].text(i, row.bad_rate * 100 + 0.05, f"{row.bad_rate:.2%}\nn={row.n / 1e3:,.0f}K", ha="center", fontsize=9)
    axes[1].set_ylabel("Delinquency on later loan (%)")
    axes[1].set_xlabel("Pre-signing activity trend")
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].grid(axis="x", visible=False)
    save_fig(fig, TASK_FIGURES / "04_engagement_dropoff_vs_delinquency.png", None)


# %% [markdown]
# ## What reduces engagement
#
# Compare later call pickup for customers with and without each pre-snapshot factor.

# %%
def chart_engagement_reducers(q5: pd.DataFrame):
    data = q5.sort_values("diff_pp", ascending=False).reset_index(drop=True)   # most negative ends up on top
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    reducers = ("Ignored 5+ calls", "Opted out", "Collection calls")
    colors = [PAL["orange"] if factor in reducers else PAL["grey"] for factor in data.factor]
    ax.barh(data.factor, data.diff_pp, color=colors, height=0.6)
    ax.axvline(0, color=PAL["navy"], lw=1)
    for i, row in data.iterrows():
        ax.text(row.diff_pp + (0.4 if row.diff_pp >= 0 else -0.4), i, f"{row.diff_pp:+.1f} pp", va="center",
                ha="left" if row.diff_pp >= 0 else "right", fontsize=10)
    ax.set_xlim(data.diff_pp.min() * 1.25, max(data.diff_pp.max(), 0) + 4)
    ax.set_xlabel("Change in later call-answer rate (pp)")
    ax.grid(axis="y", visible=False)
    save_fig(fig, TASK_FIGURES / "06_engagement_reducers.png", None, left=0.18)


# %% [markdown]
# ## Engagement score and validation
#
# The score is built from five components. The adjacent panels check whether earlier engagement ranks later applications and later sales-call responses.

# %%
def chart_score(pop, val, fwd):
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    bins = np.arange(0, 101, 2)
    peak = 0
    for tier, (lower, upper) in {"Low": (0, 50), "Medium": (50, 80), "High": (80, 101)}.items():
        scores = pop.Engagement_Score[(pop.Engagement_Score >= lower) & (pop.Engagement_Score < upper)]
        counts, _, _ = axes[0].hist(scores, bins=bins, color=TIER_COLORS[tier])
        peak = max(peak, counts.max())
    axes[0].set_yscale("log")
    axes[0].set_ylim(top=peak * 8)   # headroom above the tallest bar for the tier labels
    shares = pop.eng_short.value_counts(normalize=True)
    counts = pop.eng_short.value_counts()
    for tier, position in [("Low", 33), ("Medium", 65), ("High", 90)]:
        header_color = PAL["grey"] if tier == "Medium" else TIER_COLORS[tier]
        axes[0].text(position, 0.97, f"{tier} · {shares.get(tier, 0):.1%}\nn={counts.get(tier, 0) / 1e3:,.0f}K",
                     ha="center", fontsize=10, fontweight="bold", color=header_color, va="top",
                     transform=axes[0].get_xaxis_transform())
    axes[0].set_xlabel("Engagement score (0–100)")
    axes[0].set_ylabel("Customers · log scale")

    def tier_bars(ax, table, column, ylabel):
        table = table.reindex(["Low", "Medium", "High"])   # same left-to-right order as the histogram
        ax.bar(table.index, table[column] * 100, color=[TIER_COLORS[tier] for tier in table.index], width=0.6)
        top = table[column].max() * 100
        for i, (value, n) in enumerate(zip(table[column], table.n)):
            ax.text(i, value * 100 + top * 0.02, f"{value:.1%}\nn={n / 1e3:,.0f}K", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, top * 1.15)
        ax.set_xlabel("Engagement tier")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="x", visible=False)

    tier_bars(axes[1], pd.DataFrame(val["tier_table"]).set_index("tier"), "apply_rate", "Later application rate (%)")
    tier_bars(axes[2], pd.DataFrame(fwd["tier_table"]).set_index("tier"), "pos", "Positive call response (%)")
    save_fig(fig, TASK_FIGURES / "05_engagement_score_distribution_validation.png", None, wspace=0.3)


# %% [markdown]
# # Challenge 2 — engage with intent
#
# **Story 2 of 3:** after Task 1 identifies where repayment support may be needed, engagement helps choose whether and how to reach a customer. It measures observed interaction—not customer value or intent.

# %%
def engagement_report(metrics: dict) -> str:
    validation = metrics["validation"]
    forward = metrics["forward_call_validation"]
    q1, q2, q4 = metrics["q1"], metrics["q2"], metrics["q4"]
    tier_table = pd.DataFrame(validation["tier_table"])
    tier_table["apply_rate"] = tier_table["apply_rate"].map(lambda value: f"{value:.1%}")
    tier_table["sign_rate"] = tier_table["sign_rate"].map(lambda value: f"{value:.1%}")
    weights = pd.Series(metrics["weights"], name="weight").to_frame()
    component_auc = pd.Series(validation["component_auc_apply"], name="apply_auc").sort_values(ascending=False).to_frame()
    leads = pd.DataFrame(metrics["q3"])
    eligible_leads = leads[leads.n_leads >= 1000].sort_values("signed_within_90d", ascending=False)
    best_source = eligible_leads.iloc[0] if not eligible_leads.empty else None
    source_line = (
        f"{best_source.LEAD_SOURCE} has the highest observed offered-product signing rate among sources with at least 1,000 leads "
        f"({best_source.signed_within_90d:.1%}, n={best_source.n_leads:,})."
        if best_source is not None else "No lead source met the minimum sample threshold of 1,000 leads."
    )
    tier_by_name = {row["tier"]: row for row in validation["tier_table"]}
    application_lift = tier_by_name["High"]["apply_rate"] / max(tier_by_name["Low"]["apply_rate"], 1e-9)
    forward_tiers = {row["tier"]: row for row in forward["tier_table"]}
    response_lift = forward_tiers["High"]["pos"] / max(forward_tiers["Low"]["pos"], 1e-9)
    rising = next(row for row in q4["temporal"] if row["trend"] == "Rising")
    declining = next(row for row in q4["temporal"] if row["trend"] == "Declining")
    app_active = q1["conv_active_0.5plus"]
    app_none = q1["conv_non_users"]
    return "\n".join(
        [
            "# Task 2 — Engage with intent",
            "",
            "## Business question",
            "",
            "Once Task 1 identifies repayment evidence, which customers show enough recent interaction to guide a relevant, consent-aware contact—and does the score relate to later outcomes?",
            "",
            "## Analysis",
            "",
            "The score combines digital adoption, recency, frequency, responsiveness, and product breadth. Features are frozen before the application outcome window; call data use their separate observation period. Responsiveness is omitted and its weight redistributed when the customer has no contact evidence.",
            "",
            weights.to_markdown(),
            "",
            "## Findings and business insight",
            "",
            f"- **The score separates later behavior:** application AUC is {validation['auc_apply']:.3f}; High-tier customers apply at {tier_by_name['High']['apply_rate']:.1%} versus {tier_by_name['Low']['apply_rate']:.1%} in Low ({application_lift:.1f}×). On calls after the snapshot, positive-response rates are {forward_tiers['High']['pos']:.1%} versus {forward_tiers['Low']['pos']:.1%} ({response_lift:.1f}×, AUC {forward['auc_positive']:.3f}).\n"
            f"- Application and signing rates by tier:\n\n{tier_table[['tier', 'n', 'apply_rate', 'sign_rate']].to_markdown(index=False)}\n"
            f"- **Frequency and breadth add more ranking signal than recency alone:** component AUCs are:\n\n{component_auc.to_markdown()}\n"
            f"- **Conversation depth is more informative than dial count in this sample:** duration AUC {q2['auc_avg_duration']:.3f}, first-call robustness AUC {q2['auc_first_call_duration_excl_first_call_conversions']:.3f}, versus attempts AUC {q2['auc_attempts']:.3f}. Duration partly occurs during the conversion process, so it is not a causal lever.\n"
            f"- **App activity is a modest signal:** customers with ≥0.5 sessions/week convert on sales calls at {app_active:.2%}, versus {app_none:.2%} for non-users ({q1['n_called']:,} called customers). This supports a channel test, not a blanket app campaign.\n"
            f"- **Customer-initiated leads may deserve a distinct path:** {source_line}\n"
            f"- **Rising pre-loan event activity is a monitoring cue:** delinquency is {rising['bad_rate']:.2%} in Rising versus {declining['bad_rate']:.2%} in Declining activity ({q4['n_temporal']:,} contracts). The association does not establish cause.",
            "",
            "## Business implication and handoff",
            "",
            "Prioritize opt-in, low-cost re-engagement for low-activity customers when Task 1 shows adequate health; route established repayment concerns to support rather than sales. Task 3 combines both scores into the final customer-level action.",
            "",
            "## Limits",
            "",
            f"Application validation uses a later window beginning {validation['asof']}. Call responsiveness is validated on {forward['n']:,} customers with later calls. All reported differences are associations; contact exposure and customer selection may explain part of them.",
            "",
        ]
    )


def display_saved_figure(path: Path) -> None:
    if "get_ipython" in globals():
        from IPython.display import Image, display

        display(Image(filename=str(path)))


# %% [markdown]
# ## Build the engagement score
#
# Score features frozen at the current snapshot, then validate against later applications and later call responses before analyzing contact patterns.

# %%
if __name__ == "__main__":
    ensure_dirs(DATA_CLEAN, TASK_FIGURES, TASK_SCORES, TASK_REPORT.parent)
    set_style()
    con = get_con()
    features = build_features(con, C.SNAPSHOT_DATE, "2025-01-01", "2026-09-01", use_app=True)
    components = score(features)
    population = pd.concat([features, components], axis=1)
    population["Engagement_Tier"] = assign_tier(population.Engagement_Score, C.ENG_TIERS)
    population["eng_short"] = population.Engagement_Tier.map(C.ENG_SHORT)
    validation = validate(con)
    forward = forward_call_validation(con)
    keep = [
        "SKP_CLIENT", "Engagement_Score", "Engagement_Tier", "eng_short", *ENG_WEIGHTS.keys(), "e4_available",
        "app_sessions_wk", "days_since_last_activity", "active_months_12m", "events_12m", "digital_events_12m",
        "product_families_24m", "applications_12m", "leads_24m", "lead_deals_24m", "latest_max_offer",
        "max_offer_24m", "latest_lead_product", "out_attempts", "out_connected", "inbound_calls",
        "sales_outcomes", "positive_reactions", "conversions", "opted_out", "collection_calls",
    ]
    population[keep].to_parquet(TASK_SCORES / "eng_scores.parquet", index=False)
    chart_score(population, validation, forward)
    display_saved_figure(TASK_FIGURES / "05_engagement_score_distribution_validation.png")

# %% [markdown]
# **Interpretation.** Compare each tier with later application and call-response rates. The score is useful only if it adds a different signal from Task 1's repayment-health score.

# %% [markdown]
# ## 1. Compare app activity with sales-call response
#
# Limit the comparison to customers with a sales outcome; collection calls are excluded from the conversion cohort.

# %%
if __name__ == "__main__":
    q1 = q1_active_vs_passive(con, population)
    q3 = q3_lead_source(con, population)
    chart_active_passive(q1, q3)
    display_saved_figure(TASK_FIGURES / "01_app_users_vs_telesales_conversion.png")

# %% [markdown]
# **Interpretation.** App activity and call conversion co-occur in this sample. The comparison does not show that increasing app use would cause a customer to convert.

# %% [markdown]
# ## 2. Separate call frequency from conversation depth
#
# Compare number of attempts with connected-call duration, including a robustness view based on first-call duration.

# %%
if __name__ == "__main__":
    q2 = q2_call_dynamics(con)
    chart_call_dynamics(q2)
    display_saved_figure(TASK_FIGURES / "02_call_duration_vs_lead_conversion.png")

# %% [markdown]
# **Interpretation.** Duration can partly reflect the conversion process itself. Treat it as an operational pattern to investigate, not as a treatment effect or customer trait.

# %% [markdown]
# ## 3. Compare lead-source follow-through
#
# Follow the offered product into contract signing and check whether activity continues after the lead.

# %%
if __name__ == "__main__":
    chart_lead_source(q3)
    for name in LEAD_SOURCE_FIGURES.values():
        display_saved_figure(TASK_FIGURES / name)

# %% [markdown]
# **Interpretation.** A high deal rate alone does not establish durable engagement. Compare signing and later activity with sample size before recommending channel investment.

# %% [markdown]
# ## 4. Check pre-loan activity patterns
#
# Compare app sessions with a pre-signing event trajectory. The time-ordering is more informative than a same-period association, but remains non-causal.

# %%
if __name__ == "__main__":
    q4 = q4_early_warning(con, population)
    chart_early_warning(q4)
    display_saved_figure(TASK_FIGURES / "04_engagement_dropoff_vs_delinquency.png")

# %% [markdown]
# **Interpretation.** Treat activity changes as a monitoring prompt. Task 3 combines this engagement signal with financial-health evidence before suggesting a next action.

# %% [markdown]
# ## 5. Find what reduces engagement
#
# Factors are measured before the snapshot; the outcome is the share of later outbound calls the customer answers.

# %%
if __name__ == "__main__":
    q5 = q5_engagement_reducers(con, population)
    chart_engagement_reducers(q5)
    display_saved_figure(TASK_FIGURES / "06_engagement_reducers.png")

# %% [markdown]
# **Interpretation.** Non-response persists: customers who ignored repeated dials keep ignoring later calls. Digital inactivity and lead expiry do not lower phone reachability.

# %% [markdown]
# ## Save evidence and the stage report
#
# Store aggregate question tables in JSON, customer score components in Parquet, and the business interpretation in one Markdown report.

# %%
if __name__ == "__main__":
    tiers = population.eng_short.value_counts().reindex(["High", "Medium", "Low"])
    metrics = {
        "weights": ENG_WEIGHTS,
        "tier_counts": tiers.to_dict(),
        "tier_share": (tiers / tiers.sum()).to_dict(),
        "score_mean": population.Engagement_Score.mean(),
        "score_median": population.Engagement_Score.median(),
        "component_means": population[list(ENG_WEIGHTS)].mean().to_dict(),
        "e4_coverage": population.e4_available.mean(),
        "never_active": float(population.days_since_last_activity.isna().mean()),
        "active_last_90d": float((population.days_since_last_activity <= 90).mean()),
        "app_users": float((population.app_sessions_wk > 0).mean()),
        "validation": validation,
        "forward_call_validation": forward,
        "q1": {key: value for key, value in q1.items() if key not in ("by_app_band", "lead_source_deal")}
              | {"by_app_band": q1["by_app_band"].astype({"app_band": str}).to_dict("records"),
                 "lead_source_deal": q1["lead_source_deal"].to_dict("records")},
        "q2": {key: value.astype(str).to_dict("records") if isinstance(value, pd.DataFrame) else value for key, value in q2.items()},
        "q3": q3.to_dict("records"),
        "q4": {key: value.astype({column: str for column in value.columns if value[column].dtype.name == "category"}).to_dict("records")
               if isinstance(value, pd.DataFrame) else value for key, value in q4.items()},
        "q5": q5.to_dict("records"),
    }
    save_json(metrics, DATA_CLEAN / "metrics_phase3.json")
    TASK_REPORT.parent.mkdir(parents=True, exist_ok=True)
    TASK_REPORT.write_text(engagement_report(metrics), encoding="utf-8")
    con.close()
