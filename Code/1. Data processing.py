# %% [markdown]
# # 1. Data processing
#
# Read the source tables into temporary DuckDB staging tables, validate them, and write only the cleaned tables and quality summaries used downstream.

# %%
"""
PHASE 1 - Data ingestion, schema validation, key integrity and data hygiene.

Outputs
  Data_clean/<table>_clean.parquet           cleaned tables used in analysis
  Data_clean/metrics_phase1.json             quality profiles and cleaning decisions
  output/EDA/data_quality_report.md           readable quality summary
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import sys
from pathlib import Path


# %% [markdown]
# ## Locate project files and configure storage
#
# Paths are resolved from the script or notebook location; derived tables live under `Data_clean/`.

# %%
def find_project_root():
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    for directory in (start, *start.parents):
        if ((directory / "Data").is_dir() or (directory / "Data_raw").is_dir()) and (directory / "Code").is_dir():
            return directory
    raise RuntimeError(f"Could not find project root from {start}; expected Data/ or Data_raw/ and Code/")


ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RAW = ROOT / ("Data" if (ROOT / "Data").is_dir() else "Data_raw")
DATA_CLEAN = ROOT / "Data_clean"
EDA_TABLES = ROOT / "output" / "EDA" / "tables"
EDA_FIGURES = ROOT / "output" / "EDA" / "figures"
DQ_REPORT = ROOT / "output" / "EDA" / "data_quality_report.md"


def pq(name: str) -> str:
    if name.startswith("raw_"):
        return name
    return f"'{(DATA_CLEAN / f'{name}.parquet').as_posix()}'"


from Code.config import config as C
from Code.config.utils import ensure_dirs, get_con, save_json


# --------------------------------------------------------------------------------------
# %% [markdown]
# ## Find source tables and read the data dictionary
#
# The five named CSVs are the required inputs. If `StructureTable.xlsx` is present, its column names and type families are checked against the actual files.

# %%
# 1. Discover input files
# --------------------------------------------------------------------------------------
def discover_files() -> dict:
    found = {}
    for key, stem in C.DATASETS.items():
        cands = [p for p in RAW.rglob("*") if p.is_file() and not p.name.startswith("~$")
                 and p.stem.upper() == stem and p.suffix.lower() in (".csv", ".xlsx", ".xls")]
        if cands:
            found[key] = sorted(cands, key=lambda p: p.suffix.lower() != ".csv")[0]   # prefer csv
    missing = [k for k in C.DATASETS if k not in found]
    if missing:
        zips = list(RAW.rglob("*.zip"))
        if zips:
            out = RAW / "_extracted"
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall(out)
            return discover_files() if any(out.rglob("*.csv")) else found
        raise FileNotFoundError(f"Datasets not found under {RAW}: {missing}")
    schema = [p for p in RAW.rglob("StructureTable.xlsx") if not p.name.startswith("~$")]
    found["schema"] = schema[0] if schema else None
    return found


# --------------------------------------------------------------------------------------
# 2. Schema (StructureTable.xlsx)
# --------------------------------------------------------------------------------------
def read_schema(path) -> pd.DataFrame:
    rows = []
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    for key, sheet in C.SCHEMA_SHEETS.items():
        df = sheets.get(sheet)
        if df is None:
            continue
        hdr_r, hdr_c = np.where(df.astype(str).values == "Column Name")
        r, c = int(hdr_r[0]), int(hdr_c[0])
        desc = str(df.iloc[0].dropna().iloc[0]) if df.iloc[0].notna().any() else ""
        for i in range(r + 1, len(df)):
            col = df.iat[i, c]
            if pd.isna(col):
                continue
            rows.append({"dataset": key, "table_description": desc, "column": str(col).strip(),
                         "expected_dtype": str(df.iat[i, c + 1]).strip(),
                         "description": str(df.iat[i, c + 2]).strip().replace("\n", " | ")})
    return pd.DataFrame(rows)


# %% [markdown]
# ## Load source tables into temporary DuckDB tables
#
# Raw CSV files are read into the current connection only. The stage persists cleaned Parquet tables, not duplicate raw Parquet copies.

# %%
# 3. Temporary source tables
def load_raw(con, files: dict) -> None:
    for key, path in files.items():
        if key == "schema" or path is None:
            continue
        if path.suffix.lower() == ".csv":
            escaped_path = path.as_posix().replace("'", "''")
            source = f"read_csv_auto('{escaped_path}', sample_size=-1, header=true)"
        else:
            con.register("tmp_xlsx", pd.read_excel(path))
            source = "tmp_xlsx"
        con.execute(f"CREATE OR REPLACE TEMP TABLE raw_{key} AS SELECT * FROM {source}")
        if source == "tmp_xlsx":
            con.unregister("tmp_xlsx")


DUCK_TO_PANDAS = {"BIGINT": "int64", "INTEGER": "int64", "DOUBLE": "float64", "FLOAT": "float64",
                  "VARCHAR": "object", "TIMESTAMP": "object", "DATE": "object", "BOOLEAN": "bool"}


# %% [markdown]
# ## Audit schema, keys, missingness, and numeric extremes
#
# These checks describe the supplied data. They flag unusual values; they do not remove records or impute missing values.

# %%
def validate_schema(con, schema: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in C.DATASETS:
        actual = con.execute(f"DESCRIBE SELECT * FROM {pq('raw_' + key)}").df()
        act = dict(zip(actual["column_name"], actual["column_type"]))
        exp = schema[schema["dataset"] == key]
        for _, r in exp.iterrows():
            a = act.get(r["column"])
            mapped = DUCK_TO_PANDAS.get(a, a)
            status = "MISSING_IN_DATA" if a is None else (
                "OK" if mapped == r["expected_dtype"] or (r["expected_dtype"] == "object" and a == "TIMESTAMP")
                or (r["expected_dtype"] in ("int64", "float64") and mapped in ("int64", "float64")) else "TYPE_DIFF")
            rows.append({"dataset": key, "column": r["column"], "expected_dtype": r["expected_dtype"],
                         "observed_type": a, "status": status})
        for col in act:
            if col not in set(exp["column"]):
                rows.append({"dataset": key, "column": col, "expected_dtype": None, "observed_type": act[col],
                             "status": "NOT_IN_SCHEMA"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 4. Profiling
# --------------------------------------------------------------------------------------
def profile_missing(con) -> pd.DataFrame:
    rows = []
    for key in C.DATASETS:
        t = pq("raw_" + key)
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        desc = con.execute(f"DESCRIBE SELECT * FROM {t}").df()
        exprs = []
        for col, typ in zip(desc["column_name"], desc["column_type"]):
            e = f"sum(({col} IS NULL)::INT) AS \"{col}__null\""
            if typ == "VARCHAR":
                e += f", sum((upper(trim({col})) IN ('XNA','','NONE','#NAME?'))::INT) AS \"{col}__placeholder\""
            elif typ in ("BIGINT", "DOUBLE", "INTEGER"):
                e += f", sum(({col} = 0)::INT) AS \"{col}__zero\""
            exprs.append(e)
        res = con.execute(f"SELECT {', '.join(exprs)} FROM {t}").df().iloc[0]
        for col, typ in zip(desc["column_name"], desc["column_type"]):
            rows.append({
                "dataset": key, "column": col, "type": typ, "rows": n,
                "null_pct": round(100 * res[f"{col}__null"] / n, 2),
                "placeholder_pct (XNA/none/blank)": round(100 * res.get(f"{col}__placeholder", np.nan) / n, 2)
                if f"{col}__placeholder" in res else None,
                "zero_pct": round(100 * res.get(f"{col}__zero", np.nan) / n, 2) if f"{col}__zero" in res else None,
            })
    return pd.DataFrame(rows)


def key_integrity(con) -> pd.DataFrame:
    n_cust = con.execute(f"SELECT count(DISTINCT SKP_CLIENT) FROM {pq('raw_customer')}").fetchone()[0]
    rows = []
    for key in C.DATASETS:
        t = pq("raw_" + key)
        r = con.execute(f"""
            SELECT count(*) n_rows, count(DISTINCT SKP_CLIENT) n_clients,
                   count(DISTINCT SKP_CLIENT) FILTER (WHERE SKP_CLIENT NOT IN (SELECT SKP_CLIENT FROM {pq('raw_customer')})) orphans,
                   count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM {t})) exact_dup_rows
            FROM {t}""").fetchone()
        rows.append({"dataset": key, "rows": r[0], "distinct_SKP_CLIENT": r[1], "orphan_clients_not_in_CUSTOMER": r[2],
                     "pct_of_customers_covered": round(100 * r[1] / n_cust, 2), "exact_duplicate_rows": r[3]})
    kd = pd.DataFrame(rows)
    # secondary keys
    extra = con.execute(f"""SELECT count(*) - count(DISTINCT SKP_CREDIT_CASE) FROM {pq('raw_contract')}""").fetchone()[0]
    callid = con.execute(f"""SELECT count(*) - count(DISTINCT CALL_ID) FROM {pq('raw_call')}""").fetchone()[0]
    kd["secondary_key_note"] = ""
    kd.loc[kd.dataset == "contract", "secondary_key_note"] = f"{extra:,} rows share an SKP_CREDIT_CASE with another row"
    kd.loc[kd.dataset == "call", "secondary_key_note"] = f"{callid:,} rows re-use a CALL_ID (different timestamps -> kept)"
    kd.loc[kd.dataset == "customer", "secondary_key_note"] = "SKP_CLIENT is unique (1 row per customer)"
    return kd


def outlier_report(con) -> pd.DataFrame:
    checks = [("contract", "AMT_INCOME_MAIN", "AMT_INCOME_MAIN > 0"), ("contract", "AMT_INCOME_HOUSEHOLD", "AMT_INCOME_HOUSEHOLD > 0"),
              ("contract", "AMT_CREDIT", "AMT_CREDIT > 0"), ("contract", "ANNUITY", "ANNUITY IS NOT NULL"),
              ("contract", "INIT_PAY", "INIT_PAY IS NOT NULL"), ("contract", "PAYMENT_NUM", "PAYMENT_NUM IS NOT NULL"),
              ("call", "DURATIONS", "DURATIONS > 0"), ("leads", "MAX_OFFER", "MAX_OFFER > 0"),
              ("customer", "AVG_SESSION_PER_WEEK_2025", "AVG_SESSION_PER_WEEK_2025 > 0"),
              ("customer", "CNT_CHILDREN", "CNT_CHILDREN IS NOT NULL")]
    rows = []
    for key, col, cond in checks:
        src = f"(SELECT * FROM {pq('raw_' + key)} QUALIFY row_number() OVER (PARTITION BY SKP_CREDIT_CASE ORDER BY (NAME_EDUCATION_TYPE<>'XNA') DESC, AMT_INCOME_MAIN DESC)=1)" \
            if key == "contract" else pq("raw_" + key)
        r = con.execute(f"""
            WITH d AS (SELECT {col}::DOUBLE v FROM {src} WHERE {cond})
            SELECT count(*), min(v), quantile_cont(v,0.01), quantile_cont(v,0.25), median(v), quantile_cont(v,0.75),
                   quantile_cont(v,0.99), quantile_cont(v,0.999), max(v), avg(v) FROM d""").fetchone()
        n, mn, p1, q1, med, q3, p99, p999, mx, mean = r
        iqr = q3 - q1
        hi_fence = q3 + 3 * iqr
        n_hi = con.execute(f"SELECT count(*) FROM {src} WHERE {cond} AND {col} > {hi_fence}").fetchone()[0]
        n_cap = con.execute(f"SELECT count(*) FROM {src} WHERE {cond} AND {col} > {p999}").fetchone()[0]
        rows.append({"dataset": key, "column": col, "filter": cond, "n": n, "min": mn, "p01": p1, "q1": q1, "median": med,
                     "q3": q3, "p99": p99, "p99_9": p999, "max": mx, "mean": mean,
                     "extreme_outliers_(>Q3+3IQR)": n_hi, "pct_extreme": round(100 * n_hi / max(n, 1), 2),
                     "capped_at_p99.9": n_cap})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. Cleaning -> analytical tables
# --------------------------------------------------------------------------------------
def disposition_case_sql(col="RESPONSE_RESULT_2") -> str:
    parts = []
    for cat, codes in C.DISPOSITION_RULES.items():
        lst = ", ".join("'" + c.replace("'", "''") + "'" for c in codes)
        parts.append(f"WHEN {col} IN ({lst}) THEN '{cat}'")
    for prefix, cat in C.DISPOSITION_PREFIX_RULES:
        parts.append(f"WHEN starts_with({col}, '{prefix}') THEN '{cat}'")
    return "CASE " + " ".join(parts) + f" ELSE '{C.DEFAULT_DISPOSITION}' END"


# %% [markdown]
# ## Apply documented cleaning rules
#
# Deduplicate contract cases, preserve useful status and timing fields, flag sentinel dates, and derive analysis features. Each treatment is summarized in the quality report.

# %%
def build_clean_tables(con) -> dict:
    stats_ = {}
    sig = ", ".join(f"'{s}'" for s in C.SIGNED_STATUSES)
    dflt = ", ".join(f"'{s}'" for s in C.DEFAULT_STATUSES)

    # ---- income cap (p99.9 of positive personal income on de-duplicated contracts)
    inc_cap = con.execute(f"""
        SELECT quantile_cont(AMT_INCOME_MAIN, 0.999) FROM (SELECT * FROM {pq('raw_contract')}
        QUALIFY row_number() OVER (PARTITION BY SKP_CREDIT_CASE ORDER BY (NAME_EDUCATION_TYPE<>'XNA') DESC, AMT_INCOME_MAIN DESC)=1)
        WHERE AMT_INCOME_MAIN > 0""").fetchone()[0]
    stats_["income_cap_p999"] = inc_cap

    con.execute(f"""
        COPY (
          WITH dedup AS (
            SELECT * FROM {pq('raw_contract')}
            QUALIFY row_number() OVER (PARTITION BY SKP_CREDIT_CASE
                    ORDER BY (NAME_EDUCATION_TYPE <> 'XNA') DESC, AMT_INCOME_MAIN DESC, AMT_INCOME_HOUSEHOLD DESC) = 1),
          base AS (
          SELECT
            SKP_CREDIT_CASE, SKP_CLIENT, coalesce(PRODUCT, 'UNKNOWN') AS PRODUCT, NAME_CREDIT_STATUS,
            NAME_INCOME_TYPE, CODE_PROFESSION,
            coalesce(SIGN_CONTRACT_TIME, APPROVE_CONTRACT_TIME, APPLY_CONTRACT_TIME) AS cdate,
            APPLY_CONTRACT_TIME AS apply_time,
            NAME_CREDIT_STATUS IN ({sig}) AS is_signed,
            NAME_CREDIT_STATUS IN ('Active','Signed') AS is_open,
            NAME_CREDIT_STATUS = 'Rejected' AS is_rejected,
            CASE WHEN AMT_INCOME_MAIN > 0 THEN least(AMT_INCOME_MAIN, {inc_cap}) END AS income_main,
            AMT_INCOME_MAIN > {inc_cap} AS income_capped,
            CASE WHEN AMT_INCOME_HOUSEHOLD > 0 THEN least(AMT_INCOME_HOUSEHOLD::DOUBLE, {inc_cap} * 3) END AS income_household,
            CASE WHEN AMT_CREDIT > 0 THEN AMT_CREDIT END AS amt_credit,
            PAYMENT_NUM AS tenor, INIT_PAY AS init_pay, ANNUITY AS annuity, AMT_BILLING AS amt_billing,
            coalesce(ANNUITY, AMT_BILLING) AS monthly_obligation,
            FLAG_INS AS flag_ins,
            FIRST_DUE, SECOND_DUE, THIRD_DUE, FOURTH_DUE,
            (FIRST_DUE + SECOND_DUE + THIRD_DUE + FOURTH_DUE) AS n_due,
            greatest(FIRST_DUE, SECOND_DUE, THIRD_DUE, FOURTH_DUE) AS any_due,
            (NAME_CREDIT_STATUS IN ({dflt}))::INT AS is_default,
            (greatest(FIRST_DUE, SECOND_DUE, THIRD_DUE, FOURTH_DUE) = 1 OR NAME_CREDIT_STATUS IN ({dflt}))::INT AS bad
          FROM dedup)
          SELECT *,
            CASE WHEN income_main > 0 AND annuity > 0 THEN annuity / income_main END AS dti_main,
            CASE WHEN income_household > 0 AND annuity > 0 THEN annuity / income_household END AS dti_household,
            CASE WHEN PRODUCT IN ('CD','TW') AND amt_credit > 0 AND init_pay IS NOT NULL THEN init_pay / amt_credit END AS dp_ratio,
            CASE WHEN tenor IS NOT NULL THEN cdate + to_months(tenor::INT) END AS end_date,
            CASE WHEN annuity > 0 AND tenor > 0 AND amt_credit > 0 THEN (annuity * tenor - amt_credit) / amt_credit END AS gross_yield
          FROM base
        ) TO {pq('contract_clean')} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    stats_["contract_rows_raw"] = con.execute(f"SELECT count(*) FROM {pq('raw_contract')}").fetchone()[0]
    stats_["contract_rows_clean"] = con.execute(f"SELECT count(*) FROM {pq('contract_clean')}").fetchone()[0]

    con.execute(f"""
        COPY (
          SELECT SKP_CLIENT,
                 NULLIF(NAME_GENDER, 'XNA') AS NAME_GENDER,
                 NULLIF(NAME_EDUCATION_TYPE, 'XNA') AS NAME_EDUCATION_TYPE,
                 CASE WHEN DATE_BIRTH BETWEEN TIMESTAMP '1930-01-01' AND TIMESTAMP '2007-06-05' THEN DATE_BIRTH END AS DATE_BIRTH,
                 CASE WHEN CNT_CHILDREN <= 15 THEN CNT_CHILDREN END AS CNT_CHILDREN,
                 CASE WHEN FLAG_CAR_OWNER IN ('1','Y','YES') THEN 1 WHEN FLAG_CAR_OWNER IN ('0','N','NO') THEN 0 END AS FLAG_CAR_OWNER,
                 NULLIF(NAME_SALARY_FREQUENCY, 'XNA') AS NAME_SALARY_FREQUENCY,
                 CNT_PERSON_DEPENDENT,
                 CASE WHEN ADDRESS IS NULL OR ADDRESS LIKE '#%' THEN NULL ELSE trim(split_part(ADDRESS, ',', -1)) END AS PROVINCE,
                 coalesce(AVG_SESSION_PER_WEEK_2025, 0) AS app_sessions_wk
          FROM {pq('raw_customer')}
        ) TO {pq('customer_clean')} (FORMAT PARQUET, COMPRESSION ZSTD)""")

    con.execute(f"""
        COPY (
          SELECT DISTINCT
                 SKP_CLIENT, DTIME_CREATED AS lead_time,
                 CASE WHEN year(DATE_ACCEPTED) >= 2999 THEN NULL ELSE DATE_ACCEPTED END AS accepted_time,
                 year(DATE_ACCEPTED) >= 2999 AS sob_direct_support,
                 coalesce(DESC_FIN_REASON, 'Unknown') AS fin_reason,
                 DESC_FIN_REASON = 'Deal' AS is_deal,
                 DESC_FIN_REASON = 'Not Interest' AS is_not_interest,
                 DESC_FIN_REASON IN ('Expire Lifetime','Expired Acceptance','Expired Offer','Expired Campaign') AS is_expired,
                 CODE_PRODUCT_TYPE AS lead_product, FLAG_ASSIGNED, MAX_OFFER AS max_offer,
                 LEAD_SOURCE
          FROM {pq('raw_leads')}
        ) TO {pq('leads_clean')} (FORMAT PARQUET, COMPRESSION ZSTD)""")

    con.execute(f"""
        COPY (
          SELECT SKP_CLIENT, DTIME_EVENT AS event_time, ACTION, PRODUCT_CODE,
                 CASE WHEN ACTION LIKE '%,%' THEN 'MULTI'
                      WHEN ACTION LIKE 'APP%' THEN 'APP' WHEN ACTION LIKE 'WEB%' THEN 'WEB'
                      WHEN ACTION LIKE 'SA %' THEN 'SA' WHEN ACTION LIKE 'TLS%' THEN 'TLS'
                      WHEN ACTION LIKE 'Partner%' THEN 'PARTNER' WHEN ACTION LIKE 'SOB%' THEN 'SOB'
                      ELSE 'BRANCH' END AS channel,
                 (ACTION LIKE 'APP%' OR ACTION LIKE 'WEB%' OR ACTION LIKE 'SOB%') AS is_digital_self_serve,
                 CASE WHEN PRODUCT_CODE IN ('CLX','ACL','ACLX','SAI') THEN 'CL'
                      WHEN PRODUCT_CODE LIKE 'CC%' THEN 'CC' WHEN PRODUCT_CODE = 'BNPL' THEN 'BNPL'
                      ELSE PRODUCT_CODE END AS product_family
          FROM {pq('raw_past_behavior')}
        ) TO {pq('behavior_clean')} (FORMAT PARQUET, COMPRESSION ZSTD)""")

    case = disposition_case_sql()
    con.execute(f"""
        COPY (
          WITH base AS (
          SELECT SKP_CLIENT, CALL_ID, CALL_TYPE, TIME_OF_CALL_START AS call_start, TIME_OF_CALL_END AS call_end,
                 DURATIONS AS duration_s,
                 abs(DURATIONS - epoch(TIME_OF_CALL_END - TIME_OF_CALL_START)) > 60 AS duration_mismatch,
                 PRODUCT_TYPE, CALL_SOURCE, OFF_BY, OFFER_TYPE, RESPONSE_RESULT_2, CODE_DISPOSITION, CALL_RESULT_SYSTEM,
                 {case} AS disposition_group,
                 CALL_TYPE = 'Outbound' AS is_outbound,
                 CALL_TYPE = 'Inbound' AS is_inbound
          FROM {pq('raw_call')})
          SELECT *,
            -- connected = a human conversation took place (system ANSWER or any agent disposition other than no-contact)
            (CALL_RESULT_SYSTEM = 'ANSWER' OR disposition_group NOT IN ('NO_CONTACT')) AND duration_s > 0 AS is_connected,
            disposition_group IN ('SALES_CONVERSION','SALES_INTEREST','SALES_REFUSAL','OPT_OUT') AS is_sales_outcome,
            disposition_group = 'SALES_CONVERSION' AS is_conversion,
            disposition_group IN ('SALES_CONVERSION','SALES_INTEREST') AS is_positive_reaction
          FROM base
        ) TO {pq('call_clean')} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    disp = con.execute(f"""SELECT disposition_group, count(*) n, count(DISTINCT RESPONSE_RESULT_2) n_codes,
                                  string_agg(DISTINCT RESPONSE_RESULT_2, ' | ') FILTER (WHERE true) codes
                           FROM {pq('call_clean')} GROUP BY 1 ORDER BY 2 DESC""").df()
    disp["codes"] = disp["codes"].str.slice(0, 600)
    stats_["call_disposition_taxonomy"] = disp.to_dict("records")
    return stats_


# %% [markdown]
# ## Document data-quality findings
#
# Keep the audit readable in Markdown and machine-readable in one JSON file rather than exporting many small CSV tables.

# %%
# 6. Data-quality narrative
# --------------------------------------------------------------------------------------
def dq_findings(con, stats_) -> dict:
    f = {}
    f["contract_dup_rows_removed"] = stats_["contract_rows_raw"] - stats_["contract_rows_clean"]
    f["contract_rows_raw"] = stats_["contract_rows_raw"]
    f["contract_rows_clean"] = stats_["contract_rows_clean"]
    f["income_cap_p999"] = stats_["income_cap_p999"]
    r = con.execute(f"SELECT count(*), sum((FLAG_CAR_OWNER='X')::INT) FROM {pq('raw_customer')}").fetchone()
    f["car_owner_unusable_pct"] = 100 * r[1] / r[0]
    f["customers"] = r[0]
    f["household_income_nonzero_pct"] = con.execute(
        f"SELECT 100*avg(coalesce(income_household>0, false)::INT) FROM {pq('contract_clean')}").fetchone()[0]
    f["income_main_available_pct"] = con.execute(
        f"SELECT 100*avg(coalesce(income_main>0, false)::INT) FROM {pq('contract_clean')}").fetchone()[0]
    f["app_users_pct"] = con.execute(f"SELECT 100*avg((app_sessions_wk>0)::INT) FROM {pq('customer_clean')}").fetchone()[0]
    f["leads_exact_dups"] = con.execute(f"SELECT count(*) FROM {pq('raw_leads')}").fetchone()[0] - \
        con.execute(f"SELECT count(*) FROM {pq('leads_clean')}").fetchone()[0]
    f["leads_sentinel_3000_pct"] = con.execute(
        f"SELECT 100*avg(sob_direct_support::INT) FROM {pq('leads_clean')}").fetchone()[0]
    f["call_window"] = con.execute(f"SELECT min(call_start)::VARCHAR, max(call_start)::VARCHAR FROM {pq('call_clean')}").fetchone()
    f["contract_window"] = con.execute(f"SELECT min(cdate)::VARCHAR, max(cdate)::VARCHAR FROM {pq('contract_clean')}").fetchone()
    f["behavior_window"] = con.execute(f"SELECT min(event_time)::VARCHAR, max(event_time)::VARCHAR FROM {pq('behavior_clean')}").fetchone()
    f["leads_window"] = con.execute(f"SELECT min(lead_time)::VARCHAR, max(lead_time)::VARCHAR FROM {pq('leads_clean')}").fetchone()
    f["call_duration_mismatch_pct"] = con.execute(f"SELECT 100*avg(duration_mismatch::INT) FROM {pq('call_clean')}").fetchone()[0]
    f["call_unconnected_pct"] = con.execute(f"SELECT 100*avg((NOT is_connected)::INT) FROM {pq('call_clean')}").fetchone()[0]
    f["paid_off_due_rate"] = con.execute(
        f"SELECT avg(any_due) FROM {pq('contract_clean')} WHERE NAME_CREDIT_STATUS='Paid off'").fetchone()[0]
    f["written_off_due_rate"] = con.execute(
        f"SELECT avg(any_due) FROM {pq('contract_clean')} WHERE NAME_CREDIT_STATUS='Written off'").fetchone()[0]
    f["finished_due_rate"] = con.execute(
        f"SELECT avg(any_due) FROM {pq('contract_clean')} WHERE NAME_CREDIT_STATUS='Finished'").fetchone()[0]
    f["tet_peak_share"] = con.execute(
        f"SELECT avg((cdate >= '2024-12-01' AND cdate < '2025-02-01')::INT) FROM {pq('contract_clean')} WHERE is_signed").fetchone()[0]
    f["revolving_no_annuity_pct"] = con.execute(
        f"SELECT 100*avg((annuity IS NULL)::INT) FROM {pq('contract_clean')} WHERE PRODUCT IN ('BNPL','CC')").fetchone()[0]
    return f


def write_dq_markdown(f, kd, schema_val, outl):
    diffs = schema_val[schema_val.status != "OK"]
    lines = [
        "# Data Quality Report (Phase 1)",
        "",
        "Auto-generated by `Code/1. Data processing.py`. All numbers come from the raw files in `Data/`.",
        "",
        "## 1. Key integrity (SKP_CLIENT is the entity key)",
        "",
        kd.to_markdown(index=False),
        "",
        "## 2. Schema validation vs StructureTable.xlsx",
        "",
        f"{(schema_val.status == 'OK').sum()} of {len(schema_val)} documented columns match name & type family. Deviations:",
        "",
        diffs.to_markdown(index=False) if len(diffs) else "_none_",
        "",
        "## 3. Material findings and the treatment applied",
        "",
        "| # | Finding | Evidence | Treatment |",
        "|---|---|---|---|",
        f"| 1 | CONTRACT has duplicated credit cases | {f['contract_rows_raw']:,} rows but only {f['contract_rows_clean']:,} unique SKP_CREDIT_CASE ({f['contract_dup_rows_removed']:,} extra rows). Duplicates differ only in the applicant-profile columns (one copy has `XNA` / 0 income, the other is populated). | Keep 1 row per case, preferring the populated profile (non-XNA education, highest income). |",
        f"| 2 | `FLAG_CAR_OWNER` is unusable | {f['car_owner_unusable_pct']:.1f}% of customers carry the literal value `X` (documented as 1/0). | Excluded from scoring (cannot be a safety-net signal). |",
        f"| 3 | Household income is sparsely populated | Only {f['household_income_nonzero_pct']:.1f}% of contracts have `AMT_INCOME_HOUSEHOLD` > 0 vs {f['income_main_available_pct']:.1f}% for personal income. | Personal income is the DTI denominator; household income tested separately (Challenge 1 Q2). |",
        f"| 4 | Income outliers | Personal income up to 9 billion VND/month. | Capped at the 99.9th percentile ({f['income_cap_p999']:,.0f} VND); 0 treated as missing. |",
        f"| 5 | Sentinel dates in LEADS | {f['leads_sentinel_3000_pct']:.1f}% of `DATE_ACCEPTED` = 3000-01-01 (SOB leads supported directly by SA). | Converted to NULL + flag `sob_direct_support`. {f['leads_exact_dups']:,} exact duplicate lead rows removed. |",
        f"| 6 | Different time windows | Contracts {f['contract_window'][0][:10]} -> {f['contract_window'][1][:10]}; Behaviour {f['behavior_window'][0][:10]} -> {f['behavior_window'][1][:10]}; Leads {f['leads_window'][0][:10]} -> {f['leads_window'][1][:10]}; **Calls {f['call_window'][0][:10]} -> {f['call_window'][1][:10]}**. | Snapshot date = {C.SNAPSHOT_DATE} for contract/lead/behaviour recency. Call features use their own full window (intensity & response rates, no recency). |",
        f"| 7 | `Paid off` is not a 'good' status | {100 * f['paid_off_due_rate']:.0f}% of Paid-off contracts had an overdue flag (Written off: {100 * f['written_off_due_rate']:.0f}%, Finished: {100 * f['finished_due_rate']:.1f}%). It behaves like a post-collection settlement. | Treated as signed-with-history; its due flags count as delinquency. Written off / Sold = default. |",
        f"| 8 | BNPL / CC have no annuity or tenor | {f['revolving_no_annuity_pct']:.0f}% of revolving contracts lack `ANNUITY`. | Monthly obligation = `ANNUITY`, else first billed amount `AMT_BILLING`. |",
        f"| 9 | Call dialler noise | {f['call_unconnected_pct']:.0f}% of call rows are dial attempts with no conversation; {f['call_duration_mismatch_pct']:.1f}% have `DURATIONS` vs end-start gaps > 60s; CALL_ID is re-used across dates. | Each row kept as one attempt; `is_connected` flag; 270 disposition codes mapped to 9 business groups (taxonomy is in `Data_clean/metrics_phase1.json`). |"
        f"| 10 | Strong seasonality | {100 * f['tet_peak_share']:.0f}% of all signed contracts were signed Dec-2024/Jan-2025 (pre-Tet peak). | Out-of-time validation windows include this peak; product-level comparisons use full history. |",
        "| 11 | Demographics | Gender, birth date, children, address are present. | **Not used in any score** (fairness constraint). Used only for post-hoc bias checks. |",
        "| 12 | Right-censoring of due flags | FOURTH_DUE is ~0 for contracts signed after Jan-2025 (not enough instalments elapsed). | Delinquency analyses use contracts signed before " + C.OBS_COMPLETE_BEFORE + ". |",
        "",
        "## 4. Numerical outlier scan",
        "",
        outl.round(2).to_markdown(index=False),
        "",
        "See `Data_clean/metrics_phase1.json` for column-level missingness, schema validation, and key-integrity details.",
    ]
    DQ_REPORT.write_text("\n".join(lines), encoding="utf-8")


# %% [markdown]
# ## Build the clean tables and quality record
#
# Raw CSVs are staged only for this connection; this step writes cleaned Parquet tables, not duplicate raw copies.

# %%
def run() -> dict:
    ensure_dirs(DATA_CLEAN, EDA_TABLES, EDA_FIGURES)
    con = get_con()
    try:
        files = discover_files()
        schema = read_schema(files["schema"]) if files.get("schema") else pd.DataFrame()
        load_raw(con, files)
        schema_val = validate_schema(con, schema)
        miss = profile_missing(con)
        keys = key_integrity(con)
        outliers = outlier_report(con)
        stats = build_clean_tables(con)
        findings = dq_findings(con, stats)
        write_dq_markdown(findings, keys, schema_val, outliers)
        save_json(
            {
                "findings": findings,
                "key_integrity": keys.to_dict("records"),
                "schema_validation": schema_val.to_dict("records"),
                "missingness": miss.to_dict("records"),
                "outliers": outliers.to_dict("records"),
                "call_disposition_taxonomy": stats.get("call_disposition_taxonomy", []),
            },
            DATA_CLEAN / "metrics_phase1.json",
        )
    finally:
        con.close()
    return findings


# %% [markdown]
# ## Run Stage 1
#
# Execute this cell before EDA. Source files remain untouched; raw staging tables exist only for this DuckDB connection.

# %%
if __name__ == "__main__":
    stage1_findings = run()
