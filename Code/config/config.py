"""
Central configuration for the Customer Intelligence 360 pipeline (BI10 Round 2).

Every business assumption used anywhere in the pipeline lives here so judges can
audit (and change) it in one place. Nothing in this file is learned from data.
"""

# Logical dataset name -> file stem expected under Data/ (CSV or XLSX).
DATASETS = {
    "contract": "CONTRACT",
    "customer": "CUSTOMER",
    "leads": "LEADS",
    "past_behavior": "PAST_BEHAVIOR",
    "call": "CALL",
}
# StructureTable.xlsx sheet name for each dataset
SCHEMA_SHEETS = {
    "contract": "Contract",
    "customer": "Customer",
    "leads": "Leads",
    "past_behavior": "Past Behavior",
    "call": "Call",
}

# --------------------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------------------
DUCKDB_MEMORY_LIMIT = "2GB"     # keeps the pipeline runnable on an 8GB laptop
DUCKDB_THREADS = 4
RANDOM_SEED = 42

# --------------------------------------------------------------------------------------
# Business definitions
# --------------------------------------------------------------------------------------
# Snapshot of the contract / lead / behaviour extracts (max timestamp observed = 2025-06-04)
SNAPSHOT_DATE = "2025-06-05"
# Call extract covers a different window (2025-01-01 -> 2026-08-31); see data-quality report.

SIGNED_STATUSES = ("Active", "Finished", "Written off", "Paid off", "Signed", "Sold")
OPEN_STATUSES = ("Active", "Signed")
DEFAULT_STATUSES = ("Written off", "Sold")          # charged-off / debt sold
INSTALMENT_PRODUCTS = ("CD", "TW", "CL", "CW")       # fixed annuity + tenor
REVOLVING_PRODUCTS = ("BNPL", "CC")                  # billing-based
DOWNPAYMENT_PRODUCTS = ("CD", "TW")
PRODUCT_LABELS = {
    "CD": "CD - Consumer Durables",
    "TW": "TW - Two-Wheeler",
    "BNPL": "BNPL - Buy Now Pay Later",
    "CC": "CC - Credit Card",
    "CL": "CL - Cash Loan",
    "CW": "CW - Cash Walk-in (other)",
}

# Contracts need >= 4 instalments of observation for FIRST..FOURTH_DUE to be meaningful
OBS_COMPLETE_BEFORE = "2025-02-01"

# --------------------------------------------------------------------------------------
# Out-of-time validation design (Challenge 1 & 2)
# --------------------------------------------------------------------------------------
FH_ASOF = "2024-10-01"                 # features frozen at this date
FH_TARGET_START = "2024-10-01"         # outcome = delinquency on contracts signed in window
FH_TARGET_END = "2025-02-01"
DUE_FLAG_MATURITY_DAYS = 120           # due flags only "known" 4 months after signing
STATUS_MATURITY_DAYS = 365             # write-off status only "known" 12 months after signing

# NB: every customer in the extract applied in Dec-2024/Jan-2025 (Tet cohort), so the engagement validation
# freezes features right after that peak and asks: who comes back to apply again in the next 4 months?
ENG_ASOF = "2025-02-01"                # engagement features frozen at this date
ENG_TARGET_END = SNAPSHOT_DATE         # outcome = new application / signing between ENG_ASOF and snapshot

# --------------------------------------------------------------------------------------
# Score tiers (fixed by the brief)
# --------------------------------------------------------------------------------------
FH_TIERS = [(80, 101, "High Financial Health"), (50, 80, "Moderate Health"), (0, 50, "Financial Stress / Low Health")]
ENG_TIERS = [(80, 101, "High Engagement"), (50, 80, "Medium Engagement"), (0, 50, "Low / Dormant Engagement")]
FH_SHORT = {"High Financial Health": "High", "Moderate Health": "Medium", "Financial Stress / Low Health": "Low"}
ENG_SHORT = {"High Engagement": "High", "Medium Engagement": "Medium", "Low / Dormant Engagement": "Low"}

# --------------------------------------------------------------------------------------
# Engagement score parameters (explainable transforms)
# --------------------------------------------------------------------------------------
RECENCY_HALF_LIFE_DAYS = 120           # recency score halves every 120 days without activity
FREQ_FULL_MARKS_MONTHS = 6             # active in >= 6 of last 12 months = full frequency marks
BREADTH_FULL_MARKS = 3                 # >= 3 distinct product families in 24m = full marks
INBOUND_BONUS_MAX_CALLS = 2            # +25 points per call, capped at +50.

# --------------------------------------------------------------------------------------
# ROI assumptions (Challenge 3) - all in VND. Change here, everything re-computes.
# --------------------------------------------------------------------------------------
COST_PER_CALL_ATTEMPT = 6_000          # dialer + agent minutes + telco per outbound attempt
COST_PER_DIGITAL_TOUCH = 300           # push / SMS / in-app message
OFFER_UTILISATION = 0.60               # share of MAX_OFFER actually disbursed when converting
LGD = 0.60                             # loss given default on unsecured consumer credit
EARLY_DELINQ_TO_DEFAULT = 0.35         # fallback only - phase 4 measures the roll-to-write-off rate from data
FUNDING_COST_RATE = 0.08               # cost of funds over the average loan life

# Disposition taxonomy for RESPONSE_RESULT_2 (built from the full code list; see phase1)
DISPOSITION_RULES = {
    "SALES_CONVERSION": [
        "Sign successful", "Made_App_Successful", "Transfer for signing", "Sign Agree", "Client agrees",
        "Send_to_SA", "Signed already", "Appointment with SA", "Transfer to Sale", "Transfer for sales",
        "Push Sign", "Agree retention", "Agree retention with voucher", "Agree retention without voucher",
    ],
    "SALES_INTEREST": [
        "More consultation", "AskRecall", "Client considers", "Pending", "Ask to call back", "CallBack",
        "Ask fer information", "Support after application", "ResultAskRecallRpcRestrict",
    ],
    "SALES_REFUSAL": [
        "Refuse", "Sign refuse", "Not eligible", "No longer in need", "No need of using card", "Low credit limit",
        "Wrong product", "Termination", "Termination Non Eligible Offer", "Not eligible for termination",
        "Disagree termination", "No answer Retention", "Dropped_OTP",
    ],
    "OPT_OUT": ["Do Not Call", "Ngung_nhan_quang_cao_moi_vay_I", "Ngung_nhan_quang_cao_moi_vay_C",
                "ICre_Ngung_nhan_quang_cao_moi_vay"],
    "NO_CONTACT": ["none", "No_Answer", "Client busy", "Wrong Phone", "Wrong_phone_C", "No response",
                   "CallDisconnected", "Drop_call_I", "ICre_Drop", "DE_DROP"],
    "THIRD_PARTY": ["Relative", "Relatives", "ClientFamily", "OtherPerson"],
}
# Prefix-based rules evaluated after the exact lists above
DISPOSITION_PREFIX_RULES = [
    ("OWN", "COLLECTION"), ("Collection_C", "COLLECTION"), ("Chuyen_Collection", "COLLECTION"),
    ("ICre_Transfer to Collection", "COLLECTION"), ("Yeu_cau_mien_giam", "COLLECTION"),
    ("ICre_Yeu_cau_mien", "COLLECTION"), ("CallBacknonRPC", "COLLECTION"), ("ResultAskNonRPC", "COLLECTION"),
    ("ICre_", "SERVICE"), ("BNPL_", "SERVICE"), ("CCre_", "SERVICE"), ("HAPP_", "SERVICE"), ("OB_", "SERVICE"),
    ("DE_", "SERVICE"), ("CA_", "SERVICE"), ("Survey", "SERVICE"), ("Follow", "SERVICE"), ("MRC", "SERVICE"),
]
DEFAULT_DISPOSITION = "SERVICE"        # remaining *_I / *_C inbound service reasons
