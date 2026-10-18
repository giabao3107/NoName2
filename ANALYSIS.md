# BI10 Round 2 — Full Project Analysis

**Business Intelligence Season 10, Round 02 — Synthesize**
Customer Intelligence 360°: one financial-health score, one engagement score, one segmentation, and one next-best-action per customer.

This document is the end-to-end analysis narrative: data audit → exploratory analysis → Challenge 1 (financial health) → Challenge 2 (engagement) → Challenge 3 (segmentation and action) → the action framework and its operating boundaries. Every number below is reproduced from the generated artifacts (`Data_clean/metrics_phase*.json`, `output/*/report/report.md`, `output/Task3/scores/final_customer_intelligence_output.csv`). Where a result is an association rather than a causal effect, it is labelled as such.

---

## 1. Executive summary

The company can act on **1,000,528 customers** through two transparent 0–100 scores and a 3 × 3 routing matrix.

- **Financial health (Task 1).** A five-pillar, points-based scorecard ranks a period it never saw: out-of-time Gini **0.322**, KS **0.238** (Tet-2025 report half, 270,277 customers). It ties a re-estimated logistic regression (**0.323**) and beats gradient boosting on the same inputs (**0.320**), so complexity buys nothing. Tiers carry economic meaning (High ≥ 80, Moderate 50–79, Low < 50).
- **Engagement (Task 2).** A five-component score predicts later applications (**AUC 0.649**) and later sales-call positive response (**AUC 0.689**, High 32.4% vs Low 7.0%).
- **Segmentation and action (Task 3).** A rule-based 3 × 3 matrix over both scores routes every customer to one of ten service actions. An unsupervised K-means challenger separates better geometrically (**silhouette 0.413 vs 0.355** at equal segment counts) but has no post-action outcome label to validate actions, so the **explainable matrix stays operational** and K-means is retained as exploratory profiling only.
- **The largest lever is re-engagement:** **582,524 customers (58.2%)** have High/Moderate health and Low engagement.
- **Repayment support is a narrow, reachable queue:** **60,597** Low-health customers have established loan history, of whom **15,544** are reachable through Medium/High engagement.
- **Thin-file is not stress:** 71.4% of the Low-health tier has no signed loan history and is routed to education, not collections.

Project narrative: **protect the base → engage with intent → match the action to the need.**

---

## 2. Business context and brief

The company holds customer profiles, historical financial behaviour, leads, contracts, and call records but lacks a unified view of how healthy, how engaged, and how actionable each customer is. The brief requires three deliverables that must be **explainable**, must not confuse **correlation with causation**, must avoid **demographic risk scoring without justification**, and must produce **actionable** recommendations. Machine learning is explicitly optional.

Three challenges:

1. **Financial Health Score (0–100)** — factors, segmentation, key drivers, and justification.
2. **Engagement Score** — recency, frequency, interaction, lead activity, product interaction; validated against business outcomes.
3. **Segmentation & Next Best Action** — combine both scores, find opportunities, recommend an action, and explain why.

---

## 3. Data audit (Challenge context)

### 3.1 Sources

| Source        |  Raw rows | Distinct customers | Coverage of customer file | Observation window      |
| ------------- | --------: | -----------------: | ------------------------: | ----------------------- |
| CONTRACT      | 6,234,093 |          1,000,528 |                      100% | 2009-04-21 → 2025-06-04 |
| CUSTOMER      | 1,000,528 |          1,000,528 |                      100% | static profile          |
| LEADS         | 2,143,891 |            645,344 |                     64.5% | 2019-10-09 → 2025-06-04 |
| PAST_BEHAVIOR | 4,412,610 |            473,086 |                     47.3% | 2022-12-01 → 2025-06-03 |
| CALL          | 4,000,000 |            360,661 |                     36.1% | 2025-01-01 → 2026-08-31 |

All 59 documented columns match `StructureTable.xlsx` on name and type family.

### 3.2 Key integrity

- `SKP_CLIENT` is the entity key; **zero orphan clients** across all sources.
- CONTRACT holds **4,280,377 unique credit cases** but 6,234,093 rows: 1,953,716 extra rows share a case key. Duplicates differ only in applicant-profile columns (one copy `XNA`/zero income, the other populated), so one row per case is kept, preferring the populated profile.
- Keys are clean elsewhere: 39,249 exact duplicate lead rows and 310 exact duplicate contract rows removed; 12,527 reused `CALL_ID` values differ by timestamp and are kept.

### 3.3 Material findings and treatment

| #   | Finding                      | Evidence                                 | Treatment                                                         |
| --- | ---------------------------- | ---------------------------------------- | ----------------------------------------------------------------- |
| 1   | Duplicated contract cases    | 1,953,716 extra rows                     | Dedupe per case, keep populated profile                           |
| 2   | `FLAG_CAR_OWNER` unusable    | 100% literal `X`                         | Excluded from scoring                                             |
| 3   | Household income sparse      | 0.9% populated vs 99.7% personal         | Personal income is the DTI denominator                            |
| 4   | Income outliers              | Up to 9 billion VND/month                | Capped at p99.9 = 90,000,000 VND                                  |
| 5   | Sentinel dates in LEADS      | 27.5% `DATE_ACCEPTED` = 3000-01-01 (SOB) | NULL + `sob_direct_support` flag                                  |
| 6   | Different time windows       | CALL sits 2025–2026; others 2009–2025    | Snapshot 2025-06-05; calls scored on their own window             |
| 7   | `Paid off` is not "good"     | 82% had an overdue flag                  | Treated as signed-with-history; dues count as delinquency         |
| 8   | BNPL/CC have no annuity      | 100% lack `ANNUITY`                      | Monthly obligation = `ANNUITY` else `AMT_BILLING`                 |
| 9   | Dialler noise                | 87% of calls never connected             | `is_connected` flag; 270 codes → 9 groups                         |
| 10  | Strong seasonality           | 26% of signings in Dec-2024/Jan-2025     | Out-of-time windows include the Tet peak                          |
| 11  | Demographics present         | Gender, birth date, children, address    | Never score inputs; post-hoc fairness checks only                 |
| 12  | Right-censoring of due flags | FOURTH_DUE ≈ 0 after Jan-2025            | Delinquency labels stop before 2025-02-01 (refined to 2025-01-20) |

**Censoring is the single most important audit result.** The 4th/3rd-instalment late ratio is 1.21 in normal weeks and collapses for loans signed from **2025-01-20** because those loans cannot yet have missed instalment 4. Every label and contract comparison therefore stops at that date.

---

## 4. Exploratory data analysis

- **Product outcomes (mature signed contracts):** TW 4.97%, CW 4.77%, CC 2.45%, BNPL 2.21%, CL 2.19%, CD 2.13%. Two-wheeler credit is the riskiest product; consumer durables the safest.
- **Digital channels (events / customers):** APP 1,630,349 / 232,674; SA 867,787 / 201,283; WEB 746,021 / 169,734; TLS 726,233 / 148,336; PARTNER 396,829 / 87,345; SOB 44,623 / 39,516; BRANCH 761 / 760.
- **App sessions per week:** median 0, p99 1.73, only 9,875 customers above p99 — app use is a minority behaviour.
- **Coverage caveat:** coverage is the share of the customer file appearing in a source, not a data-quality measure.

Key structured outputs: `output/EDA/eda_report.md`, `output/EDA/data_quality_report.md`, and `Data_clean/metrics_phase1.json` / `metrics_eda.json`.

---

## 5. Challenge 1 — Customer Financial Health Score

### 5.1 Method (fully explainable, no black box)

- **Features** are frozen as of each window start (late flags only after 120 days, write-offs after 365 days, no future contract status). **Label**: late on instalments 1–4, or written off, on loans signed inside the window.
- **Design**: three training windows (2023-10 → 2024-02, 2024-02 → 2024-06, 2024-06 → 2024-10) = 354,105 customer-windows, stress **1.27%**; out-of-time test = the Tet wave 2024-10-01 → 2025-01-20 (541,128 customers), split into a scale-anchoring half and an untouched **report half of 270,277**.
- **Points**: monotone buckets, points linear in the log-odds of each bucket's bad rate (100 = safest, 0 = riskiest); pillar = mean of its factors; composite = published weights; FHS = composite on a **risk-anchored scale** (50 = 1.5× average stress, 80 = 0.5×; **18 points halve the odds**).
- **Transparent policy caps** for write-offs and current arrears.

| Pillar                | Weight | Factors                                                                        |
| --------------------- | -----: | ------------------------------------------------------------------------------ |
| Repayment History     |    35% | # past late loans, # loans completed with no overdue, months since a late loan |
| Credit Standing       |    26% | share of applications rejected, months since first signed loan                 |
| Debt Burden (DTI)     |    13% | instalments ÷ income                                                           |
| Down-payment Buffer   |    13% | down-payment ratio (CD/TW loans)                                               |
| Pre-approved Capacity |    13% | largest pre-approved offer (LEADS)                                             |

### 5.2 Results

- **Out-of-time ranking:** lowest decile stressed at **6.80%** vs **0.77%** in the highest (**8.9×**); Gini **0.322**, KS **0.238**, AUC **0.661**.
- **Tiers:** High **268,670 (26.9%)**, Moderate **519,814 (52.0%)**, Low **212,044 (21.2%)**; out-of-time stress 1.13% / 3.10% / 6.25%.
- **Bands (A–E):** 1.13% → 2.29% → 3.91% → 5.96% → 8.50%, a 7.5× spread; band E holds 5,266 customers today.
- **Policy caps:** 0.2% of the test sample hit a knock-out rule and were stressed at 16.3%.

### 5.3 Validation and robustness

| Candidate (same customers)         | Gini (all) | Gini (history) |
| ---------------------------------- | ---------: | -------------: |
| v1 design (6 factors)              |      0.301 |          0.261 |
| Repayment history only             |      0.273 |          0.208 |
| Equal pillar weights               |      0.310 |          0.292 |
| Logistic regression, same buckets  |      0.323 |          0.295 |
| Gradient boosting, same raw inputs |      0.320 |          0.320 |
| **Final scorecard**                |  **0.322** |      **0.307** |

- Dropping any one pillar moves Gini only to 0.301–0.310; 200 random ±25% weight perturbations hold Gini in 0.313–0.323 with rank correlation ≥ 0.987.
- Stability: PSI train→test **0.064**, test→today (customers with history) **0.104**.

### 5.4 Business insights

- **Debt burden has a tipping point:** stress is flat until DTI ≈ **22.5%**, then climbs 4.5× faster (+0.19% vs +0.88% per 10 DTI points); above 40% it is 4.65%.
- **First loans are riskier than repeat loans** (4.51% vs 2.19%); first-time two-wheeler reaches 7.7%.
- **Down payment is evidence, not a linear rule:** TW with 10–30% down is 2.3× as stressed as 50%+; yet CD with <10% down runs at only 1.46% (77% are repeat customers).
- **Insurance marks risk, it does not remove it:** insured contracts are more stressed in every product and every product × DTI cell, so insurance is deliberately **not** a score input.
- **"No history" hides two opposites:** customers previously **rejected** stress at 6.79% while **cancelled-only** customers stress at 3.16%, close to customers with history (2.40%; z = 6.0, p ≈ 2e-09).

### 5.5 Fairness

Mean FHS is close (women 65.1, men 66.5), but within the Low tier men are stressed at 7.88% vs 4.85% for women. Within-group Gini is 0.299 (women) / 0.347 (men) and 0.186–0.317 across age bands. **Decision: gender and age stay out of the score**; the gap is published as a monitoring item and Low-tier actions must remain supportive (no automatic declines).

---

## 6. Challenge 2 — Customer Engagement Score

### 6.1 Method (five transparent components, no model)

| Component           | Weight | Definition                                                                                                                |
| ------------------- | -----: | ------------------------------------------------------------------------------------------------------------------------- |
| E1 Digital adoption |    20% | best of app sessions/week (2 = full, log scale) or self-serve applications in 12m (3 = full)                              |
| E2 Recency          |    20% | 100 × 0.5^(days since last self-initiated activity ÷ 120)                                                                 |
| E3 Frequency        |    25% | active months in the last 12 months (6+ = full)                                                                           |
| E4 Responsiveness   |    20% | reach rate, positive sales-call reaction, lead reply rate (+25 per inbound call, max +50); missing → weight redistributed |
| E5 Product breadth  |    15% | distinct product families touched in 24 months (3+ = full)                                                                |

Features freeze at `ENG_ASOF` **2025-02-01**; the outcome window runs to `ENG_TARGET_END` **2025-06-05** (~4 months of new applications/signings after the Tet peak).

### 6.2 Results and validation

- **Later applications:** AUC **0.649**; apply rate High **36.3%** (n=50,841) vs Low **12.4%** (n=752,918), 2.9×; sign rate 23.4% vs 8.5%.
- **Later sales calls:** positive-response rate High **32.4%** vs Low **7.0%** (4.6×), AUC **0.689** (n=115,044).
- **Component ranking power (apply AUC):** E3 frequency 0.661, E5 breadth 0.627, E1 digital 0.575, E2 recency 0.503, E4 responsiveness 0.500.
- **Conversation depth beats dial count:** duration AUC 0.922, first-call robustness 0.720, attempts 0.591 — but duration partly _is_ the conversion process, so it is not a causal lever.
- **App activity is modest:** 3.39% vs 2.58% conversion for ≥0.5 sessions/week vs non-users (n=131,368).
- **Engagement reducers:** customers who ignored 5+ calls answer later at **−19.8 pp** (n=32,349); opted-out −3.9 pp; collection calls −1.9 pp.
- **Lead sources:** SOB has the highest offered-product signing rate (67.3%, n=35,535) among sources with ≥1,000 leads.
- Score distribution: mean 38.1, median 30.6; E4 present for 68.7% of customers.

### 6.3 Business insight

Prioritise **opt-in, low-cost re-engagement** for low-activity customers when Task 1 shows adequate health; route established repayment concerns to **support rather than sales**. Rising pre-loan activity is a monitoring cue (delinquency 2.54% Rising vs 2.12% Declining) but does not establish cause.

---

## 7. Challenge 3 — Segmentation and Next Best Action

### 7.1 Method

Both score files are joined **one-to-one** on `SKP_CLIENT` after explicit checks for unique keys and identical populations. Each customer is placed in the 3 × 3 health × engagement matrix and routed by rule to an action, a channel, an opportunity type, and a row-level rationale.

Thin-file customers in the Low-health row are separated from established borrowers with observed repayment stress, so missing history is never read as a missed payment.

### 7.2 Findings

- **Re-engagement is the largest service opportunity:** **582,524 customers (58.2%)** have High/Moderate health and Low engagement.
- **Low-health is mostly thin-file:** **151,447 of 212,044 (71.4%)** have no signed loan history.
- **Observed repayment stress is a distinct queue:** **60,597** Low-health customers have established history; **15,544** of these are reachable (Medium/High engagement).
- **Protect engaged relationships:** **44,909 (4.5%)** combine High/Moderate health with High engagement.

**Matrix distribution (top cells):** Medium×Low 392,016 · High×Low 190,508 · Low×Low 180,599 · Medium×Medium 99,464 · High×Medium 61,587 · Medium×High 28,334 · Low×Medium 25,998 · High×High 16,575 · Low×High 5,447.

### 7.3 Matrix vs machine learning

An unsupervised K-means challenger was fit on the two standardized score axes (train-only scaling, seed 42) and selected by training silhouette, then evaluated on a held-out sample.

| Method                 | Segments | Holdout silhouette |
| ---------------------- | -------: | -----------------: |
| 3 × 3 score matrix     |        9 |              0.355 |
| K-means selected k = 3 |        3 |              0.456 |
| K-means matched k = 9  |        9 |          **0.413** |

- K-means separates better geometrically, even at the matched 9-segment count.
- But for k = 3, clusters match the dominant matrix cell for only **53.2%** of the held-out sample, and one cluster is 37.3% thin-file.
- **Decision:** silhouette measures geometric separation, not repayment, applications, or action success. Task 3 has **no observed post-action outcome label** for supervised selection, so the rule matrix stays operational and K-means remains exploratory profiling.

---

## 8. Action framework

### 8.1 The ten routes

| Health × Engagement              | Next best action                                                      | Channel              | Opportunity              |
| -------------------------------- | --------------------------------------------------------------------- | -------------------- | ------------------------ |
| High × High                      | Needs-based product review after standard eligibility checks          | Relationship service | Retention                |
| High × Medium                    | Relevant product review with suitability and affordability checks     | Customer service     | Relationship growth      |
| High × Low                       | Opt-in digital check-in; do not infer dissatisfaction                 | Digital              | Re-engagement            |
| Medium × High                    | Tailored service review; check affordability before any credit offer  | Customer service     | Retention                |
| Medium × Medium                  | Concise account/product guidance; contact only on a clear need        | Digital              | Relationship growth      |
| Medium × Low                     | Low-frequency opt-in reminder focused on account utility              | Digital              | Re-engagement            |
| Low (established) × High         | Repayment-support review; suppress new-credit promotion               | Customer support     | Support                  |
| Low (established) × Medium       | Repayment-support options + service contact; suppress promotion       | Customer support     | Support                  |
| Low (established) × Low          | Self-service repayment resources; avoid sales outreach                | Self-service         | Support                  |
| Low (thin-file)                  | Neutral financial education; affordability checks precede credit talk | Digital              | Credit-history education |
| Low (thin-file, rejected-before) | Education + documented affordability check before any offer           | Digital              | Credit-history education |

### 8.2 Workload

| Action                                                           | Customers |
| ---------------------------------------------------------------- | --------: |
| Send a low-frequency, opt-in reminder focused on account utility |   392,016 |
| Opt-in digital check-in                                          |   190,508 |
| Neutral financial education (thin-file)                          |   151,447 |
| Concise account/product guidance                                 |    99,464 |
| Relevant product review with affordability checks                |    61,587 |
| Keep self-service repayment resources; avoid sales outreach      |    45,053 |
| Tailored service review; check affordability                     |    28,334 |
| Needs-based product review after eligibility checks              |    16,575 |
| Repayment-support options + service contact                      |    12,602 |
| Repayment-support review; suppress promotion                     |     2,942 |

### 8.3 Operating boundaries

These are **rule-based service-routing suggestions**, not estimated treatment effects or credit decisions. Before activation the company must review **consent, suitability, affordability, capacity, and policy**. Next-best actions are not proven causal uplift, and the score associations do not establish causes. Task 3's customer-level file carries the segment, action, channel, and rationale needed to audit every assignment.

---

## 9. Reproducibility and provenance

**Run order (from the project root):**

```bash
python "Code/1. Data processing.py"   # validate sources, write cleaned tables
python "Code/2. EDA.py"               # coverage, windows, missingness, behaviour
python "Code/3. Task 1.py"            # financial-health score + validation
python "Code/4. Task 2.py"            # engagement score + validation
python "Code/5. Task 3.py"            # matrix, actions, final CSV
python -m unittest discover -s tests -v
```

**Artifacts**

| Path                                                                      | Contents                                                 |
| ------------------------------------------------------------------------- | -------------------------------------------------------- |
| `Data_clean/*_clean.parquet`                                              | cleaned analytical tables                                |
| `Data_clean/metrics_phase1..4.json`, `metrics_eda.json`                   | quality and analysis evidence                            |
| `output/EDA/eda_report.md`, `data_quality_report.md`                      | EDA narrative                                            |
| `output/Task1/report/report.md`, `output/Task1/scores/fh_scores.parquet`  | financial health                                         |
| `output/Task2/report/report.md`, `output/Task2/scores/eng_scores.parquet` | engagement                                               |
| `output/Task3/report/report.md`                                           | segmentation and actions                                 |
| `output/Task3/scores/final_customer_intelligence_output.csv`              | one customer-level submission CSV (~1.23 GB, 80 columns) |

`Code/config/config.py` holds every business assumption (snapshot dates, tiers, score parameters, ROI constants, disposition taxonomy) so it can be audited and changed in one place. Random seed is fixed at 42.

**Checks:** 13 unit tests pass; Task 3's 7 QA checks all pass; the final CSV has one row per customer (1,000,528) with no missing score, action, or rationale field.

---

## 10. Limitations and honest claims

- **Observational, not causal.** All product, insurance, down-payment, offer, and channel differences are associations. Contact exposure and selection can explain part of any gap.
- **Cohort, not the whole book.** 99.98% of customers applied in Dec-2024/Jan-2025, when monthly applications ran ~12× a normal month; older vintages are a survivor view.
- **Label horizon.** The label sees instalments 1–4 only; later arrears are unobservable in the contract table, and CALL timestamps fail a timing test, so call dates are neither a score input nor a "future" validation.
- **Feedback loop.** The pre-approved offer reflects the lender's own past judgement and is held at a modest 13% weight pending monitoring.
- **Demographics.** Never score inputs; used only for the published fairness check.
- **Action validation is the open gap.** No post-action outcome label exists, so the recommended actions are prioritisation rules, not measured uplift. Any deployment should run a controlled test before scaling.

---

_Generated from the committed pipeline outputs. Companion documents: `README.md` (run guide), `AI_USAGE_REPORT.md` (AI disclosure), `BI10_ROUND02.md` (brief), and the per-task reports under `output/`._
