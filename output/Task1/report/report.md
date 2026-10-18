# Task 1 — Protect the base

## Business question

Which customers show observable repayment pressure, and which signals provide a defensible early-review score?

## Analysis

The score uses mature contracts and features frozen before the **2024-10-01–2025-02-01** outcome window. It is a 0–100 weighted scorecard, not a black-box approval model. Pillar weights:

|                          |   weight |
|:-------------------------|---------:|
| Repayment History        |     0.4  |
| Debt Burden (DTI)        |     0.15 |
| Down-payment Buffer      |     0.15 |
| Credit Standing & Buffer |     0.3  |

The held-out cohort contains 333,297 records and is separate from score fitting.

## Findings and business insight

- **The tiers order later repayment outcomes:** holdout delinquency is 4.88% for Low versus 1.35% for High, a 3.6× difference (Gini 0.302). Use this as a review-priority signal, not as an individual guarantee.
- **Low score does not always mean repayment stress:** 182,420 of 225,337 Low-tier customers (81.0%) have no signed-loan history. Only the remainder (42,917) have an established loan history that can be checked for repayment evidence. Keep these paths separate.
- **Debt burden is a useful monitoring boundary:** above the observed 22.5% DTI breakpoint, the mature-cohort delinquency rate is 3.08%, versus 2.16% below it (294,786 and 2,205,078 contracts). This is a portfolio pattern, not a universal affordability cutoff.
- **Product differences deserve a review, not a causal story:** TW is highest at 4.97%; CD is lowest at 2.13%. The strongest single-factor holdout Gini is `clean_track` (0.240).

## Business implication and handoff

Use verified repayment signals to prioritize support and affordability review. Route thin-file customers to a separate evidence-building or education path; do not label them delinquent. Task 2 adds engagement evidence to decide how to reach each group, and Task 3 combines both scores into a service action.

## Limits

The outcome is early delinquency on a historical cohort. Product, income, insurance, and down-payment comparisons are observational. Demographics are used only in a post-hoc fairness check, not in scoring or segmentation.
