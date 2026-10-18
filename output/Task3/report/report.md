# Task 3 — Match the action to the need

## Business question

How should the company act when financial-health evidence and engagement evidence point to different customer needs?

## Integrated analysis

Task 1 ranks observed repayment evidence; Task 2 estimates recent engagement and validates it against later applications and call responses. This stage joins both scores one-to-one, assigns the 3 × 3 matrix, distinguishes thin-file customers from established borrowers with stress signals, and attaches an action and row-level rationale. The output contains 1,000,528 customers and 9 populated matrix cells.

## Findings and business insight

- **Re-engagement is the largest service opportunity:** 567,051 customers (56.7%) have High/Moderate health and Low engagement. Favor low-cost, opt-in service prompts; do not infer dissatisfaction from inactivity.
- **The Low-health tier is mostly thin-file:** 182,420 of 225,337 Low-health customers (81.0%) have no signed loan history. Offer neutral education and assess affordability before any credit discussion.
- **Observed repayment stress is a distinct queue:** 42,917 Low-health customers have established loan history; 10,204 of these 42,917 show Medium/High engagement and are reachable for a repayment-support review without mixing in thin-file cases.
- **Protect engaged relationships:** 46,082 customers (4.6%) combine High/Moderate health with High engagement; prioritize retention and service quality rather than assuming they need a new credit offer.

## Matrix distribution

| Matrix_Segment                    | customers   |
|:----------------------------------|:------------|
| Medium health × Low engagement    | 343,428     |
| High health × Low engagement      | 223,623     |
| Low health × Low engagement       | 196,072     |
| Medium health × Medium engagement | 94,587      |
| High health × Medium engagement   | 67,471      |
| Medium health × High engagement   | 27,473      |
| Low health × Medium engagement    | 24,991      |
| High health × High engagement     | 18,609      |
| Low health × High engagement      | 4,274       |

## Proposed actions and workload

| Next_Best_Action                                                                                       | customers   |
|:-------------------------------------------------------------------------------------------------------|:------------|
| Send a low-frequency, opt-in reminder focused on account utility.                                      | 343,428     |
| Use an opt-in digital check-in; do not assume low activity means dissatisfaction.                      | 223,623     |
| Offer neutral financial education and explain that affordability checks precede any credit discussion. | 182,420     |
| Share concise account and product guidance; contact only when there is a clear customer need.          | 94,587      |
| Offer a relevant product review with suitability and affordability checks.                             | 67,471      |
| Keep self-service repayment resources available and avoid sales outreach.                              | 32,713      |
| Provide a tailored service review and check affordability before any credit offer.                     | 27,473      |
| Recognize the relationship and offer a needs-based product review after standard eligibility checks.   | 18,609      |
| Share repayment-support options and offer a service contact; suppress new-credit promotion.            | 8,589       |
| Offer a repayment-support review and suppress new-credit promotion.                                    | 1,615       |

## Model comparison

K-means was fit on the two score axes after standardization using the 50,000-customer training sample. K was selected from 2–10 by silhouette on a 3,000-row training subsample, then checked on a separate random 10,000-customer sample. Validation silhouette uses 3,000 held-out rows. Seed: 42; means and scales came from training only.

| method               |   segments |   holdout_silhouette |
|:---------------------|-----------:|---------------------:|
| 3 × 3 score matrix   |          9 |                0.37  |
| K-means selected k=3 |          3 |                0.506 |
| K-means matched k=9  |          9 |                0.413 |

K-means selected k=3. Its silhouette row is not directly comparable with the 9-cell matrix when the number of segments differs. At the matched count of 9 segments, K-means has the higher silhouette.

### Training silhouette by K

|   k |   train_silhouette |
|----:|-------------------:|
|   2 |           0.408649 |
|   3 |           0.491632 |
|   4 |           0.41868  |
|   5 |           0.424389 |
|   6 |           0.440088 |
|   7 |           0.441545 |
|   8 |           0.402394 |
|   9 |           0.423394 |
|  10 |           0.424067 |

For selected k=3, the dominant 3 × 3 matrix cell covers 56.6% of the held-out comparison sample. Cluster profiles below apply the fitted model to all 1,000,528 customers.

|   cluster |   customers |   population_pct |   mean_financial_health |   mean_engagement |   thin_file_pct | dominant_matrix_segment           |   dominant_matrix_share_pct |
|----------:|------------:|-----------------:|------------------------:|------------------:|----------------:|:----------------------------------|----------------------------:|
|         1 |      258839 |            25.87 |                    41.6 |              28.4 |           67.86 | Low health × Low engagement       |                       75.75 |
|         2 |      223802 |            22.37 |                    74.4 |              69.5 |            3.02 | Medium health × Medium engagement |                       42.02 |
|         3 |      517887 |            51.76 |                    77.8 |              29.5 |            0    | Medium health × Low engagement    |                       56.56 |

Silhouette measures geometric separation, not repayment, applications, customer response, or action success. Task 3 has no observed post-action outcome label for supervised model selection. Task 1's holdout Gini (0.302) and Task 2's application AUC (0.649) validate score ordering, not the success of Task 3 actions. Keep the 3 × 3 matrix as the operational segmentation: it maps those score tiers to actions and preserves the thin-file safeguard. Use K-means only for exploratory profiles; it does not replace the action rules.


## Quality checks

| check                                                 | status   | detail                                                  |
|:------------------------------------------------------|:---------|:--------------------------------------------------------|
| One row per customer                                  | PASS     | 1,000,528 rows; 1,000,528 customers expected            |
| Required outputs are populated                        | PASS     | Scores, segment, action, and rationale                  |
| Scores stay within 0–100                              | PASS     | Both scores are bounded to the published scale          |
| Tier labels match score cut-offs                      | PASS     | 80 and 50 point boundaries                              |
| Action and rationale are present                      | PASS     | Every row has a recommended action and an evidence note |
| Thin-file cases remain distinct from repayment stress | PASS     | No repayment failure is inferred from missing history   |
| Demographics are absent from the output               | PASS     | No demographic fields are used for scoring or actions   |

## Decision limits

These are rule-based service-routing suggestions, not estimated treatment effects or credit decisions. Review consent, suitability, affordability, capacity, and policy before activation. The score associations do not establish causes, and the team should verify action wording against its operating procedures.
