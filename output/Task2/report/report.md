# Task 2 — Engage with intent

## Business question

Once Task 1 identifies repayment evidence, which customers show enough recent interaction to guide a relevant, consent-aware contact—and does the score relate to later outcomes?

## Analysis

The score combines digital adoption, recency, frequency, responsiveness, and product breadth. Features are frozen before the application outcome window; call data use their separate observation period. Responsiveness is omitted and its weight redistributed when the customer has no contact evidence.

**Team-wide rule for call data (shared with Task 1, chart 1).** CALL timestamps are unreliable for every call type, not only collection calls: collection calls predate customers' first loan at the rate random dates would produce, and only about 4% of 'sign successful' calls fall within 14 days of an actual signing. Call evidence is therefore used only as customer-level counts and rates whose meaning does not depend on dates (E4 responsiveness, answer and conversion rates, unanswered-call counts). Comparisons labelled before/after the snapshot are read as cross-sectional, and the forward test of this score is the later-application check.

|                   |   weight |
|:------------------|---------:|
| E1_digital        |     0.2  |
| E2_recency        |     0.2  |
| E3_frequency      |     0.25 |
| E4_responsiveness |     0.2  |
| E5_breadth        |     0.15 |

## Findings and business insight

- **The score separates later behavior:** application AUC is 0.649; High-tier customers apply at 36.3% versus 12.4% in Low (2.9×). As a cross-sectional check on sales calls, positive-response rates are 32.4% in High versus 7.0% in Low (4.6×); the fair figure is the score without the call component (conversion AUC 0.645), because E4 and the outcome share one call log (0.705 with E4).
- Application and signing rates by tier:

| tier   |      n | apply_rate   | sign_rate   |
|:-------|-------:|:-------------|:------------|
| High   |  50841 | 36.3%        | 23.4%       |
| Medium | 196769 | 25.7%        | 17.3%       |
| Low    | 752918 | 12.4%        | 8.5%        |
- **Frequency and breadth add more ranking signal than recency alone:** component AUCs are:

|                   |   apply_auc |
|:------------------|------------:|
| E3_frequency      |    0.660741 |
| E5_breadth        |    0.627314 |
| E1_digital        |    0.574541 |
| E2_recency        |    0.502899 |
| E4_responsiveness |    0.499898 |
- **Conversation depth is more informative than dial count in this sample:** duration AUC 0.922, single non-converting call robustness AUC 0.720 (call order is unreliable, so this is one call's duration rather than a true first call), versus attempts AUC 0.591. Duration partly occurs during the conversion process, so it is not a causal lever.
- **App activity is a modest signal:** customers with ≥0.5 sessions/week convert on sales calls at 3.39%, versus 2.58% for non-users (131,368 called customers). This supports a channel test, not a blanket app campaign.
- **Customer-initiated leads may deserve a distinct path:** SOB has the highest observed offered-product signing rate among sources with at least 1,000 leads (67.3%, n=35,535).
- **Rising pre-loan event activity is a monitoring cue:** delinquency is 2.54% in Rising versus 2.12% in Declining activity (1,152,858 contracts). The association does not establish cause.

## Business implication and handoff

Prioritize opt-in, low-cost re-engagement for low-activity customers when Task 1 shows adequate health; route established repayment concerns to support rather than sales. Task 3 combines both scores into the final customer-level action.

## Limits

Application validation uses a later window beginning 2025-02-01 and is the forward test. The call-response check covers 115,044 customers and is cross-sectional because call dates are unreliable. All reported differences are associations; contact exposure and customer selection may explain part of them.
