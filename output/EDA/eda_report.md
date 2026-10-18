# Exploratory data analysis

## What the data can support

| dataset          |    rows |   distinct_customers |   customer_coverage_pct | date_start          | date_end            |
|:-----------------|--------:|---------------------:|------------------------:|:--------------------|:--------------------|
| Customers        | 1000528 |              1000528 |                  100    | nan                 | nan                 |
| Contracts        | 4280377 |              1000528 |                  100    | 2009-04-21 10:39:50 | 2025-06-04 22:29:47 |
| Leads            | 2104642 |               645344 |                   64.5  | 2019-10-09 11:16:52 | 2025-06-04 20:56:56 |
| Digital behavior | 4412610 |               473086 |                   47.28 | 2022-12-01 00:05:29 | 2025-06-03 22:30:14 |
| Calls            | 4000000 |               360661 |                   36.05 | 2025-01-01 03:47:11 | 2026-08-31 23:11:28 |

## Raw missingness: fields with the largest gaps

| dataset   | column                | type      |    rows |   null_pct |   placeholder_pct (XNA/none/blank) |   zero_pct |
|:----------|:----------------------|:----------|--------:|-----------:|-----------------------------------:|-----------:|
| customer  | CNT_PERSON_DEPENDENT  | BIGINT    | 1000528 |      98.51 |                             nan    |       0.03 |
| contract  | AMT_BILLING           | DOUBLE    | 6234093 |      93.52 |                             nan    |       0    |
| contract  | TRANSAC               | TIMESTAMP | 6234093 |      93.52 |                             nan    |     nan    |
| contract  | SIGN_CONTRACT_TIME    | TIMESTAMP | 6234093 |      32.2  |                             nan    |     nan    |
| call      | PRODUCT_TYPE          | VARCHAR   | 4000000 |      27.53 |                               1.51 |     nan    |
| contract  | APPROVE_CONTRACT_TIME | TIMESTAMP | 6234093 |      23.01 |                             nan    |     nan    |
| customer  | ADDRESS               | VARCHAR   | 1000528 |      14.92 |                               0    |     nan    |
| customer  | CNT_CHILDREN          | BIGINT    | 1000528 |      12.6  |                             nan    |      32.04 |
| contract  | ANNUITY               | DOUBLE    | 6234093 |      10.65 |                             nan    |       0    |
| contract  | PAYMENT_NUM           | DOUBLE    | 6234093 |      10.56 |                             nan    |       0    |
| call      | CALL_RESULT_SYSTEM    | VARCHAR   | 4000000 |       5.85 |                               0    |     nan    |
| leads     | DESC_FIN_REASON       | VARCHAR   | 2143891 |       1.71 |                               0    |     nan    |
| leads     | MAX_OFFER             | DOUBLE    | 2143891 |       0.58 |                             nan    |       0    |
| contract  | PRODUCT               | VARCHAR   | 6234093 |       0.45 |                               0    |     nan    |
| contract  | INIT_PAY              | DOUBLE    | 6234093 |       0.45 |                             nan    |      29.69 |

## Mature signed-contract outcomes by product

| product   |   mature_signed_contracts |   observed_bad_rate_pct |
|:----------|--------------------------:|------------------------:|
| TW        |                     98222 |                    4.97 |
| CW        |                     22043 |                    4.77 |
| CC        |                     90518 |                    2.45 |
| BNPL      |                    162650 |                    2.21 |
| CL        |                    349517 |                    2.19 |
| CD        |                   2036089 |                    2.13 |
| IN        |                         1 |                    0    |

## Digital behavior channels

| channel   |   events |   customers |
|:----------|---------:|------------:|
| APP       |  1630349 |      232674 |
| SA        |   867787 |      201283 |
| WEB       |   746021 |      169734 |
| TLS       |   726233 |      148336 |
| PARTNER   |   396829 |       87345 |
| SOB       |    44623 |       39516 |
| BRANCH    |      761 |         760 |
| MULTI     |        7 |           4 |

## Observation windows

| source           | first_month         | last_month          |   total_events |
|:-----------------|:--------------------|:--------------------|---------------:|
| Calls            | 2025-01-01 00:00:00 | 2026-08-01 00:00:00 |    4e+06       |
| Contracts        | 2009-04-01 00:00:00 | 2025-06-01 00:00:00 |    4.28038e+06 |
| Digital behavior | 2022-12-01 00:00:00 | 2025-06-01 00:00:00 |    4.41261e+06 |
| Leads            | 2019-10-01 00:00:00 | 2025-06-01 00:00:00 |    2.10464e+06 |

## App-session profile

Median: 0.00 sessions/week; p99: 1.73; above p99: 9,875.

## Reading these patterns

- Coverage is the share of the customer file appearing in each source, not a measure of data quality.
- Stage 1 coerces missing app-session values to zero. Treat zero and missing as combined unless source semantics are confirmed.
- Product differences and channel counts are descriptive; they do not establish causal effects.
- Raw column profiles and missingness are in `Data_clean/metrics_phase1.json`; cleaned profiles are in `Data_clean/metrics_eda.json`.
