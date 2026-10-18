# BI10 Round 2

This project turns customer, contract, lead, behavior, and call data into explainable financial-health and engagement scores, customer segments, and next-best-action recommendations.

## Project files

```text
Code/
  1. Data processing.py   Validate sources and write cleaned tables
  2. EDA.py               Explore coverage, windows, missingness, and behavior
  3. Task 1.py            Financial-health score and validation
  4. Task 2.py            Engagement score and validation
  5. Task 3.py            Customer matrix, recommendations, final CSV
  config/                 Shared business rules and analytical helpers
Data_raw/                 Raw CSV files and optional StructureTable.xlsx
Data_clean/               Cleaned Parquet and JSON evidence
output/                   Reports, figures, score files, and final customer CSV
Tools/pbi/                Dashboard guide and interactive dashboard
AI_USAGE_REPORT.md        Team-owned AI-use disclosure template
```

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install the pinned packages: `python -m pip install -r requirements.txt`.
3. Place `CALL.csv`, `CONTRACT.csv`, `CUSTOMER.csv`, `LEADS.csv`, and `PAST_BEHAVIOR.csv` in `Data_raw/`. `StructureTable.xlsx` is optional. Keep the inputs unchanged.

## Run order

Run from the project root. Each script uses notebook-compatible `# %%` cells. To see figures inline, open a `.py` file in VS Code with the Jupyter extension and choose **Run All** (or run cells in order); each plot cell displays one image and the next Markdown cell explains it. Running the commands below also saves the same title-free PNGs under `output/`.

```bash
python "Code/1. Data processing.py"
python "Code/2. EDA.py"
python "Code/3. Task 1.py"
python "Code/4. Task 2.py"
python "Code/5. Task 3.py"
```

## Outputs

- Cleaned analytical tables only: `Data_clean/*_clean.parquet`. Raw CSVs are staged in a temporary DuckDB connection; no new raw Parquet files are written.
- Quality and analysis evidence: `Data_clean/metrics_phase*.json`.
- EDA report and figures: `output/EDA/eda_report.md`, `output/EDA/figures/`.
- Financial-health result and report: `output/Task1/scores/fh_scores.parquet`, `output/Task1/report/report.md`.
- Engagement result and report: `output/Task2/scores/eng_scores.parquet`, `output/Task2/report/report.md`.
- **One customer-level submission CSV:** `output/Task3/scores/final_customer_intelligence_output.csv`. It includes scores, tiers, score components, matrix segment, action, and rationale.
- Task 3 report: `output/Task3/report/report.md`.

The project narrative is: **protect the base → engage with intent → match the action to the need**. Reports explain what each score supports and where evidence is observational. Next-best actions are service suggestions, not credit decisions or proven causal uplift.

This version no longer refreshes per-question CSVs. CSV tables already present under `output/*/tables/` are leftovers from earlier runs; review them before removing. Legacy `Data_clean/raw_*.parquet` files are likewise not read or refreshed.

## Checks

```bash
python -m unittest discover -s tests -v
```

Do not delete `Data_raw/`. Cleaned tables, JSON summaries, reports, figures, and scores can be rebuilt from the raw inputs.
