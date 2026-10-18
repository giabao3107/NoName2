# Generated intermediates

The pipeline writes cleaned per-table Parquet files, customer score intermediates, and compact JSON metrics. Raw source tables are staged in a temporary DuckDB connection and are not saved as raw Parquet copies. Any `raw_*.parquet` files already present are leftovers from an earlier pipeline version; the current workflow neither reads nor refreshes them.

These outputs can be rebuilt from the unchanged CSVs in `Data_raw/` (or `Data/` on older project copies). The analysis uses separate tables; there is no combined `data_clean.parquet`.