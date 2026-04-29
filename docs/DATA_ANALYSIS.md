# Data Analysis: `data_long.csv`

**Last updated:** 2026-04-20 (full universe run)

## Format

- **File:** `data/data_long.csv`
- **Companion:** `data/source_legend.csv` (maps `src` integer IDs to source names)
- **Format:** Long-format CSV (one observation per row), RFC 4180 quoting
- **Rows:** 4,762,520 (excluding header)
- **Produced by:** Java application in `java/` — `CsvLongParser.saveCsv()` writes `DataPoint` objects collected by `IngestManager` from 14 data sources.

## Schema

| Column      | Type     | Description |
|-------------|----------|-------------|
| `timestamp` | `long`   | Unix epoch milliseconds (UTC). Can be negative for pre-1970 data. |
| `ticker`    | `string` | Equity ticker (e.g. `ANZ.NZ`) or empty string for macro/economic data. |
| `feature`   | `string` | Metric name, normalized to `snake_case`. |
| `value`     | `string` | Numeric value in plain decimal form (no scientific notation). Always present (nulls filtered upstream). |
| `src`       | `int`    | Integer source ID. See `data/source_legend.csv` for the mapping. **Provenance only** — the Python loader (`csv_loader.py`) ignores `src` and routes rows by `feature` name + ticker presence. Use `src` for staleness reporting and audit, not for categorisation. |

## Timestamp Range

- **Earliest:** ~1961 (timestamp `-270950400000`) — long-term interest rate data
- **Latest:** 2026-04-18 (timestamp `1776471600000`) — recent market data and macro snapshots

## Tickers

150 unique ticker values (146 NZX equities + 3 international indices/commodities + empty string for macro data):

### Breakdown by Category

| Category | Count | Row Count | % of Data |
|----------|-------|-----------|-----------|
| NZX equities (`.NZ`) | 146 | 3,663,583 | 76.9% |
| Macro/economic *(empty ticker)* | 1 | 911,567 | 19.1% |
| International indices & commodities | 3 | 187,370 | 3.9% |

### Top 20 Tickers by Row Count

| Ticker | Row Count | % of Data |
|--------|-----------|-----------|
| *(empty/macro)* | 911,567 | 19.1% |
| `%5ETNX` (US 10Y Treasury) | 84,564 | 1.8% |
| `%5EFTSE` (FTSE 100) | 64,098 | 1.3% |
| `ANZ.NZ` | 51,997 | 1.1% |
| `SPK.NZ` | 51,983 | 1.1% |
| `AIA.NZ` | 51,980 | 1.1% |
| `AIR.NZ` | 51,966 | 1.1% |
| `PGW.NZ` | 51,959 | 1.1% |
| `IFT.NZ` | 51,958 | 1.1% |
| `POT.NZ` | 51,941 | 1.1% |
| `RYM.NZ` | 51,938 | 1.1% |
| `CEN.NZ` | 51,934 | 1.1% |
| `FPH.NZ` | 51,933 | 1.1% |
| `MFT.NZ` | 51,931 | 1.1% |
| `FBU.NZ` | 50,366 | 1.1% |
| `SKC.NZ` | 49,613 | 1.0% |
| `STU.NZ` | 45,368 | 1.0% |
| `SAN.NZ` | 44,214 | 0.9% |
| `SKT.NZ` | 44,067 | 0.9% |
| `PFI.NZ` | 40,159 | 0.8% |

### Non-NZX Tickers

| Ticker | Description | Date Range | Rows |
|--------|-------------|------------|------|
| `%5ETNX` | US 10-Year Treasury Yield | 1970-01-03 – 2026-04-18 | 84,564 |
| `%5EFTSE` | FTSE 100 Index | 1984-01-04 – 2026-04-18 | 64,098 |
| `ZS=F` | Soybeans Futures | 2000-09-16 – 2026-04-18 | 38,610 |

### NZX Data Coverage

| Metric | Count |
|--------|-------|
| Total NZX tickers | 146 |
| With price data (OHLCV) | 137 (93.8%) |
| With fundamental data (YfFinances) | 102 (69.9%) |
| With dividend data | 115 (78.8%) |
| With stock split data | 46 (31.5%) |

**9 NZX tickers lack price data:** `ARV.NZ`, `FRE.NZ`, `THL.NZ`, `TPW.NZ`, `TRA.NZ`, `TWR.NZ`, `VCT.NZ`, `VGL.NZ`, `ZEL.NZ`. These have only fundamental data from YfFinances and will be excluded from the investable universe.

### History Length Distribution (NZX tickers)

| History Span | Ticker Count |
|--------------|--------------|
| >20 years | 53 |
| 10–20 years | 57 |
| 5–10 years | 21 |
| 2–5 years | 10 |
| <2 years | 5 |

**Longest histories** (top 10 by price row count, all 2000–2026): `PFI.NZ` (39,985), `SKC.NZ` (39,952), `AIA.NZ` (39,948), `SPK.NZ` (39,945), `ANZ.NZ` (39,943), `FPH.NZ` (39,940), `POT.NZ` (39,938), `MFT.NZ` (39,937), `AIR.NZ` (39,936), `MHJ.NZ` (39,931).

**Shortest histories:** `BIF.NZ` (2026 only, 6 price rows), `ARV.NZ` (2025–2026, 0 price rows), `TRA.NZ` (2024–2026, 0 price rows), `GLD.NZ` (2024–2026), `BTC.NZ` (2024–2026).

## Sources

14 sources produced data in the current run. The `src` column contains integer IDs; `data/source_legend.csv` provides the mapping:

| ID | Source Name | Rows | Features | Notes |
|----|-------------|------|----------|-------|
| 0 | `nz_vehicle_registrations` | 282 | 2 | Economic activity proxy |
| 1 | `nz_tax_revenue` | 1,244 | 52 | |
| 2 | `nz_pensions` | 556 | 47 | |
| 3 | `nz_road_fatalities` | 383 | 1 | |
| 4 | `nz_labor_stats` | 10,100 | 56 | |
| 5 | `global_aquaculture_production` | 1,196 | 52 | |
| 6 | `yf_prices` | 3,545,014 | 8 | OHLCV + dividend + split |
| 7 | `nz_labor_taxation` | 2,800 | 112 | |
| 8 | `wikimedia_pageviews` | 1,172,934 | 231 | Now active (was inactive in prior runs) |
| 9 | `nz_rates_fx` | 1,796 | 3 | Interest rates |
| 10 | `nz_business_confidence` | 1,229 | 2 | OECD BCI/CCI |
| 11 | `nz_gdp` | 1,248 | 3 | Government expenditure data |
| 12 | `yf_finances` | 18,137 | 130 | Fundamental financials |
| 13 | `nz_balance_of_payments` | 5,601 | 39 | |

**Note:** `global_food_prices` exists in code but produced no data in this run. `wikimedia_pageviews` is now active and is the second-largest source by row count.

## Features

738 unique `snake_case` feature names. All features are normalized at write time by `CsvLongParser.normalizeFeatureName()` (camelCase, spaces, dashes, parentheses → `snake_case`).

### Stock Price Features (per ticker, src=6)

`open`, `high`, `low`, `close`, `adj_close`, `volume` — standard OHLCV. Additionally `dividend` and `split` features exist for corporate actions.

### Feature Name Examples

- `new_registrations_passenger_cars` (vehicle registrations)
- `nz_labor_lfparticipation_rate_age15to64_total` (labor stats)
- `bop_current_account_balance_revenue_minus_expenditure_sa` (balance of payments)
- `aquaculture_production_tonnes_dnk` (global aquaculture)
- `annual_basic_average_shares` (Yahoo Finance financials)
- `nz_road_fatalities_monthly` (road fatalities)
- `immediate_interest_rates_call_money_interbank_rate` (rates/FX)
- `oecd_bcicp`, `oecd_ccicp` (business confidence)

## Known Limitations

1. **Row ordering is undefined:** Rows are not sorted. Data is collected via parallel streams, so order depends on source execution timing. Sort at consumption time as needed.
2. **9 NZX tickers have no price data:** Likely due to Yahoo Finance API limitations (ticker symbol changes, delistings, or API errors). These tickers have only fundamental data and must be excluded from any price-dependent analysis.
3. **No publication dates:** The `timestamp` column represents the observation/period date, not the date the data became publicly available. Downstream consumers must apply conservative publication lags for point-in-time correctness (see implementation plan §3.2).
4. **Uneven history depth:** 5 NZX tickers have <2 years of history. The min-history filter in the investable universe construction should handle this, but factor computations requiring long lookbacks (e.g., 12-month momentum) will mechanically exclude recent listings.
5. **Wikimedia pageviews dominance:** At 1.17M rows (24.6% of all data), `wikimedia_pageviews` is the second-largest source. Its signal value for NZX equity forecasting is unvalidated and should be treated as experimental.

## Raw Data Analysis Workflow

Use the raw-data analysis workflow when you need a current source-of-truth view of `data/data_long.csv` before feature engineering or further research:

```bash
cd python
uv run python scripts/raw_data_analysis.py --data ..\data\data_long.csv --out reports\raw_data_analysis --run-date YYYY-MM-DD
```

Artifacts are written under `python/reports/raw_data_analysis/YYYY-MM-DD/`:

- `report.md` - canonical Markdown report with stable section headings for human and agent consumption
- `summary.json` - machine-readable summary with top findings, issue counts, research-implication buckets, and relative artifact paths
- `tables/dataset_overview.csv` - top-level dataset shape and date-range summary
- `tables/source_inventory.csv` - per-source coverage summary
- `tables/feature_inventory.csv` - per-feature inventory with source and numeric parse information
- `tables/sparsity_by_feature.csv` - feature-level sparsity summary
- `tables/sparsity_by_ticker.csv` - ticker-level sparsity summary
- `tables/temporal_patterns.csv` - per-feature cadence, gap, and irregularity summary
- `tables/stale_value_summary.csv` - longest unchanged-value run summary by series
- `tables/anomaly_flags.csv` - duplicate, conflict, and numeric anomaly flags
- `tables/leakage_flags.csv` - heuristic timestamp and cadence-based leakage-risk warnings

Use `report.md` as the canonical narrative entry point. Use `summary.json` and the CSV tables when an agent or downstream script needs deterministic machine-readable inputs for follow-up analysis.
