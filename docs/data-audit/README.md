# Data Audit Reports

> **Navigation:** [Main README](../README.md) | [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) | [Data Leakage](DATA_LEAKAGE.md) | [Data Audit](data-audit/README.md)

---

This directory contains data integrity audit reports comparing first-party data (Data obtained via Java ingestion) against independent third-party sources.

## Purpose

Before deploying the ranking model for live trading, data integrity must be verified:
1. Current prices match independent sources
2. Historical data has no corruption or gaps
3. Anomaly detection filters bad data correctly
4. No evidence of data leakage in the pipeline

## Audit Reports

| Date | Report | Summary |
|------|--------|---------|
| 2026-01-02 | [Price Validation](2026-01-02_price_validation.md) | 19/19 tickers match Google Finance (100%) |
| 2026-01-02 | [Macro Validation](2026-01-02_macro_validation.md) | Interest rates match RBNZ OCR (8/8 periods) |

## Methodology

Each audit follows a standardized process:

1. **Third-Party Fetch:** Live prices from Google Finance
2. **First-Party Extract:** Prices from `data/data_long.csv` at same timestamp
3. **Comparison:** Calculate absolute percentage difference
4. **Thresholds:**
   - `<1%` = MATCH
   - `1-5%` = CLOSE (investigate)
   - `>5%` = MISMATCH (critical)

## Related Documentation

- [Data Leakage Prevention](../DATA_LEAKAGE.md) — Leakage testing methodology
- [Testing Guide](../TESTING.md) — Full test suite documentation
- [Java Data Sources](../../java/docs/DATA_SOURCES.md) — Ingestion pipeline details
