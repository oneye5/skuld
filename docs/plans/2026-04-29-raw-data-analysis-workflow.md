# Raw Data Analysis Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable raw-data analysis workflow for `data/data_long.csv` that emits a deterministic Markdown report plus machine-readable CSV/JSON artifacts covering coverage, sparsity, temporal behavior, anomalies, and leakage heuristics.

**Architecture:** Add a small dedicated `skuld_research.raw_data_analysis` package instead of extending the existing factor diagnostics modules. Keep the workflow split into four responsibilities: dataset loading/normalization, metric builders, artifact writing, and CLI orchestration. The CLI reads `data_long.csv` plus `source_legend.csv`, runs the metric builders, writes stable artifacts under `python/reports/raw_data_analysis/<run-date>/`, and exits with a non-zero code on missing inputs.

**Tech Stack:** Python 3, `pandas`, `numpy`, `json`, `pathlib`, `argparse`, `pytest`, `uv`

---

## File Structure

- Create: `python/scripts/raw_data_analysis.py`
  CLI entry point that loads the raw dataset, runs the analysis pipeline, and writes report artifacts.
- Create: `python/src/skuld_research/raw_data_analysis/__init__.py`
  Public exports for the raw-data analysis package.
- Create: `python/src/skuld_research/raw_data_analysis/models.py`
  Dataclasses for normalized rows, summary payloads, and report section inputs.
- Create: `python/src/skuld_research/raw_data_analysis/dataset.py`
  Low-level loader for `data_long.csv` and `source_legend.csv`, including source-name mapping and timestamp parsing.
- Create: `python/src/skuld_research/raw_data_analysis/coverage.py`
  Dataset overview, source inventory, feature inventory, and sparsity metrics.
- Create: `python/src/skuld_research/raw_data_analysis/temporal.py`
  Frequency inference, gap summaries, and staleness metrics.
- Create: `python/src/skuld_research/raw_data_analysis/anomalies.py`
  Duplicate/conflict detection, numeric outlier checks, and leakage heuristics.
- Create: `python/src/skuld_research/raw_data_analysis/report.py`
  Markdown writer plus JSON summary writer for stable output contracts.
- Create: `python/src/skuld_research/raw_data_analysis/pipeline.py`
  Orchestration function that coordinates the metric builders and returns a complete analysis result.
- Create: `python/tests/test_raw_data_analysis_dataset.py`
  Tests for source-legend loading, timestamp normalization, and dataset parsing.
- Create: `python/tests/test_raw_data_analysis_coverage.py`
  Tests for coverage, sparsity, and source-name metrics.
- Create: `python/tests/test_raw_data_analysis_temporal.py`
  Tests for inferred frequency, gap detection, and stale-run metrics.
- Create: `python/tests/test_raw_data_analysis_anomalies.py`
  Tests for duplicates, conflicts, outliers, and leakage flags.
- Create: `python/tests/test_raw_data_analysis_report.py`
  Tests for deterministic Markdown/JSON output structure and artifact names.
- Create: `python/tests/test_raw_data_analysis_cli_e2e.py`
  End-to-end CLI test against a small synthetic dataset and legend.
- Modify: `python/tests/conftest.py`
  Add a richer synthetic raw-data CSV fixture that includes duplicate rows, sparse fundamentals, irregular timestamps, and suspicious values for raw-data analysis tests.

### Output Contract

The implementation must write these artifact names exactly:

- `python/reports/raw_data_analysis/<run-date>/report.md`
- `python/reports/raw_data_analysis/<run-date>/summary.json`
- `python/reports/raw_data_analysis/<run-date>/tables/dataset_overview.csv`
- `python/reports/raw_data_analysis/<run-date>/tables/source_inventory.csv`
- `python/reports/raw_data_analysis/<run-date>/tables/feature_inventory.csv`
- `python/reports/raw_data_analysis/<run-date>/tables/sparsity_by_feature.csv`
- `python/reports/raw_data_analysis/<run-date>/tables/sparsity_by_ticker.csv`
- `python/reports/raw_data_analysis/<run-date>/tables/temporal_patterns.csv`
- `python/reports/raw_data_analysis/<run-date>/tables/anomaly_flags.csv`
- `python/reports/raw_data_analysis/<run-date>/tables/leakage_flags.csv`

## Task 1: Create the normalized dataset loader

**Files:**
- Create: `python/src/skuld_research/raw_data_analysis/models.py`
- Create: `python/src/skuld_research/raw_data_analysis/dataset.py`
- Create: `python/tests/test_raw_data_analysis_dataset.py`
- Modify: `python/tests/conftest.py`

- [ ] **Step 1: Write the failing dataset-loader tests**

```python
from pathlib import Path

import pandas as pd


def test_load_analysis_dataset_maps_source_names(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)

    assert dataset.rows.loc[0, "src"] == 6
    assert dataset.rows.loc[0, "source_name"] == "yf_prices"


def test_load_analysis_dataset_parses_timestamp_to_utc_naive(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)

    assert dataset.rows["date"].dtype == "datetime64[ns]"
    assert dataset.rows["date"].min() == pd.Timestamp("2024-01-31")


def test_load_analysis_dataset_preserves_raw_value_and_numeric_value(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    row = dataset.rows.loc[dataset.rows["feature"] == "annual_basic_average_shares"].iloc[0]

    assert row["raw_value"] == "1000000"
    assert row["numeric_value"] == 1_000_000.0
```

- [ ] **Step 2: Run dataset-loader tests to verify they fail**

Run: `uv run pytest tests/test_raw_data_analysis_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'skuld_research.raw_data_analysis'`

- [ ] **Step 3: Add the synthetic fixture for raw-data analysis tests**

```python
RAW_ANALYSIS_CSV = """\
timestamp,ticker,feature,value,src
1706659200000,ANZ.NZ,adj_close,50.0,6
1706745600000,ANZ.NZ,adj_close,51.0,6
1706745600000,ANZ.NZ,adj_close,51.0,6
1706832000000,ANZ.NZ,volume,120000,6
1706659200000,SPK.NZ,adj_close,4.80,6
1709251200000,ANZ.NZ,annual_basic_average_shares,1000000,12
1709251200000,SPK.NZ,annual_basic_average_shares,0,12
1709251200000,,oecd_bcicp,100.5,10
1711929600000,ANZ.NZ,page_views,9999999,8
1714521600000,ANZ.NZ,page_views,10,8
"""


@pytest.fixture
def raw_analysis_csv_path(tmp_path: Path) -> Path:
    csv_file = tmp_path / "data_long.csv"
    csv_file.write_text(RAW_ANALYSIS_CSV, encoding="utf-8")
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n6,yf_prices\n8,wikimedia_pageviews\n10,nz_business_confidence\n12,yf_finances\n",
        encoding="utf-8",
    )
    return csv_file
```

- [ ] **Step 4: Implement the dataset models and loader**

```python
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AnalysisDataset:
    rows: pd.DataFrame
    source_legend: pd.DataFrame


def load_analysis_dataset(data_path: Path) -> AnalysisDataset:
    legend_path = data_path.with_name("source_legend.csv")
    rows = pd.read_csv(
        data_path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str, "src": "int64"},
    )
    legend = pd.read_csv(legend_path, dtype={"id": "int64", "name": str})
    rows["ticker"] = rows["ticker"].fillna("")
    rows["raw_value"] = rows["value"]
    rows["numeric_value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows["date"] = pd.to_datetime(rows["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    rows = rows.merge(legend.rename(columns={"id": "src", "name": "source_name"}), on="src", how="left")
    rows["source_name"] = rows["source_name"].fillna("unknown_source")
    return AnalysisDataset(rows=rows, source_legend=legend)
```

- [ ] **Step 5: Export the new package API**

```python
from skuld_research.raw_data_analysis.dataset import AnalysisDataset, load_analysis_dataset

__all__ = ["AnalysisDataset", "load_analysis_dataset"]
```

- [ ] **Step 6: Run dataset-loader tests to verify they pass**

Run: `uv run pytest tests/test_raw_data_analysis_dataset.py -v`
Expected: PASS with `3 passed`

- [ ] **Step 7: Commit**

```bash
git add python/src/skuld_research/raw_data_analysis/__init__.py python/src/skuld_research/raw_data_analysis/models.py python/src/skuld_research/raw_data_analysis/dataset.py python/tests/conftest.py python/tests/test_raw_data_analysis_dataset.py
git commit -m "feat: add raw data analysis dataset loader"
```

## Task 2: Build coverage, source, feature, and sparsity metrics

**Files:**
- Create: `python/src/skuld_research/raw_data_analysis/coverage.py`
- Create: `python/tests/test_raw_data_analysis_coverage.py`

- [ ] **Step 1: Write the failing coverage tests**

```python
def test_build_dataset_overview_reports_core_counts(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.coverage import build_dataset_overview
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    overview = build_dataset_overview(dataset)

    assert int(overview.loc[0, "row_count"]) == len(dataset.rows)
    assert int(overview.loc[0, "unique_sources"]) == 4
    assert int(overview.loc[0, "unique_features"]) >= 4


def test_build_source_inventory_uses_source_names(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.coverage import build_source_inventory
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    inventory = build_source_inventory(dataset)

    assert set(inventory["source_name"]) >= {"yf_prices", "yf_finances"}


def test_build_feature_sparsity_includes_missing_fraction(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.coverage import build_feature_sparsity
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    sparsity = build_feature_sparsity(dataset)

    assert "missing_fraction" in sparsity.columns
    assert (sparsity["missing_fraction"] >= 0.0).all()
    assert (sparsity["missing_fraction"] <= 1.0).all()
```

- [ ] **Step 2: Run coverage tests to verify they fail**

Run: `uv run pytest tests/test_raw_data_analysis_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function errors for `build_dataset_overview`, `build_source_inventory`, and `build_feature_sparsity`

- [ ] **Step 3: Implement the coverage module**

```python
import pandas as pd

from skuld_research.raw_data_analysis.dataset import AnalysisDataset


def build_dataset_overview(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows
    numeric_parse_rate = float(rows["numeric_value"].notna().mean())
    return pd.DataFrame(
        [{
            "row_count": len(rows),
            "date_min": rows["date"].min(),
            "date_max": rows["date"].max(),
            "unique_tickers": rows["ticker"].nunique(),
            "unique_features": rows["feature"].nunique(),
            "unique_sources": rows["source_name"].nunique(),
            "numeric_parse_rate": numeric_parse_rate,
        }]
    )


def build_source_inventory(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows
    grouped = rows.groupby("source_name", dropna=False)
    result = grouped.agg(
        row_count=("feature", "size"),
        date_min=("date", "min"),
        date_max=("date", "max"),
        ticker_count=("ticker", lambda s: (s != "").sum()),
        feature_count=("feature", "nunique"),
    ).reset_index()
    result["dataset_row_share"] = result["row_count"] / len(rows)
    return result.sort_values(["row_count", "source_name"], ascending=[False, True]).reset_index(drop=True)


def build_feature_sparsity(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows
    counts = rows.groupby("feature").agg(
        row_count=("feature", "size"),
        ticker_count=("ticker", lambda s: s.replace("", pd.NA).dropna().nunique()),
        date_count=("date", "nunique"),
        numeric_count=("numeric_value", lambda s: s.notna().sum()),
    )
    total_rows = len(rows)
    counts["missing_fraction"] = 1.0 - (counts["row_count"] / total_rows)
    return counts.reset_index().sort_values(["row_count", "feature"], ascending=[False, True])


def build_feature_inventory(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows
    result = rows.groupby(["feature", "source_name"], dropna=False).agg(
        row_count=("feature", "size"),
        ticker_count=("ticker", lambda s: s.replace("", pd.NA).dropna().nunique()),
        date_count=("date", "nunique"),
        numeric_parse_rate=("numeric_value", lambda s: float(s.notna().mean())),
    ).reset_index()
    return result.sort_values(["row_count", "feature"], ascending=[False, True]).reset_index(drop=True)
```

- [ ] **Step 4: Add ticker-level sparsity tests and implementation**

```python
def test_build_ticker_sparsity_excludes_empty_macro_ticker(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.coverage import build_ticker_sparsity
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    ticker_sparsity = build_ticker_sparsity(dataset)

    assert "" not in set(ticker_sparsity["ticker"])
    assert set(ticker_sparsity["ticker"]) >= {"ANZ.NZ", "SPK.NZ"}
```

```python
def build_ticker_sparsity(dataset: AnalysisDataset) -> pd.DataFrame:
    equity_rows = dataset.rows.loc[dataset.rows["ticker"] != ""]
    grouped = equity_rows.groupby("ticker")
    result = grouped.agg(
        row_count=("feature", "size"),
        feature_count=("feature", "nunique"),
        date_count=("date", "nunique"),
    ).reset_index()
    result["row_share"] = result["row_count"] / len(equity_rows)
    return result.sort_values(["row_count", "ticker"], ascending=[False, True]).reset_index(drop=True)
```

- [ ] **Step 5: Run coverage tests to verify they pass**

Run: `uv run pytest tests/test_raw_data_analysis_coverage.py -v`
Expected: PASS with `4 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/skuld_research/raw_data_analysis/coverage.py python/tests/test_raw_data_analysis_coverage.py
git commit -m "feat: add raw data coverage metrics"
```

## Task 3: Build temporal behavior metrics

**Files:**
- Create: `python/src/skuld_research/raw_data_analysis/temporal.py`
- Create: `python/tests/test_raw_data_analysis_temporal.py`

- [ ] **Step 1: Write the failing temporal tests**

```python
def test_infer_feature_temporal_patterns_labels_monthly_fundamentals(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    patterns = build_temporal_patterns(dataset)

    shares_row = patterns.loc[patterns["feature"] == "annual_basic_average_shares"].iloc[0]
    assert shares_row["frequency_label"] in {"monthly_or_slower", "quarterly_or_slower"}


def test_temporal_patterns_reports_max_gap_days(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    patterns = build_temporal_patterns(dataset)

    assert "max_gap_days" in patterns.columns
    assert (patterns["max_gap_days"] >= 0).all()
```

- [ ] **Step 2: Run temporal tests to verify they fail**

Run: `uv run pytest tests/test_raw_data_analysis_temporal.py -v`
Expected: FAIL with missing `build_temporal_patterns`

- [ ] **Step 3: Implement frequency inference and gap summaries**

```python
import numpy as np
import pandas as pd

from skuld_research.raw_data_analysis.dataset import AnalysisDataset


def _label_frequency(median_gap_days: float) -> str:
    if median_gap_days <= 2:
        return "daily"
    if median_gap_days <= 10:
        return "weekly_or_irregular"
    if median_gap_days <= 45:
        return "monthly_or_slower"
    return "quarterly_or_slower"


def build_temporal_patterns(dataset: AnalysisDataset) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for feature, feature_rows in dataset.rows.sort_values("date").groupby("feature"):
        unique_dates = feature_rows["date"].drop_duplicates().sort_values()
        gaps = unique_dates.diff().dropna().dt.days.astype(float)
        median_gap = float(gaps.median()) if not gaps.empty else np.nan
        max_gap = float(gaps.max()) if not gaps.empty else 0.0
        records.append(
            {
                "feature": feature,
                "observation_dates": int(unique_dates.size),
                "median_gap_days": median_gap,
                "max_gap_days": max_gap,
                "frequency_label": _label_frequency(median_gap) if not np.isnan(median_gap) else "singleton",
            }
        )
    return pd.DataFrame.from_records(records).sort_values("feature").reset_index(drop=True)
```

- [ ] **Step 4: Add stale-run coverage and implement it**

```python
def test_build_stale_value_summary_flags_repeated_values(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_stale_value_summary

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    stale = build_stale_value_summary(dataset)

    assert "max_repeat_run" in stale.columns
    assert stale["max_repeat_run"].max() >= 1
```

```python
def build_stale_value_summary(dataset: AnalysisDataset) -> pd.DataFrame:
    numeric_rows = dataset.rows.loc[dataset.rows["numeric_value"].notna()].copy()
    numeric_rows = numeric_rows.sort_values(["ticker", "feature", "date"])
    records: list[dict[str, object]] = []
    for (ticker, feature), group in numeric_rows.groupby(["ticker", "feature"], dropna=False):
        if ticker == "":
            continue
        repeated = group["numeric_value"].eq(group["numeric_value"].shift())
        run_id = repeated.ne(repeated.shift()).cumsum()
        run_lengths = repeated.groupby(run_id).sum()
        records.append(
            {
                "ticker": ticker,
                "feature": feature,
                "max_repeat_run": int(run_lengths.max()) if not run_lengths.empty else 0,
            }
        )
    return pd.DataFrame.from_records(records)
```

- [ ] **Step 5: Run temporal tests to verify they pass**

Run: `uv run pytest tests/test_raw_data_analysis_temporal.py -v`
Expected: PASS with `3 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/skuld_research/raw_data_analysis/temporal.py python/tests/test_raw_data_analysis_temporal.py
git commit -m "feat: add raw data temporal metrics"
```

## Task 4: Build anomaly and leakage heuristic metrics

**Files:**
- Create: `python/src/skuld_research/raw_data_analysis/anomalies.py`
- Create: `python/tests/test_raw_data_analysis_anomalies.py`

- [ ] **Step 1: Write the failing anomaly tests**

```python
def test_build_duplicate_flags_detects_duplicate_rows(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.anomalies import build_duplicate_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    duplicates = build_duplicate_flags(dataset)

    assert not duplicates.empty
    assert duplicates.loc[0, "duplicate_count"] >= 1


def test_build_numeric_outlier_flags_detects_extreme_pageview_jump(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.anomalies import build_numeric_outlier_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    outliers = build_numeric_outlier_flags(dataset)

    assert "page_views" in set(outliers["feature"])


def test_build_leakage_flags_marks_unsafe_timestamp_semantics(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.anomalies import build_leakage_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    flags = build_leakage_flags(dataset)

    assert "annual_basic_average_shares" in set(flags["feature"])
```

- [ ] **Step 2: Run anomaly tests to verify they fail**

Run: `uv run pytest tests/test_raw_data_analysis_anomalies.py -v`
Expected: FAIL with missing anomaly-builder functions

- [ ] **Step 3: Implement duplicate/conflict and numeric outlier checks**

```python
import pandas as pd

from skuld_research.raw_data_analysis.dataset import AnalysisDataset


def build_duplicate_flags(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows
    keys = ["timestamp", "ticker", "feature", "raw_value", "src"]
    duplicates = rows.loc[rows.duplicated(subset=keys, keep=False)]
    if duplicates.empty:
        return pd.DataFrame(columns=["ticker", "feature", "duplicate_count"])
    return (
        duplicates.groupby(["ticker", "feature"], dropna=False)
        .size()
        .reset_index(name="duplicate_count")
        .sort_values("duplicate_count", ascending=False)
        .reset_index(drop=True)
    )


def build_numeric_outlier_flags(dataset: AnalysisDataset) -> pd.DataFrame:
    numeric_rows = dataset.rows.loc[dataset.rows["numeric_value"].notna()].copy()
    records: list[dict[str, object]] = []
    for feature, group in numeric_rows.groupby("feature"):
        values = group["numeric_value"]
        median = values.median()
        mad = (values - median).abs().median()
        if mad == 0 or pd.isna(mad):
            continue
        score = 0.6745 * (values - median).abs() / mad
        flagged = group.loc[score > 8.0]
        for _, row in flagged.iterrows():
            records.append(
                {
                    "date": row["date"],
                    "ticker": row["ticker"],
                    "feature": feature,
                    "numeric_value": row["numeric_value"],
                    "flag_type": "robust_zscore",
                }
            )
    return pd.DataFrame.from_records(records)
```

- [ ] **Step 4: Implement conservative leakage heuristics**

```python
def build_leakage_flags(dataset: AnalysisDataset) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for feature, group in dataset.rows.groupby("feature"):
        median_gap = group["date"].sort_values().drop_duplicates().diff().dropna().dt.days.median()
        source_names = set(group["source_name"])
        if "yf_finances" in source_names:
            records.append(
                {
                    "feature": feature,
                    "risk_level": "warning",
                    "reason": "Fundamental field uses observation dates without publication timestamps; unsafe by default for raw feature engineering.",
                }
            )
        elif pd.notna(median_gap) and median_gap > 60:
            records.append(
                {
                    "feature": feature,
                    "risk_level": "review",
                    "reason": "Slow-moving field should be checked for publication timing before use.",
                }
            )
    return pd.DataFrame.from_records(records).drop_duplicates().sort_values(["risk_level", "feature"])
```

- [ ] **Step 5: Run anomaly tests to verify they pass**

Run: `uv run pytest tests/test_raw_data_analysis_anomalies.py -v`
Expected: PASS with `3 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/skuld_research/raw_data_analysis/anomalies.py python/tests/test_raw_data_analysis_anomalies.py
git commit -m "feat: add raw data anomaly heuristics"
```

## Task 5: Build the report writer and pipeline orchestrator

**Files:**
- Modify: `python/src/skuld_research/raw_data_analysis/models.py`
- Create: `python/src/skuld_research/raw_data_analysis/report.py`
- Create: `python/src/skuld_research/raw_data_analysis/pipeline.py`
- Create: `python/tests/test_raw_data_analysis_report.py`

- [ ] **Step 1: Write the failing report tests**

```python
import json


def test_write_raw_data_report_creates_stable_headings(tmp_path, raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis
    from skuld_research.raw_data_analysis.report import write_raw_data_report

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    result = run_raw_data_analysis(dataset)
    out_dir = tmp_path / "raw_data_analysis" / "2026-04-29"

    report_path, summary_path = write_raw_data_report(result, out_dir)

    content = report_path.read_text(encoding="utf-8")
    assert "# Raw Data Analysis Report" in content
    assert "## Dataset Overview" in content
    assert "## Leakage Risk Review" in content
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["report_path"].endswith("report.md")
```

- [ ] **Step 2: Run report tests to verify they fail**

Run: `uv run pytest tests/test_raw_data_analysis_report.py -v`
Expected: FAIL with missing `run_raw_data_analysis` and `write_raw_data_report`

- [ ] **Step 3: Implement the pipeline result builder**

```python
from skuld_research.raw_data_analysis.anomalies import (
    build_duplicate_flags,
    build_leakage_flags,
    build_numeric_outlier_flags,
)
from skuld_research.raw_data_analysis.coverage import (
    build_dataset_overview,
    build_feature_inventory,
    build_feature_sparsity,
    build_source_inventory,
    build_ticker_sparsity,
)
from skuld_research.raw_data_analysis.dataset import AnalysisDataset
from skuld_research.raw_data_analysis.models import RawDataAnalysisResult
from skuld_research.raw_data_analysis.temporal import build_stale_value_summary, build_temporal_patterns


def run_raw_data_analysis(dataset: AnalysisDataset) -> RawDataAnalysisResult:
    source_inventory = build_source_inventory(dataset)
    feature_inventory = build_feature_inventory(dataset)
    sparsity_by_feature = build_feature_sparsity(dataset)
    return RawDataAnalysisResult(
        dataset_overview=build_dataset_overview(dataset),
        source_inventory=source_inventory,
        feature_inventory=feature_inventory,
        sparsity_by_feature=sparsity_by_feature,
        sparsity_by_ticker=build_ticker_sparsity(dataset),
        temporal_patterns=build_temporal_patterns(dataset),
        stale_value_summary=build_stale_value_summary(dataset),
        anomaly_flags=pd.concat(
            [build_duplicate_flags(dataset), build_numeric_outlier_flags(dataset)],
            ignore_index=True,
            sort=False,
        ),
        leakage_flags=build_leakage_flags(dataset),
    )
```

- [ ] **Step 4: Implement the Markdown/JSON writer with stable artifact names**

```python
import json
from pathlib import Path


def write_raw_data_report(result: RawDataAnalysisResult, out_dir: Path) -> tuple[Path, Path]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    table_map = {
        "dataset_overview.csv": result.dataset_overview,
        "source_inventory.csv": result.source_inventory,
        "feature_inventory.csv": result.feature_inventory,
        "sparsity_by_feature.csv": result.sparsity_by_feature,
        "sparsity_by_ticker.csv": result.sparsity_by_ticker,
        "temporal_patterns.csv": result.temporal_patterns,
        "anomaly_flags.csv": result.anomaly_flags,
        "leakage_flags.csv": result.leakage_flags,
    }
    for file_name, frame in table_map.items():
        frame.to_csv(tables_dir / file_name, index=False)

    report_path = out_dir / "report.md"
    report_lines = [
        "# Raw Data Analysis Report",
        "",
        "## Dataset Overview",
        "",
        result.dataset_overview.to_markdown(index=False),
        "",
        "## Source Inventory",
        "",
        result.source_inventory.to_markdown(index=False),
        "",
        "## Feature Inventory",
        "",
        result.feature_inventory.head(20).to_markdown(index=False),
        "",
        "## Temporal Behavior",
        "",
        result.temporal_patterns.head(20).to_markdown(index=False),
        "",
        "## Outliers and Anomalies",
        "",
        result.anomaly_flags.head(20).to_markdown(index=False),
        "",
        "## Leakage Risk Review",
        "",
        result.leakage_flags.to_markdown(index=False),
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "report_path": str(report_path),
                "table_paths": {name: str(tables_dir / name) for name in table_map},
                "row_count": int(result.dataset_overview.loc[0, "row_count"]),
                "flagged_anomalies": int(len(result.anomaly_flags)),
                "flagged_leakage_features": int(len(result.leakage_flags)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report_path, summary_path
```

- [ ] **Step 5: Run report tests to verify they pass**

Run: `uv run pytest tests/test_raw_data_analysis_report.py -v`
Expected: PASS with `1 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/skuld_research/raw_data_analysis/report.py python/src/skuld_research/raw_data_analysis/pipeline.py python/tests/test_raw_data_analysis_report.py
git commit -m "feat: add raw data analysis report writer"
```

## Task 6: Add the CLI and end-to-end verification

**Files:**
- Create: `python/scripts/raw_data_analysis.py`
- Create: `python/tests/test_raw_data_analysis_cli_e2e.py`

- [ ] **Step 1: Write the failing CLI test**

```python
from pathlib import Path


def test_raw_data_analysis_cli_writes_expected_artifacts(tmp_path: Path, raw_analysis_csv_path: Path):
    from scripts.raw_data_analysis import main

    out_dir = tmp_path / "reports"
    exit_code = main(
        [
            "--data",
            str(raw_analysis_csv_path),
            "--out",
            str(out_dir),
            "--run-date",
            "2026-04-29",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "2026-04-29" / "report.md").exists()
    assert (out_dir / "2026-04-29" / "summary.json").exists()
    assert (out_dir / "2026-04-29" / "tables" / "source_inventory.csv").exists()
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run: `uv run pytest tests/test_raw_data_analysis_cli_e2e.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing `main`

- [ ] **Step 3: Implement the CLI entry point**

```python
import argparse
import sys
from pathlib import Path

from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis
from skuld_research.raw_data_analysis.report import write_raw_data_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skuld raw data analysis workflow")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-date", default="2026-04-29")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        return 1
    dataset = load_analysis_dataset(args.data)
    result = run_raw_data_analysis(dataset)
    out_dir = args.out / args.run_date
    report_path, summary_path = write_raw_data_report(result, out_dir)
    print(f"Report written to {report_path}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI test to verify it passes**

Run: `uv run pytest tests/test_raw_data_analysis_cli_e2e.py -v`
Expected: PASS with `1 passed`

- [ ] **Step 5: Run the focused raw-data-analysis test suite**

Run: `uv run pytest tests/test_raw_data_analysis_dataset.py tests/test_raw_data_analysis_coverage.py tests/test_raw_data_analysis_temporal.py tests/test_raw_data_analysis_anomalies.py tests/test_raw_data_analysis_report.py tests/test_raw_data_analysis_cli_e2e.py -v`
Expected: PASS with all six test files green

- [ ] **Step 6: Run the workflow on the real dataset**

Run: `uv run python scripts/raw_data_analysis.py --data ..\data\data_long.csv --out reports\raw_data_analysis --run-date 2026-04-29`
Expected: exit code 0 and the following files created under `python/reports/raw_data_analysis/2026-04-29/`:
- `report.md`
- `summary.json`
- `tables/source_inventory.csv`
- `tables/feature_inventory.csv`
- `tables/temporal_patterns.csv`
- `tables/anomaly_flags.csv`
- `tables/leakage_flags.csv`

- [ ] **Step 7: Spot-check key outputs against known dataset facts**

Run: `uv run pytest tests/test_real_data_smoke.py::test_load_real_csv -v`
Expected: PASS, confirming the real CSV still loads after adding the new workflow.

Manual checks in generated artifacts:
- `dataset_overview.csv` row count should match the documented dataset size order of magnitude in `docs/DATA_ANALYSIS.md`
- `source_inventory.csv` should include `yf_prices`, `wikimedia_pageviews`, and `yf_finances`
- `leakage_flags.csv` should include fundamental fields from `yf_finances`

- [ ] **Step 8: Commit**

```bash
git add python/scripts/raw_data_analysis.py python/tests/test_raw_data_analysis_cli_e2e.py python/reports/raw_data_analysis
git commit -m "feat: add raw data analysis workflow cli"
```

## Task 7: Document the final workflow usage

**Files:**
- Modify: `docs/DATA_ANALYSIS.md`

- [ ] **Step 1: Write the failing documentation expectation as a checklist comment in the PR or task notes**

```text
The docs must explain how to run the raw data analysis workflow, where artifacts are written, and how agents should use the Markdown report plus CSV/JSON outputs as the raw-data source of truth.
```

- [ ] **Step 2: Add the usage section to `docs/DATA_ANALYSIS.md`**

```markdown
## Raw Data Analysis Workflow

Use the reusable raw-data workflow when you need a current source-of-truth view of `data/data_long.csv`:

```bash
cd python
uv run python scripts/raw_data_analysis.py --data ..\data\data_long.csv --out reports\raw_data_analysis --run-date YYYY-MM-DD
```

Artifacts are written to `python/reports/raw_data_analysis/YYYY-MM-DD/`:

- `report.md` — canonical agent-readable Markdown summary
- `summary.json` — machine-readable top-level metrics and artifact paths
- `tables/*.csv` — detailed coverage, sparsity, temporal, anomaly, and leakage tables
```

- [ ] **Step 3: Run the report and CLI tests again after the docs change**

Run: `uv run pytest tests/test_raw_data_analysis_report.py tests/test_raw_data_analysis_cli_e2e.py -v`
Expected: PASS with both tests green.

- [ ] **Step 4: Commit**

```bash
git add docs/DATA_ANALYSIS.md
git commit -m "docs: add raw data analysis workflow usage"
```

## Parallel sub-agent dispatch recommendation

Use one sub-agent per independent domain after Task 1 lands:

- Coverage agent: Task 2
- Temporal agent: Task 3
- Anomalies agent: Task 4
- Main agent: Tasks 5-7 plus integration review

This split avoids multiple agents editing the same files while still parallelizing the heavy reasoning work.

## Self-review checklist

- Spec coverage: This plan covers the raw dataset loader, source legend mapping, coverage/sparsity metrics, temporal behavior metrics, anomalies/leakage heuristics, deterministic report writing, machine-readable artifact generation, CLI orchestration, real-data execution, and workflow documentation.
- Placeholder scan: No `TODO`, `TBD`, or vague “handle appropriately” instructions remain; each task includes target files, commands, and example code.
- Type consistency: The plan consistently uses `AnalysisDataset`, `RawDataAnalysisResult`, `load_analysis_dataset`, `run_raw_data_analysis`, and `write_raw_data_report` across all tasks.
