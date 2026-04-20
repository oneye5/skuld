# Milestone 1: Python Project Skeleton + Data Contracts + PIT Loader

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `uv` workspace with three Python packages (`common`, `research`, `portfolio`), implement the data contract types, CSV loader, validation layer, and point-in-time loader — all test-first.

**Architecture:** Long-format CSV (`data/data_long.csv`, 4.7M rows) → raw CSV loader → validation → PIT-filtered `PITSnapshot`. The PIT loader enforces `timestamp < asof` so downstream consumers never see future data. Contracts live in `skuld-common`; loader lives in `skuld-research`. `skuld-portfolio` is scaffolded but empty.

**Tech Stack:** Python 3.11, `uv` workspace, `pandas`, `pytest`, `ruff`, `pyright`

**Source data format (for reference):**
```
timestamp,ticker,feature,value,src
1613037600000,NTL.NZ,high,0.04812400043010712,6
```
- `timestamp`: Unix epoch milliseconds (UTC)
- `ticker`: `XXX.NZ` for NZX equities, empty string for macro, `%5ETNX`/`%5EFTSE`/`ZS=F` for international
- `feature`: snake_case metric name. Price features (src=6): `open`, `high`, `low`, `close`, `adj_close`, `volume`, `dividend`, `split`
- `value`: numeric string
- `src`: integer source ID (0–13; see `data/source_legend.csv`)

---

## File Structure

```
skuld/
├── pyproject.toml                          # workspace root
├── common/
│   ├── pyproject.toml
│   └── src/skuld_common/
│       ├── __init__.py
│       └── contracts.py                    # PITSnapshot dataclass
├── research/
│   ├── pyproject.toml
│   └── src/skuld_research/
│       ├── __init__.py
│       └── data/
│           ├── __init__.py
│           ├── csv_loader.py               # raw long-CSV → typed DataFrames
│           ├── validation.py               # negative prices, gaps, staleness
│           └── pit_loader.py               # as_of(t) → PITSnapshot
├── portfolio/
│   ├── pyproject.toml
│   └── src/skuld_portfolio/
│       └── __init__.py
└── tests/
    ├── conftest.py                         # shared fixtures (synthetic data)
    ├── test_contracts.py
    ├── test_csv_loader.py
    ├── test_validation.py
    └── test_pit_loader.py
```

Tests live in a top-level `tests/` directory (not per-package) because the loader tests need to import from both `skuld_common` and `skuld_research`. The `research` package's `pyproject.toml` declares `pytest` as a dev dependency and the test directory is configured in the root `pyproject.toml`.

---

### Task 1: Workspace Scaffolding

**Files:**
- Create: `pyproject.toml` (workspace root)
- Create: `common/pyproject.toml`
- Create: `common/src/skuld_common/__init__.py`
- Create: `research/pyproject.toml`
- Create: `research/src/skuld_research/__init__.py`
- Create: `research/src/skuld_research/data/__init__.py`
- Create: `portfolio/pyproject.toml`
- Create: `portfolio/src/skuld_portfolio/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create root `pyproject.toml`**

```toml
# pyproject.toml
[project]
name = "skuld"
version = "0.1.0"
requires-python = ">=3.11"

[tool.uv.workspace]
members = ["common", "research", "portfolio"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]

[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "basic"
```

- [ ] **Step 2: Create `common/pyproject.toml`**

```toml
# common/pyproject.toml
[project]
name = "skuld-common"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/skuld_common"]
```

- [ ] **Step 3: Create `research/pyproject.toml`**

```toml
# research/pyproject.toml
[project]
name = "skuld-research"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "skuld-common",
    "pandas>=2.0",
    "numpy>=1.24",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "pyright>=1.1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/skuld_research"]

[tool.uv.sources]
skuld-common = { workspace = true }
```

- [ ] **Step 4: Create `portfolio/pyproject.toml`**

```toml
# portfolio/pyproject.toml
[project]
name = "skuld-portfolio"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "skuld-common",
    "pandas>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/skuld_portfolio"]

[tool.uv.sources]
skuld-common = { workspace = true }
```

- [ ] **Step 5: Create `__init__.py` files and empty test conftest**

`common/src/skuld_common/__init__.py`:
```python
"""Skuld common types and contracts."""
```

`research/src/skuld_research/__init__.py`:
```python
"""Skuld research — backtesting, walk-forward, factor models."""
```

`research/src/skuld_research/data/__init__.py`:
```python
"""Data loading, validation, and point-in-time filtering."""
```

`portfolio/src/skuld_portfolio/__init__.py`:
```python
"""Skuld portfolio — recommendation generator."""
```

`tests/conftest.py`:
```python
"""Shared test fixtures for Skuld."""
```

- [ ] **Step 6: Lock dependencies and verify workspace**

Run:
```
cd d:\Projects\StandAloneProjects\skuld
uv sync
uv run python -c "import skuld_common; import skuld_research; print('imports OK')"
```
Expected: `uv sync` succeeds, imports print `imports OK`.

- [ ] **Step 7: Commit**

```
git add pyproject.toml common/ research/ portfolio/ tests/
git commit -m "feat(m1): uv workspace scaffolding with common, research, portfolio packages"
```

---

### Task 2: Data Contracts (`PITSnapshot`)

**Files:**
- Create: `common/src/skuld_common/contracts.py`
- Create: `tests/test_contracts.py`

- [ ] **Step 1: Write the failing test**

`tests/test_contracts.py`:
```python
"""Tests for PITSnapshot contract type."""

import pandas as pd

from skuld_common.contracts import PITSnapshot


def test_pit_snapshot_construction():
    """PITSnapshot can be constructed with valid DataFrames."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    snap = PITSnapshot(
        prices=pd.DataFrame(
            {"ANZ.NZ": [50.0, 51.0]},
            index=pd.to_datetime(["2025-01-13", "2025-01-14"]),
        ),
        volumes=pd.DataFrame(
            {"ANZ.NZ": [100_000.0, 120_000.0]},
            index=pd.to_datetime(["2025-01-13", "2025-01-14"]),
        ),
        fundamentals=pd.DataFrame(
            {"annual_net_income_common_stockholders": [1_000_000.0]},
            index=pd.MultiIndex.from_tuples(
                [("ANZ.NZ", pd.Timestamp("2024-06-30"))],
                names=["ticker", "publication_date"],
            ),
        ),
        macro=pd.DataFrame(
            {"oecd_bcicp": [100.5]},
            index=pd.to_datetime(["2025-01-10"]),
        ),
        corporate_actions=pd.DataFrame(
            {
                "ticker": ["ANZ.NZ"],
                "ex_date": [pd.Timestamp("2024-12-01")],
                "type": ["dividend"],
                "factor": [0.50],
            }
        ),
        asof=ts,
    )
    assert snap.asof == ts
    assert list(snap.prices.columns) == ["ANZ.NZ"]
    assert snap.prices.shape == (2, 1)


def test_pit_snapshot_rejects_future_prices():
    """PITSnapshot raises if prices contain dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    future_prices = pd.DataFrame(
        {"ANZ.NZ": [50.0, 51.0, 52.0]},
        index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
    )
    try:
        PITSnapshot(
            prices=future_prices,
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "future" in str(e).lower() or "asof" in str(e).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'skuld_common.contracts'`

- [ ] **Step 3: Write the implementation**

`common/src/skuld_common/contracts.py`:
```python
"""Core data contract types for Skuld pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PITSnapshot:
    """All values knowable strictly before `asof`. Enforced, not asked nicely.

    Attributes:
        prices: index=date, columns=ticker, values=adj_close
        volumes: index=date, columns=ticker, values=volume
        fundamentals: MultiIndex (ticker, publication_date), columns=feature
        macro: index=date, columns=macro_feature
        corporate_actions: columns: ticker, ex_date, type, factor
        asof: the timestamp this snapshot was built for
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    fundamentals: pd.DataFrame
    macro: pd.DataFrame
    corporate_actions: pd.DataFrame
    asof: pd.Timestamp

    def __post_init__(self) -> None:
        asof_naive = self.asof.tz_localize(None) if self.asof.tzinfo else self.asof
        if not self.prices.empty and len(self.prices.index) > 0:
            max_date = pd.Timestamp(self.prices.index.max())
            if max_date.tzinfo:
                max_date = max_date.tz_localize(None)
            if max_date >= asof_naive:
                raise ValueError(
                    f"prices contain dates >= asof ({self.asof}). "
                    f"Max price date: {max_date}. "
                    f"PIT invariant violated: no future data allowed."
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```
git add common/src/skuld_common/contracts.py tests/test_contracts.py
git commit -m "feat(m1): PITSnapshot contract with no-future-data invariant"
```

---

### Task 3: Synthetic Test Fixtures

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Build shared fixtures**

`tests/conftest.py`:
```python
"""Shared test fixtures for Skuld.

All fixtures produce synthetic data — no real CSV files needed.
Timestamps are chosen to exercise edge cases (future data, gaps, negatives).
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

# -- Synthetic CSV content -----------------------------------------------

# 10 rows: 5 price features for 2 tickers over 2 days, plus 1 future row,
# 1 negative-price row, 1 fundamental, 1 macro, 1 dividend, 1 split.
# asof will be 2025-01-15 00:00 UTC = 1736899200000 ms
ASOF_MS = 1736899200000  # 2025-01-15T00:00:00Z
ASOF_TS = pd.Timestamp("2025-01-15", tz="UTC")

# 2025-01-13 = 1736726400000, 2025-01-14 = 1736812800000
DAY_13_MS = 1736726400000
DAY_14_MS = 1736812800000
# 2025-01-15 = 1736899200000 (== asof, should be excluded)
DAY_15_MS = 1736899200000
# 2024-06-30 = 1719705600000 (fundamental period-end)
FUND_MS = 1719705600000

SYNTHETIC_CSV = f"""\
timestamp,ticker,feature,value,src
{DAY_13_MS},ANZ.NZ,adj_close,50.0,6
{DAY_13_MS},ANZ.NZ,volume,100000,6
{DAY_14_MS},ANZ.NZ,adj_close,51.0,6
{DAY_14_MS},ANZ.NZ,volume,120000,6
{DAY_13_MS},SPK.NZ,adj_close,4.80,6
{DAY_13_MS},SPK.NZ,volume,200000,6
{DAY_14_MS},SPK.NZ,adj_close,4.85,6
{DAY_14_MS},SPK.NZ,volume,210000,6
{DAY_15_MS},ANZ.NZ,adj_close,52.0,6
{DAY_15_MS},ANZ.NZ,volume,130000,6
{DAY_13_MS},ANZ.NZ,adj_close,-5.0,6
{FUND_MS},ANZ.NZ,annual_net_income_common_stockholders,1000000,12
{DAY_13_MS},,oecd_bcicp,100.5,10
{DAY_14_MS},ANZ.NZ,dividend,0.50,6
{DAY_13_MS},SPK.NZ,split,2.0,6
"""


@pytest.fixture
def synthetic_csv_path(tmp_path: Path) -> Path:
    """Write synthetic CSV to a temp file and return its path."""
    csv_file = tmp_path / "data_long.csv"
    csv_file.write_text(SYNTHETIC_CSV)
    return csv_file


@pytest.fixture
def synthetic_csv_io() -> io.StringIO:
    """Return synthetic CSV as a StringIO for in-memory tests."""
    return io.StringIO(SYNTHETIC_CSV)


@pytest.fixture
def asof_timestamp() -> pd.Timestamp:
    return ASOF_TS
```

- [ ] **Step 2: Verify fixtures are loadable**

Run: `uv run pytest tests/test_contracts.py -v --co`
Expected: collects existing tests without error (fixtures are just available, not used yet).

- [ ] **Step 3: Commit**

```
git add tests/conftest.py
git commit -m "feat(m1): synthetic test fixtures for CSV loader and PIT tests"
```

---

### Task 4: Raw CSV Loader

**Files:**
- Create: `research/src/skuld_research/data/csv_loader.py`
- Create: `tests/test_csv_loader.py`

The CSV loader reads the long-format CSV and splits it into typed DataFrames by source and feature, **without any PIT filtering**. That's the next layer's job.

- [ ] **Step 1: Write failing tests**

`tests/test_csv_loader.py`:
```python
"""Tests for the raw CSV loader."""

from pathlib import Path

import pandas as pd

from skuld_research.data.csv_loader import RawData, load_raw_csv


def test_load_returns_raw_data(synthetic_csv_path: Path):
    """load_raw_csv returns a RawData with prices, volumes, etc."""
    raw = load_raw_csv(synthetic_csv_path)
    assert isinstance(raw, RawData)


def test_prices_extracted(synthetic_csv_path: Path):
    """Prices DataFrame has adj_close pivoted to ticker columns."""
    raw = load_raw_csv(synthetic_csv_path)
    # 3 adj_close rows for ANZ.NZ (day13, day14, day15), 2 for SPK.NZ
    # Negative-price row is a second adj_close for ANZ day13 — loader keeps it,
    # validation removes it later.
    assert "ANZ.NZ" in raw.prices.columns
    assert "SPK.NZ" in raw.prices.columns
    assert raw.prices.index.name == "date"


def test_volumes_extracted(synthetic_csv_path: Path):
    """Volumes DataFrame has volume pivoted to ticker columns."""
    raw = load_raw_csv(synthetic_csv_path)
    assert "ANZ.NZ" in raw.volumes.columns
    assert raw.volumes["ANZ.NZ"].iloc[0] > 0


def test_fundamentals_extracted(synthetic_csv_path: Path):
    """Fundamentals have a MultiIndex of (ticker, publication_date)."""
    raw = load_raw_csv(synthetic_csv_path)
    assert raw.fundamentals.index.names == ["ticker", "publication_date"]
    assert "annual_net_income_common_stockholders" in raw.fundamentals.columns


def test_macro_extracted(synthetic_csv_path: Path):
    """Macro data has date index and feature columns."""
    raw = load_raw_csv(synthetic_csv_path)
    assert "oecd_bcicp" in raw.macro.columns


def test_corporate_actions_extracted(synthetic_csv_path: Path):
    """Corporate actions include dividends and splits."""
    raw = load_raw_csv(synthetic_csv_path)
    assert len(raw.corporate_actions) == 2  # 1 dividend + 1 split
    types = set(raw.corporate_actions["type"])
    assert types == {"dividend", "split"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_csv_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'skuld_research.data.csv_loader'`

- [ ] **Step 3: Implement the CSV loader**

`research/src/skuld_research/data/csv_loader.py`:
```python
"""Load the long-format CSV into typed DataFrames, split by source/feature.

No PIT filtering here — that's pit_loader's job. This module only parses,
pivots, and categorises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Source IDs (from data/source_legend.csv)
SRC_PRICES = 6
SRC_FUNDAMENTALS = 12

# Price features that become per-ticker columns
PRICE_FEATURE = "adj_close"
VOLUME_FEATURE = "volume"
CORPORATE_ACTION_FEATURES = {"dividend", "split"}


@dataclass
class RawData:
    """All data from the CSV, categorised but unfiltered."""

    prices: pd.DataFrame  # index=date, columns=ticker, values=adj_close
    volumes: pd.DataFrame  # index=date, columns=ticker, values=volume
    fundamentals: pd.DataFrame  # MultiIndex (ticker, publication_date), columns=feature
    macro: pd.DataFrame  # index=date, columns=feature
    corporate_actions: pd.DataFrame  # columns: ticker, ex_date, type, factor


def load_raw_csv(path: Path) -> RawData:
    """Load long-format CSV and split into categorised DataFrames.

    Args:
        path: Path to data_long.csv

    Returns:
        RawData with all observations categorised.
    """
    df = pd.read_csv(
        path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str, "src": "int8"},
    )
    # Fill NaN tickers (macro rows) with empty string
    df["ticker"] = df["ticker"].fillna("")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)

    prices = _pivot_price_feature(df, PRICE_FEATURE)
    volumes = _pivot_price_feature(df, VOLUME_FEATURE)
    fundamentals = _build_fundamentals(df)
    macro = _build_macro(df)
    corporate_actions = _build_corporate_actions(df)

    return RawData(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=macro,
        corporate_actions=corporate_actions,
    )


def _pivot_price_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Pivot a single price-source feature into date × ticker."""
    mask = (df["src"] == SRC_PRICES) & (df["feature"] == feature) & (df["ticker"] != "")
    subset = df.loc[mask, ["date", "ticker", "value"]]
    if subset.empty:
        return pd.DataFrame()
    pivoted = subset.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    pivoted.index.name = "date"
    pivoted = pivoted.sort_index()
    return pivoted


def _build_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Build fundamentals with MultiIndex (ticker, publication_date)."""
    mask = (df["src"] == SRC_FUNDAMENTALS) & (df["ticker"] != "")
    subset = df.loc[mask, ["ticker", "date", "feature", "value"]]
    if subset.empty:
        return pd.DataFrame(
            columns=pd.Index([], dtype=str),
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"]),
        )
    pivoted = subset.pivot_table(
        index=["ticker", "date"], columns="feature", values="value", aggfunc="last"
    )
    pivoted.index = pivoted.index.set_names(["ticker", "publication_date"])
    return pivoted


def _build_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Build macro DataFrame: date × feature for rows with empty ticker."""
    mask = df["ticker"] == ""
    subset = df.loc[mask, ["date", "feature", "value"]]
    if subset.empty:
        return pd.DataFrame()
    pivoted = subset.pivot_table(index="date", columns="feature", values="value", aggfunc="last")
    pivoted.index.name = "date"
    pivoted = pivoted.sort_index()
    return pivoted


def _build_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Extract dividend and split rows into a flat DataFrame."""
    mask = (df["src"] == SRC_PRICES) & (df["feature"].isin(CORPORATE_ACTION_FEATURES))
    subset = df.loc[mask, ["ticker", "date", "feature", "value"]].copy()
    subset = subset.rename(columns={"date": "ex_date", "feature": "type", "value": "factor"})
    subset = subset.reset_index(drop=True)
    return subset
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_csv_loader.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```
git add research/src/skuld_research/data/csv_loader.py tests/test_csv_loader.py
git commit -m "feat(m1): raw CSV loader — parse long-format into typed DataFrames"
```

---

### Task 5: Data Validation Layer

**Files:**
- Create: `research/src/skuld_research/data/validation.py`
- Create: `tests/test_validation.py`

Implements: negative-price detection, zero-volume handling, gap detection (>5 consecutive missing trading days), stale-data alerts. All functions take DataFrames and return validation reports — they don't mutate data.

- [ ] **Step 1: Write failing tests**

`tests/test_validation.py`:
```python
"""Tests for the data validation layer."""

import pandas as pd
import numpy as np

from skuld_research.data.validation import (
    ValidationReport,
    detect_negative_prices,
    detect_gaps,
    detect_stale_sources,
)


def test_detect_negative_prices_finds_them():
    """Negative prices are detected and reported."""
    prices = pd.DataFrame(
        {"ANZ.NZ": [50.0, -5.0, 51.0], "SPK.NZ": [4.8, 4.9, -1.0]},
        index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
    )
    report = detect_negative_prices(prices)
    assert report.issue_count == 2
    assert "ANZ.NZ" in report.details
    assert "SPK.NZ" in report.details


def test_detect_negative_prices_clean_data():
    """No negatives → zero issues."""
    prices = pd.DataFrame(
        {"ANZ.NZ": [50.0, 51.0]},
        index=pd.to_datetime(["2025-01-13", "2025-01-14"]),
    )
    report = detect_negative_prices(prices)
    assert report.issue_count == 0


def test_detect_gaps_finds_large_gap():
    """A gap of >5 trading days is detected."""
    # 10 trading days, then skip 6 trading days, then resume
    dates_before = pd.bdate_range("2025-01-02", periods=10)
    dates_after = pd.bdate_range("2025-01-22", periods=5)
    all_dates = dates_before.append(dates_after)
    prices = pd.DataFrame(
        {"ANZ.NZ": range(len(all_dates))},
        index=all_dates,
        dtype=float,
    )
    report = detect_gaps(prices, max_gap_days=5)
    assert report.issue_count >= 1
    assert "ANZ.NZ" in report.details


def test_detect_gaps_no_gap():
    """Consecutive trading days → no gap."""
    dates = pd.bdate_range("2025-01-02", periods=20)
    prices = pd.DataFrame({"ANZ.NZ": range(20)}, index=dates, dtype=float)
    report = detect_gaps(prices, max_gap_days=5)
    assert report.issue_count == 0


def test_detect_stale_sources():
    """Source whose latest timestamp is older than threshold is flagged."""
    # Source data: last observation 60 days ago, threshold 7 days
    now = pd.Timestamp("2025-03-01")
    source_latest = {"yf_prices": pd.Timestamp("2025-01-01"), "nz_gdp": pd.Timestamp("2025-02-25")}
    report = detect_stale_sources(source_latest, as_of=now, max_age_days=7)
    assert report.issue_count == 1
    assert "yf_prices" in report.details


def test_detect_stale_sources_all_fresh():
    """All sources within threshold → no issues."""
    now = pd.Timestamp("2025-03-01")
    source_latest = {"yf_prices": pd.Timestamp("2025-02-28"), "nz_gdp": pd.Timestamp("2025-02-25")}
    report = detect_stale_sources(source_latest, as_of=now, max_age_days=7)
    assert report.issue_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'skuld_research.data.validation'`

- [ ] **Step 3: Implement validation**

`research/src/skuld_research/data/validation.py`:
```python
"""Data validation utilities.

All functions inspect data and return reports — they never mutate the input.
Consumers decide what to do with the report (log, raise, exclude rows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    """Result of a single validation check."""

    check_name: str
    issue_count: int = 0
    details: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return self.issue_count == 0


def detect_negative_prices(prices: pd.DataFrame) -> ValidationReport:
    """Flag any negative values in a prices DataFrame.

    Args:
        prices: index=date, columns=ticker, values=price

    Returns:
        Report listing affected (ticker, date) pairs.
    """
    report = ValidationReport(check_name="negative_prices")
    for ticker in prices.columns:
        neg_mask = prices[ticker] < 0
        if neg_mask.any():
            neg_dates = prices.index[neg_mask].strftime("%Y-%m-%d").tolist()
            report.details[ticker] = neg_dates
            report.issue_count += int(neg_mask.sum())
    return report


def detect_gaps(
    prices: pd.DataFrame, max_gap_days: int = 5
) -> ValidationReport:
    """Flag tickers with gaps of >max_gap_days consecutive business days.

    Args:
        prices: index=date (sorted), columns=ticker
        max_gap_days: threshold for gap detection

    Returns:
        Report listing affected tickers and gap periods.
    """
    report = ValidationReport(check_name="price_gaps")
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 2:
            continue
        # Compute business-day differences between consecutive observations
        dates = pd.DatetimeIndex(series.index).sort_values()
        bday_diffs = [
            len(pd.bdate_range(dates[i], dates[i + 1], inclusive="neither"))
            for i in range(len(dates) - 1)
        ]
        gaps = [
            (dates[i], dates[i + 1], bday_diffs[i])
            for i, d in enumerate(bday_diffs)
            if d > max_gap_days
        ]
        if gaps:
            report.details[ticker] = [
                f"{g[0].strftime('%Y-%m-%d')} → {g[1].strftime('%Y-%m-%d')} ({g[2]} bdays)"
                for g in gaps
            ]
            report.issue_count += len(gaps)
    return report


def detect_stale_sources(
    source_latest: dict[str, pd.Timestamp],
    as_of: pd.Timestamp,
    max_age_days: int = 7,
) -> ValidationReport:
    """Flag sources whose latest data is older than threshold.

    Args:
        source_latest: mapping of source name → latest observation timestamp
        as_of: reference date (typically today)
        max_age_days: days before a source is considered stale

    Returns:
        Report listing stale sources with their age.
    """
    report = ValidationReport(check_name="stale_sources")
    for source, latest in source_latest.items():
        age = (as_of - latest).days
        if age > max_age_days:
            report.details[source] = [f"last data: {latest.strftime('%Y-%m-%d')}, age: {age} days"]
            report.issue_count += 1
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```
git add research/src/skuld_research/data/validation.py tests/test_validation.py
git commit -m "feat(m1): validation layer — negative prices, gaps, staleness"
```

---

### Task 6: PIT Loader

**Files:**
- Create: `research/src/skuld_research/data/pit_loader.py`
- Create: `tests/test_pit_loader.py`

The PIT loader takes the `RawData` from Task 4 and filters it to produce a `PITSnapshot` where every observation is strictly before `asof`.

- [ ] **Step 1: Write failing tests**

`tests/test_pit_loader.py`:
```python
"""Tests for the point-in-time loader.

The critical invariant: as_of(t) never returns data with timestamps >= t.
"""

from pathlib import Path

import pandas as pd

from skuld_research.data.csv_loader import load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_common.contracts import PITSnapshot
from tests.conftest import ASOF_TS, DAY_13_MS, DAY_14_MS


def test_pit_loader_returns_snapshot(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """as_of returns a PITSnapshot."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    assert isinstance(snap, PITSnapshot)
    assert snap.asof == asof_timestamp


def test_future_data_excluded(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Row at t=asof (2025-01-15) must NOT appear in the snapshot.

    The synthetic CSV has an ANZ.NZ adj_close=52.0 at 2025-01-15T00:00Z
    which equals asof. The PIT invariant is strictly-before, so this row
    must be absent.
    """
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    # Prices should have only day13 and day14 for ANZ.NZ, not day15
    max_price_date = snap.prices.index.max()
    assert max_price_date < asof_timestamp.tz_localize(None), (
        f"PIT violation: max price date {max_price_date} >= asof {asof_timestamp}"
    )


def test_negative_prices_excluded(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Negative price rows are filtered out of the snapshot."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    assert (snap.prices >= 0).all().all(), "Negative prices found in PIT snapshot"


def test_fundamentals_filtered(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Fundamentals with publication_date >= asof are excluded."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    if not snap.fundamentals.empty:
        pub_dates = snap.fundamentals.index.get_level_values("publication_date")
        asof_naive = asof_timestamp.tz_localize(None)
        assert (pub_dates < asof_naive).all(), "Fundamental data at or after asof found"


def test_macro_filtered(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Macro data with date >= asof is excluded."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    if not snap.macro.empty:
        asof_naive = asof_timestamp.tz_localize(None)
        assert (snap.macro.index < asof_naive).all(), "Macro data at or after asof found"


def test_corporate_actions_filtered(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Corporate actions with ex_date >= asof are excluded."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    if not snap.corporate_actions.empty:
        asof_naive = asof_timestamp.tz_localize(None)
        assert (snap.corporate_actions["ex_date"] < asof_naive).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pit_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'skuld_research.data.pit_loader'`

- [ ] **Step 3: Implement the PIT loader**

`research/src/skuld_research/data/pit_loader.py`:
```python
"""Point-in-time loader.

Wraps RawData and produces PITSnapshot instances filtered to strictly-before
a given timestamp. This is the central anti-lookahead control.
"""

from __future__ import annotations

import pandas as pd

from skuld_common.contracts import PITSnapshot
from skuld_research.data.csv_loader import RawData
from skuld_research.data.validation import detect_negative_prices


class PITLoader:
    """Produces point-in-time snapshots from raw data.

    Usage:
        raw = load_raw_csv(path)
        loader = PITLoader(raw)
        snap = loader.as_of(pd.Timestamp("2025-01-15", tz="UTC"))
    """

    def __init__(self, raw: RawData) -> None:
        self._raw = raw

    def as_of(self, t: pd.Timestamp) -> PITSnapshot:
        """Return all data knowable strictly before `t`.

        Args:
            t: The as-of timestamp. Must be timezone-aware (UTC).

        Returns:
            PITSnapshot with all frames filtered to dates < t.
        """
        t_naive = t.tz_localize(None) if t.tzinfo else t

        prices = self._filter_by_index(self._raw.prices, t_naive)
        prices = self._remove_negative_prices(prices)
        volumes = self._filter_by_index(self._raw.volumes, t_naive)
        fundamentals = self._filter_fundamentals(self._raw.fundamentals, t_naive)
        macro = self._filter_by_index(self._raw.macro, t_naive)
        corporate_actions = self._filter_corporate_actions(
            self._raw.corporate_actions, t_naive
        )

        return PITSnapshot(
            prices=prices,
            volumes=volumes,
            fundamentals=fundamentals,
            macro=macro,
            corporate_actions=corporate_actions,
            asof=t,
        )

    @staticmethod
    def _filter_by_index(df: pd.DataFrame, t_naive: pd.Timestamp) -> pd.DataFrame:
        """Keep only rows where index < t_naive."""
        if df.empty:
            return df
        return df.loc[df.index < t_naive]

    @staticmethod
    def _remove_negative_prices(prices: pd.DataFrame) -> pd.DataFrame:
        """Replace negative prices with NaN, then drop all-NaN rows."""
        if prices.empty:
            return prices
        cleaned = prices.where(prices >= 0)
        return cleaned.dropna(how="all")

    @staticmethod
    def _filter_fundamentals(df: pd.DataFrame, t_naive: pd.Timestamp) -> pd.DataFrame:
        """Keep fundamentals where publication_date < t_naive."""
        if df.empty:
            return df
        pub_dates = df.index.get_level_values("publication_date")
        mask = pub_dates < t_naive
        return df.loc[mask]

    @staticmethod
    def _filter_corporate_actions(df: pd.DataFrame, t_naive: pd.Timestamp) -> pd.DataFrame:
        """Keep corporate actions where ex_date < t_naive."""
        if df.empty:
            return df
        return df.loc[df["ex_date"] < t_naive].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pit_loader.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```
git add research/src/skuld_research/data/pit_loader.py tests/test_pit_loader.py
git commit -m "feat(m1): PIT loader — as_of(t) with strict no-lookahead invariant"
```

---

### Task 7: Full Test Suite + Validation Smoke Test Against Real Data

**Files:**
- Create: `tests/test_real_data_smoke.py`

This is a smoke test that runs against the actual `data/data_long.csv`. It is **not** a unit test — it validates that the loader and validation layer work on real data. Skipped if the CSV is absent (CI-safe).

- [ ] **Step 1: Write the smoke test**

`tests/test_real_data_smoke.py`:
```python
"""Smoke tests against real data_long.csv.

Skipped if the file is absent — these are local-only validation checks.
"""

from pathlib import Path

import pandas as pd
import pytest

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data_long.csv"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="data/data_long.csv not present"
)


def test_load_real_csv():
    """Real CSV loads without crashing."""
    from skuld_research.data.csv_loader import load_raw_csv

    raw = load_raw_csv(DATA_PATH)
    assert not raw.prices.empty, "No price data loaded"
    assert len(raw.prices.columns) > 100, f"Only {len(raw.prices.columns)} tickers loaded"


def test_pit_loader_on_real_data():
    """PIT snapshot of real data at a recent date returns data."""
    from skuld_research.data.csv_loader import load_raw_csv
    from skuld_research.data.pit_loader import PITLoader

    raw = load_raw_csv(DATA_PATH)
    loader = PITLoader(raw)
    snap = loader.as_of(pd.Timestamp("2025-06-01", tz="UTC"))
    assert not snap.prices.empty
    # PIT invariant: no price after 2025-06-01
    max_date = snap.prices.index.max()
    assert max_date < pd.Timestamp("2025-06-01"), f"PIT violation: {max_date}"


def test_negative_price_detection_real():
    """Run negative-price validation on real data and report."""
    from skuld_research.data.csv_loader import load_raw_csv
    from skuld_research.data.validation import detect_negative_prices

    raw = load_raw_csv(DATA_PATH)
    report = detect_negative_prices(raw.prices)
    # We don't assert zero issues — we just confirm the check runs.
    # Print results for human review.
    if not report.is_clean:
        print(f"\nWARNING: {report.issue_count} negative price observations found:")
        for ticker, dates in report.details.items():
            print(f"  {ticker}: {dates[:5]}{'...' if len(dates) > 5 else ''}")
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (smoke tests may take a few seconds on 4.7M rows).

- [ ] **Step 3: Commit**

```
git add tests/test_real_data_smoke.py
git commit -m "feat(m1): smoke tests against real data_long.csv"
```

---

## Done-When Checklist

Per the implementation plan's Milestone 1 definition:

- [ ] A synthetic-data fixture proves a row dated `t+1` cannot appear in `loader.as_of(t)` → `test_future_data_excluded` in Task 6
- [ ] `uv run pytest` passes from a clean checkout → Task 7 full suite
- [ ] The validation layer flags a synthetic negative-price row → `test_detect_negative_prices_finds_them` in Task 5
- [ ] `uv` workspace with `common/`, `research/`, `portfolio/` packages → Task 1
