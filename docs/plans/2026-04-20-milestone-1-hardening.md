# Milestone 1 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Python workspace under `python/`, extend PITSnapshot validation to all frames, fix detect_gaps performance, add public API re-exports.

**Architecture:** Phase 1 moves files (all subsequent paths change). Phase 2 has four independent tasks: validation, performance, re-exports, test updates.

**Tech Stack:** Python 3.11, uv workspace, pandas, numpy, pytest

**Spec:** `docs/specs/2026-04-20-milestone-1-hardening.md`

---

## File Structure

```
skuld/
├── python/                          # NEW — uv workspace root
│   ├── pyproject.toml               # MOVED from root
│   ├── uv.lock                      # MOVED from root
│   ├── common/                      # MOVED from root
│   │   ├── pyproject.toml
│   │   └── src/skuld_common/
│   │       ├── __init__.py          # MODIFIED (re-exports)
│   │       └── contracts.py         # MODIFIED (extended validation)
│   ├── research/                    # MOVED from root
│   │   ├── pyproject.toml
│   │   └── src/skuld_research/
│   │       ├── __init__.py
│   │       └── data/
│   │           ├── __init__.py      # MODIFIED (re-exports)
│   │           ├── csv_loader.py
│   │           ├── validation.py    # MODIFIED (perf fix)
│   │           └── pit_loader.py
│   ├── portfolio/                   # MOVED from root
│   │   ├── pyproject.toml
│   │   └── src/skuld_portfolio/
│   │       └── __init__.py
│   └── tests/                       # MOVED from root
│       ├── conftest.py
│       ├── test_contracts.py        # MODIFIED (new validation tests)
│       ├── test_csv_loader.py
│       ├── test_pit_loader.py
│       ├── test_real_data_smoke.py  # MODIFIED (path fix)
│       └── test_validation.py
├── data/                            # STAYS
├── docs/                            # STAYS
├── java/                            # STAYS
└── sample_data.txt                  # STAYS
```

---

### Task 1: Move Python workspace under `python/`

**Files:**
- Move: `pyproject.toml` → `python/pyproject.toml`
- Move: `uv.lock` → `python/uv.lock`
- Move: `common/` → `python/common/`
- Move: `research/` → `python/research/`
- Move: `portfolio/` → `python/portfolio/`
- Move: `tests/` → `python/tests/`
- Modify: `python/tests/test_real_data_smoke.py` (fix DATA_PATH)

- [ ] **Step 1: Create python/ directory and move files**

```powershell
cd d:\Projects\StandAloneProjects\skuld
New-Item -ItemType Directory -Path python -Force
Move-Item pyproject.toml python/
Move-Item uv.lock python/
Move-Item common python/
Move-Item research python/
Move-Item portfolio python/
Move-Item tests python/
```

- [ ] **Step 2: Move .venv if it exists (or recreate later)**

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
```

- [ ] **Step 3: Fix DATA_PATH in smoke test**

In `python/tests/test_real_data_smoke.py`, change:
```python
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data_long.csv"
```
to:
```python
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "data_long.csv"
```

- [ ] **Step 4: Recreate venv and verify**

```powershell
cd d:\Projects\StandAloneProjects\skuld\python
uv sync --all-packages
uv run pytest tests/ -v --tb=short
```

Expected: 23 passed

- [ ] **Step 5: Commit**

```powershell
cd d:\Projects\StandAloneProjects\skuld
git add -A
git commit -m "refactor: move Python workspace under python/ directory"
```

---

### Task 2: Extend PITSnapshot validation

**Files:**
- Modify: `python/common/src/skuld_common/contracts.py`
- Modify: `python/tests/test_contracts.py`

- [ ] **Step 1: Write failing tests for new validation**

Add to `python/tests/test_contracts.py`:
```python
def test_pit_snapshot_rejects_future_volumes():
    """PITSnapshot raises if volumes contain dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(
                {"ANZ.NZ": [100_000.0, 120_000.0, 130_000.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "volumes" in str(e).lower()


def test_pit_snapshot_rejects_future_macro():
    """PITSnapshot raises if macro contains dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(
                {"oecd_bcicp": [100.5, 101.0]},
                index=pd.to_datetime(["2025-01-14", "2025-01-15"]),
            ),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "macro" in str(e).lower()


def test_pit_snapshot_rejects_future_fundamentals():
    """PITSnapshot raises if fundamentals have publication_date >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(
                {"annual_net_income_common_stockholders": [1_000_000.0]},
                index=pd.MultiIndex.from_tuples(
                    [("ANZ.NZ", pd.Timestamp("2025-01-15"))],
                    names=["ticker", "publication_date"],
                ),
            ),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "fundamentals" in str(e).lower()


def test_pit_snapshot_rejects_future_corporate_actions():
    """PITSnapshot raises if corporate_actions have ex_date >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(
                {
                    "ticker": ["ANZ.NZ"],
                    "ex_date": [pd.Timestamp("2025-01-15")],
                    "type": ["dividend"],
                    "factor": [0.50],
                }
            ),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "corporate_actions" in str(e).lower()


def test_pit_snapshot_reports_all_violations():
    """PITSnapshot reports all violations in a single error, not just the first."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(
                {"ANZ.NZ": [50.0, 51.0, 52.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            volumes=pd.DataFrame(
                {"ANZ.NZ": [100_000.0, 120_000.0, 130_000.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        msg = str(e).lower()
        assert "prices" in msg and "volumes" in msg, (
            f"Expected both 'prices' and 'volumes' in error, got: {e}"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && uv run pytest tests/test_contracts.py -v`
Expected: 5 new tests FAIL

- [ ] **Step 3: Implement extended validation**

Replace `__post_init__` in `python/common/src/skuld_common/contracts.py`:
```python
    def __post_init__(self) -> None:
        asof_naive = self.asof.tz_localize(None) if self.asof.tzinfo else self.asof
        violations: list[str] = []

        # Check index-based frames: prices, volumes, macro
        for name, df in [("prices", self.prices), ("volumes", self.volumes), ("macro", self.macro)]:
            if not df.empty and len(df.index) > 0:
                max_date = pd.Timestamp(df.index.max())
                if max_date.tzinfo:
                    max_date = max_date.tz_localize(None)
                if max_date >= asof_naive:
                    violations.append(
                        f"{name}: max date {max_date} >= asof {self.asof}"
                    )

        # Check fundamentals (MultiIndex with publication_date level)
        if not self.fundamentals.empty and len(self.fundamentals.index) > 0:
            pub_dates = self.fundamentals.index.get_level_values("publication_date")
            max_pub = pd.Timestamp(pub_dates.max())
            if max_pub.tzinfo:
                max_pub = max_pub.tz_localize(None)
            if max_pub >= asof_naive:
                violations.append(
                    f"fundamentals: max publication_date {max_pub} >= asof {self.asof}"
                )

        # Check corporate_actions (ex_date column)
        if not self.corporate_actions.empty and "ex_date" in self.corporate_actions.columns:
            max_ex = pd.Timestamp(self.corporate_actions["ex_date"].max())
            if max_ex.tzinfo:
                max_ex = max_ex.tz_localize(None)
            if max_ex >= asof_naive:
                violations.append(
                    f"corporate_actions: max ex_date {max_ex} >= asof {self.asof}"
                )

        if violations:
            raise ValueError(
                f"PIT invariant violated — no future data allowed.\n"
                + "\n".join(f"  - {v}" for v in violations)
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && uv run pytest tests/test_contracts.py -v`
Expected: 7 passed (2 old + 5 new)

- [ ] **Step 5: Run full suite**

Run: `cd python && uv run pytest tests/ -v`
Expected: 28 passed

- [ ] **Step 6: Commit**

```bash
git add python/common/src/skuld_common/contracts.py python/tests/test_contracts.py
git commit -m "feat: extend PITSnapshot validation to all temporal frames"
```

---

### Task 3: Fix `detect_gaps` performance

**Files:**
- Modify: `python/research/src/skuld_research/data/validation.py`

- [ ] **Step 1: Replace bdate_range loop with np.busday_count**

In `python/research/src/skuld_research/data/validation.py`, add `import numpy as np` at top and replace the `detect_gaps` function body:

```python
import numpy as np
```

Replace the gap detection loop:
```python
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
        dates = pd.DatetimeIndex(series.index).sort_values()
        dates_arr = dates.values.astype("datetime64[D]")
        bday_gaps = np.busday_count(dates_arr[:-1], dates_arr[1:])
        large_idx = np.where(bday_gaps > max_gap_days)[0]
        if len(large_idx) > 0:
            report.details[ticker] = [
                f"{dates[i].strftime('%Y-%m-%d')} → {dates[i + 1].strftime('%Y-%m-%d')} ({bday_gaps[i]} bdays)"
                for i in large_idx
            ]
            report.issue_count += len(large_idx)
    return report
```

- [ ] **Step 2: Run validation tests**

Run: `cd python && uv run pytest tests/test_validation.py -v`
Expected: 6 passed

- [ ] **Step 3: Commit**

```bash
git add python/research/src/skuld_research/data/validation.py
git commit -m "perf: vectorize detect_gaps with np.busday_count"
```

---

### Task 4: Public API re-exports

**Files:**
- Modify: `python/common/src/skuld_common/__init__.py`
- Modify: `python/research/src/skuld_research/data/__init__.py`

- [ ] **Step 1: Add re-exports to skuld_common**

`python/common/src/skuld_common/__init__.py`:
```python
"""Skuld common types and contracts."""

from skuld_common.contracts import PITSnapshot

__all__ = ["PITSnapshot"]
```

- [ ] **Step 2: Add re-exports to skuld_research.data**

`python/research/src/skuld_research/data/__init__.py`:
```python
"""Data loading, validation, and point-in-time filtering."""

from skuld_research.data.csv_loader import RawData, load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.validation import (
    ValidationReport,
    detect_gaps,
    detect_negative_prices,
    detect_stale_sources,
)

__all__ = [
    "RawData",
    "load_raw_csv",
    "PITLoader",
    "ValidationReport",
    "detect_gaps",
    "detect_negative_prices",
    "detect_stale_sources",
]
```

- [ ] **Step 3: Verify imports work**

Run:
```powershell
cd python
uv run python -c "from skuld_common import PITSnapshot; print('PITSnapshot:', PITSnapshot)"
uv run python -c "from skuld_research.data import RawData, load_raw_csv, PITLoader; print('OK')"
```

- [ ] **Step 4: Run full suite**

Run: `cd python && uv run pytest tests/ -v`
Expected: 28 passed

- [ ] **Step 5: Commit**

```bash
git add python/common/src/skuld_common/__init__.py python/research/src/skuld_research/data/__init__.py
git commit -m "feat: add public API re-exports for skuld_common and skuld_research.data"
```
