# Milestone 1 Hardening: Validation, Performance, API Surface, Directory Layout

**Date:** 2026-04-20
**Status:** Approved
**Scope:** Post-M1 improvements before starting Milestone 2

---

## 1. Directory Restructuring

Move all Python workspace files under `python/` so the repo root has clear top-level concerns.

**Before:**
```
skuld/
├── common/          ← Python package
├── data/            ← shared data
├── docs/
├── java/
├── portfolio/       ← Python package
├── pyproject.toml   ← Python workspace root
├── research/        ← Python package
├── sample_data.txt
├── tests/           ← Python tests
└── uv.lock
```

**After:**
```
skuld/
├── .agents/
├── data/            ← shared (Java writes, Python reads)
├── docs/
├── java/
├── python/          ← uv workspace root
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── common/
│   ├── research/
│   ├── portfolio/
│   └── tests/
└── sample_data.txt
```

**Path fixes required:**
- `tests/test_real_data_smoke.py`: `DATA_PATH` changes from `parent.parent / "data"` to `parent.parent.parent / "data"` (one more level up)
- `.venv/` recreated inside `python/`

---

## 2. PITSnapshot Validation (Extend `__post_init__`)

Currently only `prices` is validated for future data. Extend to all temporal frames.

**Checks to add:**
| Frame | Check |
|---|---|
| volumes | `index.max() < asof_naive` |
| macro | `index.max() < asof_naive` |
| fundamentals | `publication_date` level max `< asof_naive` |
| corporate_actions | `ex_date` column max `< asof_naive` |

**Behaviour:** Collect all violations, raise a single `ValueError` listing them all. Not fail-fast.

**Dataclass decision:** Keep frozen dataclass. No Pydantic. Construction overhead matters in backtest loops.

---

## 3. `detect_gaps` Performance Fix

**Problem:** Current implementation calls `pd.bdate_range(dates[i], dates[i+1], inclusive="neither")` in a Python loop for each pair of consecutive dates per ticker. O(n) Python-level calls with expensive bdate_range construction.

**Fix:** Replace with `np.busday_count(dates_arr[:-1], dates_arr[1:])` — vectorized C-level business-day counting across the entire date series at once.

```python
import numpy as np

dates_arr = series.index.values.astype("datetime64[D]")
bday_gaps = np.busday_count(dates_arr[:-1], dates_arr[1:])
large_gaps = np.where(bday_gaps > max_gap_days)[0]
```

---

## 4. Public API Re-exports

**`skuld_common/__init__.py`:**
```python
from skuld_common.contracts import PITSnapshot
```

**`skuld_research/data/__init__.py`:**
```python
from skuld_research.data.csv_loader import RawData, load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.validation import (
    ValidationReport,
    detect_negative_prices,
    detect_gaps,
    detect_stale_sources,
)
```

Other `__init__.py` files unchanged (no public API yet).

---

## 5. Test Updates

- Existing tests must pass after all changes (23 tests).
- Add tests for the new validation checks in `PITSnapshot`:
  - volumes with future dates → ValueError
  - macro with future dates → ValueError
  - fundamentals with future publication_date → ValueError
  - corporate_actions with future ex_date → ValueError
  - Multiple violations reported in a single error message
