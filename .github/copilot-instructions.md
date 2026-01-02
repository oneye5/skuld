# Copilot instructions (skuld)

This repo contains:
- **Python ML pipeline** in `python/ml-pipeline/` (primary day-to-day work).
- **Java ingestion** in `java/` (Maven project).

## CRITICAL: Development Philosophy

**1. Purpose: Improve Ranking Model Performance**
- The goal is to **improve the ranking model's performance** (Sharpe ratio, Precision, etc.) on the validation set.

**2. Research-Driven Development**
- Before writing code, **review research** (university papers, financial ML forums, blogs).
- Look for proven features in similar domains (e.g., "momentum features for mid-cap equities", "handling outliers in financial time series").
- Propose features based on **domain logic**, not random guessing.

**3. Optimize for Iteration Speed (Avoid the 15-minute wait)**
- The full pipeline (`main.py`) takes ~15 minutes. **Do not run it for every small change.**
- **Write Helper Scripts:** Create small scripts in `python/ml-pipeline/output/debug/` or `tests/` to verify your specific transformation on a small DataFrame subset.
- **Unit Tests First:** Write a test for your new feature transformer. If the test passes, the logic is likely correct.
- Only run the full pipeline when you are confident the code works and you need to see the *performance impact*.

**4. Incremental Changes**
- Make **one small change at a time** (e.g., add one feature, change one scaler).
- Measure the direction of performance. Did Sharpe ratio go up? Did Precision improve?
- If it didn't help, revert it before trying the next idea. Keep the codebase clean.

---

## Documentation

### Reference documentation

Read relevant docs before making changes:

| Document | When to read |
|----------|--------------|
| [README.md](../README.md) | Project overview, quick start, key metrics |
| [docs/RANKING_PIPELINE_GUIDE.md](../docs/RANKING_PIPELINE_GUIDE.md) | Pipeline architecture, running evaluation, configuration |
| [docs/FEATURES.md](../docs/FEATURES.md) | Adding or modifying features |
| [docs/CLUSTERING.md](../docs/CLUSTERING.md) | Cluster methodology and leakage-safe workflow |
| [docs/DATA_LEAKAGE.md](../docs/DATA_LEAKAGE.md) | Leakage prevention strategies (read before any data flow changes) |
| [docs/TESTING.md](../docs/TESTING.md) | Test patterns, adding new tests |
| [docs/ANNUAL_STATISTICS.md](../docs/ANNUAL_STATISTICS.md) | Performance metrics and Monte Carlo usage |
| [java/docs/ARCHITECTURE.md](../java/docs/ARCHITECTURE.md) | Java ingestion architecture |
| [java/docs/DATA_SOURCES.md](../java/docs/DATA_SOURCES.md) | Adding or modifying data sources |

### Documentation maintenance

**When to update docs:**
- Adding a new feature category → update `FEATURES.md`
- Changing pipeline stages or config → update `RANKING_PIPELINE_GUIDE.md`
- Adding leakage safeguards → update `DATA_LEAKAGE.md`
- Adding new data sources → update `DATA_SOURCES.md`
- Changing test patterns → update `TESTING.md`

**When to create new docs:**
- New major subsystem that doesn't fit existing docs
- Complex methodology that needs its own reference (like clustering)

**Documentation style:**
- No emojis
- Use ranges instead of exact counts (e.g., "300+ tests" not "386 tests")
- Use ASCII diagrams for architecture/flow visualization
- Keep navigation links at the top of each doc

---

## Python: how to run code (Windows + `uv`)

### Working directory matters
Run Python commands **from** the ML pipeline folder:
- `D:\Projects\StandAloneProjects\skuld\python\ml-pipeline`

Most scripts/tests assume this working directory and rely on local path setup (see `conftest.py` and runnable modules).

### Install/sync dependencies
Use `uv` for environment + dependency management.
- Sync environment (after pulling changes or updating deps):
  - `uv sync`

If you add a dependency:
- Prefer `uv add <package>` (or update `pyproject.toml`), then run `uv sync`.

### Run the pipeline

**Ranking Pipeline (Primary)**:
- `uv run python scripts/run_model_evaluation.py` — Full evaluation with rolling windows (main entry point)

**Debug/Analysis Scripts**:
- `uv run python scripts/debug_anomaly_detection.py` — Debug price anomaly detection
- `uv run python scripts/analyze_quintile_by_year.py` — Analyze performance by year
- `uv run python scripts/profile_*.py` — Performance profiling tools

**Rule:** when Copilot suggests running Python code, prefer `uv run <file.py>` (not `python <file.py>`).

### Common failure modes to check first
- Missing/incorrect raw data file: the loader expects `skuld/data/data_long.csv` by default.
- CWD mismatch: if imports/data paths break, confirm you’re running from `python/ml-pipeline/`.

---

## Python: how to run tests

Tests are configured in `python/ml-pipeline/pyproject.toml` and live under `python/ml-pipeline/tests/`.

Run all tests:
- `uv run pytest`

Run a single test file:
- `uv run pytest tests/test_scaler.py -v`

Run a single test by name/keyword:
- `uv run pytest -k scaler -v`

When making changes:
- Start with the **smallest** relevant test selection.
- Only run the full suite once the targeted tests pass.

---

## Where to find results

**Ranking Pipeline** — After running evaluation:
- `python/ml-pipeline/output/runs/ranking_<timestamp>/metrics.json` — IC, ICIR, Sharpe
- `python/ml-pipeline/output/runs/ranking_<timestamp>/predictions.csv` — All predictions
- `python/ml-pipeline/output/runs/ranking_<timestamp>/quintile_returns.csv` — Returns by quintile
- `python/ml-pipeline/output/runs/ranking_<timestamp>/plots/` — Visualizations

---

## Codebase conventions to respect

- **Modularity:** prefer pure functions and “data in → data out” modules.
- **Centralized config:** constants belong in `python/ml-pipeline/config/*.py`.
- **No leakage:** scalers/feature transforms must be fit on training data only.
- **Validation:** use `core/validation.py` decorators and functions for input checking.
- **Logging:** use `core/logging_config.py` instead of print statements.
- **Experiment tracking:** use `core/experiment_tracking.py` for reproducibility.
- **Tests mirror structure:** keep `tests/` organized to match the module layout.

---

## Key modules

| Module | Purpose |
|--------|---------|
| `core/validation.py` | Data validation decorators, lookahead bias checks |
| `core/experiment_tracking.py` | Experiment manifests, git tracking, comparison |
| `core/logging_config.py` | Structured logging, timing utilities |
| `core/preprocessor.py` | NaN handling, forward fill (sorted by timestamp) |
| `features/cross_sectional.py` | Per-timestamp ranking features |
| `learner/ranking.py` | LGBMRanker wrapper |

---

## Java (secondary)

If working in `java/`, prefer Maven commands from that folder:
- `mvn test`
- `mvn package`

Keep Python and Java changes separate unless the task requires both.
