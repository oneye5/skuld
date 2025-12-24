# Copilot instructions (skuld)

This repo contains:
- **Python ML pipeline** in `python/ml-pipeline/` (primary day-to-day work).
- **Java ingestion** in `java/` (Maven project).

## 🚨 CRITICAL: Development Philosophy 🚨

**1. THE MODEL IS FROZEN**
- **Do NOT change the model architecture (XGBoost/CatBoost/etc) or hyperparameters.**
- **Do NOT change the target definition** unless explicitly instructed.
- We assume the current model config is "good enough" to detect signal if the data is good.
- **Goal:** Improve performance solely through **Feature Engineering, Data Cleaning, and Scaling**.

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

## Canonical docs to read first

The project blueprint is in:
- [python/ml-pipeline/documentation/README.md](../python/ml-pipeline/documentation/README.md)

Before changing pipeline semantics, review that doc’s sections on:
- **Data format** (`skuld/data/data_long.csv`) and long→wide conversion rules
- **Leakage prevention** (fit scalers on training only; rolling-window isolation)
- **Macro vs ticker handling** via the `MACRO_` prefix
- **Rolling window** approach and model config defaults

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
The main entrypoint is:
- `uv run .\main.py`

Other runnable entrypoints live in `python/ml-pipeline/runnables/`.

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
- `uv run pytest .\tests\data-preparation\labeling\test_labeler.py -v`

Run a single test by name/keyword:
- `uv run pytest -k labeler -v`

When making changes:
- Start with the **smallest** relevant test selection.
- Only run the full suite once the targeted tests pass.

---

## Where to find results

After a full run (`main.py`), check `python/ml-pipeline/output/runs/<timestamp>/`:
1. **`evaluation/metrics.json`**: High-level stats (Precision, Recall, F1, AUC).
2. **`evaluation/trades.csv`**: Simulated PnL. Check the Sharpe Ratio and Max Drawdown.
3. **`config.json`**: Verifies what config was used.

---

## Codebase conventions to respect

- **Modularity:** prefer pure functions and “data in → data out” modules.
- **Centralized config:** constants belong in `python/ml-pipeline/config/*.py`.
- **No leakage:** scalers/feature transforms must be fit on training data only.
- **Tests mirror structure:** keep `tests/` organized to match the module layout.

Note: some directories use hyphens (e.g. `data-preparation/`). Imports are handled via explicit path setup; if you add new import roots, ensure tests/runnables can still import them cleanly.

---

## Java (secondary)

If working in `java/`, prefer Maven commands from that folder:
- `mvn test`
- `mvn package`

Keep Python and Java changes separate unless the task requires both.
