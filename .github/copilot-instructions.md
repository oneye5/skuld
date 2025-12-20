# Copilot instructions (skuld)

This repo contains:
- **Python ML pipeline** in `python/ml-pipeline/` (primary day-to-day work).
- **Java ingestion** in `java/` (Maven project).

When working in this workspace, optimize for an **iterative, agentic workflow**:
1. **Reproduce** the current behavior (run the smallest command/test that shows the issue).
2. **Write/adjust tests first** when changing behavior (TDD is preferred).
3. Make the **smallest safe change**, then re-run the same test/command.
4. Expand to a broader test run only after the focused check passes.
5. If something fails, **surface the exact error**, locate the root cause, and iterate.

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
