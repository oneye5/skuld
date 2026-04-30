# Strategy Specs Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename preregistered strategy YAMLs into a clearer lifecycle-based `strategy-specs` layout with concise hyphenated names.

**Architecture:** Keep the existing `BacktestSpec` schema and runner behavior. Change file organization, discovery, references, and tests so strategy specs are discovered recursively under `python/configs/strategy-specs/`.

**Tech Stack:** Python, Pydantic, PyYAML, pytest, Markdown docs.

---

## File Map

- Create `python/configs/strategy-specs/README.md` to document lifecycle policy and inventory.
- Move YAML files from `python/configs/preregistered/` to `python/configs/strategy-specs/{archive,candidates,production}/`.
- Modify YAML `name:` values to match filename stems using hyphens.
- Modify `python/src/skuld_research/config/loader.py` to recursively discover strategy specs and keep a compatibility function name.
- Modify CLI examples/help in `python/scripts/backtest.py` and `python/scripts/recommend.py`.
- Modify tests that reference exact spec paths or names.
- Modify docs/reports references that point to old paths/names where they remain current documentation.

## Tasks

### Task 1: Move Specs And Rename Internal Names

**Files:**
- Move: `python/configs/preregistered/2026-04-26_momentum_only.yaml` -> `python/configs/strategy-specs/archive/m8-mom.yaml`
- Move: `python/configs/preregistered/2026-04-26_momentum_low_vol.yaml` -> `python/configs/strategy-specs/archive/m8-mom-lowvol.yaml`
- Move: `python/configs/preregistered/2026-04-26_momentum_size.yaml` -> `python/configs/strategy-specs/archive/m8-mom-size.yaml`
- Move: `python/configs/preregistered/2026-04-26_momentum_low_vol_size.yaml` -> `python/configs/strategy-specs/archive/m8-mom-lowvol-size.yaml`
- Move: `python/configs/preregistered/2026-04-26_phase1_baseline.yaml` -> `python/configs/strategy-specs/archive/mom-baseline.yaml`
- Move: `python/configs/preregistered/2026-04-26_phase1_baseline_with_overlay.yaml` -> `python/configs/strategy-specs/archive/mom-overlay.yaml`
- Move: `python/configs/preregistered/2026-04-26_momentum_smoothed3.yaml` -> `python/configs/strategy-specs/candidates/mom-smooth3.yaml`
- Move: `python/configs/preregistered/2026-04-26_momentum_smoothed3_quarterly.yaml` -> `python/configs/strategy-specs/candidates/mom-smooth3-quarterly.yaml`
- Move: `python/configs/preregistered/2026-04-27_phase1_no_mcap.yaml` -> `python/configs/strategy-specs/candidates/mom-no-mcap.yaml`
- Move: `python/configs/preregistered/2026-04-27_phase1_ar_spread.yaml` -> `python/configs/strategy-specs/candidates/mom-ar-spread.yaml`

- [ ] Create `python/configs/strategy-specs/archive`, `python/configs/strategy-specs/candidates`, and `python/configs/strategy-specs/production`.
- [ ] Move each YAML to its target path.
- [ ] Update each YAML `name:` to the target filename stem.
- [ ] Replace comments that say `Read-only — modifications are rejected by tests/test_preregistered_immutability.py` with `Read-only — create a new strategy spec instead of editing historical specs.`

Expected name mapping:

```text
momentum_only -> m8-mom
momentum_low_vol -> m8-mom-lowvol
momentum_size -> m8-mom-size
momentum_low_vol_size -> m8-mom-lowvol-size
phase1_baseline -> mom-baseline
phase1_baseline_with_overlay -> mom-overlay
momentum_smoothed3 -> mom-smooth3
momentum_smoothed3_quarterly -> mom-smooth3-quarterly
phase1_no_mcap -> mom-no-mcap
phase1_ar_spread -> mom-ar-spread
```

### Task 2: Document Strategy Spec Lifecycle

**Files:**
- Create: `python/configs/strategy-specs/README.md`

- [ ] Add a README explaining `archive/`, `candidates/`, and `production/`.
- [ ] Include the current inventory and why each spec lives where it does.
- [ ] Include the archive/promotion policy from `docs/specs/2026-04-30-strategy-specs-rename.md`.

### Task 3: Update Loader Discovery

**Files:**
- Modify: `python/src/skuld_research/config/loader.py`
- Modify: `python/tests/test_config_loader.py`
- Modify: `python/tests/test_preregistered_immutability.py`

- [ ] Change `iter_preregistered_specs()` to search recursively under `configs/strategy-specs/`.
- [ ] Update docstrings to say strategy specs rather than preregistered specs.
- [ ] Keep the function name for compatibility unless a broader public API rename is needed.
- [ ] Update immutability test path from `python/configs/preregistered/` to `python/configs/strategy-specs/`.
- [ ] Update config loader tests to expect `m8-mom` and recursive paths.

### Task 4: Update Scripts And Tests For New Paths

**Files:**
- Modify: `python/scripts/backtest.py`
- Modify: `python/scripts/recommend.py`
- Modify: `python/scripts/_m8_evaluate.py`
- Modify: `python/tests/test_overlay_spec_hash_compat.py`
- Modify: `python/tests/test_write_recommendations_csv.py`
- Modify: `python/tests/test_backtest_cli_e2e.py`

- [ ] Replace old example paths with new `configs/strategy-specs/...` paths.
- [ ] Update M8 evaluation paths to use archive specs.
- [ ] Update test spec paths to new locations.
- [ ] Update expected hash values only after running the hash test once to get the new actual value.

### Task 5: Update Documentation References

**Files:**
- Modify: `docs/specs/2026-04-30-phase1-dominance-diagnostic.md`
- Modify: `.github/copilot-instructions.md`
- Modify if needed: generated reports under `python/reports/` if tests or docs treat them as current references.

- [ ] Replace old subject path with `configs/strategy-specs/candidates/mom-ar-spread.yaml`.
- [ ] Replace old strategy name `phase1_ar_spread` with `mom-ar-spread` where referring to the renamed spec.
- [ ] Avoid rewriting historical result tables unless they are expected by tests.

### Task 6: Verify And Fix Fallout

**Files:**
- Any failing test file caused by stale paths/names/hashes.

- [ ] Run `uv run pytest tests/test_config_loader.py tests/test_preregistered_immutability.py tests/test_overlay_spec_hash_compat.py tests/test_write_recommendations_csv.py tests/test_backtest_cli_e2e.py` from `python/`.
- [ ] If hash-lock tests fail only because `name:` changed intentionally, update the expected hash to the new actual hash.
- [ ] Run `uv run pytest -m "not slow"` from `python/`.
- [ ] Run `uv run ruff check .` from `python/`.

## Self-Review

- Spec coverage: the plan covers new layout, hyphenated names, lifecycle README, recursive discovery, path/reference updates, and tests.
- Placeholder scan: no placeholders remain.
- Type consistency: no schema changes are required; path discovery remains behind existing loader APIs.
