# Tier 2 Performance Barriers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Tier 2 fixes from `docs/specs/2026-05-02-performance-barriers-deep-analysis.md`: data-layer anomaly masking, enforced ADV participation caps, TD gate replacement, and a duration-aware 60/40 benchmark.

**Architecture:** Keep the fixes small and local to the existing pipeline. Extend the spec and prepared-panel path for anomaly filtering, turn the optimizer's existing optional liquidity cap into a real engine input, split TD evaluation from Romano-Wolf dominance into a dedicated HAC excess-return gate, and make the 60/40 benchmark configurable between yield-only and duration-aware bond returns.

**Tech Stack:** Python, pandas, pydantic, pytest, scipy, existing Skuld backtest / benchmark / gating modules.

---

### Task 1: Data-Layer Anomaly Masking

**Files:**
- Modify: `python/src/skuld_research/config/spec.py`
- Modify: `python/src/skuld_research/config/hashing.py`
- Modify: `python/src/skuld_common/contracts.py`
- Modify: `python/src/skuld_research/data/prepared_panel.py`
- Test: `python/tests/test_prepared_panel.py`
- Test: `python/tests/test_config_spec.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_prepared_panel_masks_extreme_one_sided_daily_move_without_volume_confirmation():
    ...

def test_prepared_panel_masks_extreme_monthly_move_without_corporate_action():
    ...

def test_anomaly_filter_spec_round_trip_and_hash_neutral_when_disabled():
    ...
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `uv run pytest tests/test_prepared_panel.py tests/test_config_spec.py -k anomaly`
Expected: FAIL because the spec and prepared-panel pipeline do not yet expose or apply the anomaly mask.

- [ ] **Step 3: Add the minimal spec + panel implementation**

```python
class AnomalyFilterSpec(BaseModel):
    kind: Literal["none", "mask_extremes"] = "none"
    daily_abs_return_threshold: float = 2.0
    monthly_abs_return_threshold: float = 5.0
    volume_gate_threshold: float = 0.20
    require_volume_confirmation: bool = True
    corporate_action_buffer_days: int = 5


def _apply_anomaly_mask(...):
    ...  # mask one-sided extreme daily/monthly moves to NaN
```

- [ ] **Step 4: Thread the filtered prices through `build_prepared_panel`**

```python
filtered_prices = _apply_anomaly_mask(
    prices_daily,
    volumes_daily,
    snap.corporate_actions,
    anomaly_filter,
)
returns_daily = filtered_prices.pct_change(fill_method=None)
```

- [ ] **Step 5: Re-run the targeted tests to verify GREEN**

Run: `uv run pytest tests/test_prepared_panel.py tests/test_config_spec.py -k anomaly`
Expected: PASS.

- [ ] **Step 6: Run broader verification for the touched area**

Run: `uv run pytest tests/test_prepared_panel.py tests/test_config_spec.py tests/test_runner_run_from_spec.py`

### Task 2: Enforce ADV Participation Caps In The Backtest

**Files:**
- Modify: `python/src/skuld_research/config/spec.py`
- Modify: `python/src/skuld_research/backtest/engine.py`
- Modify: `python/src/skuld_research/portfolio/optimizer.py`
- Modify: `python/src/skuld_research/config/runner.py`
- Test: `python/tests/test_portfolio_constructor.py`
- Test: `python/tests/test_backtest_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_target_portfolio_caps_weight_by_adv_and_nav():
    ...

def test_backtest_engine_passes_rebalance_adv_into_optimizer():
    ...
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `uv run pytest tests/test_portfolio_constructor.py tests/test_backtest_engine.py -k adv`
Expected: FAIL because the engine does not yet pass rebalance-date ADV and the cap is not configurable.

- [ ] **Step 3: Add the minimal implementation**

```python
class BacktestEngineSpec(BaseModel):
    adv_participation_cap: float | None = 0.01


adv_at_t = _lookup_adv(panel, t)
target = build_target_portfolio(
    ...,
    adv=adv_at_t,
    portfolio_nav=nav,
    adv_participation_cap=cfg.adv_participation_cap,
)
```

- [ ] **Step 4: Keep the cap local to the optimizer**

```python
if adv is not None and portfolio_nav is not None and adv_participation_cap is not None:
    liq_cap = adv_participation_cap * adv[ticker] / portfolio_nav
```

- [ ] **Step 5: Re-run the targeted tests to verify GREEN**

Run: `uv run pytest tests/test_portfolio_constructor.py tests/test_backtest_engine.py -k adv`
Expected: PASS.

- [ ] **Step 6: Run broader verification for portfolio construction**

Run: `uv run pytest tests/test_portfolio_constructor.py tests/test_backtest_engine.py tests/test_runner_run_from_spec.py`

### Task 3: Replace TD Dominance With A HAC Excess-Return Gate

**Files:**
- Modify: `python/src/skuld_common/contracts.py`
- Modify: `python/src/skuld_research/config/spec.py`
- Add: `python/src/skuld_research/stats/excess_return.py`
- Modify: `python/src/skuld_research/stats/gating.py`
- Modify: `python/src/skuld_research/config/runner.py`
- Modify: `python/src/skuld_research/reporting/report_builder.py`
- Test: `python/tests/test_stats_gating.py`
- Test: `python/tests/test_reporting_builder.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_td_gate_uses_one_sided_excess_return_test_not_romano_wolf():
    ...

def test_report_builder_uses_td_excess_return_bar():
    ...
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `uv run pytest tests/test_stats_gating.py tests/test_reporting_builder.py -k td`
Expected: FAIL because gating only exposes TD through Romano-Wolf and reporting still compares Sharpe-vs-Sharpe.

- [ ] **Step 3: Add the minimal statistical helper**

```python
@dataclass(frozen=True)
class ExcessReturnTestResult:
    mean_excess_annual: float
    t_stat: float
    p_value: float
    passes: bool


def one_sided_hac_excess_return(strategy: pd.Series, benchmark: pd.Series, alpha: float = 0.05) -> ExcessReturnTestResult:
    ...
```

- [ ] **Step 4: Replace only the TD gate, keep Romano-Wolf for the risky benchmarks**

```python
td_result = one_sided_hac_excess_return(oos_returns, td_returns, alpha=alpha)
bars["td_excess_return"] = (...)
dominance_benchmarks = {k: v for k, v in benchmarks.items() if k != "NZ TD floor"}
```

- [ ] **Step 5: Update reporting to read the new gate instead of a Sharpe comparison**

```python
td_passed, td_reason = gating.bars["td_excess_return"]
pass_fail_bars.append(("Sanity floor (TD excess return)", td_passed, td_reason))
```

- [ ] **Step 6: Re-run the targeted tests to verify GREEN**

Run: `uv run pytest tests/test_stats_gating.py tests/test_reporting_builder.py -k td`
Expected: PASS.

- [ ] **Step 7: Run broader gating verification**

Run: `uv run pytest tests/test_stats_gating.py tests/test_reporting_builder.py tests/test_runner_run_from_spec.py`

### Task 4: Add Duration-Aware 60/40 Benchmark Support

**Files:**
- Modify: `python/src/skuld_research/config/spec.py`
- Modify: `python/src/skuld_research/benchmarks/sixty_forty.py`
- Modify: `python/src/skuld_research/config/runner.py`
- Test: `python/tests/test_benchmarks_sixty_forty.py`
- Test: `python/tests/test_runner_run_from_spec.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_sixty_forty_duration_mode_adds_price_return_from_yield_changes():
    ...

def test_sixty_forty_yield_only_mode_preserves_existing_behavior():
    ...
```

- [ ] **Step 2: Run the targeted tests to verify RED**

Run: `uv run pytest tests/test_benchmarks_sixty_forty.py -k duration`
Expected: FAIL because the benchmark only supports yield-only bond returns.

- [ ] **Step 3: Add minimal config + benchmark support**

```python
class BenchmarksSpec(BaseModel):
    sixty_forty_bond_duration_years: float = 0.0


bond_returns_monthly = coupon_carry - duration_years * bond_yield_change
```

- [ ] **Step 4: Preserve backward compatibility when duration is zero**

```python
if duration_years <= 0:
    bond_returns_monthly = (1.0 + bond_rates_monthly) ** (1.0 / 12.0) - 1.0
else:
    bond_returns_monthly = _duration_aware_bond_returns(...)
```

- [ ] **Step 5: Re-run the targeted tests to verify GREEN**

Run: `uv run pytest tests/test_benchmarks_sixty_forty.py -k duration`
Expected: PASS.

- [ ] **Step 6: Run broader benchmark verification**

Run: `uv run pytest tests/test_benchmarks_sixty_forty.py tests/test_runner_run_from_spec.py`

### Final Verification

**Files:**
- Verify touched code and tests only.

- [ ] **Step 1: Run the full targeted Tier 2 verification suite**

Run: `uv run pytest tests/test_prepared_panel.py tests/test_config_spec.py tests/test_portfolio_constructor.py tests/test_backtest_engine.py tests/test_stats_gating.py tests/test_reporting_builder.py tests/test_benchmarks_sixty_forty.py tests/test_runner_run_from_spec.py`

- [ ] **Step 2: Run lint for touched modules**

Run: `uv run ruff check src tests`

- [ ] **Step 3: Re-read the Tier 2 requirements and confirm coverage**

Checklist:

```text
- Multi-day winsorize / volume-gated price acceptance implemented in data layer
- ADV participation cap enforced in optimizer / engine path
- TD dominance replaced by one-sided excess-return gate
- 60/40 benchmark supports duration P&L
```
