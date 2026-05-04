# mom-s8 Canonical Metrics Memo

Date: 2026-05-04

## Scope

This memo records the current canonical evaluation state of `python/configs/strategy-specs/candidates/mom-s8.yaml` after reconciling script drift, Sharpe/bootstrap inconsistency, and benchmark rate-unit bugs.

All figures below refer to the canonical `run_from_spec(...)` path, not ad hoc analysis scripts that rebuild config manually.

## Executive Summary

- `mom-s8` currently passes canonical gating.
- The strategy's rolling OOS Sharpe is `0.9057`.
- The 400bps stress-haircut OOS Sharpe is `0.5323`.
- The stationary bootstrap 95% CI for OOS Sharpe is `[0.3520, 1.4750]`.
- The strategy now passes both benchmark-dominance bars:
  - `dominance_NZX equal-weighted`: pass
  - `dominance_60/40`: pass

## Canonical Strategy Metrics

Source: `run_from_spec(spec, raw_csv_path=Path("data/data_long.csv"), write_ledger=False)`

- Rolling OOS Sharpe: `0.905662`
- Rolling OOS Sharpe (flat haircut): `0.532261`
- Rolling OOS Sharpe (delisting adjusted): `0.859...`
- Rolling OOS stationary bootstrap CI: `[0.352000, 1.474986]`

Interpretation:

- The signal remains positive after risk-free adjustment.
- The confidence interval is materially wider than the naive IID bootstrap and should be treated as the canonical uncertainty measure.
- The lower bound remains above zero, but not by a large margin.

## Canonical Benchmarks

### NZ TD floor

- Sharpe: `0.138394`
- Mean annual return: `0.035897`
- Max drawdown: `0.000000`

### NZX equal-weighted

- Sharpe: `-0.124317`
- Mean annual return: `0.027154`
- Max drawdown: `-0.217163`

### 60/40

- Sharpe: `0.187014`
- Mean annual return: `0.047698`
- Max drawdown: `-0.118946`

## Canonical Gating Outcome

Overall: `PASS`

Bars:

- `sanity_floor`: pass — `Sharpe 0.53 > 0.00`
- `bootstrap_ci`: pass — `95% CI low 0.35 > 0`
- `deflated_sharpe`: pass — `p=0.0405 <= 0.05`
- `td_excess_return`: pass — `Mean excess 9.61% > 0, p=0.0005 <= 0.05`
- `dominance_NZX equal-weighted`: pass — `p_adj=0.0015 <= 0.05`
- `dominance_60/40`: pass — `p_adj=0.0065 <= 0.05`

## Corrections Made

### 1. Script/config drift

`scripts/walk_forward_eval.py` had been reconstructing a partial `BacktestConfig` and omitted canonical fields such as execution-policy wiring, ADV panel handling, and related settings. That path reported a stronger OOS Sharpe (`~0.946`) than the canonical runner.

The script now uses the canonical `run_from_spec(...)` path.

### 2. Sharpe/bootstrap inconsistency

The stationary bootstrap had been computed on raw returns while the reported Sharpe subtracted the annual risk-free rate.

This was fixed by adding `rf_annual` handling to `stationary_bootstrap_sharpe(...)` and wiring that through gating and walk-forward reporting.

### 3. Benchmark rate-unit bugs

Both benchmark modules incorrectly treated sub-1.0 rate observations as decimal yields instead of percentage points.

Examples:

- `0.94` in `long_term_interest_rates` was treated as `94%` instead of `0.94%`
- `0.33` in `short_term_interest_rates` was treated as `33%` instead of `0.33%`

This produced:

- impossible 60/40 monthly returns below `-100%`
- inflated TD-floor returns
- incorrect dominance conclusions

The rate normalization logic in both benchmark modules has been corrected.

## Current Trust Level

Reasonable conclusion:

- The strategy has a credible positive OOS signal under the current methodology.
- The benchmark pass is now materially more trustworthy than before the measurement fixes.
- The point estimate should still not be treated as a live-performance promise.

Conservative reading:

- Use the haircut Sharpe (`0.53`) and lower CI region (`~0.35`) as the practical anchor, not the point estimate (`0.91`).

## Verification Status

Focused regression coverage passed after the fixes:

- `python/tests/test_benchmarks_sixty_forty.py`
- `python/tests/test_benchmarks_td_floor.py`
- `python/tests/test_stats_bootstrap.py`
- `python/tests/test_stats_gating.py`

Result: `33 passed`

## Open Caveats

- The raw macro ingest still logs duplicate `(date, feature)` collisions; current behavior is last-row-wins.
- The strategy remains sensitive to small-market regime shifts and thin-universe structure.
- This memo records the current canonical numbers; it does not replace a full production methodology report.
