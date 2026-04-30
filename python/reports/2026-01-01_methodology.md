# Methodology Report

**Config hash:** 4dd5177fe965a6a703fe04c9ad0ec24e30b2eab6f00876657004a930b74ef489
**Git SHA:** 8858478-dirty
**As-of date:** 2026-01-01
**Panel coverage:** 1970-01-03 to 2025-12-31
**Master seed:** 42
**Prior trials:** 30

### RNG Master Seed Derivation

Child RNG seeds derived via numpy.random.SeedSequence(master_seed).spawn(4):
1. bootstrap (stationary bootstrap for Sharpe CI)
2. mc_delisting (Monte Carlo delisting simulation for survivorship adjustment)
3. optimiser_tiebreak (portfolio optimizer tie-breaking when multiple solutions exist)
4. dominance (Romano-Wolf stepwise procedure bootstrap resampling)

Changing this order is a breaking change to reproducibility.

## Strategy

**Name:** m8-mom

### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Strategy | -0.302 | -0.765 | -0.360 | -27.83% | -27.83% | -41.72% | 11.01% | 9024 | 34.84% | 2.054 | -0.094 | 1 | 1 |

### Rolling Driver (Gating Reference)

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Strategy | -0.133 | -0.516 | -0.181 | -21.06% | -21.06% | -29.04% | 17.17% | 9553 | 50.51% | 1.700 | -0.066 | 9 | 3 |

### Per-Regime Sharpe (Rolling Driver)

- **bull:** -0.050
- **chop:** -0.331

## Benchmarks

### NZ TD floor

**Coverage:** 1970-01-30 to 2025-12-31

**Notes:**
- Notional term deposit, 4.0% default floor

#### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZ TD floor | 1.166 | 1.166 | 1.166 | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 100.00% | 4.114 | 0.000 | 2 | 0 |

#### Rolling Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZ TD floor | 1.207 | 1.207 | 1.207 | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 100.00% | 2.710 | 0.000 | 12 | 0 |

### NZX equal-weighted

**Coverage:** 2000-02-29 to 2025-12-31

**Notes:**
- 20M NZD mcap floor
- ADV filter not implemented

#### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZX equal-weighted | -0.483 | -0.483 | -0.483 | -30.50% | -30.50% | -30.50% | 0.00% | 0 | 25.72% | 1.163 | -0.089 | 2 | 0 |

#### Rolling Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZX equal-weighted | 0.070 | 0.070 | 0.070 | -21.57% | -21.57% | -21.57% | 0.00% | 0 | 55.56% | 0.473 | 0.027 | 12 | 0 |

### 60/40

**Coverage:** 2008-01-31 to 2025-12-31

**Notes:**
- Bond rates auto-converted from percentage to decimal
- yield-only bond proxy (no duration P&L)
- flat 50bps annual haircut applied uniformly

#### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60/40 | 0.195 | 0.195 | 0.195 | -13.27% | -13.27% | -13.27% | 0.00% | 0 | 64.94% | -1.068 | 0.097 | 2 | 0 |

#### Rolling Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60/40 | 0.728 | 0.728 | 0.728 | -4.34% | -4.34% | -4.34% | 0.00% | 0 | 68.93% | 0.001 | 0.865 | 12 | 0 |

## Dominance (Romano-Wolf Stepwise)

| Benchmark | Adjusted p-value | Dominates |
|---|---|---|
| NZ TD floor | 0.9965 | False |
| NZX equal-weighted | 0.9710 | False |
| 60/40 | 0.9900 | False |

## Gating Decision

**Overall:** FAIL

### Bars

- **sanity_floor:** ✗ — Sharpe -0.52 ≤ 0.00
- **bootstrap_ci:** ✗ — 95% CI low -0.50 ≤ 0
- **deflated_sharpe:** ✗ — p=0.9993 > 0.05
- **dominance_NZ TD floor:** ✗ — p_adj=0.9965 > 0.05
- **dominance_NZX equal-weighted:** ✗ — p_adj=0.9710 > 0.05
- **dominance_60/40:** ✗ — p_adj=0.9900 > 0.05

**Notes:** Kept folds: 9; Rejected folds: 3; n_trials: 39 (prior: 30, ledger: 9); Per-regime Sharpe: bull=-0.05, chop=-0.33

## Pass / Fail

- **Sanity floor (TD floor):** ✗ FAIL — Strategy Sharpe -0.181 ≤ TD floor Sharpe 1.207
- **Primary benchmark (NZX equal-weighted) via Romano-Wolf:** ✗ FAIL — Dominates=False, p_adj=0.9710 > 0.05
- **Deflated Sharpe:** ✗ FAIL — Deflated Sharpe p=0.9993 > 0.05

### Rejected Folds

- fold 0: 100% empty months
- fold 1: 100% empty months
- fold 2: 100% empty months
