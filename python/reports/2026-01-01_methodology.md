# Methodology Report

**Config hash:** 4dd5177fe965a6a703fe04c9ad0ec24e30b2eab6f00876657004a930b74ef489
**Git SHA:** acdc0a7-dirty
**As-of date:** 2026-01-01
**Panel coverage:** 1970-01-02 to 2025-12-31
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
| Strategy | -0.116 | -0.563 | -0.172 | -17.45% | -17.45% | -33.21% | 7.84% | 6957 | 27.10% | 6.965 | -0.059 | 1 | 1 |

### Rolling Driver (Gating Reference)

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Strategy | 0.066 | -0.299 | 0.021 | -11.20% | -11.20% | -22.21% | 12.02% | 6975 | 39.39% | 5.681 | 0.065 | 9 | 3 |

### Per-Regime Sharpe (Rolling Driver)

- **bull:** 0.303
- **chop:** -0.635

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
- 10,000 share ADV floor

#### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZX equal-weighted | -0.287 | -0.287 | -0.287 | -28.65% | -28.65% | -28.65% | 0.00% | 0 | 27.33% | 1.260 | -0.057 | 2 | 0 |

#### Rolling Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZX equal-weighted | 0.351 | 0.351 | 0.351 | -15.43% | -15.43% | -15.43% | 0.00% | 0 | 59.03% | 0.440 | 0.188 | 12 | 0 |

### 60/40

**Coverage:** 2008-01-31 to 2025-12-31

**Notes:**
- Bond rates auto-converted from percentage to decimal
- yield-only bond proxy (no duration P&L)
- flat 50bps annual haircut applied uniformly

#### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60/40 | 0.389 | 0.389 | 0.389 | -13.00% | -13.00% | -13.00% | 0.00% | 0 | 67.53% | -0.399 | 0.206 | 2 | 0 |

#### Rolling Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60/40 | 0.986 | 0.986 | 0.986 | -4.34% | -4.34% | -4.34% | 0.00% | 0 | 73.79% | 0.410 | 1.292 | 12 | 0 |

## Dominance (Romano-Wolf Stepwise)

| Benchmark | Adjusted p-value | Dominates |
|---|---|---|
| NZX equal-weighted | 0.8175 | False |
| 60/40 | 0.8185 | False |

## Gating Decision

**Overall:** FAIL

### Bars

- **sanity_floor:** ✗ — Sharpe -0.30 ≤ 0.00
- **bootstrap_ci:** ✗ — 95% CI low -0.41 ≤ 0
- **deflated_sharpe:** ✗ — p=0.9848 > 0.05
- **td_excess_return:** ✗ — Mean excess -5.62% ≤ 0 or p=0.8910 > 0.05
- **dominance_NZX equal-weighted:** ✗ — p_adj=0.8175 > 0.05
- **dominance_60/40:** ✗ — p_adj=0.8185 > 0.05

**Notes:** Kept folds: 9; Rejected folds: 3; n_trials: 41 (prior: 30, ledger: 11); Per-regime Sharpe: bull=0.30, chop=-0.63

## Pass / Fail

- **Sanity floor:** ✗ FAIL — Sharpe -0.30 ≤ 0.00
- **Bootstrap CI:** ✗ FAIL — 95% CI low -0.41 ≤ 0
- **Deflated Sharpe:** ✗ FAIL — p=0.9848 > 0.05
- **TD excess return:** ✗ FAIL — Mean excess -5.62% ≤ 0 or p=0.8910 > 0.05
- **Benchmark (NZX equal-weighted) via Romano-Wolf:** ✗ FAIL — p_adj=0.8175 > 0.05
- **Benchmark (60/40) via Romano-Wolf:** ✗ FAIL — p_adj=0.8185 > 0.05

### Rejected Folds

- fold 0: 100% empty months
- fold 1: 100% empty months
- fold 2: 100% empty months
