# Methodology Report

**Config hash:** 63d50800e374ab3f1147ccd9880962e6a7e1c599beffeac01f732dc9836c68fb
**Git SHA:** d40c5fa-dirty
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

**Name:** phase1_no_mcap

### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Strategy | 0.372 | 0.249 | 0.357 | -60.74% | -60.74% | -64.15% | 22.02% | 279360 | 49.51% | 4.073 | 0.199 | 2 | 0 |

### Rolling Driver (Gating Reference)

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Strategy | 0.053 | -0.168 | 0.025 | -47.86% | -47.86% | -51.77% | 23.50% | 28730 | 51.52% | 6.962 | 0.020 | 21 | 0 |

### Per-Regime Sharpe (Rolling Driver)

- **bear:** -2.231
- **bull:** 0.063
- **chop:** 0.104

## Benchmarks

### NZ TD floor

**Coverage:** 1970-01-30 to 2025-12-31

**Notes:**
- Notional term deposit, 4.0% default floor

#### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZ TD floor | 9.641 | 9.398 | 9.398 | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 100.00% | -0.509 | 0.000 | 2 | 0 |

#### Rolling Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZ TD floor | 8.821 | 8.583 | 8.583 | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 100.00% | -0.162 | 0.000 | 21 | 0 |

### NZX equal-weighted

**Coverage:** 2000-02-29 to 2025-12-31

**Notes:**
- 20M NZD mcap floor
- ADV filter not implemented

#### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZX equal-weighted | -0.281 | -0.900 | -0.900 | -27.33% | -27.33% | -27.33% | 0.00% | 0 | 26.69% | 0.982 | -0.066 | 2 | 0 |

#### Rolling Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NZX equal-weighted | -0.136 | -0.696 | -0.696 | -20.97% | -20.97% | -20.97% | 0.00% | 0 | 32.94% | 0.792 | -0.046 | 21 | 0 |

### 60/40

**Coverage:** 2008-01-31 to 2025-12-31

**Notes:**
- Bond rates auto-converted from percentage to decimal
- yield-only bond proxy (no duration P&L)
- flat 50bps annual haircut applied uniformly

#### Two-Fold Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60/40 | 0.195 | -0.409 | -0.409 | -13.27% | -13.27% | -13.27% | 0.00% | 0 | 64.94% | -1.068 | 0.097 | 2 | 0 |

#### Rolling Driver

| Metric | Sharpe Raw | Sharpe Flat HC | Sharpe Delisting Adj | Max DD Obs | Max DD MC Med | Max DD MC P90 | Avg Turnover | Total Cost NZD | Hit Rate | Skewness | Calmar | Kept Folds | Rejected Folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60/40 | 0.195 | -0.409 | -0.409 | -13.27% | -13.27% | -13.27% | 0.00% | 0 | 64.94% | -1.068 | 0.097 | 21 | 0 |

## Dominance (Romano-Wolf Stepwise)

| Benchmark | Adjusted p-value | Dominates |
|---|---|---|
| NZ TD floor | 1.0000 | False |
| NZX equal-weighted | 0.9845 | False |
| 60/40 | 0.9965 | False |

## Gating Decision

**Overall:** FAIL

### Bars

- **sanity_floor:** ✗ — Sharpe -0.17 ≤ 0.00
- **deflated_sharpe:** ✗ — p=0.9921 > 0.05
- **dominance_NZ TD floor:** ✗ — p_adj=1.0000 > 0.05
- **dominance_NZX equal-weighted:** ✗ — p_adj=0.9850 > 0.05
- **dominance_60/40:** ✗ — p_adj=0.9960 > 0.05

**Notes:** Kept folds: 21; Rejected folds: 0; n_trials: 33 (prior: 30, ledger: 3); Per-regime Sharpe: bull=0.06, bear=-2.23, chop=0.10

## Pass / Fail

- **Sanity floor (TD floor):** ✗ FAIL — Strategy Sharpe 0.025 ≤ TD floor Sharpe 8.583
- **Primary benchmark (NZX equal-weighted) via Romano-Wolf:** ✗ FAIL — Dominates=False, p_adj=0.9845 > 0.05
- **Deflated Sharpe:** ✗ FAIL — Deflated Sharpe p=0.9921 > 0.05
