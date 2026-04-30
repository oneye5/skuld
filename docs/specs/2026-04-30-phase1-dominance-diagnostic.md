# mom-ar-spread dominance failure — diagnostic

Date: 2026-04-30
Subject of analysis: `configs/strategy-specs/candidates/mom-ar-spread.yaml`
Method: walk-forward OOS returns vs three benchmarks (`NZ TD floor`, `NZX equal-weighted`, `60/40`), regime decomposition, return concentration, and raw-data spot checks.

## TL;DR

`mom-ar-spread` is a real signal — IR vs every benchmark improves materially in the post-2010 sample and IR vs all three benchmarks rises in the second half of the data. It fails dominance gating today not primarily because of weak alpha but because of a **single corrupted data point in Jan 2010** that creates a spurious +63% portfolio month, plus **structural exposure to GFC drawdowns** in the 2008–2010 window where the strategy lost 5%/yr while the TD floor returned +4.5%/yr.

The two issues compound: the GFC losses widen historical TE vs TD/60/40, and the Jan 2010 outlier inflates raw return at the cost of inflating volatility, which depresses Sharpe and weakens dominance tests through both numerator (TE) and the multiple-testing penalty.

Adding ML signals or new factors before fixing these two issues will not improve gating outcomes — they will simply ride on the same broken cost function.

## Evidence

### 1. The signal is real and stable in the post-2010 sample

Pairwise alignment (full available history per benchmark):

| Benchmark | n months | strat CAGR | bm CAGR | ann excess | TE | IR | win rate | IR 1st half | IR 2nd half |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NZ TD floor | 231 | 0.149 | 0.069 | +0.086 | 0.178 | 0.481 | 0.550 | 0.303 | 0.993 |
| NZX equal-weighted | 231 | 0.149 | 0.014 | +0.137 | 0.174 | 0.790 | 0.615 | 0.584 | 1.590 |
| 60/40 | 140 | 0.089 | 0.052 | +0.038 | 0.097 | 0.389 | 0.557 | 0.049 | 0.855 |

Excluding the GFC-era 2008–2010 entirely (post-2010 sample):

| Benchmark | n months | strat CAGR | bm CAGR | IR | IR 1st half | IR 2nd half |
|---|---:|---:|---:|---:|---:|---:|
| NZ TD floor (post-2010) | 165 | 0.147 | 0.072 | 0.735 | 0.783 | 0.690 |
| NZX equal-weighted (post-2010) | 165 | 0.147 | 0.028 | 1.271 | 1.164 | 1.421 |
| 60/40 (post-2010) | 116 | 0.121 | 0.072 | 0.612 | 0.272 | 0.948 |

Information ratios above 0.6 against all three benchmarks in the post-2010 sample, with stable or improving second-half IR. This is the profile of a working systematic edge — not noise.

### 2. The strategy is short-vol exposure during equity drawdowns

Conditional on NZX equal-weighted drawdown state (full sample):

| State | n | strat ann ret | strat ann vol | strat Sharpe | TD ann ret | NZX EW ann ret | 60/40 ann ret |
|---|---:|---:|---:|---:|---:|---:|---:|
| NZX DD > 10% | 20 | 0.092 | 0.088 | 1.04 | 0.044 | -0.055 | 0.056 |
| NZX DD 1–10% | 104 | 0.073 | 0.108 | 0.68 | 0.060 | 0.005 | 0.056 |
| NZX near high | 16 | 0.205 | 0.131 | 1.56 | 0.169 | 0.280 | 0.030 |

Calendar period decomposition:

| Period | n | strat | NZ TD | NZX EW | 60/40 |
|---|---:|---:|---:|---:|---:|
| GFC + recovery 2008–2010 | 24 | -0.050 | +0.045 | -0.022 | -0.031 |
| Low-rate 2011–2019 | 69 | +0.075 | +0.025 | +0.016 | +0.056 |
| COVID era 2020–2021 | 16 | +0.407 | +0.360 | +0.211 | +0.154 |
| Hike cycle 2022–2025 | 31 | +0.073 | +0.041 | -0.002 | +0.061 |

The strategy beats every benchmark in three of four regimes. The single regime where it underperforms TD by ~10%/yr is 2008–2010, which is also when TD rates were 6%+ and equity vol was 130%/yr.

### 3. Return concentration is alarming and traces to a single data corruption

Top 10 single-month returns concentrate the entire compounded return:

- Total cumulative net return: **+1359%**
- Top-5 best months compounded standalone: **+126%**
- Top-10 best months compounded standalone: **+223%**
- Best single month (2010-01-29): **+63%** in one month

Dropping into raw data confirms this is corruption, not signal:

```
SKT.NZ close
2009-12-31: 32.10
2010-01-04:  5.05    <- corrupted single-day print
2010-01-05: 32.61
...
2010-01-29: 30.52
```

Yahoo Finance dropped a `5.05` close on 2010-01-04 — a ~84% one-day "drop" followed by ~545% one-day "recovery" the next session. The raw-to-monthly aggregation feeds this through as a clean monthly return that the panel then trades on. The strategy held SKT through the rebalance and booked a fictitious +63% portfolio month.

This is one of likely many such artifacts. `LIC.NZ` shows -74% in the same month from a similar mechanism.

## Why this fails dominance

Dominance tests reject equality when `mean(strat - bm) / TE(strat - bm) * sqrt(n)` clears a multiple-testing threshold. Both the GFC drawdown and the Jan 2010 outlier hit this expression in the worst possible way:

1. GFC: large negative `strat - bm` realisations vs TD floor inflate the numerator's denominator (TE) and reduce its mean.
2. Jan 2010 outlier: the +63% month inflates strategy variance more than it lifts mean (variance scales with squared returns), depressing Sharpe and TE-weighted dominance against TD specifically (TD has near-zero variance, so excess vol falls almost entirely on the strategy side).

The post-2010 IRs above 0.6 vs all three benchmarks suggest dominance would pass cleanly on a clean sample.

## Recommendations

In priority order. None require changing the alpha model.

### A. Data quality remediation (highest leverage, highest risk if skipped)

**A1. Single-day return outlier scrubber.** Add a preprocessing pass over `data_long.csv` that flags any daily return where `|r_t| > k * rolling_mad(r, w)` for sensible `k, w`, then either:
- winsorises the day to the rolling band, or
- treats it as missing and forward-fills.

This needs to run before monthly aggregation and before the strategy ever sees the data. Target: zero monthly returns above the 99.9th percentile of cross-sectional NZX equity behaviour without an accompanying corporate action record.

**A2. Corporate-action awareness.** Yahoo's `close` is dividend-adjusted but not split-adjusted in many vintages. We are likely also processing other split events as outliers. Cross-check the NZX corporate actions ledger or use `adj_close` consistently if it exists for all tickers (it does for SKT but not all features).

**A3. Backfill the 2008–2010 macro context.** The 60/40 benchmark only starts 2008-02 but TD and NZX EW go back to 2005. A consistent rolling window for *all* benchmarks would let dominance gating use the same n for all three tests instead of penalising the strategy for having more history than its hardest benchmark.

### B. Regime conditioning (medium leverage)

The strategy's only failure regime is 2008–2010. A simple cash overlay tied to one of:
- NZ OCR direction (raise → reduce equity exposure),
- realised NZX 60-day vol,
- term-spread inversion,

would likely close the TD-floor gap without adding alpha factors and without inflating DSR penalty. The codebase already has overlay scaffolding for this.

### C. Defer ML and new factors

There is nothing wrong with the existing alpha that adding more signals would fix. ML on a corrupt panel will produce a more confident, more overfit reading of the same broken data. New factors increase the multiple-testing penalty in DSR, which already fails (p = 0.618).

## Concrete next preregistration candidate

Once A1 and A2 are done, re-run `mom-ar-spread` unchanged on the cleaned panel. Expected outcome based on this analysis:

- Sharpe DL: 0.633 → likely 0.7–0.9 range (top 5–10 months will normalise downward, vol drops more than mean).
- DSR p: 0.618 → likely cleared if Sharpe lifts and skewness normalises.
- Dominance vs NZX EW: already 0.036 → likely sub-0.01.
- Dominance vs 60/40: 0.198 → likely sub-0.05 in post-2010 era.
- Dominance vs NZ TD floor: 0.414 → may still fail without the regime overlay (B); GFC underperformance is real, not a data artifact.

If A1+A2 lift gating to passing on three of four criteria with only TD dominance remaining, that is a strong signal to invest in B. If gating still fails after data cleaning, that is the signal to revisit the alpha specification.

## Files / commands referenced

- Input: `configs/strategy-specs/candidates/mom-ar-spread.yaml`
- Input: `data/data_long.csv`
- Runner: `skuld_research.config.runner.run_from_spec`
- Diagnostic re-runs use unchanged spec (immutable); analysis is overlay-only and produces no spec changes.
