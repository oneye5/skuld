# Mom-s8 Improvement Plan

Status: Complete
Related: `docs/plans/2026-05-05-phase2-alpha-bakeoff-design.md`, `python/reports/phase2_exploration.md`

## Context

The Phase 2 price-only alpha sweep did not produce a promotable challenger. `mom-s8` remains the production baseline. Further small momentum variants are likely to add fishing risk without improving deployable performance.

## Objective

Improve `mom-s8` deployable performance without expanding the production alpha trial count unnecessarily.

## Motivation

The failed alpha sweep is informative: the current signal is hard to beat, but many candidates lost through turnover, cost drag, noisy formation paths, or weak incremental return rather than a complete absence of signal. The next highest-value work is therefore not more small signal variants. It is understanding whether `mom-s8` is losing deployable value through construction, investability, or regime exposure.

## Workstreams

1. **Diagnose `mom-s8` return sources.** Measure attribution by period, ticker, rebalance, factor leg, turnover, cost drag, max-position binding, and universe breadth. This identifies whether returns come from repeatable broad effects or a few fragile names/periods.
2. **Portfolio-construction sweep.** Keep the `mom-s8` score fixed and vary rebalance frequency, no-trade threshold, turnover budget, max position, score tilt, and smoothing. Motivation: if the alpha is real but costly, better sizing/trading rules can improve net Sharpe without adding a new alpha hypothesis.
3. **Investability and data-quality filters.** Test stricter ADV, active trading days, stale-price exclusion, and chronic anomaly exclusion. Motivation: NZX thin trading can create false momentum, inflated highs, and expensive rebalances; removing weakly tradable names may improve both signal quality and execution realism.
4. **Regime deployment rules.** Only test after diagnostics show state dependence. Candidate states: broad market trend, realised volatility, cross-sectional dispersion, and universe breadth. Motivation: momentum often fails in reversals and crowded drawdowns, but regime overlays can easily overfit if not tied to observed failure modes.
5. **Defer new alpha families until data support improves.** Volume factors require volume in `PreparedPanel`; fundamentals require conservative publication-date handling. Motivation: these may be more independent than price momentum, but implementing them before the data contract is PIT-safe would create false confidence.

## Decision Rules

- Prefer changes that preserve the frozen `mom-s8` score and improve implementation, sizing, or tradability.
- Do not promote any result based only on in-sample improvement or a single lucky subperiod.
- Treat each materially different production candidate as a new hypothesis; keep broad search in `exploration` scope.
- Stop a workstream if it improves gross returns but worsens turnover, capacity, or paired net-return stability.

## Implementation Notes

### WS1 — Diagnostics infrastructure (done 2026-05-06)

- `BacktestResult.cap_binding_count: pd.Series` added to `skuld_common/contracts.py`; engine tracks the number of tickers at or above `max_position - 1e-4` each period (engine.py lines 268, 390, 529, 555).
- `AttributionReport` extended with `ticker_contributions` (date × ticker weighted return to signal-EW), `breadth_series` (universe member count per rebalance date), and `factor_leg_alpha_ann` (per-factor standalone CAGR alpha vs market proxy). `attribute_returns` gains optional `component_score_panels` argument.
- Bug fixed: `_apply_stale_price_mask` was using `within_streak_pos > 0` instead of `> 1`, causing the first day of each streak to be incorrectly masked. Fixed to `> 1`.

### WS2 — Construction sweep infrastructure (done 2026-05-06)

- `build_construction_variants(base_spec, quick=False)` added to `factor_experiment.py`. Full grid: `max_position` × `score_lambda` × `smoothing_alpha` × `no_trade_threshold_frac` × `turnover_budget_frac` × `rebalance_freq` = 4×4×3×3×3×2 = 864 variants. `quick=True` yields 16 variants.

### WS3 — Investability filter variants (done 2026-05-06)

- `stale_price_streak_days: int` added to `AnomalyFilterSpec` (default 0 = no-op). Filter masks consecutive identical closing prices >= N trading days; only the second+ day of each streak is NaN'd.
- Candidate specs created under `python/configs/strategy-specs/candidates/`:
  - `ws3-mom-s8-adv25k.yaml` — ADV floor 25k (vs 10k baseline)
  - `ws3-mom-s8-hist180.yaml` — min_history_days 180 (vs 126)
  - `ws3-mom-s8-chronic3.yaml` — chronic_ticker_max_extreme_days 3 (vs 5)
  - `ws3-mom-s8-strict.yaml` — all three filters combined (most restrictive)
  - All four specs enable `stale_price_streak_days: 5` and carry `ledger_scope: exploration`.

### WS3 — Investability filter results (done 2026-05-06)

Baseline `mom-s8`: flat-haircut Sharpe 0.532, turnover 17.8%.

Decision criterion: breadth ≥ 6 mean tickers AND Sharpe HC within −0.05 of baseline.

| Variant | Breadth (mean) | Sharpe HC | Δ HC | Verdict |
|---|---:|---:|---:|---|
| `ws3-mom-s8-adv25k` | 59.4 | 0.486 | −0.046 | pass (borderline) |
| `ws3-mom-s8-hist180` | 69.6 | 0.531 | −0.001 | pass |
| `ws3-mom-s8-chronic3` | 69.4 | 0.452 | −0.080 | fail |
| `ws3-mom-s8-strict` | 57.4 | 0.426 | −0.106 | fail |

**WS3 recommendation**: adopt `ws3-mom-s8-hist180` (essentially neutral, raises minimum history). `adv25k` is borderline — higher liquidity floor raises turnover and costs without Sharpe gain; not recommended for promotion.

**Promoted**: `python/configs/strategy-specs/production/mom-s9.yaml` — applies `min_history_days: 180` and `stale_price_streak_days: 5` to `mom-s8`. `n_trials_prior` unchanged at 30 (data quality change, not a new alpha hypothesis).

### WS4 — Regime overlay (done 2026-05-06)

- `ws4-mom-s8-overlay.yaml` uses `overlay: kind: nzx_ma200_agg_momentum, defensive_cash_fraction: 0.30`; triggered when EW market proxy < 200-day MA AND cross-sectional mean momentum < 0.
- Bear-regime Sharpe was −1.702 (WS1), which motivated running the overlay despite the deferred note.

| Metric | Value |
|---|---:|
| Sharpe HC | 0.513 |
| Δ HC vs baseline | −0.019 |
| Paired ann. delta | −0.2% |
| Paired 95% CI | [−0.10%, +0.08%] |
| Turnover | 17.4% |
| Max drawdown | −25.3% |

**WS4 verdict**: FAIL — overlay does not improve risk-adjusted return. Bull-period overhead outweighs the bear-regime defensive benefit at 30% cash target.

### WS5 — New alpha factors (done 2026-05-06)

- `EpsMomentumFactor`: YoY trailing diluted EPS growth, capped at ±10×, PIT-safe.
- `VolumeTrendFactor`: log(ADV_20d / ADV_60d), `min_trading_days=30`.
- Registered in `spec.py` and `config/factors.py`. `PreparedPanel.volumes` added as optional field; `build_prepared_panel` passes `volumes_daily` through.

Decision criterion: flat-haircut Sharpe ≥ baseline AND paired ann. delta ≥ 0.

| Variant | Sharpe HC | Δ HC | Paired ann. Δ | Verdict |
|---|---:|---:|---:|---|
| `ws5-mom-s8-eps` | 0.518 | −0.014 | +0.3% | fail |
| `ws5-mom-s8-voltrd` | 0.513 | −0.019 | −0.2% | fail |

**WS5 verdict**: FAIL — neither factor improves on baseline when combined with `mom-s8`. Signal dilution likely outweighs any incremental information.
