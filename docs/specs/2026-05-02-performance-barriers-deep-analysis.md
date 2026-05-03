# Performance Barriers — Deep Multi-Agent Analysis

**Date:** 2026-05-02
**Scope:** Read-only deep dive across data, benchmarks/gating, execution realism, and attribution + factor-experiment results.
**Strategies covered:** all current candidates under `python/configs/strategy-specs/candidates/`, with primary focus on `mom-ar-spread-scrubbed` and `mom-dividend-yield-ar-spread-scrubbed`, plus the most recent `factor-sweep-*` results.
**Status:** Findings only. No implementation.

---

## TL;DR — Three findings dominate

1. **The cost model is silently misconfigured.** The AR-spread panel is universally floored to 5 bps/side because `compute_abdi_ranaldo_spread_panel` rolls over the wrong index. Every `*-ar-spread*` candidate is being priced at 5 bps spread, not the intended AR estimate, not the 200 bps fallback. **The published `mom-ar-spread-scrubbed` Sharpe of 0.77 is built on inert spread costs.**
2. **The gating stack is harder than it should be, and partly self-inflicted.** Deflated SR is fed the post-haircut Sharpe (0.46), not the delisting-adjusted Sharpe (0.77). The TD-floor dominance gate is structurally near-impassable for an unlevered equity book. The 60/40 dominance gate uses a yield-only bond proxy that is borderline-by-construction. The spec field `n_trials_prior` is a no-op (silent bug).
3. **Data contamination is large, concentrated, and fixable.** ~750 daily |r|>50% events and 22 ticker-months above +500% survive into the panel. Three microcap names alone (MEE, NTL, ALF) drag Sharpe by an estimated -0.05 to -0.10. The current scrubber catches only single-day round-trips; it misses one-sided level shifts, decimal-shift errors, stale-price tails, and share-count discontinuities.

If only one thing is fixed, it should be the AR-spread panel bug, because every cost-related conclusion downstream of it is wrong.

---

## 1. Data and preprocessing

### Inventory (NZX-only, asof 2026-01-01)

| Severity | Count | Tickers |
|---|---:|---:|
| Daily abs return > 30% | 1,523 | 64 |
| Daily abs return > 50% | 750 | 42 |
| Daily abs return > 100% | 224 | 28 |
| Daily abs return > 200% | 154 | 22 |
| Monthly abs return > 50% | 309 | 46 |
| Monthly abs return > 500% | 22 | 10 |
| Stale runs >=10 identical closes (vol > 0) | 2,679 | 95 |
| Zero-volume observations | 126,210 (~19% of all bar-days) | 18 tickers > 50% zero-vol |
| Scrubber events fired | 252 (only 3 of top-50 daily anomalies) |

### Failure mode

**One-sided level shifts dominate.** 45 of the top 50 daily anomalies are one-sided spikes with no reversion. The current scrubber only catches symmetric round-trips with reversal tolerance < 10%, so it covers roughly the wrong tail.

Notable concentrated contamination:

- TWF.NZ 2018-04-18: +10,344x (likely prior-day price `0.0002` parsing error).
- BAI.NZ 2001-07-18: +499x, persists at new level.
- MEE.NZ 2006-12-05: +325x, persists.
- ALF.NZ 2009-11-25: +725x after two consecutive -99.88% prints, classic split-not-applied.
- MHJ.NZ 2001-2003: 22 decimal-shift events oscillating between two scales.
- 10+ unexplained share-count jumps (BAI, BIT, CRP, GEN, MEE, NTL, PHL, RTO).

### Quantified factor impact

Masking ticker-months containing daily |r|>50% lifts mean momentum IC from **0.054 → 0.064 (+17%)** and IR by +12%. No year degrades meaningfully. Lift concentrates in 2001-2002, 2005, 2011, 2014, 2019-2020.

Approximately **12% of momentum-driven trades** in any month are entered or exited within +/- 2 months of an anomalous price print on the same name. **66% of all rebalance months have at least one such trade.** This is a steady drag, not catastrophic events.

### Coverage gaps in current scrubber

The scrubber misses:

- One-sided level shifts (45 of top 50 daily anomalies).
- Multi-day round-trips (>200 near-misses with combined reversal in [10%, 30%]).
- Decimal-shift / unit errors (48 events at exact 10x/100x ratios with no matching split).
- Stale-price runs (10k+ runs of >=5 identical closes with non-zero volume).
- Share-count / market-cap discontinuities (10+ unexplained >=2x jumps).
- Zero-volume anomaly conditioning (anomalies are 1.3x more frequent on zero-vol days).
- The cross-source `adjustments` audit layer is `kind: audit` only — its results never feed back into prices.

### Top three data wins (ranked by expected Sharpe lift)

1. **Multi-day return-magnitude winsorize** (NaN-mask, not price replacement) for any monthly |r_m|>500% or daily |r_d|>200% with no corporate action within +/- 5 days. Expected Sharpe +0.10 to +0.15.
2. **Volume-gated price acceptance** for any move |r|>20% (require non-zero volume on day t and t+1). Expected Sharpe +0.05 to +0.10 on top of #1.
3. **Promote `adjustments.kind: audit` -> `repair` and add a decimal-shift detector.** Expected Sharpe +0.03 to +0.05, concentrated in 2001-2003.

Combined estimated IC lift: 0.054 -> ~0.075 (+40%).

---

## 2. Benchmark and gating sensitivity

### Benchmark construction summary

- **NZ TD floor (`nz_td_floor`):** `panel.macro["short_term_interest_rates"]` (OECD STIR) -> month-end resample -> ffill <= 3 months -> `default_floor=4%` fallback (never hits in OOS coverage) -> `(1+r)^(1/12) - 1`. Mean STIR 2010-2024 = 2.67%. Synthetic series Sharpe ~2.37 with effectively zero drawdown and 100% positive months.
- **NZX equal-weighted (`nzx_equal_weighted_fixed_universe`):** mcap floor only (>=20M NZD), independent of `panel.universe_mask`. **ADV filter explicitly raises `NotImplementedError`.** Inherits full strategy cost stack (200 bps spread + 400 bps haircut), which is overly punitive for a benchmark.
- **60/40 (`sixty_forty`):** equity leg = `FNZ.NZ` (coverage starts 2008-01, n=154 OOS months). Bond leg is **yield-only** (no duration P&L), flat 50 bps haircut, no rebalancing drag.

### TD-floor dominance is structurally near-impassable

| TD scenario | TD Sharpe | p_adj (TD) | Dominates? |
|---|---:|---:|:-:|
| Default (current) | 2.37 | 0.190 | No |
| -200 bps shift | 1.68 | 0.076 | No |
| -100 bps shift | 2.01 | 0.124 | No |
| +100 bps shift | 2.73 | 0.280 | No |
| +200 bps shift | 3.09 | 0.387 | No |
| Switch STIR -> OCR/interbank | 2.05 | 0.286 | No |

Naive paired t(strat - TD) = +2.33 (one-sided p ~ 0.0098, +5.9%/yr excess) — the strategy *does* beat TD on excess return, but the family-wise Romano-Wolf test against a zero-vol asset is the wrong tool for this comparison.

MC-error (20 seeds, 2000 resamples): TD p_adj std = 0.0086. Failure to dominate is robust, not seed noise.

### 60/40 is borderline-by-artifact

| 60/40 scenario | 60/40 Sharpe | p_adj | Dominates? |
|---|---:|---:|:-:|
| Yield-only, 50 bps (current) | 0.78 | 0.070 | No |
| Yield-only, 0 bps haircut | 0.86 | 0.090 | No |
| Yield-only, 200 bps haircut | 0.55 | **0.025** | **Yes** |
| Duration = 4y, 50 bps | 0.76 | 0.063 | No |
| Duration = 6y, 50 bps | 0.74 | 0.062 | No |
| Duration = 6y, 100 bps | 0.66 | **0.043** | **Yes** |
| Duration = 8y, 50 bps | 0.71 | 0.058 | No |

Realistic NZ bond duration plus realistic 100 bps ETF haircut flips the gate to PASS. Current configuration sits inside the calibration noise band (MC std 0.004; current p = 0.0646 ± 0.009).

### Deflated Sharpe is fed the wrong Sharpe

`gating.py:80-87` passes `oos_sharpe_flat_haircut` (0.46) to the deflated SR test, but reports and downstream displays quote `oos_sharpe_delisting_adjusted` (0.77). Sensitivity at fixed n_obs=231, skew=0.341, kurt=7.17:

| Sharpe input | n_trials | Deflated p | Pass? |
|---|---:|---:|:-:|
| flat-haircut 0.46 (current) | 39 | 0.565 | No |
| flat-haircut 0.46 | 10 | 0.328 | No |
| flat-haircut 0.46 | 5 | 0.203 | No |
| **delisting-adjusted 0.77** | 39 | **0.114** | No |
| delisting-adjusted 0.77 | 10 | 0.035 | **Yes** |
| delisting-adjusted 0.77 | 5 | 0.014 | **Yes** |

**Two bugs intersect here:**

- `gating.py:13,70` imports the module-level `n_trials_prior` constant from `stats/ledger.py:11` and **ignores `spec.n_trials_prior`**. The YAML knob is a silent no-op.
- The Sharpe definition piped into deflated SR is `oos_sharpe_flat_haircut` (which double-counts costs against an already cost-adjusted measure), not the delisting-adjusted Sharpe used elsewhere.

This single calibration choice moves deflated p from 0.565 to 0.114. Combined with the silent `n_trials_prior` bug, **this is the single most consequential calibration decision in the gating stack.**

### Bootstrap CI and Romano-Wolf

- Bootstrap CI low ranges 0.45-0.66 across all (block, n_resamples) tested; ALL pass `> 0`. Default block = `n^(1/3) ≈ 6.13` is reasonable for monthly data. n_resamples sensitivity tiny (CI moves <0.01 from 500 to 10,000).
- Romano-Wolf p_adj converges by 2,000 resamples. MC std at 2,000 resamples: TD = 0.009, 60/40 = 0.004, NZX = 0.002. All gate decisions stable to seed.

### Robust vs fragile gates

| Gate | Current verdict | Robust? |
|---|---|---|
| sanity_floor | PASS | trivially |
| bootstrap_ci | PASS (low ~ 0.56-0.66) | highly |
| **deflated_sharpe** | FAIL p=0.565 | **fragile** (Sharpe-input + cost-double-counting) |
| **dominance vs TD** | FAIL p=0.190 | **structurally hard** for any plausible TD path |
| dominance vs NZX EW | PASS p=0.012 | robust |
| **dominance vs 60/40** | FAIL p=0.065 | **fragile** (yield-only bond proxy) |

Three of six gates are fragile or structurally mis-specified.

### Recommendations

1. Decide and document a single Sharpe definition for deflated SR (recommend delisting-adjusted).
2. Fix the silent `n_trials_prior` bug in `gating.py:13,70`.
3. Replace TD dominance with either a one-sided HAC excess-return test or move TD into the deflated-Sharpe denominator as the risk-free rate.
4. Implement duration P&L in the 60/40 bond leg.
5. Calibrate NZX-EW costs independently (50-80 bps, not 600).
6. Implement the documented ADV filter for NZX-EW.
7. Use a regime-conditional or shorter STIR window for TD's `default_floor` (4% has not been the NZ short rate since 2008).
8. Consider replacing deflated-SR with a PSR threshold (e.g. PSR > 0.90 against benchmark Sharpe = 0).

---

## 3. Execution and microstructure realism

### Critical bug: AR-spread panel is universally inert

`compute_abdi_ranaldo_spread_panel` rolls over the raw OHLC index, which has 37,963 mixed-timestamp rows for ~15,682 unique calendar dates. Per-ticker valid-row density never satisfies `min_periods=20` in the 60-row rolling window. **Result: 100% of 6.83M panel cells are floored to 5 bps/side.** SPK.NZ on a clean per-ticker daily series ranges 9-86 bps with mean ~33.

**Every `*-ar-spread*` backtest is being priced at 5 bps spread, 40x cheaper than the 200 bps fallback and ~6-10x cheaper than realistic AR estimates.**

Verified two independent ways:

- Full panel describe: 100% of 6.83M cells == 5.0.
- Per-ticker clean series for SPK.NZ: 9-86 bps range, ~33 bps mean.

### Cost decomposition by NAV (`mom-ar-spread-scrubbed`)

| NAV | Total cost | Spread | Fees | Sharpe gross | Sharpe net |
|---:|---:|---:|---:|---:|---:|
| 10k | 7,243 | 520 (7%) | 6,723 (93%) | +1.15 | +0.82 |
| 25k | 32,229 | 1,287 (4%) | 30,941 (96%) | +1.15 | +0.55 |
| 100k | 180,489 | 5,101 (3%) | 175,388 (97%) | +1.15 | +0.32 |
| 250k | 477,984 | 12,730 (3%) | 465,254 (97%) | +1.15 | +0.27 |
| 1M | 1,965,227 | 50,864 (3%) | 1,914,363 (97%) | +1.15 | +0.24 |

Two compounding errors:

- **Per-order Sharesies $25 cap is not modeled** (`costs/model.py:20`). The engine bills 190 bps on all turnover above the $5k monthly coverage with no per-order ceiling. At $1M NAV this overstates fees by an estimated 50-70%.
- **AR spread is currently 5 bps, not the realistic ~30 bps.** Spread cost is understated.

These errors partially mask each other but do not cancel.

### Liquidity participation is unenforced

`optimizer.py:14-16` defers ADV-participation capping to "Milestone 4". Only the universe filter `min_adv_dollars=10000` exists. Trade reconstruction at NAV $10k:

| Participation cap | Trades violating |
|---|---|
| > 1% ADV | 35% of trades |
| > 5% ADV | 19% |
| > 10% ADV | 13% |
| > 25% ADV | 5% |

NAV-scaled liquidity demand:

| NAV | Median % ADV | p99 | Max | Trades >10% | >25% | >100% |
|---:|---:|---:|---:|---:|---:|---:|
| 10k | 0.29 | 142 | 138,651 | 304 | 121 | 32 |
| 100k | 2.92 | 1,425 | 1,400,000 | 830 | 606 | 304 |
| 1M | 29.2 | 14,252 | 13,900,000 | 1,528 | 1,225 | 830 |

At $1M NAV, the **median** trade requests 29% of daily $-volume; 35% of trades demand >100% ADV. The $1M backtest is a price-taker fantasy.

### Production / research divergence

`skuld_portfolio/execution_planner/plan_trades.py` does not receive a `per_ticker_spread_bps` panel and uses the flat `cost_model.spread_bps`. Even if the AR panel worked, research and production would price spreads on different bases. The spec name `mom-ar-spread-scrubbed` advertises AR pricing that neither side actually applies.

### Other execution gaps

- No turnover budget per period; nothing caps total monthly turnover beyond per-trade `no_trade_threshold_frac` and `size_floor_nzd`.
- No price impact / slippage model. Even fixed AR spread is quoted-effective, not impact-adjusted for 25%+ ADV demand.
- ADV definition is `mean of non-zero-volume days` over a 20-day window — undocumented choice; permissive variant.
- `size_floor_cost_multiple` interacts with broken AR: with AR == 5 bps the cost-multiple floor never binds; once fixed, many small trades will be deferred.
- `min_obs=20` in a 60-row window is borderline even on a clean daily grid.

### Recommended fix order

1. Repair `compute_abdi_ranaldo_spread_panel` (densify per-ticker daily index before rolling). Add a regression test asserting >50% of values exceed the floor on real NZX OHLC.
2. Add per-order Sharesies $25 cap to `CostModel`.
3. Wire a 1% ADV participation cap into `build_target_portfolio` (consume `adv` and `portfolio_nav`).
4. Plumb the working AR spread panel through `skuld_portfolio` planner.
5. Add a turnover budget per rebalance (e.g. 30% monthly cap).
6. Add a square-root impact model on top of AR spread.
7. Document and standardize the ADV definition (recommend strict 21d $-ADV excluding zero-volume days).

### Expected behavioral changes after fixes

- AR-spread fix at NAV $10k: spread cost rises 520 -> ~7,000-12,000; many small trades fail size-floor and defer; turnover drops 20-40%; net Sharpe falls from 0.82 toward ~0.5-0.6 (this is the truthful number).
- Per-order cap at NAV $1M: fees drop 1.91M -> ~0.4-0.7M; net Sharpe rises from 0.24 toward ~0.5+.
- 1% ADV cap: 35% of $10k trades violate; backtest realism improves; published returns fall.

---

## 4. Attribution and factor sweep

### Sweep diagnostics (139 variants, all fail gating)

`results.csv` only carries Sharpe and gating; no MDD/turnover/deflated-p; gating reason is missing.

Mean OOS Sharpe (DL) by axis, discovery lane (n=112):

| Axis | Levels | Effect |
|---|---|---|
| **rebalance** | bme 0.20 vs **bqe 0.72** | **+0.52** |
| **spread model** | flat 0.06 vs **abdi_ranaldo 0.86** | **+0.80** (caveat: AR is currently inert -> this is a 5 bps vs 200 bps comparison, not a realism comparison) |
| mcap floor | 0 -> 0.39, **20 -> 0.52** | +0.13 |
| **overlay** | overlay vs no-overlay | **0.00** (cash overlay is dead code) |
| factors | mom-s3 0.62, mom-s6 0.61, mom-divyield 0.49, mom-size 0.49, mom 0.43, mom-lowvol 0.33, mom-lowvol-size 0.24 | – |

Refinement lane (n=27): cash floor and spread_scale are inert. Only `no_trade_threshold` matters: 0.0025 -> 0.769, 0.005 -> 0.773, 0.01 -> 0.788. All within [0.769, 0.788].

Top variants (none gate):

- `disc-mom-divyield-bqe-mcap0-abdi_ranaldo` DL 1.355
- `disc-mom-s3-bqe-mcap20-abdi_ranaldo` DL 1.350
- `disc-mom-size-bqe-mcap20-abdi_ranaldo` DL 1.319

**Implication:** dominant levers are structural (rebalance cadence, cost model bug). Cash overlay is dead. Smoothed momentum (`mom-s3`, `mom-s6`) is stronger than promoted basic `mom` and deserves a candidate slot — but headline numbers are inflated by the AR bug.

### Per-name attribution (`mom-ar-spread-scrubbed`)

- 141 unique tickers; sum of contributions 7.20 (12.7% ann compound).
- **Top winners:** ATM +0.47, TWR +0.43, MFT +0.32, PHL +0.30, CHI +0.26, RYM +0.25, SKO +0.24, BRM +0.23, SKL +0.20, BIT +0.19. All real, recognizable.
- **Top losers:** MCK -0.10, WBC -0.10, **NTL -0.08**, BRW -0.06, **MEE -0.05**, SAN -0.05, **ALF -0.05**, TEM -0.04. Three of the worst are contaminated microcaps.

Defensive variant `mom-dividend-yield-ar-spread-scrubbed` shares the same losers — the dividend overlay does NOT prune contaminated tickers.

### Worst-month forensics

| Month | Port return | Holdings | Drivers |
|---|---:|---:|---|
| 2008-10 | -9.9% | 9 | AOF -19, MCK -16, MFT -15, TEM -14, PGW -14 (broad GFC) |
| 2008-11 | -8.1% | 8 | AOF -21, WBC -16, PGW -9, CEN -9 |
| 2008-02 | -7.2% | 9 | RAK -28, AIA -15 |
| 2018-03 | -5.5% | **4** | MLN -31% (concentration: one name = 31% of loss) |
| 2013-06 | -5.4% | 12 | TRA -16, HLG -11, ATM -10 |
| 2018-10 | -5.1% | 21 | broad-based (no single >2% contrib) |

Worst months are real GFC drawdowns plus one concentration event, not data artifacts. The engine is doing what it should; GFC is genuinely bad for momentum on NZX.

### Offensive vs defensive

| Metric | Scrubbed | Divyield | 50/50 blend |
|---|---:|---:|---:|
| Ann compound | 12.7% | 11.8% | 12.3% |
| Ann vol | 11.2% | 10.8% | 10.8% |
| Sharpe | 0.82 | 0.77 | 0.81 |
| Max DD | -26.5% | **-21.4%** | -21.8% |
| Correlation | — | — | **0.94** |

Divyield - scrubbed mean diff +0.000/month, t = -0.94. Statistically indistinguishable on average; differences are timing-only.

Sub-period:

| Period | n | Scrubbed Sharpe | Divyield Sharpe | Divyield - Scrubbed (ann) |
|---|---:|---:|---:|---:|
| GFC 2008-09 | 22 | -0.90 | -0.47 | **+6.4 ppt** |
| Eurocrisis 2011 | 11 | -2.27 | -3.10 | -1.7 ppt |
| COVID 2020 | 11 | +2.31 | +1.96 | -5.8 ppt |
| Bear chop 2022-23 | 22 | +0.22 | -0.82 | **-7.7 ppt** |
| Bull 2012-21 | 110 | +1.56 | +1.41 | -2.1 ppt |
| All | 231 | +0.82 | +0.77 | -0.9 ppt |

Engine's regime tagging shows divyield is genuinely defensive in `bear` (scrubbed -1.47, divyield +0.55) but a structural drag elsewhere. 0.94 correlation means the 50/50 blend gives no Sharpe lift. **Promote only as a regime-conditional overlay (engine-bear only), not as a standalone candidate.**

### Data-induced loss flags

Three contaminated microcaps drag a combined ~ -0.18 contribution (~ -2.5% ann) on scrubbed and ~ -0.19 on divyield:

| Ticker | Contamination |
|---|---|
| MEE.NZ | +325x on 2006-12-05; +19x on 2019-11-28; multiple +9x, +3x prints |
| NTL.NZ | +9.4x on 2000-01-04; fortnightly -85.5% / +5-10x oscillation through 2011 |
| ALF.NZ | +725x on 2009-11-25 after two -99.88% prints; +9x, +8x scattered |

Removing these three is expected to lift offensive Sharpe by 0.05-0.10.

### Concentration risk

2018-03 had only 4 names; one stock (MLN -31%) drove the entire monthly loss. `bqe + mcap20` small-fold variants need a `min_names` floor or tighter per-position weight cap when n<10.

### Per-year stability

Both strategies share the same crisis years:

- **Worst:** 2011 (-18% / -15%, Sharpe -2.16/-2.37), 2008 (-30%/-26%, Sharpe -1.25/-1.18)
- **Best:** 2025 (+170%/+176%), 2020 (+172%/+165%), 2019 (+90%/+83%), 2012 (+56%/+64%)
- Both flat-to-bad in 2022 and 2023 — momentum's chop years

---

## 5. Cross-cutting prioritized findings

Ranked by expected Sharpe / dominance impact, weighted by implementation effort and risk of mis-attribution.

~~### Tier 1 — silently corrupting every published number (must fix before any further evaluation)

1. **Repair `compute_abdi_ranaldo_spread_panel`.** The AR panel is universally floored to 5 bps. Every cost-related sweep result and every cost-related candidate ranking is currently wrong. After fix, expect Sharpe to drop ~0.2-0.3 across cost-sensitive specs — that is the truth.
2. **Fix `gating.py` Sharpe-input choice and the `n_trials_prior` no-op.** Single biggest swing in the gating stack: deflated p moves from 0.565 to 0.114 just by passing the right Sharpe.
3. **Add per-order Sharesies $25 cap.** Without this, every NAV-scaling result above $25k is unrealistically pessimistic.~~ Done

### Tier 2 — large structural wins (evidence-backed, low overfit risk)

4. **Multi-day winsorize plus volume-gated price acceptance** in the data layer. Expected IC lift +30-40%, expected Sharpe lift +0.10-0.20.
5. **Wire ADV participation cap** into the optimizer (1% default). Expected: small Sharpe drag at $10k, large realism gain at >$100k.
6. **Replace TD dominance with a one-sided excess-return test** or move TD into the deflated-SR risk-free rate. Removes a structurally impassable gate.
7. **Add duration P&L to 60/40 bond leg.** Expected: 60/40 dominance flips to PASS for `mom-ar-spread-scrubbed` after combining with realistic 100 bps ETF haircut.

### Tier 3 — design and methodology

8. **Promote smoothed-momentum candidates (`mom-s3`, `mom-s6`).** Sweep evidence shows these dominate basic `mom` even before the AR fix. Re-evaluate after Tier 1 fixes to avoid promoting AR-bug artifacts.
9. **Fix or remove the cash overlay.** Currently bit-identical to non-overlay across all sweep pairs. Either dead code or never-triggering rule.
10. **Use divyield as a regime-conditional overlay, not standalone.** GFC-defensive, chop-drag, 0.94 correlation with offensive variant.
11. **Add `min_names` floor or tighter per-position cap** for small-fold periods (e.g. 2018-03 with 4 names).
12. **Add `gating_reason` and per-criterion fields** to the experiment output. All 139 variants currently report `gating_passes=False` with no reason; sweep is information-impoverished.
13. **Calibrate NZX-EW benchmark costs independently** (50-80 bps, not 600); implement its documented ADV filter.

### Tier 4 — research hygiene

14. Document the ADV definition (strict vs permissive) in `prepared_panel.py`.
15. Add a regression test asserting AR-spread panel coverage on real NZX OHLC.
16. Add a turnover budget per rebalance.
17. Add a simple square-root impact model on top of AR spread.

---

## 6. Headline takeaway

The strategy is closer to passing than the headline `passes=False` suggests, but **not in the way the current numbers imply.** The published Sharpe of 0.77 is propped up by the AR-spread bug; the true post-cost Sharpe at NAV $10k is likely ~0.5-0.6 after Tier 1 fixes. However, three of the six gates are fragile or mis-specified (deflated SR Sharpe-input, TD dominance structure, 60/40 bond proxy), and fixing them moves the strategy clearly past two of the three currently-failing gates without changing the underlying signal.

**The right execution order is: fix Tier 1 first to get truthful numbers, then re-evaluate before any further candidate sweeps.** Sweeping on top of the current AR bug will systematically reward whatever spec depends most on the inert spread.
I
