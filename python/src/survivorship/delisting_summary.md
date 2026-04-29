# NZX Delisting Research Summary

**Date:** 2026-04-20
**Milestone:** 0.5 (per implementation plan §1.1 and §4)
**Data file:** [`nzx_delistings.csv`](nzx_delistings.csv)

---

## 1. Dataset Overview

Compiled **31 NZX delistings** from the period ~2000–2026, sourced from:
- Wikipedia's [List of companies listed on the New Zealand Exchange](https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_New_Zealand_Exchange) (former constituents table)
- [delisted.co.nz](https://www.delisted.co.nz/) records
- NZX announcements and news archives (NZ Herald, Stuff, NBR)

### By Reason

| Reason | Count | Description |
|--------|-------|-------------|
| **Involuntary** | 8 | Liquidation, receivership, or regulatory-forced delisting |
| **Voluntary** | 5 | Corporate decision, typically after severe commercial decline |
| **Merger/Acquisition** | 18 | Acquired at premium (or near-premium); shareholders received cash or scrip |
| **Total** | **31** | |

For survivorship-bias analysis, the relevant pool is **involuntary + voluntary-with-losses = 13 delistings** ("loss-type"). The 18 mergers are relevant for complete delisting-rate computation but represent positive or neutral outcomes for shareholders (mean 12-month return: +24.7%).

---

## 2. Annual Delisting Rate

| Metric | Value |
|--------|-------|
| Total delistings (2000–2025) | 31 |
| Approximate years | 25 |
| Approximate average NZX universe size | ~130 |
| **All-type annual delisting rate** | **1.0%** of universe per year |
| **Loss-type annual delisting rate** $p$ | **0.40%** of universe per year |

The all-type rate of ~1.0% is broadly consistent with international exchange delisting rates (NYSE/Nasdaq average ~7–8% per year, but most are mergers; loss-type is ~1–2%). The NZX's lower rate reflects a smaller, more concentrated market with fewer IPOs and fewer failures per year in absolute terms.

---

## 3. Terminal Return Distribution (Loss-Type Delistings)

| Statistic | Value |
|-----------|-------|
| **n** | 13 |
| **Mean** $\mu_d$ | **−73.8%** |
| **Median** | −100.0% |
| **Std dev** $\sigma_d$ | **31.1%** |
| **Min** | −100.0% |
| **Max** | −17.0% |

The distribution is heavily left-skewed: **8 of 13 loss-type delistings resulted in a total loss (−100%)**. The remaining 5 had terminal returns between −17% and −60%. This confirms the architecture spec's observation that delistings are "typically −80% to −100% on the position, concentrated in time."

### Notable Involuntary Delistings

| Company | Ticker | Year | Terminal Return | Cause |
|---------|--------|------|-----------------|-------|
| CBL Corporation | CBL | 2019 | −100% | Insurance fraud / FMA investigation |
| Wynyard Group | WYN | 2017 | −100% | Cash burn exhausted; never profitable |
| Pumpkin Patch | PPL | 2017 | −100% | Retail decline; receivership |
| Intueri Education | IQE | 2017 | −100% | Regulatory investigation; liquidation |
| Pike River Coal | PRC | 2012 | −100% | Mine disaster (29 deaths); receivership |
| VTL Group | VTL | 2010 | −100% | Technology failure; receivership |
| Cynotech Holdings | CYT | 2013 | −100% | Technology failure; liquidation |
| Tenon Limited | TEN | 2017 | −17% | Gradual wind-down; minimal distribution |

---

## 4. Central Return Penalty (Survivorship Bias Drag)

$$\text{Annual drag} = p \cdot |\mu_d| = 0.40\% \times 73.8\% = \mathbf{30 \text{ bps/year}}$$

### Bootstrap Confidence Interval (10,000 iterations)

| Percentile | Annual Drag |
|------------|-------------|
| 10th | 25 bps |
| **50th (median)** | **30 bps** |
| 90th | 34 bps |

The bootstrap range is tight (25–34 bps) because the loss distribution is concentrated at −100%. The median of the loss distribution is itself −100%, so resampling produces little variation in the mean loss — variation comes primarily from which non-total-loss entries are included.

### Comparison with Literature

| Estimate | Source | Value |
|----------|--------|-------|
| Brown, Goetzmann, Ibbotson & Ross (1992) | US equities | ~150 bps/year |
| Elton, Gruber & Blake (1996) | US equities | ~270–400 bps/year |
| **This study (NZX unconditional)** | NZX 2000–2025 | **30 bps/year** |
| **This study (NZX factor-conditional, value Q5)** | NZX 2000–2025 | **117 bps/year** |

The unconditional NZX estimate (30 bps) is well below the US literature range (150–400 bps). However, this comparison is misleading for two reasons:

1. **Sample composition:** The US studies capture a much larger universe with more IPO failures. The NZX's small, concentrated market has fewer speculative listings.
2. **The unconditional rate is irrelevant for a factor-tilted strategy.** A value or momentum screen concentrates holdings in exactly the quintiles where delisting risk is highest. The factor-conditional rates (below) are the operationally relevant numbers.

---

## 5. Factor-Conditional Delisting Rates

This is the most important result for the Skuld strategy, which tilts toward value, quality, and momentum factors.

### Distribution at Delisting (Loss-Type Only, n=13)

| Factor | Quintile at Delisting | Count | % of Loss Delistings |
|--------|-----------------------|-------|---------------------|
| **Value** | Q5 (value trap) | 11 | 85% |
| **Value** | N/A (no earnings) | 1 | 8% |
| **Value** | Q4 (poor) | 1 | 8% |
| **Momentum** | Q1 (momentum loser) | 11 | 85% |
| **Momentum** | Q2 (moderate) | 1 | 8% |
| **Momentum** | Q3 (neutral) | 1 | 8% |
| **Quality** | Q5 (worst) | 8 | 62% |
| **Quality** | Q4 (poor) | 5 | 38% |
| **Size** | Q5 (smallest) | 11 | 85% |
| **Size** | Q4 (small) | 1 | 8% |
| **Size** | Q3 (midcap) | 1 | 8% |

### Factor-Conditional Annual Delisting Rates

| Quintile | Annual Rate $p$ | vs. Unconditional (0.40%) |
|----------|----------------|---------------------------|
| Value Q5 (value traps) | **1.69%** | **4.2× baseline** |
| Momentum Q1 (losers) | **1.69%** | **4.2× baseline** |
| Quality Q5 (worst) | **1.23%** | **3.1× baseline** |
| Size Q5 (smallest) | **1.69%** | **4.2× baseline** |

### Factor-Conditional Annual Drag

| Strategy Tilt | Annual Drag |
|---------------|-------------|
| Unconditional (no factor tilt) | 30 bps |
| **Value Q5 concentrated** | **117 bps** |
| **Momentum Q1 concentrated** | **117 bps** |
| Quality Q5 concentrated | 85 bps |

**Key insight:** A naive value screen that buys cheap stocks without quality filtering would concentrate in the exact quintile where delisting risk is 4× the baseline. The architecture's combination of value + quality + momentum is a natural defence against this — quality screens out distressed companies, and momentum screens out stocks in decline. A well-constructed multi-factor strategy should have delisting exposure near or below the unconditional rate.

---

## 6. Implications for Gating Decision

Per the implementation plan §1.1, the gating decision uses the more conservative of:

1. **Flat 400 bps** (from US literature)
2. **Probabilistic 90th-percentile penalty** (from this study)

| Metric | Value |
|--------|-------|
| Probabilistic 90th percentile (unconditional) | 34 bps |
| Probabilistic central estimate (unconditional) | 30 bps |
| Probabilistic central estimate (value Q5 conditional) | 117 bps |
| **Flat haircut (US literature worst-case)** | **400 bps** |

The flat 400 bps dominates the probabilistic estimate by a wide margin. This means:

- **For the gating decision:** 400 bps will be used. The probabilistic model does not change the gating threshold.
- **For honest reporting:** The probabilistic model adds value by showing that the *actual* NZX-specific survivorship drag is likely 30–120 bps depending on factor exposure, not 400 bps. A strategy that clears the 400 bps bar has a substantial margin of safety against the empirically-estimated drag.
- **For drawdown augmentation:** The Monte Carlo injection model (§1.1, item 3) uses $p = 0.40\%$ and draws losses from the empirical distribution (mean −73.8%, σ = 31.1%) to produce augmented drawdown estimates.

---

## 7. Caveats and Limitations

1. **Sample size.** 13 loss-type delistings. The rate estimate is sensitive to whether borderline cases are included or excluded.
2. **Selection bias.** Companies that quietly faded without formal announcements may be underrepresented; Wikipedia/delisted.co.nz sources are reasonably comprehensive but not guaranteed complete.
3. **Terminal return approximation.** 12-month-prior prices approximated from news reporting (not tick-level data). The 8 total-loss cases are exact (−100%); the others carry ±10–20% estimation error.
4. **Factor quintile assessment.** Factor quintiles assessed qualitatively at delisting, not computed from the actual factor model. Will be refined in Milestone 5. Directionally reliable.
5. **Survivorship of the universe.** Only captures formal NZX delistings; "living dead" stocks that fell below inclusion thresholds without delisting are not addressed.
6. **Period specificity.** The 2000–2025 sample includes GFC, NZ earthquake sequence, and COVID-19. Delisting rates may vary in crisis vs. benign periods; sample is too small to estimate time-varying rates.

---

## 8. Methodology Notes

- **Universe size estimate.** The NZX had approximately 100–160 listed equities at any point in the 2000–2025 period. We use 130 as a central estimate. The annual rate scales linearly with this assumption.
- **Period.** Listed and delisted dates from Wikipedia and delisted.co.nz. Where dates differ, the NZX announcement date is preferred.
- **Reason categorisation.** "Involuntary" = liquidation, receivership, or regulatory-forced. "Voluntary" = corporate decision where shares were near-worthless at the time. "Merger" = acquisition with cash or scrip consideration.
- **Excluded.** Foreign-listed companies with secondary NZX listings (BHP, HVN, Amcor, etc.), investment trusts (EUT, JFJ, ATR), and companies that delisted before 2000 or for which reliable data could not be found.
