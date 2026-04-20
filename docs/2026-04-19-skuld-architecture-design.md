# Skuld — High-Level Architecture & Methodology Spec

**Status:** Draft for review (rev. 3)
**Date:** 2026-04-19
**Supersedes/extends:** [plan.md](plan.md)


## 1. Purpose

Provide statistically-backed buy/sell/hold recommendations and portfolio
allocations for a personal NZX-only equity portfolio held on Sharesies. The
output of the application is a set of recommended actions to take at each
rebalance event; a human (the user) reviews and executes them.

The application's job is **not** to decide whether NZ equities are a good
asset class — that decision is made outside the app. Its job is: *given that
capital will be deployed in the NZX universe, do it more intelligently than
buying the index, net of all costs.*

---

## 2. Scope & Constraints

### In scope
- NZX-listed tickers from `Tickers.java` (~150 names; ~70-90% with usable data).
- Long-only positions plus a cash sleeve (allocation % decided by the app).
- Recommendations only — no order execution.
- Rebalance cadence is flexible; turnover is constrained by the Sharesies
  fee structure rather than a fixed schedule.

### Out of scope
- ASX, US, or any non-NZX universe.
- Order execution, broker integration.
- Choice of cash instrument (TD / HISA / bond ETF / etc. — user's discretion).
- Tax optimisation.
- Shorting, leverage, derivatives.

### Hard constraints
- **Sharesies fee model:** flat $15/month for ≤ $5,000 of monthly volume;
  1.9% on volume above that. Treated as a turnover budget, not a cadence rule.
- **NZX liquidity:** small/mid caps have wide spreads. Realistic round-trip
  cost is plausibly 0.5-2% per side beyond the platform fee. All backtests
  must model this.
- **Data history is shallow:** many NZX listings have <10 years of clean
  history. Backtest folds will be limited.

---

## 3. Success Criteria

The benchmark stack is layered. Beating the floor is necessary but not
sufficient; the meaningful claim is beating the harder baselines.

### Sanity floor (must clear, validates the asset class not the strategy)
- 1-year NZ term-deposit rate over rolling 3-year windows. If unmet, the
  *asset class* is not paying — independent of strategy quality.
- **60% NZX50 (`FNZ.NZ` or equivalent) / 40% NZ government bonds**, monthly
  rebalanced, **net of modelled Sharesies fees and NZX spreads**.
  Acknowledged as a soft bar: a long-only 150-name portfolio will beat a
  cap-weighted NZX50 partly via the size factor mechanically. Beating this
  benchmark is necessary but does not validate the methodology.

### Primary benchmark (must beat — the meaningful claim)
- **NZX equal-weighted basket** of all NZX-listed equities with market
  cap > NZ$20M and trailing 20-day ADV > NZ$10k/day, monthly rebalanced,
  net of the same modelled costs. This universe definition is **fixed and
  independent of the strategy's own liquidity filter** — it uses its own
  hard-coded thresholds so the benchmark cannot be gamed by tuning the
  strategy's universe parameters.
  This neutralises the size factor and forces the strategy to demonstrate
  edge beyond "hold the small-cap tilt."
- **Closest published smart-beta methodology applied to the NZ universe**
  (e.g. Smartshares / iShares value, momentum, or quality methodologies
  re-implemented on the NZX universe). If the strategy cannot beat a
  rules-based published factor index net of costs, it has no edge — it
  has a backtest.

### Judging metric
- **Risk-adjusted return** (Sharpe and Sortino) over **rolling 3-year
  out-of-sample windows**, reported with **block-bootstrap confidence
  intervals** (Politis & Romano 1994) — not point estimates — given the
  small effective fold count (see §6).

### Honesty disclosure (always reported, never targeted)
- Absolute return vs. S&P 500 over the same period. Not a benchmark — a
  permanent reminder of the opportunity cost of the NZX-only constraint.
  Pedersen (*Efficiently Inefficient*) suggests long-only, single-small-
  market constraints structurally cap achievable Sharpe at well below
  globally-documented factor levels.

### Gating discipline
- Every signal, overlay, and added complexity must demonstrably improve
  out-of-sample risk-adjusted return net of costs in walk-forward testing
  **and clear a multiple-testing-adjusted significance bar** (deflated
  Sharpe ratio, see §6), or it is removed.
- The cash overlay faces the **highest** gating bar, not the lowest, on
  the basis that small-sample regime triggers are the component most
  empirically prone to OOS failure (see §4).

---

## 4. Methodology

### Strategy class
**Long-only, low-turnover, liquidity-aware, factor-tilted equity portfolio
with a rules-based cash overlay.** Composable signal architecture so
additional signal sources (trend-following, ML residual, macro regime tilt)
can be added later without disturbing the core.

### Why this class
- Universe is ~150 names with ~10y history → too small for ML to reliably
  outperform a well-built factor model. Strong inductive bias (theory-driven
  features, few parameters) buys sample efficiency we cannot afford to lose.
  López de Prado (*Advances in Financial Machine Learning*, 2018, ch. 14)
  is explicit that ML on equity cross-sections of this size will not
  survive a deflated-Sharpe adjustment.
- Factor investing has 50+ years of out-of-sample replication across
  markets — Fama–French (1993, 2015); Asness, Moskowitz & Pedersen,
  "Value and Momentum Everywhere" (*JoF* 2013) replicating premia in
  40+ markets including small developed markets structurally similar to
  NZ; AQR "Fact, Fiction, and Value Investing" (2015). It is the
  strongest baseline available.
- NZX is illiquid and inefficient relative to global markets — the regime
  most favourable to disciplined systematic retail strategies, and the one
  most punishing to high-turnover ML approaches.
- Recommendations are expressed in human-meaningful concepts (value,
  momentum, quality, low-vol scores), supporting the user's manual
  fundamental-review step.

### NZX-specific evidence caveat (explicit)
- Direct NZ factor literature is thin (Bryant & Eleswarapu 1997 on the
  NZ size effect; some RBNZ working papers; Australian evidence in
  Brailsford, Gaunt, O'Brien). The Phase 1 design **assumes that factor
  premia documented globally hold qualitatively in NZ**. The available
  NZ sample is insufficient to confirm this independently. This
  assumption is a known risk and must be stated in every methodology
  report — not buried.

### Phased build
- **Phase 1 — Factor model.** 4-6 named factors, HRP portfolio construction
  (López de Prado, *J. Portfolio Management* 2016 — beats mean-variance
  and inverse-variance OOS precisely when the covariance matrix is noisy
  and ill-conditioned, which describes a 150×150 NZ matrix exactly), and
  a **rules-based cash overlay subject to the §3 high gating bar**. Must
  beat the primary benchmark to proceed.
- **Phase 2 — ML residual layer (gated).** ML model trained to predict
  the residual the factor model misses, in the spirit of Gu, Kelly & Xiu,
  "Empirical Asset Pricing via Machine Learning" (*RFS* 2020). Blended in
  if and only if it improves out-of-sample Sharpe net of costs **and
  clears the deflated-Sharpe bar** (§6).
- **Future overlays.** Trend-following signal, macro regime tilt, sentiment
  signals — each added under the same gating discipline.

### Cash overlay: explicit risk acknowledgement
The rules-based cash overlay is the **highest-leverage, lowest-evidence**
component of the design. Faber (2007) timing rules work on the S&P;
replications on small markets (Clare, Seaton, Smith & Thomas 2013) show
the edge is much noisier. The NZX50's ~16-year history yields very few
independent regime transitions to fit against. This component is therefore:
- Held to the highest gating bar (§3).
- Considered against an alternative: a **defensive sleeve inside the
  equity portfolio** (low-vol + quality tilt, per Asness/Frazzini/Pedersen
  "Quality Minus Junk" 2019 and Frazzini/Pedersen "Betting Against Beta"
  2014), which captures most drawdown protection without the small-sample
  fitting risk. Decision deferred to the implementation plan (§8).

### Rejected alternatives
- **ML-first.** Sample size insufficient; overfitting risk unacceptable;
  recommendations would not be auditable.
- **Statistical arbitrage / pairs trading.** Requires shorting (Sharesies
  doesn't allow); NZX too illiquid; insufficient co-integrated pairs.
- **Pure macro regime-switching.** Effective n of regime shifts in available
  data is too small to fit a classifier reliably.

---

## 5. Architecture

### Pipeline overview

```
                        ┌────────────────────────┐
                        │ point-in-time data     │
                        │ loader (no lookahead)  │
                        └───────────┬────────────┘
                                    │ raw fields
                                    ▼
                        ┌────────────────────────┐
                        │ shared feature prep    │
                        │ (clean, align, lag)    │
                        └───────────┬────────────┘
                                    │ trustworthy panel
                                    ▼
        ┌──────────────┬───────────┴──────────┬────────────────┐
        ▼              ▼                      ▼                ▼
┌─────────────┐ ┌─────────────┐       ┌─────────────┐  ┌─────────────┐
│ signal A    │ │ signal B    │  ...  │ signal N    │  │ (future)    │
│ own feature │ │ own feature │       │ own feature │  │ signals     │
│ prep + gen  │ │ prep + gen  │       │ prep + gen  │  │             │
└──────┬──────┘ └──────┬──────┘       └──────┬──────┘  └──────┬──────┘
       │               │                     │                │
       └───────────────┴──────┬──────────────┴────────────────┘
                              ▼ scores per ticker (uniform contract)
                     ┌────────────────┐
                     │ signal combiner│
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ portfolio      │
                     │ constructor    │
                     │ (HRP / risk    │
                     │  parity)       │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ cash overlay   │
                     │ (equity vs %   │
                     │  cash)         │
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ execution      │
                     │ planner        │
                     │ (respect fee   │
                     │  budget)       │
                     └────────┬───────┘
                              ▼
                       recommendations
```

### Component contracts

| Component | Input | Output | Notes |
|---|---|---|---|
| Point-in-time data loader | `(t, fields)` | Raw values knowable on date `t` | Single source of truth for "what was knowable when". Enforces no-lookahead in code, not by convention. |
| Shared feature prep | Raw point-in-time data | Cleaned, aligned, publication-lagged panel | Thin layer. Cleans/aligns; does not invent features. May offer a memoized derived-series library (returns, vols) for signals to opt into. |
| Signal generator (each) | Cleaned panel + own feature prep | `pd.Series[ticker, score]` per `t` | Each signal owns its own feature engineering. All signals share the same output contract — interchangeable and composable. |
| Signal combiner | List of score Series | Single combined score Series per `t` | Default: z-score each, average. Weighted blend later, gated on out-of-sample evidence. |
| Portfolio constructor | Combined scores + covariance estimate | `pd.Series[ticker, weight]` | HRP or risk-parity weighted by score percentile. Concentration and liquidity caps enforced here. |
| Cash overlay | Portfolio weights + aggregate signal/regime state | Adjusted weights with cash % | Rules-based (e.g. NZX50 vs 200d-MA, aggregate signal strength). No predictive regime-switching in Phase 1. |
| Execution planner | Current portfolio + target portfolio + fee model + assumed cash yield | Trade list (+ deferred trades) | Optimises over the Sharesies fee cliff at $5k/mo. May defer trades to next month if economical. |

### Composability principle

Every signal generator implements:

```
SignalGenerator:
    name: str
    required_data: list[DataRequest]   # declared up-front
    score(t, universe) -> pd.Series    # index=ticker, values=float
```

Adding a new signal source (ML residual, trend, sentiment, …) is a new module
plus a registration line. No existing code changes. The combiner and all
downstream stages are signal-agnostic.

### "One application per core process"

Three applications, cleanly separated, sharing only the data contract
(via a shared `common/` package):

- **`common/`** — shared data contract types, config schema, validation
  utilities. Lightweight; no heavy dependencies. Both apps below depend on it.
- **`research/`** — backtests, walk-forward analysis, exploratory work,
  artefacts. Never runs in production. Outputs frozen signal configurations
  and a methodology report.
- **`portfolio/`** — the monthly decision app. Consumes frozen signal
  configurations, current portfolio, and current data. Emits
  recommendations. Deterministic, auditable, boring.

All Python packages managed as a `uv` workspace. The existing **`java/`**
ingest layer remains as-is and feeds both.

---

## 6. Data Quality and Statistical Requirements

These are not optional. Each one, violated, has a documented history of
turning a winning backtest into a losing strategy.

### Data integrity
1. **Point-in-time integrity.** No data point may be visible to logic at
   date `t` unless it was knowable on date `t`. Fundamentals indexed by
   publication date, not period-end. Enforced by the loader contract.
2. **Corporate actions handling.** Total-return series are constructed
   with explicit handling of splits, special dividends, capital returns
   (frequent on NZX — e.g. Spark, Auckland Airport historically), and
   rights issues. Naive price-only series forbidden. Adjustment factors
   are recorded per-event and auditable.
3. **Survivorship-bias remediation.** Empirical
   estimates put survivorship bias on equity backtests at ~150–400 bps/yr
   (Brown, Goetzmann, Ibbotson & Ross 1992; Elton, Gruber & Blake 1996).
   NZX has had material delistings in the relevant window (CBL, Pumpkin
   Patch, Wynyard, etc.) that a value/momentum screen could plausibly
   have been long into. Obtaining delisted-name data for the NZX is not
   feasible within the project's resource budget. **Resolution: a
   probabilistic delisting model built from a one-time research sample
   of NZX delistings, combined with a conservative flat annual return
   penalty of 400 bps/yr** (worst-case end of the evidence range) for
   gating decisions. The probabilistic model reports risk metrics as
   ranges rather than point estimates, and augments drawdown via Monte
   Carlo delisting injection. See the implementation plan §1.1 for full
   specification.
   **Known limitations** (disclosed in every methodology report):
   - The flat penalty component is an average; the probabilistic model
     addresses factor-conditional bias but relies on a small research
     sample of NZX delistings.
   - Covariance and correlation estimates (used by HRP) remain unaffected
     and therefore do not reflect delisting-event dynamics.
   - If the strategy passes gating thresholds only by a margin smaller
     than the penalty's uncertainty band, the result is
     inconclusive and must be flagged as such.
4. **Imputation policy.** With ~70-90% data coverage, missing-fundamentals
   handling is a first-class decision and must be specified per-factor:
   sector-median imputation, forward-fill within a max staleness window,
   or exclusion from that factor's ranking on that date. Default:
   exclusion from ranking (the name receives a neutral score on that
   factor). The chosen policy is part of the pre-registered configuration
   (item 13 below).
5. **Realistic transaction costs.** Backtests model Sharesies fees plus
   per-ticker spread estimates (Frazzini, Israel & Moskowitz, "Trading
   Costs" 2018, documents that ignoring these has flipped the sign of
   published factor returns at retail-realistic AUM). Naive close-to-close
   pricing forbidden.

### Universe construction
6. **Cross-sectional z-scoring with shrinkage.** Factor exposures are
   ranked within each date's universe and **shrunk toward the sector or
   universe mean** (Jorion 1986-style or Bayesian) before combining, to
   reduce the impact of noisy raw scores in a thin universe.
7. **Winsorization.** Extreme z-scores capped (e.g. ±3) before combining.
8. **Sector neutrality** (recommended). Factor scores computed within
   sector to prevent sector bets masquerading as factor bets.
9. **Minimum history per stock.** Names with insufficient history (e.g.
   <2 years) excluded per-rebalance.
10. **Liquidity filter — quantified.** A name is excluded from the
    investable universe on date `t` if the **intended position size would
    exceed 1% of trailing 20-day average daily dollar volume** (Almgren–
    Chriss-style impact bound). NZX small caps frequently trade <NZ$50k/day;
    without an explicit number the backtest silently assumes liquidity
    that does not exist.

### Statistical discipline
11. **Walk-forward validation, with explicit fold-count honesty.** Rolling
    train/test folds with the test partition strictly excluded from any
    fitting decision. Given ~10 years of clean NZX history and a ≥3-year
    training window plus 3-year OOS window, the design yields **only 2–4
    non-overlapping folds** — borderline for any statistical claim. All
    OOS performance metrics are therefore reported with **block-bootstrap
    confidence intervals** (Politis & Romano 1994), not point estimates.
    **Combinatorial Purged Cross-Validation** (López de Prado 2018, ch. 12)
    is the preferred variant; vanilla walk-forward is a fallback.
12. **Multiple-testing discipline (deflated Sharpe).** With 4-6 factors ×
    combination weights × overlay variants × universe filters, the trial
    count balloons fast. Every reported Sharpe must be accompanied by a
    **deflated Sharpe ratio** (Bailey & López de Prado 2014) accounting for
    the number of variants tried, or — at minimum — a Bonferroni-style
    adjustment over a pre-declared trial count. "Improves OOS Sharpe" is
    not a sufficient gate; "improves OOS Sharpe net of costs by enough to
    survive the multiple-testing adjustment" is.
13. **Pre-registration.** Before the full-period backtest is run, the
    factor list, hyperparameters, gating thresholds, imputation policy,
    liquidity threshold, and overlay rules are written to a frozen
    configuration file and committed. Any post-hoc change invalidates
    the OOS claim. This is the single cheapest bias-mitigation available
    (López de Prado, repeatedly).
14. **Configurable cash yield assumption.** Backtests treat cash as
    earning a user-supplied risk-free rate so defensive periods are
    evaluated fairly.

### Operational hygiene
15. **Rebalance trigger — no-trade region.** Rebalancing is not unconditional.
    A rebalance is executed only when the **expected post-cost alpha from
    drift correction exceeds expected round-trip cost** (Markowitz / Van
    Dijk-style no-trade region). Otherwise the existing portfolio is held.
16. **Position-size floor.** Trades whose dollar value falls below a
    configured economic floor (driven by the Sharesies fee structure and
    spread cost) are deferred or aggregated, regardless of signal strength.
17. **Override-leakage policy.** The user's manual review may override any
    recommendation. Overrides and their outcomes are **logged but never
    fed back into model training, factor selection, or hyperparameter
    choice**. Doing so would constitute OOS data leakage from the model's
    perspective and silently invalidate every subsequent backtest.

---

## 7. Risk Posture

Aligned with the user's stated risk-cautious stance:

- **Validation before deployment.** No strategy variant is used for live
  recommendations until it has cleared walk-forward gating against the
  primary benchmark.
- **Performance attribution every period.** Returns decomposed by factor
  and signal so "the model made money" can be distinguished from "the
  model rode market beta."
- **Drawdown ceilings.** Strategy variants exceeding a configured
  historical drawdown threshold are rejected at the gating step.
- **Manual review remains the final gate.** App emits recommendations;
  user reviews and may override based on qualitative judgement
  (industry view, fundamental concerns, personal intuition). Output
  format must support this — every recommendation cites the named
  factor exposures driving it.

---

## 8. Open Questions ~~Deferred to~~ Resolved in Implementation Plan

> **All items below were resolved in [2026-04-20-skuld-implementation-plan.md](2026-04-20-skuld-implementation-plan.md).** Kept here for traceability.

- **Survivorship bias penalty calibration.** → Resolved as probabilistic delisting model + 400 bps worst-case gating (impl plan §1.1).
- **Cash overlay vs. defensive equity sleeve.** → Both, layered (impl plan §1.2).
- **Backtesting library choice.** → Custom thin engine + riskfolio-lib (impl plan §1.3).
- **Portfolio optimiser specifics.** → Configurable, default HRP (impl plan §1.4).
- **Spread cost model.** → Flat 200 bps round-trip, configurable (impl plan §1.5).
- **Initial factor set.** → Momentum, value, quality, low-vol, size (impl plan §1.6).
- **Pre-registration artefact.** → YAML + SHA-256 hash (impl plan §1.7).
- **Cash overlay rule.** → NZX50 < 200d-MA AND aggregate momentum z < 0 (impl plan §1.2).
- **No-trade-region thresholds.** → 2× round-trip cost; $50 / 5× floor (impl plan §1.8).
- **Recommendation output format.** → CSV with full schema (impl plan §3.9).

---

## 9. Non-Goals (explicit YAGNI)

- Real-time / intraday data.
- Order execution or broker API integration.
- Multi-account or multi-user support.
- A user interface beyond the recommendation report.
- Tax-aware lot selection.
- Predictive regime-switching models.
- Any signal class not on the contender list in [plan.md](plan.md) section 4
  unless first added to that list with justification.

---

## 10. Next Step

If approved, the next deliverable is an **implementation plan** that
resolves the open questions in §8, sequences the Phase 1 build into
testable milestones, and defines the gating evidence required to proceed
to Phase 2.
