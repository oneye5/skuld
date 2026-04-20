# Skuld — High-Level Implementation Plan

**Status:** Draft for review
**Date:** 2026-04-20
**Builds on:** [2026-04-19-skuld-architecture-design.md](2026-04-19-skuld-architecture-design.md), [plan.md](plan.md), [DATA_ANALYSIS.md](DATA_ANALYSIS.md)

This document resolves the open questions in §8 of the architecture spec, fixes the data contracts that flow between every pipeline stage, and sequences the Phase 1 build into testable milestones. **Bite-sized TDD task lists are derived per milestone in a follow-up plan before code is written for that milestone** (per the `writing-plans` skill).

---

## 1. Resolved Decisions (closing architecture §8)

Each decision is followed by the evidence it rests on. Decisions made under "worst-case bias" are explicitly flagged.

### 1.1 Survivorship-bias remediation: **probabilistic delisting model + flat haircut**

**Background:** Brown, Goetzmann, Ibbotson & Ross (1992) and Elton, Gruber & Blake (1996) jointly bracket survivorship bias on equity backtests at ~150–400 bps/yr. The NZX has had material delistings inside the available history window (CBL Insurance, Pumpkin Patch, Wynyard, Intueri, Eroad's near-miss) that a value or momentum screen could plausibly have been long into.

**Problem with a flat haircut alone:** A flat annual return penalty addresses average-return bias but structurally understates drawdown and tail risk. Delistings are typically −80% to −100% on the position, concentrated in time. Smoothing them into a lower average return is dishonest about the risk profile — the Sharpe ratio looks better than the lived experience. Additionally, covariance and correlation estimates (used by HRP) remain unaffected by a flat penalty and therefore do not reflect delisting-event dynamics.

**Approach: one-time delisting research sample + probabilistic risk overlay.**

1. **One-time agentic research task (Milestone 0.5).** Before any backtesting, compile a dataset of NZX delistings over the available history window (~2000–2026) via public records (NZX announcements archive, Companies Office, news archives). For each delisting, record: ticker, date, reason (voluntary/involuntary/merger), terminal return (last traded price vs. price 12 months prior), and which factor quintiles the stock occupied at delisting (value trap? momentum loser?). This is a manual research task, not an automated data feed — it produces a small static CSV (`research/data/nzx_delistings.csv`) committed to the repo.

2. **Derive delisting statistics.** From the research sample, compute:
   - **Annual delisting rate** $p$ (delistings per year / universe size).
   - **Conditional loss distribution** given delisting: mean $\mu_d$, standard deviation $\sigma_d$.
   - **Factor-conditional delisting rates**: $p_{\text{value\_q5}}$, $p_{\text{momentum\_q1}}$, etc. — the probability of delisting given the stock is in a particular factor quintile. This captures the documented correlation between survivorship bias and value/momentum factors (the flat haircut's blindspot).

3. **Reporting: ranges, not point estimates.** Instead of a single haircut number, the methodology report presents:
   - **Central return penalty**: $p \cdot \mu_d$ (expected annual drag from delisting risk).
   - **Return penalty range**: 10th–90th percentile via bootstrap resampling of the delisting sample.
   - **Drawdown augmentation**: for each walk-forward fold, report both the observed max drawdown and an **augmented max drawdown** computed by Monte Carlo injection of delisting events at rate $p$ with loss drawn from the empirical $\mu_d, \sigma_d$. This gives a realistic range of drawdown outcomes that the survivor-only backtest cannot.
   - **Factor-adjusted Sharpe**: the headline Sharpe is reported alongside a **delisting-risk-adjusted Sharpe** that applies factor-conditional delisting rates to the strategy's actual factor exposures, not a blanket number. This is more honest than a flat 400 bps for a strategy that may tilt toward or away from delisting-prone quintiles.

4. **Sensitivity and worst-case.** The OOS gating step reports results at 150, 300, and 400 bps flat haircut *as well as* the probabilistic model's central estimate and 90th-percentile estimate — so it is explicit which gating decisions are sensitive to model choice (architecture §8, item 1). Per the user's worst-case preference, the **gating decision** uses the more conservative of the flat 400 bps and the probabilistic 90th-percentile penalty.

**Why this is better than a flat haircut alone:** A flat haircut is a blunt instrument borrowed from US large-cap literature and applied to a 146-name small-market universe where the dynamics are materially different. The probabilistic model: (a) uses actual NZX delisting data rather than US estimates, (b) captures the fat-tailed nature of delisting losses in drawdown reporting, (c) accounts for factor-conditional delisting risk (value traps, momentum crashes), and (d) reports ranges rather than false-precision point estimates. The one-time research cost is ~2–4 hours; the payoff is honest risk reporting for the life of the project.

### 1.2 Cash sleeve: **rules-based fraction, user-chosen instrument**

The user clarified that "cash" is whatever cash-equivalent the user prefers (TD, HISA, bond ETF). Skuld decides the *fraction* of NAV held as cash; the user decides the *instrument*. This collapses the architecture's "cash overlay vs. defensive equity sleeve" question into a **both, layered** answer:

1. **Always-on defensive tilt inside the equity sleeve.** Low-vol and quality factors carry positive default weight in the factor combiner (§3.4). This captures most drawdown protection without a small-sample regime trigger — the well-evidenced part of defensive investing (Frazzini/Pedersen "Betting Against Beta" 2014; Asness/Frazzini/Pedersen "Quality Minus Junk" 2019).
2. **Rules-based cash fraction (Phase 1.5, gated).** Cash fraction defaults to a configured floor (e.g. 5%) and rises when a regime trigger fires. Trigger: **NZX50 below its 200-day moving average AND aggregate cross-sectional momentum z-score < 0**. The dual condition is deliberate — Faber (2007)'s single-MA rule replicates poorly on small markets per Clare/Seaton/Smith/Thomas (2013); requiring two independent confirmations reduces false positives at the cost of slower entry, an acceptable trade-off for a monthly-rebalance retail strategy.
3. The cash overlay (item 2) is **only switched on if it clears the §3 gating bar in walk-forward testing**. Until then, the defensive equity tilt (item 1) carries all defensive duty and the cash fraction stays at the configured floor.

The user inputs the dollar value of cash holdings (Sharesies wallet + external cash equivalents). Liquidity of non-share cash assets is the user's responsibility, per spec.

### 1.3 Backtesting framework: **thin custom monthly engine + `riskfolio-lib` for optimisation**

Evaluated against the §6 data-integrity requirements:

| Library | PIT enforcement | Corporate actions | Custom HRP | Verdict |
|---|---|---|---|---|
| `vectorbt` | Vectorised; PIT requires careful index discipline by the user — easy to leak | Manual | Manual | Overkill, leakage-prone |
| `backtrader` | Event-driven, OK by default | Manual | Manual | Heavy; daily-bar event loop wasted on monthly rebalance |
| `zipline-reloaded` | Strong PIT primitives | Built-in (US-centric) | Manual | Heaviest; corporate-actions support assumes US data feeds |
| `bt` (pmorissette) | OK | Manual | Manual | Closest fit; still doesn't model the Sharesies fee cliff cleanly |
| **Custom thin engine** | Enforced by loader contract (§3.1) | Built into loader | `riskfolio-lib` | **Chosen** |

**Rationale:** A monthly rebalance loop on ~150 names is ~300 lines. The leakage surface is the *loader contract*, not the *backtest engine*. Owning the engine lets us enforce the loader contract directly, model the Sharesies fee cliff at $5k/mo natively, and keep the no-trade-region logic auditable. The portfolio optimiser — the genuinely complex bit — is delegated to `riskfolio-lib` ([Cajas, 2021](https://riskfolio-lib.readthedocs.io/)), which implements HRP, HERC, risk parity, mean-variance with multiple shrinkage estimators, and Black-Litterman behind a uniform interface.

This honours the "don't reinvent the wheel" mandate where it counts (optimisation, statistics) while keeping the leakage-critical surface in our hands.

### 1.4 Portfolio optimiser: **configurable, default HRP**

Optimiser is selected per-config, not hard-coded. Supported methods (all via `riskfolio-lib`):

| Method | When it wins | Reference |
|---|---|---|
| **HRP** *(default)* | Noisy, ill-conditioned covariance — exactly the 150×150 NZ case | López de Prado (*JPM* 2016) |
| HERC | HRP variant with hierarchical risk contribution | Raffinot (2018) |
| Risk parity | Stable, transparent; no covariance inversion | Maillard/Roncalli/Teïletche (2010) |
| Mean-variance + Ledoit-Wolf shrinkage | Benchmark; stress-test |  Ledoit & Wolf (2004) |
| Black-Litterman | If a tactical view layer is added later | Black & Litterman (1992) |

Each method is a backtestable variant under the §6.12 deflated-Sharpe trial budget. Phase 1 ships with HRP as the default and at least one alternative (risk parity) wired up so the comparison is real.

### 1.5 Spread cost model: **flat 200 bps round-trip** *(worst-case)*, configurable

- Per-ticker spread data for NZX small caps is not cleanly available historically; a per-ticker model would be a fitted parameter we cannot validate.
- A flat assumption removes that fitting risk at the cost of overstating costs for liquid large-caps and possibly understating for illiquid micro-caps.
- **200 bps round-trip = 100 bps per side**, sourced as a worst-case anchor from NZX market quality data and analogue ASX small-cap effective-spread studies (which document mid-cap effective spreads of 30–80 bps and small-cap effective spreads of 100–200+ bps; e.g. Comerton-Forde & Putniņš work on Australian small caps).
- The §6.10 liquidity filter (intended position ≤ 1% of trailing 20-day ADV) already excludes the worst micro-caps where the flat 200 bps would be optimistic. The two controls are complementary.
- Configurable via the pre-registered config; spread is multiplied by a user-supplied scale factor for sensitivity runs (0.5×, 1.0×, 1.5×).

### 1.6 Initial factor set

| Factor | Definition | Lag | Imputation | Reference |
|---|---|---|---|---|
| Momentum | 12-month return excluding most recent month (12-1) | 1 trading day | Exclude from ranking if <11m history | Asness/Moskowitz/Pedersen (*JoF* 2013) |
| Value | 0.5·(book-to-market z) + 0.5·(earnings yield z) | Fundamentals lagged by publication date (no period-end peeking) | Sector-median for the missing leg if only one is missing; exclude if both missing | Fama-French (1993, 2015); AQR "Fact, Fiction" (2015) |
| Quality | 0.5·(gross-profits-to-assets z) + 0.5·(−accruals z) | Same as value | Sector-median if one leg missing; exclude if both | Novy-Marx (*JFE* 2013); Sloan (1996) on accruals |
| Low-volatility | Inverse of trailing 12-month realised daily-return volatility | 1 trading day | Exclude if <6m of returns | Frazzini/Pedersen (*JFE* 2014) |
| Size | −log(market cap), winsorised | 1 trading day | Exclude if market cap missing | Fama-French (1993) — included for the cross-sectional tilt; not for size-premium harvest |

Combined as: per-date, per-sector cross-sectional z-score → winsorise at ±3 → shrink toward sector mean (Jorion 1986 style, shrinkage intensity in config) → equal-weighted average (Phase 1 default). Weighted blend deferred to Phase 1.5 and gated.

### 1.7 Pre-registration artefact

`research/configs/preregistered/YYYY-MM-DD_<name>.yaml` — a single YAML file containing every choice that could otherwise be silently changed: factor list, factor weights, shrinkage parameters, optimiser, optimiser params, liquidity threshold, spread bps, survivorship penalty bps, no-trade-region threshold, position-size floor, cash-overlay rule (if active), universe filter, rebalance cadence, walk-forward fold definition. The file is git-committed; its SHA-256 hash is embedded in every backtest report and recommendation CSV that uses it.

### 1.8 No-trade region & position-size floor

- **No-trade region:** rebalance a position only if expected post-cost alpha from drift correction > **2 × round-trip cost** (van Dijk / Markowitz style). With 200 bps spread + Sharesies fee, the threshold is meaningful and prevents fee-bleed from cosmetic drift.
- **Position-size floor:** a buy/sell trade is executed only if its dollar value ≥ **max($50, 5 × estimated round-trip cost in dollars)**. Smaller trades are deferred or aggregated next month. The $50 floor reflects the fee-breakeven at 200 bps; the 5× multiple ensures the trade is economically meaningful, not just fee-positive.

### 1.9 Recommendation output: **CSV** (per user)

Schema in §3.7.

### 1.10 Portfolio input: **Sharesies CSV export + user-supplied cash value**

Two inputs to the `portfolio/` app at run time:

1. Sharesies "Investment activity" or "Holdings" CSV export (whichever Sharesies offers as a holdings snapshot; verified at implementation time against a real export — the parser is built test-first against a real file).
2. A small YAML the user maintains: `cash_nzd: <number>` covering Sharesies wallet balance + any external cash equivalents the user is treating as the cash sleeve.

---

## 2. Repository Layout (target)

```
skuld/
├── pyproject.toml                # workspace root (uv workspace members)
├── uv.lock                       # deterministic lockfile
├── java/                         # existing — unchanged ingest layer
├── data/                         # existing — long-format CSV from Java
├── docs/                         # existing
├── common/                       # NEW — shared types + validators
│   ├── pyproject.toml
│   └── src/skuld_common/
│       ├── contracts.py          # PITSnapshot, PreparedPanel, etc.
│       ├── config.py             # pre-registered YAML schema + loader
│       └── validation.py         # data validation utilities
├── research/                     # NEW — backtests, walk-forward, artefacts
│   ├── pyproject.toml
│   ├── configs/
│   │   └── preregistered/        # frozen, hash-stamped YAMLs
│   ├── data/
│   │   └── nzx_delistings.csv    # one-time research sample (Milestone 0.5)
│   ├── src/skuld_research/
│   │   ├── data/                 # PIT loader, panel builder, corporate actions
│   │   ├── factors/              # one module per factor; uniform interface
│   │   ├── combiner/
│   │   ├── universe/             # liquidity filter, min-history filter
│   │   ├── portfolio/            # optimiser wrappers (riskfolio-lib)
│   │   ├── overlay/              # cash overlay rules
│   │   ├── costs/                # spread + Sharesies fee model
│   │   ├── backtest/             # monthly rebalance engine
│   │   ├── stats/                # deflated Sharpe, block bootstrap, walk-forward
│   │   └── reporting/            # methodology report generator
│   └── tests/
└── portfolio/                    # NEW — monthly recommendation app
    ├── pyproject.toml
    ├── src/skuld_portfolio/
    │   ├── inputs/               # Sharesies CSV parser, cash YAML reader
    │   ├── pipeline/             # uses frozen config from research/
    │   ├── execution_planner/    # fee-cliff-aware trade list
    │   └── output/               # CSV writer
    └── tests/
```

Both Python apps consume the long-format CSV produced by the existing Java ingestion.

**Shared code: `skuld-common` as a workspace package.** The data contract types (`PITSnapshot`, `PreparedPanel`, `CombinedScores`, `TargetPortfolio`, `TradeList`, config schema) live in a third package `common/` at the repo root, managed as a `uv` workspace member. Both `research/` and `portfolio/` declare a path dependency on it (`skuld-common = { path = "../common" }` in their respective `pyproject.toml`). This ensures:
- Type definitions are single-sourced — a change to a contract type is a compile-time break in both consumers.
- No version drift between research and portfolio interpretations of the same data.
- The package is lightweight (types + validators only, no heavy dependencies).

```
skuld/
├── common/                       # NEW — shared types + validators
│   ├── pyproject.toml
│   └── src/skuld_common/
│       ├── contracts.py           # PITSnapshot, PreparedPanel, etc.
│       ├── config.py              # pre-registered YAML schema + loader
│       └── validation.py          # data validation utilities
```

**Language choice:** Python for all three packages. Justified by the dependency on `riskfolio-lib`, `pandas`, `numpy`, `statsmodels`, and the broader scientific-Python ecosystem for which there is no Java equivalent of comparable quality for this specific workload. The Java ingestion layer remains Java.

**Tooling: `uv` (Astral).** All Python packages are managed with `uv`. Dependencies are locked via `uv.lock` at the workspace root. Scripts are run with `uv run`. A root `pyproject.toml` defines the workspace:

```toml
[tool.uv.workspace]
members = ["common", "research", "portfolio"]
```

This gives deterministic, fast dependency resolution and avoids the `pip`/`venv` fragmentation that plagues multi-package Python repos. `uv` is chosen over `poetry`/`pdm` for its speed (10–100× faster cold installs), Cargo-style lockfile semantics, and native workspace support.

---

## 3. Data Contracts at Each Pipeline Stage

This section is the load-bearing part of the plan: every interface boundary has a typed contract that downstream code can rely on. Schemas are given as pandas frame layouts because that is the lingua franca of the libraries we are adopting.

### 3.1 Stage 0 — Raw long-format CSV (already produced by Java)

Source: `data/data_long.csv` (4,762,520 rows; 146 NZX tickers + 3 international + macro). See [DATA_ANALYSIS.md](DATA_ANALYSIS.md) for full statistics. Schema unchanged from current Java output:

| Column | Dtype | Notes |
|---|---|---|
| `timestamp` | int64 | Unix epoch ms, UTC |
| `ticker` | string | `XXX.NZ` or empty for macro |
| `feature` | string | snake_case |
| `value` | float64 | parsed from string at load |
| `src` | int8 | source ID; legend in `data/source_legend.csv` |

### 3.2 Stage 1 — Point-in-time loader

```python
class PITLoader:
    def as_of(self, t: pd.Timestamp) -> PITSnapshot: ...

class PITSnapshot:
    """All values knowable strictly before `t`. Enforced, not asked nicely."""
    prices: pd.DataFrame      # index=date, columns=ticker, values=adj_close
    volumes: pd.DataFrame     # index=date, columns=ticker, values=adv_dollars
    fundamentals: pd.DataFrame  # MultiIndex (ticker, publication_date), columns=field
    macro: pd.DataFrame       # index=date, columns=macro_feature
    corporate_actions: pd.DataFrame  # columns: ticker, ex_date, type, factor
    asof: pd.Timestamp        # the `t` this snapshot was built for
```

**Invariant tested in CI:** for every column in every frame returned, `column_publication_date < asof`. A test fixture intentionally containing a future-dated row must produce a snapshot in which that row is absent.

**Publication-date convention:** The raw CSV `timestamp` column is treated as the publication date — i.e., the date the observation became knowable. This is a simplifying assumption: for price data (src=6) it is exact (the close price is known at market close on that date); for macro data, the publishing agencies' release dates are close to the timestamp recorded by the Java ingestion layer. For fundamental data (src=12, YfFinances), Yahoo's timeseries API returns period-end dates, not filing dates. Since the PIT loader filters on `timestamp < asof`, and fundamental data timestamps are period-end dates, using `timestamp` directly as publication date is **conservative-by-default for fundamentals** — the period-end is always *before* the actual publication date, so we never see data earlier than it was published. The risk is reversed: we see it *later* than it was actually available (by the filing lag), which is a performance drag, not a leakage risk. This is acceptable for Phase 1.

**Rationale for not adding a separate lag:** Applying an additional +60-day lag on top of period-end timestamps would double-penalise — the data is already indexed by period-end (which predates publication), so adding a further 60 days would make fundamentals arrive ~120 days late in the backtest. The correct fix would be to source actual filing dates per company, which is a Phase 2 data-quality improvement. For Phase 1, `timestamp` as-is provides a PIT-safe lower bound.

### 3.3 Stage 2 — Shared feature prep

```python
class PreparedPanel:
    returns_daily: pd.DataFrame      # index=date, columns=ticker, total-return
    returns_monthly: pd.DataFrame    # month-end; total-return
    market_cap: pd.DataFrame         # index=date, columns=ticker, NZD
    sector: pd.Series                # index=ticker, values=GICS sector or 'Unknown'
    universe_mask: pd.DataFrame      # index=rebalance_date, columns=ticker, bool
    asof: pd.Timestamp
```

**Responsibilities:** clean, align, build total-return series with corporate actions applied (splits, dividends, capital returns, rights issues per architecture §6.2), compute market cap, attach sector. **Does not invent factor features.** Memoised derived series live here so multiple factor modules can share work without duplicating it.

### 3.4 Stage 3 — Signal generators (one per factor)

```python
@runtime_checkable
class SignalGenerator(Protocol):
    name: str
    required_data: list[DataRequest]
    def score(self, panel: PreparedPanel, t: pd.Timestamp,
              universe: list[str]) -> pd.Series:
        """Returns Series indexed by ticker, values=raw factor score.
        NaN for tickers excluded from this factor's ranking on this date."""
```

Output of each `score()` call is a `pd.Series` with `index = universe`, `dtype = float64`, name = the factor name. Missing values are NaN — the combiner handles them per the imputation policy (default: exclude from ranking, score becomes 0 after z-scoring).

### 3.5 Stage 4 — Signal combiner

Input: `dict[str, pd.Series]` (one entry per active factor).
Output:

```python
class CombinedScores:
    scores: pd.Series            # index=ticker, values=combined z, NaN-free over universe
    component_scores: pd.DataFrame  # index=ticker, columns=factor name, post-shrinkage z
    asof: pd.Timestamp
```

Per-date pipeline: cross-sectional z within sector → winsorise at ±3 → shrink toward sector mean → equal-weighted average (Phase 1) → re-z the combined score so downstream code sees a unit-variance signal.

`component_scores` is preserved end-to-end so the recommendation CSV can cite the named factor exposures driving each pick (architecture §7).

### 3.6 Stage 5 — Portfolio constructor

```python
class TargetPortfolio:
    weights: pd.Series           # index=ticker, sums to (1 - cash_floor)
    cash_weight: float           # in [0, 1]
    method: str                  # 'HRP', 'RiskParity', etc.
    asof: pd.Timestamp
```

`riskfolio-lib` supplies the optimiser. Constraints enforced here, not later:

- Per-name cap: configured (default 5%).
- Per-sector cap: configured (default 25%).
- Liquidity cap: target dollar position ≤ 1% × trailing 20d ADV (§6.10).
- Score-tilt: weight is a function of (combined score, optimiser output). Phase 1 implementation: the optimiser receives **only the top-N names with positive combined scores** (N configurable, default top quintile of the universe). The optimiser produces risk-balanced weights over this filtered set. Final weights = `optimiser_weight × max(0, 1 + λ · score)`, re-normalised to sum to `1 - cash_weight`, with λ in config. The `max(0, ...)` clamp is a safety invariant: since only positive-score names enter the optimiser, `1 + λ · score` should already be positive for reasonable λ, but the clamp prevents negative weights if λ is misconfigured. **Enforced by assertion in the portfolio constructor: all weights ≥ 0, sum within tolerance of (1 − cash_weight).**

### 3.7 Stage 6 — Cash overlay

```python
def apply_cash_overlay(target: TargetPortfolio,
                       panel: PreparedPanel,
                       rule: OverlayRule) -> TargetPortfolio: ...
```

Phase 1: `rule = NoOverlay` (cash held at configured floor only).
Phase 1.5 (after gating): `rule = NzxMA200AndAggMomentumRule(...)`.

Returns a new `TargetPortfolio` with `cash_weight` increased and equity weights re-normalised to `1 - cash_weight`. The defensive *equity tilt* is not handled here; it lives in factor weights (§3.4).

### 3.8 Stage 7 — Execution planner

```python
class CurrentPortfolio:
    holdings: pd.Series          # index=ticker, values=shares held
    prices: pd.Series            # index=ticker, values=last close NZD
    cash_nzd: float

class TradeList:
    trades: pd.DataFrame
    # columns:
    #   ticker, action ('BUY'|'SELL'|'HOLD'|'DEFER'),
    #   current_shares, target_shares, delta_shares,
    #   current_value_nzd, target_value_nzd, delta_value_nzd,
    #   est_round_trip_cost_nzd, in_no_trade_region: bool,
    #   below_size_floor: bool, deferred_to_next_month: bool,
    #   sharesies_fee_band ('flat_15'|'percent_19bps')
    total_volume_nzd: float
    total_estimated_cost_nzd: float
    asof: pd.Timestamp
    config_hash: str
```

The planner solves the Sharesies fee-cliff problem: total monthly volume up to $5,000 carries a flat $15 cost; volume beyond that is taxed at 1.9%. The planner orders trades by expected post-cost alpha and **defers low-alpha trades that push volume past $5,000 unless their expected alpha exceeds the marginal 1.9% cost**.

### 3.9 Stage 8 — Output CSV

`portfolio/` app writes `output/recommendations_YYYY-MM-DD.csv` with this schema. **Every column is required; the user's manual review depends on it.**

| Column | Type | Notes |
|---|---|---|
| `rebalance_date` | ISO date | |
| `ticker` | string | empty row at end summarises CASH |
| `action` | enum | `BUY` / `SELL` / `HOLD` / `DEFER` |
| `current_shares` | int | |
| `target_shares` | int | |
| `delta_shares` | int | signed |
| `current_value_nzd` | float | |
| `target_value_nzd` | float | |
| `delta_value_nzd` | float | signed |
| `current_weight` | float | of NAV |
| `target_weight` | float | of NAV |
| `combined_score_z` | float | from §3.5 |
| `factor_momentum_z` | float | from `component_scores` |
| `factor_value_z` | float | |
| `factor_quality_z` | float | |
| `factor_lowvol_z` | float | |
| `factor_size_z` | float | |
| `est_round_trip_cost_nzd` | float | |
| `sharesies_fee_band` | enum | `flat_15` / `percent_19bps` |
| `in_no_trade_region` | bool | |
| `below_size_floor` | bool | |
| `deferred_to_next_month` | bool | |
| `rationale` | string | one-line natural-language summary, e.g. `"Top quintile momentum + value; bottom decile vol; sector OK"` |
| `config_hash` | string | SHA-256 of pre-registered YAML; same on every row of the file |

A second file `output/overrides_log_YYYY-MM-DD.csv` is opened (empty) for the user to record manual overrides. **Per architecture §6.17, the override log is never read back into research code.**

---

## 4. Milestone Sequence

Each milestone is independently testable and produces working software. A bite-sized TDD task plan is written for each milestone *before* code is written for it.

### Milestone 0.5 — NZX delisting research sample

- Compile a dataset of NZX delistings (~2000–2026) from public records: NZX announcements, Companies Office, news archives.
- For each delisting, record: ticker, date, reason, terminal return, factor-quintile membership at delisting.
- Commit as `research/data/nzx_delistings.csv`.
- Compute summary statistics: annual delisting rate, conditional loss distribution, factor-conditional rates.

**Done when:** `nzx_delistings.csv` is committed with ≥10 entries, and a summary markdown in `research/data/delisting_summary.md` reports the annual rate, mean/σ of terminal returns, and factor-conditional rates.

### Milestone 1 — Python project skeleton + data contract

- Create `common/`, `research/`, and `portfolio/` Python projects as a `uv` workspace (root `pyproject.toml` with workspace members, `uv.lock`, ruff/pyright configured).
- Implement data validation layer in `common/validation.py`: negative-price detection, zero-volume day handling, gap detection (>5 consecutive missing trading days), and stale-data alerts per source.
- Implement Stage 0 → Stage 1 (CSV loader → `PITSnapshot`) with the no-lookahead invariant test.

**Done when:** a synthetic-data fixture proves a row dated `t+1` cannot appear in `loader.as_of(t)`, `uv run pytest` passes from a clean checkout, and the validation layer flags a synthetic negative-price row.

### Milestone 2 — Shared feature prep + corporate actions

- Implement Stage 2 (`PreparedPanel`).
- Apply corporate actions from `yf_prices` and any other source carrying them.
- Implement liquidity-filter and min-history-filter universe construction.

**Done when:** for a known historical NZX corporate action (e.g. Auckland Airport's 2017 capital return), total-return series matches a manually-computed reference within 0.1%.

### Milestone 3 — One factor end-to-end (momentum)

- Implement the `SignalGenerator` protocol and the momentum factor.
- Implement the combiner with one factor.
- Implement Stage 5 with risk parity (simpler than HRP for a one-factor smoke test).
- Run a one-shot point-in-time evaluation; print resulting weights.

**Done when:** a point-in-time call produces a sensible top-decile-momentum portfolio whose weights pass all caps in §3.6.

### Milestone 4 — Backtest engine + cost model

- Implement Stage 7 cost model (spreads + Sharesies fee cliff).
- Implement the monthly rebalance engine including the no-trade region and size-floor.
- Implement the probabilistic survivorship-bias model from §1.1: flat haircut (400 bps/yr gating default) + Monte Carlo delisting injection for augmented drawdown + factor-conditional Sharpe adjustment.
- Produce the first walk-forward backtest of the momentum-only strategy.

**Done when:** a walk-forward backtest of momentum-only over the available history runs end-to-end and produces a results object with returns, costs, turnover, drawdown (both observed and augmented), and Sharpe (both raw and delisting-adjusted).

### Milestone 5 — Full Phase 1 factor model

- Add value, quality, low-vol, size factors with the imputation policies in §1.6.
- Switch optimiser to HRP via `riskfolio-lib`; keep risk parity as a configurable alternative.
- Implement deflated-Sharpe ratio and block-bootstrap CI in `stats/`.

**Done when:** walk-forward backtest of the full factor model runs and a methodology report is generated containing OOS Sharpe + 95% block-bootstrap CI + deflated Sharpe vs. each §3 benchmark, all net of the 400 bps haircut and 200 bps round-trip spread.

### Milestone 6 — Pre-registration + benchmarks

- Build the pre-registration YAML loader; embed config hash in every artefact.
- Implement the §3 benchmark stack: NZ TD floor, 60/40 reference, NZX equal-weighted, Smartshares smart-beta replication.
- Run the full Phase 1 walk-forward against all benchmarks; produce the methodology report.

**Done when:** the methodology report numerically clears (or honestly fails) the primary benchmark in §3 of the architecture spec, and every number in it can be reproduced from the committed config.

### Milestone 7 — `portfolio/` app: recommendation generator

- Implement Sharesies CSV parser (test-first against a real export).
- Implement cash YAML reader.
- Wire Stages 1 → 7 in `portfolio/` consuming the frozen config from Milestone 6.
- Implement Stage 8 CSV writer with the schema in §3.9.

**Done when:** running `portfolio/` against the user's real Sharesies export and cash YAML produces a `recommendations_YYYY-MM-DD.csv` whose every row reconciles by hand against the strategy's stated logic.

### Milestone 8 — Cash overlay (gated)

- Implement the dual-condition cash overlay rule.
- Re-run the Milestone 6 walk-forward with the overlay added.
- **Decision point:** if the overlay improves OOS Sharpe net of costs *and* clears the deflated-Sharpe bar relative to the no-overlay baseline, switch the production config to use it. Otherwise, leave Phase 1 production at the no-overlay configuration and document the negative result.

**Done when:** a yes/no decision on the cash overlay is made on evidence and the production config is updated accordingly.

### Decision tree for inconclusive Phase 1 results

With only 2–4 non-overlapping walk-forward folds, it is plausible that **no strategy variant clears the deflated-Sharpe bar** regardless of actual quality. The plan must define what happens:

1. **Strategy clears both the primary benchmark and the deflated-Sharpe bar.** → Deploy to production config. Proceed normally.
2. **Strategy clears the primary benchmark but NOT the deflated-Sharpe bar** (i.e., positive OOS alpha but insufficient statistical significance). → **Deploy with a "low-confidence" flag** in the methodology report. The recommendation CSV is still generated, but the rationale column includes a disclaimer. The user treats recommendations as suggestive, not authoritative. The strategy runs in parallel with an equal-weight benchmark for the next 12 months of live tracking to accumulate out-of-sample evidence.
3. **Strategy does NOT clear the primary benchmark** (NZX equal-weighted, net of costs). → **Do not deploy.** Default the `portfolio/` app to producing an equal-weighted portfolio of the investable universe (the benchmark itself), which is still better than ad-hoc stock picking and costs nothing to run. Document the negative result. Investigate whether the failure is driven by the cost model (spread + fees eating the alpha), insufficient factor strength in NZ, or data quality issues. This diagnosis informs whether Phase 2 (ML residual) is worth pursuing or whether the honest conclusion is that the NZX is too thin for systematic alpha.

**Rationale:** A plan that only defines the success path is a plan that has not thought about failure. The equal-weight fallback (option 3) is not a concession — it is a disciplined, low-cost, factor-neutral portfolio that the literature shows outperforms cap-weighted indices in most markets (DeMiguel, Garlappi & Uppal, "Optimal Versus Naive Diversification", *RFS* 2009). It is a defensible default.

---

## 5. Out of Scope for This Plan (Phase 2+)

Listed for clarity; not built in Phase 1:

- ML residual layer (architecture §4 Phase 2).
- Trend-following overlay.
- Macro regime tilt.
- Sentiment signals.
- Black-Litterman tactical view layer.
- Tax-aware lot selection.

Each is added under the same gating discipline used for the cash overlay in Milestone 8.

---

## 6. Operational Concerns

### 6.1 Yahoo Finance API fragility

The entire price/fundamental data pipeline depends on Yahoo Finance's unofficial API, which has a history of breaking changes and rate limiting. For Phase 1, the mitigation is **detection, not redundancy**:

- The data validation layer (Milestone 1) reports the age of each source's most recent data point. If any source's latest timestamp is older than a configured threshold (default: 7 days for prices, 90 days for fundamentals), a warning is printed at pipeline start.
- If the Java ingestion fails or returns partial data, the user re-runs it. The existing `data_long.csv` on disk serves as the implicit cache — it is only overwritten on a successful full run.
- A fallback data source is not needed at this stage. If Yahoo becomes permanently unavailable, that is a problem worth solving then, not now.

### 6.2 Data validation layer

The CSV loader (Stage 0 → Stage 1) applies validation before any downstream consumption:

- **Negative prices:** flagged and excluded. Likely API errors.
- **Zero-volume handling for ADV:** the trailing 20-day ADV used in the liquidity filter (§3.6) is computed as the mean of the 20 most recent *trading days with non-zero volume*, not calendar days. This avoids NZX small caps with intermittent zero-volume days from having their ADV diluted to near-zero, which would exclude them despite being tradable on active days. The count of zero-volume days in the trailing 20 is reported as a separate liquidity signal.
- **Gap detection:** if a ticker has >5 consecutive missing trading days (not holidays), the gap is flagged. Prices within the gap are not forward-filled — the ticker is excluded from the universe for that rebalance date.
- **Stale fundamentals:** if a ticker's most recent fundamental data is >18 months old, it is excluded from value and quality factor rankings (stale data is worse than missing data for these factors).

### 6.3 Sharesies CSV export brittleness

Sharesies can change their CSV export format without notice. The parser (Milestone 7) is built test-first against a real export file committed as a test fixture. The parser validates the header row against an expected schema and raises a clear error with instructions if the format has changed, rather than silently misinterpreting columns.

---

## 7. Self-Review Notes

- **Spec coverage:** every architecture §8 open question is resolved in §1. Every architecture §6 data-integrity requirement is reflected in a contract (PIT loader §3.2, corporate actions §3.3, survivorship penalty §1.1, imputation §1.6, transaction costs §1.5 + §3.8, liquidity filter §3.6, walk-forward + deflated Sharpe Milestone 5, pre-registration §1.7, no-trade region §1.8, override log §3.9).
- **Worst-case bias applied where the user requested it:** survivorship penalty (gating uses max of flat 400 bps and probabilistic 90th-percentile), spread cost (top of evidence range), cash overlay gating (highest bar, off by default).
- **No placeholders:** every contract has concrete column names and dtypes; every milestone has a concrete done-when.
- **Type consistency:** the `component_scores` frame defined in §3.5 surfaces as the per-factor `factor_*_z` columns in §3.9. The `config_hash` defined in §1.7 appears in §3.8 and §3.9.
- **Failure path defined:** the inconclusive-results decision tree (§4) ensures the project has a defensible outcome even if statistical significance is unachievable with the available sample.
- **Tooling locked:** `uv` for Python dependency management, `ruff`/`pyright` for linting/typing, `pytest` for testing. No ambiguity.
- **Data contracts single-sourced:** `skuld-common` workspace package prevents drift between research and portfolio interpretations.
