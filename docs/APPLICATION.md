# Application Structure

## Three Cooperating Packages

Skuld is implemented as three Python packages sharing a common contract layer. Each package has exactly one job.

**`skuld_common`** holds the shared contract types, configuration schema, and validation utilities. This is the lingua franca of the system: `PITSnapshot`, `PreparedPanel`, `CombinedScores`, `TargetPortfolio`, `TradeList`, and `BacktestSpec` all live here. Any change to a contract type is visible immediately to both consumers. `skuld_common` has no dependency on the other two packages.

**`skuld_research`** is the research and backtesting pipeline. It ingests raw data via the PIT loader, builds prepared panels, generates factor signals, runs walk-forward backtests, applies statistical gating, and produces frozen pre-registered spec files and methodology reports. It never runs in production. Its outputs — immutable YAML spec files and methodology reports — are the artefacts that authorise the production system.

**`skuld_portfolio`** is the production recommendation app. It consumes a frozen spec from `skuld_research`, a Sharesies CSV holdings export, and a user-supplied cash value. It runs the same pipeline as `skuld_research` through a shared entry point, feeds the result into the execution planner, and writes a recommendation CSV and sidecar JSON. It is deterministic, auditable, and produces no output unless a validated spec is available.

Both `skuld_research` and `skuld_portfolio` share the `run_from_spec(spec, panel)` entry point so the two apps cannot diverge in their interpretation of a spec file.

---

## The Data Flow

Java ingestion produces a single long-format CSV from ~14 data sources. The Python pipeline consumes it through the following stages, each producing a well-typed contract:

```
Java ingest  →  data_long.csv
     │
     ▼
CSV Loader         (validation: negative prices, gaps, stale data)
     │
     ▼
PIT Loader         (no-lookahead enforcement → PITSnapshot)
     │
     ▼
PreparedPanel      (corporate actions, total returns, universe mask)
     │
     ├── Signal generators (one per factor → per-ticker score Series)
     │
     ▼
Signal Combiner    (z-score, winsorise, shrink, average → CombinedScores)
     │
     ▼
Portfolio Constructor  (HRP/risk-parity/inverse-vol → TargetPortfolio)
     │
     ▼
Cash Overlay       (rules-based equity/cash split, if enabled and gated)
     │
     ▼
Execution Planner  (fee-cliff optimisation → TradeList)
     │
     ▼
Output             (recommendations CSV + meta.json sidecar)
```

The PIT loader is the single point of lookahead enforcement. Every stage downstream receives only data that was knowable before the rebalance date `t`. This invariant is tested in CI with synthetic future-dated rows.

---

## Research vs Production

**Research** runs walk-forward: the pipeline is applied at every rebalance date within a train/test fold sequence. A `BacktestSpec` YAML is pre-registered (content-addressed by SHA-256, committed, immutable by CI rule) before any full-period backtest. This prevents post-hoc parameter tuning from invalidating out-of-sample claims. The walk-forward result passes through statistical gating before a spec is considered for production.

**Production** runs point-in-time: the pipeline is applied once, at the current date, using a frozen spec that has already passed gating. The execution planner then converts the target portfolio into a trade list respecting the Sharesies fee structure.

The spec file is the handoff between the two modes. It captures every parameter: factor list, weights, universe filters, optimiser, cost model, overlay rules, gating thresholds. Nothing outside the spec is allowed to vary between a backtest run and a production recommendation run.

---

## Statistical Gating

Every candidate spec must clear four bars before being considered for production:

1. **Out-of-sample Sharpe improvement** net of transaction costs and survivorship haircut over the primary benchmark (NZX equal-weighted fixed universe).
2. **Deflated Sharpe** (Bailey & López de Prado 2014): the Sharpe is adjusted downward for the total number of specs tried, as recorded in the trial ledger. Starting count is 30 (reflecting informal hypothesis tests in Milestones 0.5–4); each new production candidate increments this count.
3. **Stationary bootstrap confidence interval** (Politis & Romano 1994): the 95% CI on OOS Sharpe must not straddle zero.
4. **Romano-Wolf multiple-testing correction**: strategy must dominate each benchmark with adjusted p-value < 0.05.

A spec that fails any of these bars is not deployed. The decision is recorded in a decision file with the gate output verbatim. Failed or parked specs remain in the codebase for reproducibility — they are not deleted.

The trial ledger is split into `production` (increments the deflation count) and `exploration` (does not). This allows free experimentation without silently inflating the production deflation penalty.

---

## Phase Status

**Phase 1 (factor model):** complete. All milestones (0.5–10) are implemented. The production baseline is **`mom-s8` momentum plus return-on-risk**. Yahoo fundamentals now reach the research pipeline, but the first two simple fundamental factors tested so far underperformed the older plain momentum comparison: `book_to_market` is kept as a research-only comparison factor, while `ocf_to_assets` is parked. Low-volatility and size FAIL the gating bar. The project is in the Phase 2 alpha-bakeoff window.

**Phase 2 active direction:** run the exploration-first alpha candidate funnel described in `docs/plans/2026-05-05-phase2-alpha-bakeoff-design.md`. Phase 2 now starts with a broad `exploration` sweep of PIT-safe `mom-s8` extensions, then promotes only a small set of distinct finalists into a frozen `production` bake-off against `mom-s8`. The executable exploration specs live under `python/configs/strategy-specs/candidates/phase2-*.yaml`, and `python/scripts/phase2_exploration.py` writes the candidate-vs-`mom-s8` summary report. Sector-dependent candidates are excluded from promotable evaluation unless PIT-safe historical sector membership becomes available. Multi-strategy combination remains deferred to a separate Phase 2B design.

The equal-weighted NZX basket remains the documented fallback if no approved production spec remains deployable.

---

## Java Ingestion Layer

The Java layer is intentionally separate and feeds the Python pipeline via the long-format CSV. It does not implement any signal or portfolio logic. Its sole responsibility is data collection and normalisation.

Data sources self-register into an `IngestManager` singleton at construction time. All sources are fetched in parallel. The output is a single CSV with a uniform schema: `(timestamp, ticker, feature, value, src)`. The Python side routes rows to the appropriate frame by `feature` name and ticker presence; the `src` column is used for audit and staleness reporting only.

---

## Key Design Principles

**Separation of concerns:** research and portfolio apps share contracts and a single `run_from_spec` entry point but are otherwise independent. The research app cannot write production recommendations; the portfolio app cannot modify pre-registered specs.

**Pre-registration before back-testing:** the complete spec — factor list, hyperparameters, universe filters, fold definitions — is frozen and committed before any full-period backtest. Post-hoc changes invalidate the OOS claim. This is the cheapest available bias mitigation.

**Override isolation:** the user's manual review may override any recommendation. Overrides are logged but never fed back into model training, factor selection, or hyperparameter choice. Doing so would constitute OOS data leakage.

**Reproducibility:** every stochastic component (bootstrap, Monte Carlo delisting injection, optimiser tie-breakers) draws from child seeds derived deterministically from a single `master_seed` in the spec. Two consecutive runs on the same spec and data produce byte-identical numeric output.
