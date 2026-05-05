# Phase 2 Alpha Candidate Funnel

Status: Design
Related: `docs/APPLICATION.md`

## Objective

Phase 2 answers one question: is there any PIT-safe extension of `mom-s8` that is strong enough to justify a fresh production challenge against the current baseline?

Phase 2 is explicitly a funnel:

1. broad `exploration` search across many candidate ideas
2. narrow promotion of the strongest distinct finalists into `production`
3. frozen, pre-registered production bake-off against `mom-s8`

The purpose of `exploration` is search, ranking, and pruning. The purpose of `production` is a defensible deployment claim.

## Constraints

- `mom-s8` remains the live baseline until a new production candidate passes.
- All new ideas start in `exploration` ledger scope.
- Only a small number of finalists should be promoted to `production`, because each production candidate increases the deflated-Sharpe trial count.
- Sector-dependent candidates are not promotable under the current data reality, because the project does not have PIT-safe historical sector membership.
- Multi-strategy production combination is out of scope for this phase.

## Data Rule

Anything that relies on sector classifications is excluded from promotable evaluation unless PIT-safe sector membership becomes available.

This includes:

- sector-relative momentum
- sector-neutral ranking
- sector overlays
- any candidate whose score construction depends on sector group membership

Sector-based analysis may still be useful for diagnostics, but it must remain `exploration` scope and must not be used to justify production promotion.

## Phase Structure

### 1. Exploration Sweep

Run a broad set of PIT-safe candidates in `exploration` scope. The goal is to map the idea space, not to make a production claim.

Exploration should answer:

- does the idea improve on `mom-s8` directionally?
- is it stable across subperiods?
- does it survive costs reasonably?
- is it meaningfully different from other candidates?
- is the extra complexity justified?

### 2. Promotion Screen

Promote only a small number of finalists into `production`.

A candidate is promotable only if it is:

- PIT-safe
- mechanically simple enough to explain and freeze
- materially different from other shortlisted candidates
- not obviously dominated by a simpler nearby variant
- reasonably stable in subperiods and under cost stress
- operationally viable on turnover, capacity, and coverage

Exploration success is not proof. Promotion means the idea is strong enough to deserve a fresh, pre-registered production test.

### 3. Production Bake-Off

Only after the shortlist is chosen:

1. freeze the finalist specs
2. register them as `production`
3. run the canonical gate stack against `mom-s8`
4. record PASS / PARK / DEMOTE decisions

Any post-results modification creates a new hypothesis and requires a new spec identity.

## Exploration Candidate Set

The exploration set should be broad and intentionally cover different ideas, not many tiny variations of the same idea.

### Baseline Extensions

`mom-s8-stack-lite`
- `mom-s8` plus 1-month reversal plus 60-day idiosyncratic volatility
- intent: test whether a small hand-built multi-signal rank beats plain momentum

`mom-s8-resid-lite`
- the same inputs as `mom-s8-stack-lite`, but combined by a small rolling ridge model
- intent: test whether a tiny learned combiner earns its complexity over fixed weights
- note: deprioritised. The NZX cross-section (~40–60 stocks per rebalance) is too thin for a rolling ridge fit to find signal rather than noise. If `mom-s8-stack-lite` fails, this variant is unlikely to clear the incremental bar independently. Operational complexity (lookahead risk in rolling fit, additional hyperparameter) is disproportionate to expected gain.

### Momentum Redefinitions

`mom-resid`
- momentum computed on returns residualized against the market rather than on raw returns
- intent: favor stock-specific strength rather than broad market drift
- priority: highest in the set. Blitz, Huij & Martens (2011) showed residual momentum earns ~2× the Sharpe of total-return momentum, confirmed out-of-sample by Huij & Lansdorp (2017) across global universes. The mechanism is direct: conventional momentum accumulates time-varying factor tilts during formation that reverse when factors reverse, driving crash episodes. On NZX specifically, where the market is small and highly correlated (yield stocks, property, utilities), raw momentum is more likely to embed a market-beta tilt. Residualising strips this.
- implementation note: full Fama-French three-factor residuals require locally constructed size and value factors, which will be noisy on ~60 NZX names. Prefer a simpler market-model residual (rolling OLS against FNZ.NZ over the same formation window). This is more robust and easier to freeze in a production spec.

`mom-betaadj`
- momentum adjusted by market beta or market-model fit
- intent: reduce the chance that apparent alpha is just compensated market exposure
- note: largely dominated by `mom-resid`. Beta-adjusting returns is the single-factor simplification of the residual momentum idea. Given `mom-resid` uses rolling OLS against the same market proxy and is the academically stronger formulation, both candidates will be highly correlated in exploration results. Run both but do not promote both. If `mom-resid` performs, `mom-betaadj` should not occupy a separate shortlist seat.

`mom-dual-horizon`
- combine two momentum horizons, such as medium-term and longer-term momentum
- intent: test whether the baseline horizon is too narrow

`mom-ex-short-spike`
- momentum that downweights or excludes the most recent short-term burst component
- intent: avoid buying names whose signal is mostly one sharp move

### Path-Quality Candidates

`mom-consistency`
- prefer stocks with steadier momentum paths rather than one-jump winners
- intent: test whether smoother trends are more robust than raw return ranking
- implementation: measure as the information ratio of monthly returns over the lookback (mean monthly return divided by its standard deviation), or alternatively as the fraction of positive months in the lookback window. Both formulations capture path breadth and are strongly correlated; the IR formulation is preferred as it captures both direction and magnitude of smoothness. The positive-month fraction variant (`mom-hitrate`) is absorbed here as an implementation alternative rather than a separate exploration candidate.

`mom-drawdownaware`
- penalize momentum names with large recent drawdowns during the lookback path
- intent: reward persistent uptrends and avoid fragile rebound-like profiles

`mom-hitrate`
- combine baseline momentum with the fraction of positive months or other path-breadth measure
- intent: distinguish broad participation from narrow episodic gains
- note: absorbed into `mom-consistency`. The positive-month fraction and the IR of monthly returns are highly correlated (typically r > 0.7). Running both as separate candidates consumes a shortlist slot without adding a meaningfully independent idea. Implement hitrate as a variant formulation within the `mom-consistency` exploration spec.

### Risk-Shaped Candidates

`mom-vol-penalized`
- baseline momentum with an explicit penalty for recent realized volatility or idiosyncratic volatility
- intent: keep the signal but prefer names with cleaner risk shape
- priority: second-highest in the set. Barroso & Santa-Clara (2015) and Daniel & Moskowitz (2016) both showed that scaling momentum positions by realized volatility nearly doubles the Sharpe ratio and eliminates most crash episodes. The mechanism is direct: momentum crashes occur when a high-return formation period tilts the portfolio toward high-beta names that reverse sharply; vol-scaling reduces exposure exactly when this risk is elevated. Note that `mom-s8` already incorporates `return_on_risk` at the signal-ranking level, but that is a ranking input rather than a score-level vol penalty. `mom-vol-penalized` as a per-stock score adjustment is a distinct improvement. On NZX, where individual stocks can be highly volatile, this is particularly motivated.

`mom-lowcrash`
- baseline momentum with a penalty for left-tail behavior, sharp reversals, or recent crashiness
- intent: reduce momentum crash exposure without changing the whole framework
- note: likely dominated by `mom-vol-penalized` on NZX. Left-tail statistics (skewness, crash frequency) estimated over 12-24 months on stocks with sporadic NZX trading carry high estimation noise. The crash protection this candidate targets is largely captured through the volatility channel in `mom-vol-penalized` — high-vol stocks tend to have worse crash profiles. If `mom-vol-penalized` is in the exploration set, `mom-lowcrash` is a lower-priority variant. A max-drawdown-during-lookback penalty achieves a similar effect with less estimation noise and may be worth testing as an implementation variant of `mom-drawdownaware` rather than as a standalone.

### Optional Fundamentals Bucket

Fundamental candidates are allowed only if the exact input is already judged PIT-safe enough for promotable research. If publication-date fidelity remains unresolved, these stay exploratory-only and should not be shortlisted for production.

### Additional Candidates

`mom-52wh`
- momentum score based on proximity of current price to the 52-week high, rather than trailing cumulative return
- intent: capture anchoring-driven underreaction as a distinct signal from trailing-return momentum
- rationale: George & Hwang (2004, *Journal of Finance*) showed that the ratio of current price to 52-week high is a stronger predictor of future returns than 12-1 returns alone, and this has replicated in international markets. The mechanism is investor anchoring on the 52-week high as a reference point, creating underreaction as prices approach it. The signal is simple, PIT-safe (price data only), lower-turnover than raw return momentum (changes more slowly), and is genuinely non-redundant with `mom-s8` in signal construction. A combined score of 52-week high proximity plus `mom-s8` return signal would represent the strongest baseline extension with the most distinct academic pedigree.

`mom-ts-filter`
- cross-sectional momentum score gated or discounted by whether the stock is in an absolute uptrend
- intent: protect against buying the best of a falling universe during broad market drawdowns
- rationale: Moskowitz, Ooi & Pedersen (2012) documented that a stock's own past return sign (absolute, not relative) predicts its future return independently of cross-sectional rank. Overlaying a time-series gate — for example, scoring a stock zero or applying a decay factor when it is below its 12-month moving average — prevents the cross-sectional strategy from rotating into names that are all trending down together. On NZX, where the small market is highly correlated and entire sectors can fall simultaneously, this overlay addresses a known weakness of pure cross-sectional strategies. Implementation is PIT-safe, simple to specify and freeze, and adds no material turnover cost.

## Exploration Priority and Shortlist Guidance

The exploration set is intentionally broad, but not all candidates have equal prior probability of clearing the +0.10 incremental bar. The following priority notes should inform how exploration resources are sequenced and how the shortlist is assembled.

**High-priority candidates** (strongest academic backing, most distinct from `mom-s8`):
- `mom-resid` — highest priority. The residual momentum literature is the most robust enhancement to cross-sectional momentum, and the NZX market-correlation problem makes it especially motivated here. Implement as a market-model residual (rolling OLS against FNZ.NZ) rather than full FF3 to keep estimation stable on the small NZX universe.
- `mom-vol-penalized` — second-highest priority. Volatility-managed momentum is one of the most replicated enhancements in the crash-protection literature. `mom-s8`'s existing `return_on_risk` factor is a ranking input, not a score-level vol penalty; the two are distinct improvements.
- `mom-52wh` — new, high-priority. Anchoring-based signal construction is non-redundant with trailing-return momentum and has strong international replication. Should be run early alongside `mom-resid`.

**Medium-priority candidates** (motivated but expect modest incremental gain over `mom-s8`):
- `mom-consistency` (absorbing hitrate as a variant formulation)
- `mom-drawdownaware` — watch NZX universe coverage; a drawdown penalty on a 40-60 stock universe may leave too few eligible names per rebalance
- `mom-ts-filter` — new; simple overlay but directionally important for bear-market protection on a correlated small market
- `mom-dual-horizon` — expect modest gain given `mom-s8` already uses smoothed formation

**Deprioritised candidates** (run in exploration but should not consume shortlist seats):
- `mom-betaadj` — dominated by `mom-resid`; correlated results expected
- `mom-lowcrash` — dominated by `mom-vol-penalized` on a small market with limited left-tail history
- `mom-s8-resid-lite` — ML combiner on ~50-stock universe is fitting to noise; if `mom-s8-stack-lite` fails this will too
- `mom-hitrate` — absorbed into `mom-consistency`; not a standalone candidate

**Realistic shortlist target:** `mom-resid`, `mom-vol-penalized`, `mom-52wh`, and one path-quality candidate (`mom-consistency` or `mom-drawdownaware`). That is 3–4 finalists within the permitted range and represents genuine idea diversity across signal construction, risk-shaping, and reference-level anchoring.

## Production Shortlist Size

The exploration set can be large. The production shortlist should be small.

Default target: promote the best 3-6 distinct candidates into the formal production bake-off.

The limit is not about compute cost. It exists to protect inference quality and avoid turning the final winner into the result of an overly wide production fishing expedition.

## Production Promotion Rule

Each promoted finalist is evaluated independently against `mom-s8`.

Canonical gates:

1. flat-haircut OOS Sharpe > 0
2. stationary-bootstrap Sharpe CI lower bound > 0
3. deflated-Sharpe p <= 0.05 using the effective trial count from the canonical runner
4. TD excess return p <= 0.05
5. Romano-Wolf dominance over NZX equal-weight and 60/40, adjusted p <= 0.05

Phase 2 incremental bar:

6. candidate flat-haircut OOS Sharpe must exceed `mom-s8` by at least `+0.10`, and the paired stationary-bootstrap CI on candidate-minus-baseline monthly OOS returns must have positive median delta and lower bound >= 0

A candidate that passes the standard gates but fails the incremental bar is not promotable. It may be an interesting exploration result, but it is not new deployable alpha.

## Required Reporting

### Exploration report

The exploration phase should produce a single summary showing:

- candidate definitions and frozen exploration specs
- headline metrics versus `mom-s8`
- turnover, cost drag, and capacity flags
- subperiod behavior
- redundancy notes between similar candidates
- shortlist recommendation with reasons for inclusion or exclusion

### Production decision records

Each promoted finalist gets its own decision record following the established factor-decision pattern.

Every production decision record must include:

- gate output verbatim
- effective trial count
- paired-delta result versus `mom-s8`
- turnover and cost drag
- capacity / binding-constraint summary
- any degraded rebalance counts where applicable

## Testing And Validation

Before trusting candidate results:

1. baseline `mom-s8` metrics must reproduce unchanged after any new infrastructure lands
2. each new candidate type must have targeted unit coverage for its score construction and leakage guards
3. candidate runs must be deterministic under the same seed
4. each shortlisted production candidate must pass an end-to-end smoke test

## Out Of Scope

- sector-based promotable candidates under current data constraints
- automatic multi-strategy production combination
- direct deployment from `exploration`
- hyperparameter fishing inside `production`
- retuning `mom-s8` itself
- production monitoring and scheduled operations
- bear-regime remediation

## Success Criteria

Phase 2 is complete when:

1. a broad PIT-safe exploration sweep has been run and documented
2. a small set of distinct finalists has been promoted into `production`
3. each finalist has a recorded PASS / PARK / DEMOTE decision under frozen conditions
4. baseline `mom-s8` metrics remain unchanged after the supporting infrastructure is merged
5. `APPLICATION.md` is updated with the resulting production state

Promotion is optional. A successful Phase 2 may end with no promoted replacement if the exploration funnel was broad, the shortlist was disciplined, and the final production tests were clean.
