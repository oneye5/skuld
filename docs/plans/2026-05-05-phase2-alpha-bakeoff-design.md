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

### Momentum Redefinitions

`mom-resid`
- momentum computed on returns residualized against the market rather than on raw returns
- intent: favor stock-specific strength rather than broad market drift

`mom-betaadj`
- momentum adjusted by market beta or market-model fit
- intent: reduce the chance that apparent alpha is just compensated market exposure

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

`mom-drawdownaware`
- penalize momentum names with large recent drawdowns during the lookback path
- intent: reward persistent uptrends and avoid fragile rebound-like profiles

`mom-hitrate`
- combine baseline momentum with the fraction of positive months or other path-breadth measure
- intent: distinguish broad participation from narrow episodic gains

### Risk-Shaped Candidates

`mom-vol-penalized`
- baseline momentum with an explicit penalty for recent realized volatility or idiosyncratic volatility
- intent: keep the signal but prefer names with cleaner risk shape

`mom-lowcrash`
- baseline momentum with a penalty for left-tail behavior, sharp reversals, or recent crashiness
- intent: reduce momentum crash exposure without changing the whole framework

### Optional Fundamentals Bucket

Fundamental candidates are allowed only if the exact input is already judged PIT-safe enough for promotable research. If publication-date fidelity remains unresolved, these stay exploratory-only and should not be shortlisted for production.

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
