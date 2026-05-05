# Strategy Specs

Strategy specs are frozen YAML definitions for research and recommendation runs. Each spec captures the universe, factors, cost model, backtest settings, walk-forward validation, survivorship adjustment, gating, and benchmark configuration used for a strategy hypothesis.

Do not edit an existing historical spec to change a result. Create a new spec when the hypothesis changes.

## Lifecycle

`candidates/` contains active research specs that may still be rerun, compared, promoted, or used as the basis for another iteration.

`production/` contains the approved spec used for recommendation generation. It should normally contain at most one strategy spec unless the project explicitly supports multiple live strategies.

`archive/` contains frozen historical specs that are no longer candidates for production but remain as audit evidence for multiple-testing discipline and reproducibility.

## Archive Policy

Move a spec to `archive/` when one of these is true:

- It fails gating and there is no intent to rerun it unchanged.
- A newer spec supersedes it by changing assumptions.
- It was only part of a batch screen.
- It depends on an invalidated universe, cost, or data assumption.
- Its result is kept only for audit history.

Do not archive a spec when one of these is true:

- It is still an active comparison baseline.
- It is expected to be rerun unchanged after data cleaning.
- It is the current production recommendation spec.
- It has not yet been evaluated.

A candidate can move to `production/` only when it passes gating, known data-quality issues affecting the result are resolved or explicitly accepted, and the production rationale is documented here.

## Naming

- Use hyphens in filenames and YAML `name:` values.
- Do not include dates in filenames.
- Do not use milestone names such as `phase1` as strategy names.
- Prefer concise mechanics-based names, such as `mom-ar-spread`.

## Current Inventory

Archive:

- `archive/m8-mom.yaml` — M8 momentum-only factor screen.
- `archive/m8-mom-lowvol.yaml` — M8 momentum plus low-volatility factor screen.
- `archive/m8-mom-size.yaml` — M8 momentum plus size factor screen.
- `archive/m8-mom-lowvol-size.yaml` — M8 momentum, low-volatility, and size factor screen.
- `archive/mom-baseline.yaml` — momentum baseline selected after M8 factor evaluation.
- `archive/mom-overlay.yaml` — momentum baseline with NZX MA200 and aggregate momentum cash overlay.
- `archive/mom-ar-spread-scrubbed-overlay.yaml` — scrubbed AR-spread variant with the defensive cash overlay; archived after methodology review, with deterministic overlay trigger/non-trigger tests retained.
- `archive/mom-smooth3.yaml` — early 3-month-smoothed momentum candidate; archived after stronger repaired-data descendants superseded it.
- `archive/mom-smooth3-quarterly.yaml` — quarterly smoothed-momentum candidate; archived after failing current gating despite an interesting point estimate.
- `archive/mom-no-mcap.yaml` — no-market-cap momentum variant; archived after weak net performance and high costs.
- `archive/mom-ar-spread.yaml` — first AR-spread candidate; archived after newer repaired-data descendants superseded it.
- `archive/mom-ar-spread-scrubbed.yaml` — scrubbed AR-spread candidate; archived after stronger repaired-data descendants superseded it.
- `archive/mom-ar-spread-scrubbed-v2.yaml` — AR-spread refinement with turnover controls; archived after stronger repaired-data descendants superseded it.
- `archive/mom-dividend-yield-ar-spread-scrubbed.yaml` — comparison-only dividend-yield variant; archived as a standalone candidate while retaining it as overlay/comparison evidence.

Candidates:

- `candidates/mom-s6.yaml` — scrubbed AR-spread candidate with 6-month momentum score smoothing.
- `candidates/mom-s7.yaml` — repaired-data momentum candidate with 6-month smoothing and `score_lambda=0.5`.

Production:

- `production/mom-s8.yaml` — approved momentum plus return-on-risk production baseline. Promoted after canonical `run_from_spec(...)` evaluation passed all current gating bars.
