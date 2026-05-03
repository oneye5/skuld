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

Candidates:

- `candidates/mom-smooth3.yaml` — momentum baseline with 3-month score smoothing.
- `candidates/mom-smooth3-quarterly.yaml` — smoothed momentum with quarterly rebalancing.
- `candidates/mom-no-mcap.yaml` — smoothed momentum with the market-cap filter disabled.
- `candidates/mom-ar-spread.yaml` — no-market-cap smoothed momentum with Abdi-Ranaldo per-ticker spread estimates.
- `candidates/mom-ar-spread-scrubbed.yaml` — AR-spread variant with round-trip raw-data scrubbing.
- `candidates/mom-s6.yaml` — scrubbed AR-spread candidate with 6-month momentum score smoothing.
- `candidates/mom-dividend-yield-ar-spread-scrubbed.yaml` — comparison-only dividend-yield variant; treat dividend yield as regime-conditional overlay guidance until engine support exists for regime-conditioned factor activation.

Production:

- No production spec is currently designated.
