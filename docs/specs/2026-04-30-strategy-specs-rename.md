# Strategy Specs Rename

## Purpose

Rename and reorganize the preregistered strategy YAML files so their paths and names explain their purpose without relying on dated or milestone-oriented labels such as `phase1`.

## Current Problem

The current folder, `python/configs/preregistered/`, correctly signals immutability but does not explain what the files are for. The filenames mix dates, milestones, and implementation details, which makes it hard to tell whether a spec is historical, active, or production-ready. Names such as `phase1_ar_spread` describe chronology rather than strategy mechanics.

## Target Layout

Use `strategy-specs` as the top-level concept and split specs by lifecycle:

```text
python/configs/strategy-specs/
  README.md
  archive/
    m8-mom.yaml
    m8-mom-lowvol.yaml
    m8-mom-size.yaml
    m8-mom-lowvol-size.yaml
    mom-baseline.yaml
    mom-overlay.yaml
  candidates/
    mom-smooth3.yaml
    mom-smooth3-quarterly.yaml
    mom-no-mcap.yaml
    mom-ar-spread.yaml
  production/
```

## Naming Rules

- Use hyphens consistently in filenames and YAML `name:` values.
- Do not include dates in filenames.
- Do not use `phase` prefixes.
- Prefer concise strategy-mechanics names, for example `mom-ar-spread` instead of `phase1_ar_spread`.
- Preserve dates and historical rationale in comments, reports, or `README.md`, not in filenames.

## Lifecycle Policy

`candidates/` contains active research specs that may still be rerun, compared, promoted, or used as the basis for another iteration.

`production/` contains the approved spec used for recommendation generation. It should normally contain at most one strategy spec unless the project explicitly supports multiple live strategies.

`archive/` contains frozen historical specs that are no longer candidates for production but remain as audit evidence for multiple-testing discipline and reproducibility.

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

A candidate can move to `production/` only when it passes gating, known data-quality issues affecting the result are resolved or explicitly accepted, and the production rationale is documented in `strategy-specs/README.md`.

## Implementation Impact

The rename intentionally changes YAML `name:` values as well as file paths. This will change spec hashes and report names going forward. Tests and references that load specific paths or assert specific hashes must be updated to the new names and hashes.

The loader should discover specs recursively under `strategy-specs/`, not only a flat `preregistered/` directory. Scripts and help text should refer to strategy specs instead of preregistered configs.

## Acceptance Criteria

- All existing strategy YAMLs are moved to the new layout.
- Every moved YAML has a hyphenated `name:` matching its filename stem.
- The `phase` prefix is removed everywhere it only refers to old naming.
- `strategy-specs/README.md` documents the lifecycle policy and current spec inventory.
- Code, tests, scripts, and docs reference the new paths.
- Relevant Python tests pass after updating expected hashes and paths.
