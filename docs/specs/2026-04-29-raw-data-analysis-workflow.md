# Raw Data Analysis Workflow Specification

## Goal

Create a reusable raw-data analysis workflow for `data/data_long.csv` that becomes the source of truth for future feature engineering and research decisions. The workflow must produce a stable, agent-readable report and machine-readable summary artifacts that describe raw-data coverage, sparsity, temporal behavior, anomalies, and leakage risks.

## Motivation

Current research infrastructure is strong at backtesting and factor diagnostics, but weak at explaining the raw dataset that feeds those systems. The project needs a canonical raw-data analysis layer that answers questions such as:

- Which sources and features are well covered versus sparse?
- How often does each feature update, and how stale does it become?
- Which observations look anomalous or operationally suspicious?
- Which raw features appear unsafe for feature engineering because of timestamp or leakage risk?

This workflow is intended to be the root source of truth for deciding which raw inputs are usable, which need guards, and which should be excluded or repaired.

## Scope

The first version covers the raw ingested dataset only:

- `data/data_long.csv`

Included analyses:

- Dataset shape and composition
- Source coverage over time
- Feature coverage over time
- Ticker coverage over time
- Missingness and sparsity patterns
- Observation frequency and update irregularity
- Staleness patterns
- Duplicate or conflicting raw observations where detectable
- Outlier and anomaly detection on raw numeric values
- Timestamp- and shape-based leakage heuristics

Explicitly out of scope for this version:

- PIT snapshot behavior
- Prepared panel construction
- Factor scoring and IC
- Backtest or portfolio diagnostics
- Production recommendation logic

## Output Requirements

The workflow must emit outputs that work for both humans and agents.

### 1. Canonical Markdown report

Produce a deterministic Markdown report with stable section headings and a consistent ordering so future agents can parse it reliably. The report should contain concise interpretation text and link directly to the machine-readable artifacts.

Suggested location:

- `python/reports/raw_data_analysis/<run-date>/report.md`

### 2. Machine-readable artifacts

Produce structured outputs alongside the report:

- CSV files for detailed metric tables
- A single JSON summary file containing top-level findings, key counts, and paths to generated artifacts

Suggested location:

- `python/reports/raw_data_analysis/<run-date>/tables/*.csv`
- `python/reports/raw_data_analysis/<run-date>/summary.json`

## Report Structure

The report should use these top-level sections.

### 1. Dataset overview

Summarize:

- Row count
- Date range
- Unique tickers
- Unique features
- Unique sources
- Proportion of numeric versus non-numeric observations where inferable

### 2. Source inventory

For each source, report:

- Total rows
- Date range
- Ticker count
- Feature count
- Share of dataset rows
- Any obvious concentration or inactivity patterns

### 3. Feature inventory

For each feature, report:

- Source
- Row count
- Ticker coverage
- Date coverage
- Inferred observation pattern such as daily, monthly, quarterly, or irregular
- Numeric parse rate where applicable

### 4. Sparsity and missingness

Quantify sparsity:

- By source
- By feature
- By ticker
- By ticker-feature pair where useful
- Over time, so sparse coverage can be distinguished from late source start dates

### 5. Temporal behavior

Characterize observation timing:

- Update interval distribution by feature
- Gap distribution
- Long stale runs
- Repeated unchanged values where suspicious
- Bursty or irregular publication patterns

### 6. Outliers and anomalies

Flag raw-data issues that deserve manual review:

- Impossible values where domain-independent checks are safe
- Extreme z-score or robust-z-score events for numeric features
- Abrupt step changes that look operationally suspicious
- Duplicated rows
- Conflicting observations for the same `(date, ticker, feature, source)` key if present

### 7. Leakage risk review

Apply conservative raw-data leakage heuristics, such as:

- Features whose timestamp pattern suggests future information may be aligned too early
- Features with suspiciously synchronized updates across many tickers
- Fundamentals-like fields with reporting cadence inconsistent with expected publication patterns
- Any field whose usable timestamp semantics are unclear and therefore unsafe by default

The report should clearly separate confirmed issues from heuristic warnings.

### 8. Research implications

Produce a prioritized interpretation section that groups raw features into categories:

- Likely usable now
- Usable only with safeguards
- Unsafe until repaired or better timestamped
- Too sparse to prioritize

This section is the bridge from raw-data analysis to future feature engineering.

## Analysis Domains

The implementation should be organized into a small number of focused analysis domains.

### Coverage and sparsity

Responsibilities:

- Dataset composition
- Source coverage
- Feature coverage
- Ticker coverage
- Missingness and sparsity summaries

### Temporal behavior

Responsibilities:

- Frequency inference
- Gap and staleness summaries
- Irregular update patterns

### Anomalies and leakage heuristics

Responsibilities:

- Numeric anomaly detection
- Duplicate/conflict checks
- Timestamp-shape heuristics for leakage risk

### Report integration

Responsibilities:

- Assemble all domain outputs into the canonical Markdown report
- Emit JSON summary metadata
- Keep section order and table names stable

## Sub-agent execution model

This task is intentionally suitable for parallel sub-agents because the analysis domains are mostly independent.

Recommended split:

- Sub-agent 1: coverage and sparsity metrics
- Sub-agent 2: temporal behavior metrics
- Sub-agent 3: anomalies and leakage heuristics
- Main agent: integrate outputs, finalize schema, ensure consistency, and write the report

Sub-agents should work against a shared output contract rather than editing the same report body directly. That contract should define expected CSV table names, JSON keys, and interpretation handoff fields.

## Implementation constraints

- Keep the first version focused on `data/data_long.csv` only
- Favor deterministic outputs over exploratory notebook workflows
- Prefer simple Python scripts or modules over complex orchestration frameworks
- Use stable file names and schema so future agents can consume outputs without guesswork
- Keep logic modular enough that additional checks can be added later without rewriting the report format

## Likely code shape

The implementation should stay small and explicit. A reasonable shape is:

- One orchestration entry point under `python/scripts/`
- Supporting analysis modules under `python/src/skuld_research/diagnostics/` or a nearby raw-data analysis package if a clearer boundary is needed
- One report writer responsible for Markdown and JSON assembly

The exact file layout can be finalized in the implementation plan, but the architecture should preserve clear separation between metric generation and report rendering.

## Verification requirements

The finished workflow should be considered complete only if it can be run end-to-end on the current dataset and produces:

- A Markdown report
- The expected CSV tables
- A JSON summary file
- Stable section headings and artifact names

Verification should include at least:

- Script execution against the real dataset
- Spot checks that key counts match documented dataset properties
- Validation that generated files exist in the expected output directory

## Success criteria

The first version succeeds if it gives the project a reliable raw-data source of truth that answers:

- What data exists?
- How much of it is actually usable?
- How does coverage vary by source, feature, ticker, and time?
- What raw-data issues are likely to distort research?
- Which raw features should be prioritized, guarded, or ignored in future feature engineering?

## Follow-on work

This specification is intentionally limited to raw-data analysis. Later work may extend the same framework to:

- PIT snapshot analysis
- Prepared panel propagation analysis
- Factor-readiness scoring
- Cross-spec research comparisons

Those should be separate specs so the raw-data source-of-truth layer remains focused and trustworthy.
