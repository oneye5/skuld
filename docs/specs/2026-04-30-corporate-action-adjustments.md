# Spec: Corporate-Action Adjustment Audit & Repair Layer

**Date:** 2026-04-30
**Status:** Approved
**Owner:** skuld_research/data
**Related:**
- `docs/specs/2026-04-30-phase1-dominance-diagnostic.md` (data-quality motivation)
- `docs/DATA_PIPELINE.md` §2.3 (currently incorrect; will be corrected as part of this work)
- `python/src/skuld_research/data/scrubber.py` (sibling, complementary, intentionally separate)

## 1. Problem

Skuld trusts Yahoo's `adj_close` for all return computations. This is implicit
and untested. Yahoo's adjusted-close chain is known to fail in several ways:

- Missed splits (history not back-adjusted through a split that did occur).
- Unit jumps (100x errors from pence/£ or cent/$ confusion).
- Bad dividend back-adjustment (residual mismatch between price drop on
  ex-date and the cash dividend).
- Duplicated or orphan corporate-action records.
- Splits recorded in the corporate-action table that are not reflected in the
  price chain (or vice-versa).

The Java ingest already fetches Yahoo dividends and splits alongside
`adj_close`. The Python loader routes them into `RawData.corporate_actions`,
but no code cross-validates the two. The dominance diagnostic
(2026-04-30-phase1-dominance-diagnostic.md) further showed that a single
corrupted print can blow through monthly compounding and contaminate factor
research.

In addition, `docs/DATA_PIPELINE.md` §2.3 currently states that the Python
pipeline performs explicit corporate-action adjustment. This is false, and
the documentation must be corrected.

## 2. Goals

1. **Detect** discrepancies between Yahoo's `adj_close` and the corporate-action
   ledger, with structured per-event audit output.
2. **Optionally repair** the price panel for high-confidence detections, with
   an idempotent, policy-controlled correction step.
3. Keep the new layer **fully decoupled** from existing pipeline contracts
   (`RawData`, `PITSnapshot`, `PreparedPanel`) so it can be invoked from any
   future pipeline (e.g. ML feature pipelines that build their own data
   transformations).
4. Mirror yfinance `repair=True`'s detection *taxonomy* without copying its
   code or accepting its known false-positive rate by default.
5. Be off by default everywhere; opt-in via the same spec mechanism as
   `ScrubbingSpec`.

## 3. Non-Goals

- Replacing or modifying the existing round-trip scrubber (`scrubber.py`).
  The two layers address different failure modes.
- Sourcing corporate actions from a non-Yahoo provider (e.g. NZX direct).
- Repairing intraday data — this layer is daily-only.
- Re-fetching raw data to fill gaps (yfinance does sub-interval refetching;
  Skuld is offline relative to its CSV input and will not call Yahoo from
  this layer).

## 4. Design

### 4.1 Module placement

New file: `python/src/skuld_research/data/adjustments.py`.

Shape, conventions, and frozen-dataclass style intentionally mirror
`python/src/skuld_research/data/scrubber.py` so the two read as siblings.

### 4.2 API

All API entry points are pure functions on DataFrames. They import nothing
from `csv_loader`, `pit_loader`, `prepared_panel`, `contracts`, or
`skuld_common`. Dependencies: pandas, numpy, stdlib.

```python
class RepairPolicy(str, Enum):
    OFF = "off"                     # identity; useful for symmetric callsites
    CONSERVATIVE = "conservative"   # repair only unambiguous detections
    AGGRESSIVE = "aggressive"       # also re-derive dividend chain when raw_close given

@dataclass(frozen=True)
class AdjustmentAuditReport:
    events: pd.DataFrame
    # columns:
    #   ticker:               object
    #   ex_date:              datetime64[ns]
    #   kind:                 object   one of the categories in §4.3
    #   severity:             object   "info" | "warn" | "error" | "skipped_no_raw"
    #   observed_ratio:       float64  adj_close[ex] / adj_close[prev_obs_day]
    #   expected_ratio:       float64  derived from corp action; NaN if N/A
    #   residual:             float64  abs(observed/expected - 1); NaN if N/A
    #   adj_close_prev:       float64
    #   adj_close_ex:         float64
    #   raw_close_prev:       float64  NaN when raw_close not provided
    #   raw_close_ex:         float64
    #   corp_action_type:     object   "dividend" | "split" | "" (orphan jumps)
    #   corp_action_factor:   float64
    #   notes:                object   short human-readable explanation

@dataclass(frozen=True)
class RepairResult:
    prices: pd.DataFrame              # repaired adjusted-close panel
    report: AdjustmentAuditReport     # all detections (including non-repaired)
    repairs: pd.DataFrame
    # columns:
    #   ticker, ex_date, kind, action,
    #   factor_applied, range_start, range_end

def audit_adjustments(
    adj_close: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    raw_close: pd.DataFrame | None = None,
    dividend_residual_tol: float = 0.25,
    split_residual_tol: float = 0.05,
    missed_split_ratios: tuple[float, ...] = (0.5, 2.0, 3.0, 4.0, 5.0, 10.0,
                                              0.1, 0.2, 0.25, 1/3),
    unit_jump_ratios: tuple[float, ...] = (100.0, 0.01),
    unit_jump_tol: float = 0.02,
) -> AdjustmentAuditReport: ...

def repair_adjustments(
    adj_close: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    raw_close: pd.DataFrame | None = None,
    policy: RepairPolicy = RepairPolicy.CONSERVATIVE,
    **audit_kwargs,
) -> RepairResult: ...
```

Inputs:
- `adj_close`: wide `date × ticker` adjusted-close panel. Same shape as
  `RawData.prices`.
- `corporate_actions`: flat frame with columns `ticker, ex_date, type, factor`.
  Same shape as `RawData.corporate_actions`. `type ∈ {"dividend", "split"}`,
  `factor` is cash amount per share for dividends and `numerator/denominator`
  for splits (e.g. 2-for-1 → 2.0; 1-for-10 reverse → 0.1).
- `raw_close`: optional wide `date × ticker` panel of *unadjusted* close.
  Required for `bad_div_adjust` detection and for `AGGRESSIVE` repair.
  Available from `load_raw_ohlc()`.

### 4.3 Detection categories (`kind`)

Each detection is classified into one category and assigned a severity.
Per-ticker computation; vectorised within ticker, looped across tickers
(matches scrubber pattern).

| `kind` | What it detects | Default severity | Repairable? |
|---|---|---|---|
| `missed_split` | observed `adj_close` ratio at an ex-date matches `missed_split_ratios` (or its reciprocal) within `split_residual_tol`, but no `split` row exists in `corporate_actions` at that date | `error` | Yes (CONSERVATIVE) |
| `unit_jump` | observed ratio matches `unit_jump_ratios` (100 or 0.01) within `unit_jump_tol` and no corp-action explains it | `error` | Yes (CONSERVATIVE) |
| `split_mismatch` | a `split` row exists but observed ratio does not match the split factor within `split_residual_tol` | `warn` | No (data ambiguity; investigate manually) |
| `bad_div_adjust` | `dividend` row exists; expected back-adjustment ratio `(raw_close[ex] - div) / raw_close[prev]` differs from observed adj_close ratio by more than `dividend_residual_tol` | `error` if `raw_close` provided, else `skipped_no_raw` | Yes (AGGRESSIVE only) |
| `orphan_action` | corp-action `ex_date` falls outside the price index for that ticker (delisted, ex-date past last close, or before first close) | `info` | No (no prices to fix) |
| `duplicate_action` | multiple corp-action rows of the same `type` on the same `(ticker, ex_date)` | `warn` | Yes (dedup the corp-actions frame, but does not mutate prices) |

Ex-date alignment: corporate-action `ex_date` is mapped to the trading day at-
or-after `ex_date` for that ticker. The "previous" observation is the trading
day strictly before that. This handles weekend/holiday ex-dates.

### 4.4 Repair mechanics

For each `error`-severity event whose `kind` is repairable under the chosen
policy:

- **`missed_split`**: divide all `adj_close` values strictly before the
  aligned ex-date by the inferred split factor for that ticker.
  `range_start = first observation`, `range_end = day_before_ex_date`.
- **`unit_jump`**: same back-adjustment by the unit factor (100 or 0.01).
- **`bad_div_adjust`** (AGGRESSIVE only): re-derive the entire adjusted-close
  series for the affected ticker from `raw_close + dividends + splits` using
  the standard backward total-return chain
  `adj[t-1] = adj[t] · (raw[t-1] / (raw[t] · split_factor)) · (1 - div/raw[t-1])`,
  applied right-to-left. Replace the column.
- **`duplicate_action`**: dedup the corp-actions frame (max factor for
  splits, sum for dividends). Reported in `repairs` but the price panel is
  not mutated by this rule alone; a follow-up audit pass against the deduped
  frame may surface new `bad_div_adjust` or `split_mismatch` events.

Repair is **idempotent**: running `repair_adjustments` on its own output
produces an empty `repairs` frame and an audit report containing only `info`
or `skipped_no_raw` entries. This is enforced by test.

`RepairPolicy.OFF` is a no-op identity that returns the input prices, the
full audit report, and an empty `repairs` frame. Useful for callers that
want to log audit results without mutating data.

### 4.5 Coupling and integration points

The new module is self-contained. Existing code is touched in exactly three
places:

1. `python/src/skuld_research/config/spec.py`: add `AdjustmentSpec` (mirroring
   `ScrubbingSpec`), and an optional `adjustments: AdjustmentSpec | None = None`
   field on `BacktestSpec`. Default `None` → no behaviour change. Off by
   default in every existing strategy YAML (no migration required).
2. `python/src/skuld_research/data/csv_loader.py`: extend `load_raw_csv` and
   `load_raw_ohlc` with an optional `adjustments: AdjustmentSpec | None = None`
   parameter, symmetric with the existing `scrub` parameter. When provided
   and `kind != "off"`, the loader calls `repair_adjustments` after the
   scrubber and attaches an `AdjustmentAuditReport` to a new optional
   `RawData.adjustment_report` field. Field is `None` when adjustments are
   not requested, preserving backward compatibility.
3. `docs/DATA_PIPELINE.md`: correct §2.3 to describe the actual current
   behaviour (Yahoo `adj_close` is trusted by default) and document the new
   opt-in audit/repair layer.

Future ML or alternative pipelines that do not consume `RawData` can call
`audit_adjustments` and `repair_adjustments` directly with whatever frames
they have. This is the explicit reason the API is pure-function-on-DataFrame.

### 4.6 Documentation deliverables

- `docs/ADJUSTMENTS.md` (new): user-facing documentation of detection
  categories, severity model, repair policies, when to enable in research
  vs production, and a worked SKT.NZ–style example.
- `docs/DATA_PIPELINE.md` §2.3: corrected.
- `python/src/skuld_research/data/scrubber.py` module docstring: add a
  one-line "see also `adjustments.py`" pointer.
- `.github/copilot-instructions.md` documentation index: add entries for
  `docs/ADJUSTMENTS.md` and this spec.

## 5. Test plan

`python/tests/test_adjustments.py` (new), conventions matching
`python/tests/test_scrubber.py`:

Audit-only:
- `test_audit_no_actions_no_events` — empty corp-actions frame produces empty report.
- `test_audit_clean_panel_no_events` — synthetic clean adj_close + matching dividends + matching splits produces empty report.
- `test_detect_missed_split_2for1` — adj_close drops 50% with no `split` row → one `missed_split` event with `severity="error"`.
- `test_detect_unit_jump_100x` — single ex-date 100x ratio with no action → one `unit_jump` event.
- `test_detect_bad_div_adjust_when_raw_close_provided` — raw + adj inconsistent with dividend factor by > tolerance → `bad_div_adjust` event with severity `error`.
- `test_skip_bad_div_adjust_without_raw_close` — same fixture, no `raw_close` → severity `skipped_no_raw`, not `error`.
- `test_detect_split_mismatch` — `split` row exists with factor 2.0 but observed ratio is 1.0 → `split_mismatch` warn.
- `test_detect_orphan_action_after_last_price` — corp action past end of price series → `orphan_action` info.
- `test_detect_duplicate_action_same_day` — two dividend rows same `(ticker, ex_date)` → `duplicate_action` warn.

Repair:
- `test_repair_off_is_identity` — `RepairPolicy.OFF` returns input unchanged with non-empty audit and empty `repairs`.
- `test_repair_conservative_fixes_missed_split` — 2-for-1 missed-split fixture: pre-split prices halved, post-split untouched, `repairs` ledger correct.
- `test_repair_conservative_fixes_unit_jump_100x` — boundary day repaired, `repairs` ledger correct.
- `test_repair_conservative_skips_bad_div_adjust` — `bad_div_adjust` event present in audit but not in `repairs`.
- `test_repair_aggressive_fixes_bad_div_adjust` — full ticker chain re-derived; check returns_daily on result is consistent with raw + actions.
- `test_repair_idempotent` — `repair_adjustments(repair_adjustments(...).prices, ...)` produces empty `repairs` and no `error`-severity events.

Orthogonality / regression:
- `test_skt_2010_corruption_not_falsely_flagged_as_split` — round-trip print fixture (the scrubber's domain) does NOT produce a `missed_split` event; this layer should leave it alone.
- `test_load_raw_csv_with_adjustments_attaches_report` — integration test exercising the optional `csv_loader` hook end-to-end on a synthetic CSV.

Lint / type:
- All new code passes `uv run ruff check .` and `uv run pyright`.

## 6. Acceptance criteria

1. `audit_adjustments` and `repair_adjustments` exist as pure functions in
   `python/src/skuld_research/data/adjustments.py` with the signatures
   above. Importing them does not import any existing pipeline contract
   module beyond pandas/numpy.
2. All test cases in §5 pass.
3. `csv_loader.load_raw_csv` and `load_raw_ohlc` accept an optional
   `adjustments` parameter. When omitted (the default), behaviour is
   bit-identical to the current implementation. Verified by an existing-
   tests-still-pass criterion: `uv run pytest` is green with no test
   modifications other than additions.
4. `AdjustmentSpec` is added to `BacktestSpec` as an optional field
   (default `None`). All existing strategy YAML specs continue to load
   without modification.
5. `docs/DATA_PIPELINE.md` §2.3 is corrected.
6. `docs/ADJUSTMENTS.md` exists and is linked from
   `.github/copilot-instructions.md`.
7. Repair is idempotent (test enforced).
8. Repair policy `OFF` is a no-op identity (test enforced).

## 7. Out-of-scope follow-ups

- Empirical study of how often each detection category fires across the
  full NZX universe. Once we have the audit data, we can decide whether
  `CONSERVATIVE` should become the default in production or stay opt-in.
- Migrating `DividendYieldFactor` to optionally consume the deduped corp-
  actions frame from a `RepairResult`.
- Adding `capital_gains` and `rights_issue` event types (Java ingest does
  not currently emit these; would require a parallel Java change).
- Sourcing actual filing dates and authoritative split records from NZX
  directly to remove the dependency on Yahoo.
