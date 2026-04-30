# Corporate-Action Adjustment Audit & Repair

Opt-in audit/repair layer that cross-validates Yahoo Finance's `adj_close`
panel against the corporate-action ledger (dividends and splits) and, when
asked, repairs detected discrepancies. Lives in
`python/src/skuld_research/data/adjustments/` and is implemented as pure
functions on DataFrames so any pipeline — including bespoke ML feature
pipelines that do not consume `RawData` — can use it.

Spec: `docs/specs/2026-04-30-corporate-action-adjustments.md`.
Sibling: `python/src/skuld_research/data/scrubber.py`.

---

## 1. Overview

Skuld trusts Yahoo's `adj_close` for all return computations. Yahoo *should*
incorporate splits, dividends, and capital returns into that series, and
`PreparedPanel` does not re-apply the corporate-action frame on top of it
(see `docs/DATA_PIPELINE.md` §2.3). That trust is implicit and historically
untested. Yahoo's adjusted-close chain is known to fail in several ways:
missed splits, 100x unit jumps, residual mismatches between dividend
back-adjustment and the actual cash dividend, duplicated or orphan
corp-action records, and split rows that never made it into the price chain.

This layer makes that trust **explicit and verifiable**:

- **Audit** — compute observed adj_close ratios at every corp-action ex-date,
  compare to the expected ratio implied by the ledger, and emit a long-form
  `events` DataFrame classifying every discrepancy.
- **Repair** (optional) — for high-confidence detections, back-scale the
  pre-event price history (e.g. halve all prints before a missed 2-for-1
  split). Repair is policy-controlled and idempotent.

### What it is *not*

- **Not a print scrubber.** Single-day round-trip anomalies (a clearly
  wrong daily close that reverses the next day) are the domain of
  `scrubber.py`. The two layers are orthogonal: `scrubber` looks at the
  shape of the daily return path; `adjustments` looks at consistency
  between `adj_close` and the corp-action ledger. Neither subsumes the
  other.
- **Not a data fetcher.** It never calls Yahoo or any other provider. It
  works on the frames the caller already has.
- **Not a full reconciliation against a primary source** (e.g. NZX direct).
  It cross-validates Yahoo against itself.

### Why opt-in

The audit always emits structured events, which is harmless. Repairs
*mutate* the price panel and therefore change every downstream factor
score. We keep both off by default so that historical research results
remain reproducible without a flag-day migration. Existing strategy YAMLs
do not need to change.

---

## 2. When to enable

| Setting | Recommended for |
|---|---|
| Disabled (no `adjustments` field in spec) | Reproducing historical backtests bit-for-bit. |
| `kind="audit"` | **Default recommendation for any new factor** that consumes `adj_close` (which is most of them). Surfaces discrepancies in the audit report without changing prices. Cheap. |
| `kind="repair"`, `policy="conservative"` | Research notebooks investigating known data-quality issues, or a new strategy where you want clean prices and have inspected the audit output and accepted the repairs. |
| `kind="repair"`, `policy="aggressive"` | Research only. Re-derives full adjusted-close chains for tickers with `bad_div_adjust` events. Do **not** enable in production without an additional human review of the resulting chain. |

In production (`skuld_portfolio`), prefer `kind="audit"` so the report is
attached to `RawData.adjustment_report` for logging, but leave the price
panel untouched unless a specific defect has been triaged.

---

## 3. Detection categories

Six discrepancy categories are emitted in the audit report's `events` frame.
Each row carries `kind`, `severity`, and enough numeric context
(`observed_ratio`, `expected_ratio`, `residual`, surrounding prices) to
reproduce the decision.

| `kind` | What it detects | Default severity | Repairable |
|---|---|---|---|
| `missed_split` | adj_close ratio at an ex-date matches a known split-shape factor (2.0, 0.5, 3.0, 1/3, …) within `split_residual_tol`, but no `split` row exists in the ledger. | `error` | Yes (CONSERVATIVE) |
| `unit_jump` | Ratio matches a 100x or 0.01x unit-confusion factor within `unit_jump_tol`, with no corp-action explanation. Typically pence/£ or cent/$ confusion. | `error` | Yes (CONSERVATIVE) |
| `split_mismatch` | A `split` row exists in the ledger but the observed adj_close ratio does not match the expected `1/factor` within `split_residual_tol`. | `warn` | No (data is ambiguous; investigate manually) |
| `bad_div_adjust` | A `dividend` row exists; the observed adj_close ratio is inconsistent with the price-drop model `(raw_close[ex] - dividend) / raw_close[prev]` beyond `dividend_residual_tol`. | `error` if `raw_close` provided, else `skipped_no_raw` | Yes (AGGRESSIVE only) |
| `orphan_action` | Corp-action `ex_date` falls outside the price index for that ticker (delisted, before first close, after last close). | `info` | No (no prices to fix) |
| `duplicate_action` | Multiple ledger rows of the same `type` on the same `(ticker, ex_date)`. | `warn` | Dedup-only (does not mutate prices) |

**Severity model:**

- `info` — observed and expected; no action recommended.
- `warn` — discrepancy detected; manual inspection suggested.
- `error` — high-confidence defect; eligible for repair under the right
  policy.
- `skipped_no_raw` — would have been an `error` but `raw_close` was not
  supplied, so the check could not be completed.

**Ex-date alignment.** The corp-action `ex_date` is mapped to the trading
day at-or-after `ex_date` for that ticker; the "previous" observation is
the trading day strictly before that. This handles weekend and holiday
ex-dates uniformly across NZX names.

---

## 4. Repair policies

```python
class RepairPolicy(str, Enum):
    OFF = "off"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
```

| Policy | Repairs applied | When to use |
|---|---|---|
| `OFF` | None. Identity transform on prices; full audit report is still returned. | Symmetric callsites that want to log audit results without mutating data. |
| `CONSERVATIVE` | `missed_split`, `unit_jump`. Back-scales pre-event prices by the inferred factor. | Default repair mode. Both detections are unambiguous: the ratio matches a discrete factor list within a tight tolerance. |
| `AGGRESSIVE` | Everything in `CONSERVATIVE`, plus `bad_div_adjust`. Re-derives the entire adjusted-close series for the affected ticker from `raw_close + dividends + splits` using the standard backward total-return chain. | Research only. Replaces a whole column, so any existing trust in Yahoo's chain for that ticker is dropped in favour of the locally-derived chain. |

**Idempotence.** Repair is idempotent: feeding `repair_adjustments` its
own output produces an empty `repairs` frame and an audit report containing
only `info` or `skipped_no_raw` events. This is enforced by test.

**`duplicate_action`** is reported as a `warn` and recorded in `repairs`
when collapsed (max factor for splits, sum for dividends), but the price
panel is **not** mutated by this rule alone. A follow-up audit pass against
the deduped ledger may then surface a `bad_div_adjust` or `split_mismatch`.

---

## 5. API reference

Two pure functions. Inputs and outputs are bare pandas frames plus frozen
dataclass wrappers — no Skuld pipeline contracts are imported.

```python
from skuld_research.data.adjustments import (
    audit_adjustments,
    repair_adjustments,
    RepairPolicy,
)

def audit_adjustments(
    adj_close: pd.DataFrame,             # wide date × ticker
    corporate_actions: pd.DataFrame,     # ticker, ex_date, type, factor
    *,
    raw_close: pd.DataFrame | None = None,
    dividend_residual_tol: float = 0.25,
    split_residual_tol: float = 0.05,
    missed_split_ratios: tuple[float, ...] = (
        0.5, 2.0, 3.0, 4.0, 5.0, 10.0, 0.1, 0.2, 0.25, 1/3,
    ),
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

`AdjustmentAuditReport.events` columns: `ticker`, `ex_date`, `kind`,
`severity`, `observed_ratio`, `expected_ratio`, `residual`,
`adj_close_prev`, `adj_close_ex`, `raw_close_prev`, `raw_close_ex`,
`corp_action_type`, `corp_action_factor`, `notes`.

`RepairResult` exposes `prices` (possibly mutated panel), `report` (full
audit), and `repairs` (long-form ledger of every applied repair with
columns `ticker`, `ex_date`, `kind`, `action`, `factor_applied`,
`range_start`, `range_end`).

### Direct use from a custom pipeline

The API is deliberately decoupled from `RawData` so that ML feature
pipelines or any other consumer can call it directly:

```python
from skuld_research.data.adjustments import (
    audit_adjustments,
    repair_adjustments,
    RepairPolicy,
)

# Audit only — never mutates input.
report = audit_adjustments(my_adj_close_panel, my_corp_actions_frame)
errors = report.events[report.events["severity"] == "error"]
print(errors[["ticker", "ex_date", "kind", "observed_ratio", "notes"]])

# Repair under a chosen policy.
result = repair_adjustments(
    my_adj_close_panel,
    my_corp_actions_frame,
    raw_close=my_raw_close_panel,           # required for AGGRESSIVE
    policy=RepairPolicy.CONSERVATIVE,
)
clean_prices = result.prices
```

---

## 6. Integration with the standard loader

For research and production runs that go through `RawData`, opt in via
`AdjustmentSpec` on `BacktestSpec`. The field defaults to `None` (off);
adding it does not change the spec hash unless `kind != "off"`.

YAML:

```yaml
# python/configs/strategy-specs/.../my-strategy.yaml
adjustments:
  kind: audit                   # off | audit | repair
  policy: conservative          # off | conservative | aggressive  (used when kind=repair)
  dividend_residual_tol: 0.25
  split_residual_tol: 0.05
  unit_jump_tol: 0.02
```

Python:

```python
from skuld_research.config.spec import AdjustmentSpec, BacktestSpec

spec = BacktestSpec(
    name="my-strategy",
    asof=date(2026, 4, 30),
    adjustments=AdjustmentSpec(kind="audit"),
    # ...
)
```

The CSV loader honours the field automatically. When `kind="audit"` or
`kind="repair"`, the resulting `RawData` carries an
`adjustment_report: AdjustmentAuditReport` field; for `kind="repair"`,
`RawData.prices` is the repaired panel. When the spec field is absent or
`kind="off"`, `RawData.adjustment_report` is `None` and prices are
bit-identical to the legacy code path.

---

## 7. Worked example: missed 2-for-1 split

Synthetic but realistic fixture. A ticker `XYZ.NZ` trades flat around
$10.00, then experiences an apparent 50% drop on 2024-06-03 with no
ledger record:

| date | adj_close (XYZ.NZ) |
|---|---|
| 2024-05-30 | 10.00 |
| 2024-05-31 | 10.05 |
| 2024-06-03 | **5.02** |
| 2024-06-04 | 5.03 |

`corporate_actions` contains no `split` row for `(XYZ.NZ, 2024-06-03)`.

`audit_adjustments` emits one event:

```
ticker:              XYZ.NZ
ex_date:             2024-06-03
kind:                missed_split
severity:            error
observed_ratio:      0.4995          # 5.02 / 10.05
expected_ratio:      0.5             # nearest match in missed_split_ratios
residual:            0.001
adj_close_prev:      10.05
adj_close_ex:        5.02
corp_action_type:    ""              # no row to attribute it to
corp_action_factor:  NaN
notes:               "ratio matches missed_split factor 2.0 within tolerance"
```

Under `RepairPolicy.CONSERVATIVE`, `repair_adjustments` divides every
`adj_close` value strictly before 2024-06-03 by 2.0:

| date | adj_close (after repair) |
|---|---|
| 2024-05-30 | 5.00 |
| 2024-05-31 | 5.025 |
| 2024-06-03 | 5.02 |
| 2024-06-04 | 5.03 |

The `repairs` ledger records:

```
ticker:           XYZ.NZ
ex_date:          2024-06-03
kind:             missed_split
action:           backscale
factor_applied:   2.0
range_start:      <first observation for XYZ.NZ>
range_end:        2024-05-31
```

Re-running the audit on the repaired panel produces no `error` events.

---

## 8. Known limitations

- **Daily-only.** Intraday data is out of scope.
- **`bad_div_adjust` requires `raw_close`.** Without it, the check is
  recorded as `skipped_no_raw` rather than `error`. `RawData.raw_prices`
  (from `load_raw_ohlc`) supplies this; ad hoc callers must provide their
  own.
- **No refetching.** Unlike `yfinance(repair=True)`, this layer never
  calls Yahoo. It works on the frames already in memory. Genuine gaps in
  the underlying source require a re-ingest.
- **No `capital_gains` or `rights_issue` handling.** The Java ingest does
  not currently emit these event types; adding them would require a
  parallel Java change. They will surface as `orphan_action` info if they
  appear in the ledger.
- **`AGGRESSIVE` policy is research-only.** It replaces entire
  adjusted-close columns with locally-derived chains. Do not enable in
  production without an additional human review and a dedicated
  validation suite for the affected tickers.
- **Cross-validates Yahoo against itself.** Both `adj_close` and the
  corp-action ledger come from Yahoo. A failure mode where both are
  wrong in a self-consistent way (e.g. a split that never made it into
  either) is not detectable here. Sourcing authoritative records from
  NZX directly is an out-of-scope follow-up tracked in the spec §7.
