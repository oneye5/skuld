# Look-ahead Bias Remediation Spec

**Date:** 2026-04-30
**Status:** Approved for implementation
**Owner:** assistant
**Related:** `docs/specs/2026-04-30-corporate-action-adjustments.md` (sibling data-quality stream)

## Problem

Source audit (2026-04-30) identified that **9 ingestion sources timestamp datapoints at the period start** rather than the date the data became publicly knowable. Backtests using these features see information months (or in the FAO case, ~18 months) before it actually existed, producing inflated paper performance.

Affected sources:

| Source | Cadence | Currently stamped | Real release lag (typical) |
|---|---|---|---|
| `NzGdp` | Quarterly | Q-start (e.g. 2024-Q1 → 2024-01-01) | Q-end + ~75 days |
| `NzBalanceOfPayments` | Quarterly | Q-start | Q-end + ~75 days |
| `NzVehicleRegistrations` | Monthly | M-start | M-end + ~25 days |
| `NzBusinessConfidence` | Monthly | M-start | M-end + ~5 days |
| `NzLaborStats` | Quarterly | Q-start | Q-end + ~45 days |
| `NzTaxRevenue` | Annual | Y-start | Y-end + ~12 months |
| `NzPensions` | Annual | Y-start | Y-end + ~12 months |
| `NzLaborTaxation` | Annual | Y-start | Y-end + ~12 months |
| `NzRoadFatalities` | Monthly | M-start | M-end + ~30 days |
| `GlobalFoodPrices` | Annual | Y-start | Y-end + ~18 months |
| `GlobalAquacultureProduction` | Annual | Y-start | Y-end + ~18 months |
| `WikimediaPageviews` | Daily | UTC midnight of the day | UTC day + 1 day (NZ T+1 morning) |

`NzRatesFx`, `YfPrices`, `YfFinances` are **not affected by this spec** — prices are intraday/EOD and stamped correctly; `YfFinances` is reported-as-of and stamped at filing date.

## Decisions (locked with user 2026-04-30)

1. **Per-source publication-lag table.** Each source gets a hardcoded typical lag based on the publishing agency's release calendar. Easy to override per source if a more accurate value is known.
2. **Overwrite `timestamp` in-place** in `DataPoint` before emission. No new column. Single-source-of-truth date semantic: "first moment this datapoint is knowable in NZ time."
3. **User runs the re-ingest** after the Java changes land. No automated re-run, no candidate re-evaluation in this stream.

## Out of scope

- Fixing OECD SDMX regex parsing fragility (separate workstream).
- The `YfPrices` Yahoo silent-forward-adjust quirk (separate workstream).
- Producing before/after backtest metrics (user-deferred).

## Design

### Lag application location

A new utility class `lazic.utils.ingest.ReleaseDate` exposes:

```java
public static LocalDateTime applyLag(LocalDateTime periodStart, ReleaseLag lag);
```

where `ReleaseLag` is a record:

```java
public record ReleaseLag(int days, int months) {
    public static ReleaseLag of(int days) { return new ReleaseLag(days, 0); }
    public static ReleaseLag months(int months) { return new ReleaseLag(0, months); }
    public static final ReleaseLag NONE = new ReleaseLag(0, 0);
}
```

Each source calls `ReleaseDate.applyLag(periodStart, MY_LAG)` immediately before constructing its `DataPoint`.

### Lag values

Centralised in source classes (per-source constant) for locality — each source documents its own publishing schedule near the constant. Example for `NzGdp`:

```java
// Stats NZ National Accounts: quarterly release, ~10 weeks after quarter-end.
// https://www.stats.govt.nz/release-calendar
private static final ReleaseLag RELEASE_LAG = new ReleaseLag(75, 0);
```

The lag is added to the **period END**, not period start. So Q1 2024 (Jan 1 – Mar 31) becomes `2024-03-31 + 75 days = 2024-06-14`. `ReleaseDate.applyLag` therefore needs the period to know its end:

```java
public static LocalDateTime applyLag(LocalDateTime periodStart, Cadence cadence, ReleaseLag lag);

public enum Cadence { DAILY, MONTHLY, QUARTERLY, ANNUAL }
```

`Cadence.QUARTERLY.endOf(2024-01-01)` returns `2024-03-31T23:59:59`; lag is added on top.

### Per-source lag table

| Source | Cadence | Lag |
|---|---|---|
| NzGdp | QUARTERLY | 75 days |
| NzBalanceOfPayments | QUARTERLY | 75 days |
| NzLaborStats | QUARTERLY | 45 days |
| NzVehicleRegistrations | MONTHLY | 25 days |
| NzBusinessConfidence | MONTHLY | 5 days |
| NzRoadFatalities | MONTHLY | 30 days |
| NzTaxRevenue | ANNUAL | 12 months |
| NzPensions | ANNUAL | 12 months |
| NzLaborTaxation | ANNUAL | 12 months |
| GlobalFoodPrices | ANNUAL | 18 months |
| GlobalAquacultureProduction | ANNUAL | 18 months |
| WikimediaPageviews | DAILY | 1 day |

These are conservative typical lags. Any source whose actual publishing schedule is known more precisely can override — the values are constants near each source's parser.

## Acceptance criteria

1. `ReleaseDate.applyLag` and `Cadence` exist with unit tests covering each cadence × leap-year × end-of-year boundary.
2. Each of the 12 affected sources references its lag constant exactly once, applied at the single point where `DataPoint` is constructed.
3. `mvn -q compile` clean.
4. The timestamp convention ("knowledge-time, not event-time") is documented in `ReleaseDate.java` javadoc.
5. A diagnostic main class (`DiagReleaseDate.java`) exists temporarily, prints one row per affected source confirming the shift, and is deleted before the work is considered complete.
6. The Python pipeline is **not modified** — it already treats `date` as knowledge-time. (Verified by skim, not by test, in this spec.)

## Risks

- **Per-source lag overrides drift from reality.** Mitigation: each constant has a comment with the source agency URL and last-verified date.
- **Some macro features may now have so much lag they're useless.** Acceptable — that's the truth. Users can compensate downstream.
- **Re-ingestion is the user's responsibility.** Old `data/data_long.csv` retains pre-fix timestamps until re-run.
