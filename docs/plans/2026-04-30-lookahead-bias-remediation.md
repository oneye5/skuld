# Look-ahead Bias Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate look-ahead bias in 12 macro/social ingestion sources by stamping each `DataPoint` at its real public-release date instead of the period start.

**Architecture:** Add `lazic.utils.ingest.ReleaseDate` utility + `Cadence` enum + `ReleaseLag` record. Each affected source declares a per-source `RELEASE_LAG` constant (with documentation comment) and routes its emission through `ReleaseDate.applyLag(periodStart, cadence, lag)`. No Python changes. No schema changes. User re-runs ingestion afterwards.

**Tech Stack:** Java 21, JUnit 5 (already in `java/pom.xml`), Maven.

**Spec:** `docs/specs/2026-04-30-lookahead-bias-remediation.md`

---

## File Structure

**Create:**
- `java/src/main/java/lazic/utils/ingest/Cadence.java`
- `java/src/main/java/lazic/utils/ingest/ReleaseLag.java`
- `java/src/main/java/lazic/utils/ingest/ReleaseDate.java`
- `java/src/test/java/lazic/utils/ingest/ReleaseDateTest.java`
- `java/src/main/java/lazic/diag/DiagReleaseDate.java` (temporary, deleted in last task)

**Modify (one constant + one method call each):**
- `java/src/main/java/lazic/sources/NzGdp.java`
- `java/src/main/java/lazic/sources/NzBalanceOfPayments.java`
- `java/src/main/java/lazic/sources/NzLaborStats.java`
- `java/src/main/java/lazic/sources/NzVehicleRegistrations.java`
- `java/src/main/java/lazic/sources/NzBusinessConfidence.java`
- `java/src/main/java/lazic/sources/NzRoadFatalities.java`
- `java/src/main/java/lazic/sources/NzTaxRevenue.java`
- `java/src/main/java/lazic/sources/NzPensions.java`
- `java/src/main/java/lazic/sources/NzLaborTaxation.java`
- `java/src/main/java/lazic/sources/GlobalFoodPrices.java`
- `java/src/main/java/lazic/sources/GlobalAquacultureProduction.java`
- `java/src/main/java/lazic/sources/WikimediaPageviews.java`
- `java/docs/DATA_SOURCES.md` (convention note)

---

## Task 1: Foundation — `Cadence`, `ReleaseLag`, `ReleaseDate`

**Files:**
- Create: `java/src/main/java/lazic/utils/ingest/Cadence.java`
- Create: `java/src/main/java/lazic/utils/ingest/ReleaseLag.java`
- Create: `java/src/main/java/lazic/utils/ingest/ReleaseDate.java`
- Test: `java/src/test/java/lazic/utils/ingest/ReleaseDateTest.java`

- [ ] **Step 1: Write the failing test**

```java
// java/src/test/java/lazic/utils/ingest/ReleaseDateTest.java
package lazic.utils.ingest;

import org.junit.jupiter.api.Test;
import java.time.LocalDateTime;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ReleaseDateTest {

    @Test
    void quarterlyShiftAddsLagToPeriodEnd() {
        // Q1 2024 starts Jan 1; ends Mar 31 23:59:59; +75 days -> Jun 14 23:59:59
        LocalDateTime q1Start = LocalDateTime.of(2024, 1, 1, 0, 0);
        LocalDateTime out = ReleaseDate.applyLag(q1Start, Cadence.QUARTERLY, ReleaseLag.of(75));
        assertEquals(LocalDateTime.of(2024, 6, 14, 23, 59, 59), out);
    }

    @Test
    void monthlyShiftAddsLagToPeriodEnd() {
        // Feb 2024 (leap) ends Feb 29; +25 days -> Mar 25
        LocalDateTime febStart = LocalDateTime.of(2024, 2, 1, 0, 0);
        LocalDateTime out = ReleaseDate.applyLag(febStart, Cadence.MONTHLY, ReleaseLag.of(25));
        assertEquals(LocalDateTime.of(2024, 3, 25, 23, 59, 59), out);
    }

    @Test
    void annualShiftAddsLagInMonths() {
        // 2023 ends Dec 31; +18 months -> 2025-06-30 23:59:59
        LocalDateTime y = LocalDateTime.of(2023, 1, 1, 0, 0);
        LocalDateTime out = ReleaseDate.applyLag(y, Cadence.ANNUAL, ReleaseLag.months(18));
        assertEquals(LocalDateTime.of(2025, 6, 30, 23, 59, 59), out);
    }

    @Test
    void dailyShiftIsIdempotentOnPeriodEnd() {
        // Daily period "ends" same day; +1 day -> next day 23:59:59
        LocalDateTime d = LocalDateTime.of(2024, 4, 30, 0, 0);
        LocalDateTime out = ReleaseDate.applyLag(d, Cadence.DAILY, ReleaseLag.of(1));
        assertEquals(LocalDateTime.of(2024, 5, 1, 23, 59, 59), out);
    }

    @Test
    void zeroLagReturnsPeriodEnd() {
        LocalDateTime q1Start = LocalDateTime.of(2024, 1, 1, 0, 0);
        LocalDateTime out = ReleaseDate.applyLag(q1Start, Cadence.QUARTERLY, ReleaseLag.NONE);
        assertEquals(LocalDateTime.of(2024, 3, 31, 23, 59, 59), out);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mvn -q -pl . test -Dtest=ReleaseDateTest` (from `java/`)
Expected: compile failure — `Cadence`, `ReleaseLag`, `ReleaseDate` not defined.

- [ ] **Step 3: Implement `Cadence`**

```java
// java/src/main/java/lazic/utils/ingest/Cadence.java
package lazic.utils.ingest;

import java.time.LocalDateTime;
import java.time.YearMonth;

public enum Cadence {
    DAILY {
        @Override public LocalDateTime endOf(LocalDateTime periodStart) {
            return periodStart.toLocalDate().atTime(23, 59, 59);
        }
    },
    MONTHLY {
        @Override public LocalDateTime endOf(LocalDateTime periodStart) {
            YearMonth ym = YearMonth.from(periodStart);
            return ym.atEndOfMonth().atTime(23, 59, 59);
        }
    },
    QUARTERLY {
        @Override public LocalDateTime endOf(LocalDateTime periodStart) {
            int month = periodStart.getMonthValue();
            int qEndMonth = ((month - 1) / 3) * 3 + 3;
            YearMonth ym = YearMonth.of(periodStart.getYear(), qEndMonth);
            return ym.atEndOfMonth().atTime(23, 59, 59);
        }
    },
    ANNUAL {
        @Override public LocalDateTime endOf(LocalDateTime periodStart) {
            return LocalDateTime.of(periodStart.getYear(), 12, 31, 23, 59, 59);
        }
    };

    public abstract LocalDateTime endOf(LocalDateTime periodStart);
}
```

- [ ] **Step 4: Implement `ReleaseLag`**

```java
// java/src/main/java/lazic/utils/ingest/ReleaseLag.java
package lazic.utils.ingest;

public record ReleaseLag(int days, int months) {
    public static ReleaseLag of(int days) { return new ReleaseLag(days, 0); }
    public static ReleaseLag months(int months) { return new ReleaseLag(0, months); }
    public static final ReleaseLag NONE = new ReleaseLag(0, 0);
}
```

- [ ] **Step 5: Implement `ReleaseDate`**

```java
// java/src/main/java/lazic/utils/ingest/ReleaseDate.java
package lazic.utils.ingest;

import java.time.LocalDateTime;

/**
 * Shifts a period-start timestamp to the date the datapoint was first publicly knowable.
 * Lag is added to the period END (computed from cadence), not the period start.
 *
 * Convention: all DataPoint timestamps in this codebase represent knowledge-time, not
 * event-time. See java/docs/DATA_SOURCES.md.
 */
public final class ReleaseDate {
    private ReleaseDate() {}

    public static LocalDateTime applyLag(LocalDateTime periodStart, Cadence cadence, ReleaseLag lag) {
        LocalDateTime end = cadence.endOf(periodStart);
        return end.plusDays(lag.days()).plusMonths(lag.months());
    }
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `mvn -q test -Dtest=ReleaseDateTest` (from `java/`)
Expected: 5/5 PASS.

- [ ] **Step 7: Run full build**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 2: Apply lag in `NzGdp`

**Files:**
- Modify: `java/src/main/java/lazic/sources/NzGdp.java`

- [ ] **Step 1: Add lag constant near the top of the class**

After the `URL` field, add:

```java
// Stats NZ National Accounts (via OECD SDMX): quarterly release ~10 weeks after quarter-end.
// https://www.stats.govt.nz/release-calendar/
private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(75);
```

Add imports at the top:

```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;
```

- [ ] **Step 2: Wrap the timestamp at the emission point**

In `parseXmlData`, locate:

```java
LocalDateTime dateTime = convertQuarterToDateTime(timePeriodStr);
```

Replace with:

```java
LocalDateTime periodStart = convertQuarterToDateTime(timePeriodStr);
LocalDateTime dateTime = ReleaseDate.applyLag(periodStart, Cadence.QUARTERLY, RELEASE_LAG);
```

- [ ] **Step 3: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 3: Apply lag in `NzBalanceOfPayments`

**Files:**
- Modify: `java/src/main/java/lazic/sources/NzBalanceOfPayments.java`

- [ ] **Step 1: Read the file to locate emission point**

Run the `read` tool on `java/src/main/java/lazic/sources/NzBalanceOfPayments.java`. Identify the line that constructs each `DataPoint(timestamp, ...)`.

- [ ] **Step 2: Add lag constant + imports**

Imports:
```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;
```

Constant near class top:
```java
// RBNZ / Stats NZ Balance of Payments: quarterly release ~75 days after quarter-end.
private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(75);
```

- [ ] **Step 3: Wrap the timestamp at the emission point**

Wherever the period-start `LocalDateTime` is computed before being passed into `new DataPoint(...)`, insert before the `DataPoint` construction:

```java
timestamp = ReleaseDate.applyLag(timestamp, Cadence.QUARTERLY, RELEASE_LAG);
```

(Use the actual local variable name; rename inline if the variable is reused later.)

- [ ] **Step 4: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 4: Apply lag in `NzLaborStats`

**Files:**
- Modify: `java/src/main/java/lazic/sources/NzLaborStats.java`

- [ ] **Step 1: Read the file**

Read `java/src/main/java/lazic/sources/NzLaborStats.java` to locate emission point.

- [ ] **Step 2: Add lag constant + imports**

```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;

// Stats NZ Household Labour Force Survey: quarterly release ~6 weeks after quarter-end.
private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(45);
```

- [ ] **Step 3: Wrap timestamp**

At the `DataPoint` construction site, wrap the timestamp via `ReleaseDate.applyLag(periodStart, Cadence.QUARTERLY, RELEASE_LAG)` exactly as in Task 2.

- [ ] **Step 4: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 5: Apply lag in `NzVehicleRegistrations`

**Files:**
- Modify: `java/src/main/java/lazic/sources/NzVehicleRegistrations.java`

- [ ] **Step 1: Read the file**

Read `java/src/main/java/lazic/sources/NzVehicleRegistrations.java`.

- [ ] **Step 2: Add lag constant + imports**

```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;

// NZTA monthly vehicle registration stats: published ~25 days after month-end.
private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(25);
```

- [ ] **Step 3: Wrap timestamp**

Same pattern: `ReleaseDate.applyLag(periodStart, Cadence.MONTHLY, RELEASE_LAG)` at the `DataPoint` construction site.

- [ ] **Step 4: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 6: Apply lag in `NzBusinessConfidence`

**Files:**
- Modify: `java/src/main/java/lazic/sources/NzBusinessConfidence.java`

- [ ] **Step 1: Read the file**

Read `java/src/main/java/lazic/sources/NzBusinessConfidence.java`.

- [ ] **Step 2: Add lag + imports**

```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;

// ANZ Business Outlook: published end of survey month, lag ~5 days.
private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(5);
```

- [ ] **Step 3: Wrap timestamp**

`ReleaseDate.applyLag(periodStart, Cadence.MONTHLY, RELEASE_LAG)` at emission.

- [ ] **Step 4: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 7: Apply lag in `NzRoadFatalities`

**Files:**
- Modify: `java/src/main/java/lazic/sources/NzRoadFatalities.java`

- [ ] **Step 1: Read the file**

Read `java/src/main/java/lazic/sources/NzRoadFatalities.java`.

- [ ] **Step 2: Add lag + imports**

```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;

// NZTA / Ministry of Transport monthly road deaths: ~30 days after month-end.
private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(30);
```

- [ ] **Step 3: Wrap timestamp**

`ReleaseDate.applyLag(periodStart, Cadence.MONTHLY, RELEASE_LAG)` at emission.

- [ ] **Step 4: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 8: Apply lag in annual NZ sources (`NzTaxRevenue`, `NzPensions`, `NzLaborTaxation`)

These three are structurally identical (annual OECD SDMX). Repeat the same edit for each.

**Files:**
- Modify: `java/src/main/java/lazic/sources/NzTaxRevenue.java`
- Modify: `java/src/main/java/lazic/sources/NzPensions.java`
- Modify: `java/src/main/java/lazic/sources/NzLaborTaxation.java`

- [ ] **Step 1: For each file, read it**

Read each file in turn.

- [ ] **Step 2: For each file, add lag + imports**

```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;

// OECD revenue/pension/labour-tax stats: annual release ~12 months after year-end.
private static final ReleaseLag RELEASE_LAG = ReleaseLag.months(12);
```

- [ ] **Step 3: For each file, wrap timestamp**

`ReleaseDate.applyLag(periodStart, Cadence.ANNUAL, RELEASE_LAG)` at emission.

- [ ] **Step 4: Compile after all three**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 9: Apply lag in `GlobalFoodPrices` and `GlobalAquacultureProduction`

**Files:**
- Modify: `java/src/main/java/lazic/sources/GlobalFoodPrices.java`
- Modify: `java/src/main/java/lazic/sources/GlobalAquacultureProduction.java`

- [ ] **Step 1: Read both files**

Read each.

- [ ] **Step 2: Add lag + imports**

```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;

// FAO annual statistics: typical publication ~18 months after reference year-end.
private static final ReleaseLag RELEASE_LAG = ReleaseLag.months(18);
```

- [ ] **Step 3: Wrap timestamp**

`ReleaseDate.applyLag(periodStart, Cadence.ANNUAL, RELEASE_LAG)` at emission.

- [ ] **Step 4: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 10: Apply lag in `WikimediaPageviews`

**Files:**
- Modify: `java/src/main/java/lazic/sources/WikimediaPageviews.java`

- [ ] **Step 1: Read the file**

Read `java/src/main/java/lazic/sources/WikimediaPageviews.java`.

- [ ] **Step 2: Add lag + imports**

```java
import lazic.utils.ingest.Cadence;
import lazic.utils.ingest.ReleaseDate;
import lazic.utils.ingest.ReleaseLag;

// Wikimedia pageviews API: UTC daily totals are stable T+1. NZ market opens 12h after
// UTC midnight, so a UTC day's totals are not actionable until at least the next NZ
// session — apply +1 day lag.
private static final ReleaseLag RELEASE_LAG = ReleaseLag.of(1);
```

- [ ] **Step 3: Wrap timestamp**

`ReleaseDate.applyLag(periodStart, Cadence.DAILY, RELEASE_LAG)` at emission.

- [ ] **Step 4: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 11: Diagnostic verification

**Files:**
- Create: `java/src/main/java/lazic/diag/DiagReleaseDate.java`

- [ ] **Step 1: Write the diagnostic main**

```java
// java/src/main/java/lazic/diag/DiagReleaseDate.java
package lazic.diag;

import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.sources.NzGdp;
import lazic.sources.NzVehicleRegistrations;
import lazic.sources.WikimediaPageviews;
import lazic.sources.GlobalFoodPrices;

import java.util.List;

/** Temporary diagnostic. Run via `mvn -q exec:java -Dexec.mainClass="lazic.diag.DiagReleaseDate"`. */
public class DiagReleaseDate {
    public static void main(String[] args) {
        List<DataSourceBase> sources = List.of(
            new NzGdp(),
            new NzVehicleRegistrations(),
            new WikimediaPageviews(),
            new GlobalFoodPrices()
        );
        for (DataSourceBase s : sources) {
            DataPoint dp = s.getDataPoints().stream().findFirst().orElse(null);
            System.out.println(s.getSourceName() + " -> " + dp);
        }
    }
}
```

- [ ] **Step 2: Run it**

Run: `mvn -q exec:java -Dexec.mainClass="lazic.diag.DiagReleaseDate"` (from `java/`)
Expected: each line shows a `timestamp` clearly inside or after the period it represents (e.g. `NzGdp` 2024-Q1 row should have a timestamp ≥ 2024-06-14, not 2024-01-01).

- [ ] **Step 3: Inspect output, confirm shift is sensible for each source**

Manual inspection step. If any timestamp is unchanged or earlier than expected, return to that source's task and fix.

- [ ] **Step 4: Delete the diagnostic**

Delete `java/src/main/java/lazic/diag/DiagReleaseDate.java`. If the `lazic.diag` directory is now empty, delete it too.

- [ ] **Step 5: Compile**

Run: `mvn -q compile` (from `java/`)
Expected: BUILD SUCCESS.

---

## Task 12: Documentation

**Files:**
- Modify: `java/docs/DATA_SOURCES.md`
- Modify: `.github/copilot-instructions.md` (docs index)

- [ ] **Step 1: Add convention note to `java/docs/DATA_SOURCES.md`**

At the top of the document, immediately after any existing intro paragraph, insert:

```markdown
## Timestamp convention

All `DataPoint.timestamp` values represent the **earliest moment the datapoint was first publicly knowable in NZ time**, not the underlying period it describes. For example, NZ Q1 2024 GDP — covering Jan–Mar — is stamped around mid-June 2024, reflecting the typical Stats NZ release lag.

Each source declares its release lag as a `RELEASE_LAG` constant near the top of the class, with a comment citing the publishing agency's release calendar. Lag is applied via `lazic.utils.ingest.ReleaseDate.applyLag(periodStart, cadence, lag)` at the single point each `DataPoint` is constructed. See spec `docs/specs/2026-04-30-lookahead-bias-remediation.md`.

Sources unaffected by this convention (price/intraday data already correctly stamped): `YfPrices`, `YfFinances`, `NzRatesFx`.
```

- [ ] **Step 2: Add spec to docs index in `.github/copilot-instructions.md`**

Find the "Plans and specs" → spec list under the `## Documentation index` section. Add an entry (kept alphabetically/chronologically correct):

```markdown
- `docs/specs/2026-04-30-lookahead-bias-remediation.md` — Spec for shifting macro-source `DataPoint.timestamp` from period-start to public-release date, eliminating look-ahead bias across 12 ingestion sources.
```

- [ ] **Step 3: Final compile + test**

Run: `mvn -q test` (from `java/`)
Expected: ReleaseDateTest 5/5 PASS, no other regressions.

---

## Self-review checklist (post-execution)

- All 12 affected sources from the spec table have been modified (Tasks 2–10 cover them; verify count: NzGdp, NzBoP, NzLaborStats, NzVehicleReg, NzBusinessConfidence, NzRoadFatalities, NzTaxRevenue, NzPensions, NzLaborTaxation, GlobalFood, GlobalAquaculture, WikimediaPageviews = 12. ✓).
- Diagnostic file deleted (Task 11 step 4).
- Documentation index updated (Task 12 step 2).
- No Python changes were made (per spec out-of-scope).
- User informed they need to run `mvn -q exec:java` to refresh `data/data_long.csv`.
