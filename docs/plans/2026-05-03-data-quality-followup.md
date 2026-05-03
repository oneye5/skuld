# Skuld Data Quality Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the four remaining quality items: (1) factor IC / backtest performance audit after prior fixes, (2) Java unit tests for NzLaborStats and DataPoint, (3) size factor proxy for pre-2022 market-cap gap, (4) wiki CSV duplicate removal.

**Architecture:** Tasks 1–2 are independent (audit + Java). Task 3 modifies `PreparedPanel` and `size.py`. Task 4 is a one-line CSV edit. All tasks are independent — they may be executed in any order or in parallel.

**Tech Stack:** Python 3.11, pandas, scipy, pytest, Java 17, Maven, Gson

---

## Task 1: Factor IC audit script

**Files:**
- Create: `scripts/factor_ic_audit.py`

### Context
`scripts/analyse_pipeline.py` only checks data health — it never computes forward-return IC or runs the backtest engine. We need a standalone script that:
- Loads the prepared panel
- For each factor, computes cross-sectional rank IC (Spearman) against 1-month forward returns at every rebalance date
- Prints mean IC, ICIR (IC / std(IC)), and hit rate per factor
- Runs the full backtest via `BacktestEngine` and prints the key metrics (`sharpe_raw`, `sharpe_flat_haircut`, `calmar_ratio`, `hit_rate`, `avg_positions`) for the default `mom-s6.yaml` spec

This is a one-way read-only audit script — no model changes.

- [ ] **Step 1: Write the script**

```python
# scripts/factor_ic_audit.py
"""
Factor IC audit: measures cross-sectional rank IC per factor and runs full backtest.

Usage:
    uv run --project python python scripts/factor_ic_audit.py

Outputs:
    - Per-factor: mean IC, ICIR, hit rate (fraction of dates with IC > 0)
    - Full backtest summary for configs/strategy-specs/candidates/mom-s6.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Allow imports from the python src tree when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "src"))

from skuld_research.config.loader import load_spec
from skuld_research.data.csv_loader import load_csv
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.backtest.engine import BacktestEngine
from skuld_research.factors.momentum import MomentumFactor
from skuld_research.factors.size import SizeFactor
from skuld_research.factors.low_volatility import LowVolatilityFactor
from skuld_research.factors.dividend_yield import DividendYieldFactor


DATA_PATH = Path("data/data_long.csv")
SPEC_PATH = Path("configs/strategy-specs/candidates/mom-s6.yaml")


def rank_ic(scores: pd.Series, fwd_returns: pd.Series) -> float | None:
    """Spearman rank IC between factor scores and forward returns, shared tickers only."""
    common = scores.index.intersection(fwd_returns.index)
    if len(common) < 5:
        return None
    ic, _ = spearmanr(scores.loc[common], fwd_returns.loc[common])
    return float(ic)


def compute_factor_ics(panel, rebalance_dates: list, factor, fwd_returns_monthly: pd.DataFrame) -> list[float]:
    ics = []
    for date in rebalance_dates:
        scores = factor.score(panel, date)
        if scores is None or scores.empty:
            continue
        # Forward return = return in the month *after* this rebalance date
        date_idx = fwd_returns_monthly.index.searchsorted(date)
        if date_idx + 1 >= len(fwd_returns_monthly):
            continue
        fwd_date = fwd_returns_monthly.index[date_idx + 1]
        fwd_ret = fwd_returns_monthly.loc[fwd_date].dropna()
        ic = rank_ic(scores, fwd_ret)
        if ic is not None:
            ics.append(ic)
    return ics


def print_ic_summary(name: str, ics: list[float]) -> None:
    if not ics:
        print(f"  {name:<30} no valid IC dates")
        return
    arr = np.array(ics)
    mean_ic = arr.mean()
    icir = mean_ic / arr.std() if arr.std() > 1e-9 else float("nan")
    hit = (arr > 0).mean()
    print(f"  {name:<30} mean_IC={mean_ic:+.4f}  ICIR={icir:+.3f}  hit={hit:.1%}  n={len(arr)}")


def main() -> None:
    print("Loading data...")
    raw = load_csv(DATA_PATH)

    spec = load_spec(SPEC_PATH)
    print(f"Building prepared panel (spec: {SPEC_PATH.name})...")
    panel = build_prepared_panel(raw, spec)

    # Monthly returns for IC computation
    monthly_close = panel.close.resample("ME").last()
    fwd_returns_monthly = monthly_close.pct_change().shift(-1)

    # Rebalance dates from spec (use end-of-month dates covering panel range)
    rebalance_dates = pd.date_range(panel.close.index[0], panel.close.index[-1], freq="ME").tolist()

    factors = [
        ("Momentum",       MomentumFactor(spec.factors.momentum)),
        ("Size",           SizeFactor(spec.factors.size)),
        ("LowVolatility",  LowVolatilityFactor(spec.factors.low_volatility)),
        ("DividendYield",  DividendYieldFactor(spec.factors.dividend_yield)),
    ]

    print("\n=== Factor IC Audit ===")
    for name, factor in factors:
        ics = compute_factor_ics(panel, rebalance_dates, factor, fwd_returns_monthly)
        print_ic_summary(name, ics)

    print("\n=== Backtest (mom-s6.yaml) ===")
    engine = BacktestEngine(spec)
    result = engine.run(panel)
    print(f"  Period:              {result.start:%Y-%m} → {result.end:%Y-%m}  ({result.n_periods} months)")
    print(f"  Sharpe (raw):        {result.sharpe_raw:.3f}")
    print(f"  Sharpe (400bps hc):  {result.sharpe_flat_haircut:.3f}")
    print(f"  Calmar:              {result.calmar_ratio:.3f}")
    print(f"  Hit rate:            {result.hit_rate:.1%}")
    print(f"  Avg positions:       {result.avg_positions:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and capture output**

```
uv run --project python python scripts/factor_ic_audit.py
```

Expected: prints IC table and backtest summary. If a factor constructor signature differs from what's shown (the script uses `spec.factors.momentum` etc.), adjust to match actual spec fields — check `python/src/skuld_research/config/spec.py` for the exact attribute names and pass them accordingly. Fix any import or attribute errors until the script runs cleanly.

- [ ] **Step 3: Verify output looks reasonable**

Sanity checks:
- Momentum IC should be positive (NZX has shown momentum persistence)
- Size IC direction is less certain on a small exchange
- All factors should have `n > 20` dates
- Sharpe raw should be in range `[-1, 5]`; if outside that range, something is wrong

---

## Task 2: Java unit tests for DataPoint and NzLaborStats

**Files:**
- Modify: `java/pom.xml` — add JUnit 5 + Surefire test dependencies
- Create: `java/src/test/java/lazic/utils/ingest/DataPointTest.java`
- Create: `java/src/test/java/lazic/sources/NzLaborStatsTest.java`
- Create: `java/src/test/resources/lazic/sources/sdmx_labor_fixture.json`

### Context
No Java tests exist. Maven has no test dependencies. The `DataPoint.equals()`/`hashCode()` fix and the `NzLaborStats.buildFeatureName()` regional-suffix fix both need regression coverage.

`NzLaborStats.buildFeatureName` is private — the test will call `getDataPoints()` on a stubbed instance that parses a small fixture JSON instead of hitting the live URL, by overriding `WebHtmlGetter.get()` or by extracting the parse logic into a package-private method.

The cleanest approach without mocking frameworks: extract `parseObservations(String rawJson)` as a package-private method in `NzLaborStats`, then test it directly.

- [ ] **Step 1: Add JUnit 5 and Surefire to pom.xml**

In `java/pom.xml`, add inside `<dependencies>`:

```xml
<dependency>
  <groupId>org.junit.jupiter</groupId>
  <artifactId>junit-jupiter</artifactId>
  <version>5.10.2</version>
  <scope>test</scope>
</dependency>
```

And add a `<build><plugins>` section (or extend existing one) with:

```xml
<build>
  <plugins>
    <plugin>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-surefire-plugin</artifactId>
      <version>3.2.5</version>
    </plugin>
  </plugins>
</build>
```

- [ ] **Step 2: Create test directory structure**

```
java/src/test/java/lazic/utils/ingest/
java/src/test/java/lazic/sources/
java/src/test/resources/lazic/sources/
```

Run: `New-Item -ItemType Directory -Force java/src/test/java/lazic/utils/ingest, java/src/test/java/lazic/sources, java/src/test/resources/lazic/sources`

- [ ] **Step 3: Write DataPointTest.java**

```java
// java/src/test/java/lazic/utils/ingest/DataPointTest.java
package lazic.utils.ingest;

import org.junit.jupiter.api.Test;
import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class DataPointTest {

    private static final LocalDateTime TS = LocalDateTime.of(2024, 1, 1, 0, 0);

    @Test
    void equalPoints_areEqual() {
        DataPoint a = new DataPoint(TS, "NZX:AIR", "close_price", 2.50);
        DataPoint b = new DataPoint(TS, "NZX:AIR", "close_price", 2.50);
        assertEquals(a, b);
    }

    @Test
    void equalPoints_haveSameHashCode() {
        DataPoint a = new DataPoint(TS, "NZX:AIR", "close_price", 2.50);
        DataPoint b = new DataPoint(TS, "NZX:AIR", "close_price", 2.50);
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    void differentValue_notEqual() {
        DataPoint a = new DataPoint(TS, null, "unemployment_rate", 11.3);
        DataPoint b = new DataPoint(TS, null, "unemployment_rate", 18400.0);
        assertNotEquals(a, b);
    }

    @Test
    void hashSet_deduplicatesExactDuplicates() {
        DataPoint a = new DataPoint(TS, null, "inflation", 3.5);
        DataPoint b = new DataPoint(TS, null, "inflation", 3.5);
        Set<DataPoint> set = new HashSet<>();
        set.add(a);
        set.add(b);
        assertEquals(1, set.size(), "HashSet must collapse identical DataPoints to one entry");
    }

    @Test
    void hashSet_keepsDistinctValues() {
        DataPoint rate = new DataPoint(TS, null, "unemployment", 11.3);
        DataPoint count = new DataPoint(TS, null, "unemployment", 18400.0);
        Set<DataPoint> set = new HashSet<>();
        set.add(rate);
        set.add(count);
        assertEquals(2, set.size(), "DataPoints with different values must remain distinct in HashSet");
    }

    @Test
    void nullTicker_handledCorrectly() {
        DataPoint a = new DataPoint(TS, null, "gdp", 200000.0);
        DataPoint b = new DataPoint(TS, null, "gdp", 200000.0);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }
}
```

- [ ] **Step 4: Extract parseObservations in NzLaborStats.java**

In `java/src/main/java/lazic/sources/NzLaborStats.java`, refactor `getDataPoints()` to delegate parsing to a new package-private method:

```java
// Add this method alongside getDataPoints():
Set<DataPoint> parseObservations(String rawData) {
    Set<DataPoint> dataPoints = new HashSet<>();
    Gson gson = new Gson();
    try {
        JsonObject root = gson.fromJson(rawData, JsonObject.class);
        JsonArray dataSets = root.getAsJsonArray("dataSets");
        if (dataSets == null || dataSets.size() == 0) return dataPoints;

        JsonObject dataSet = dataSets.get(0).getAsJsonObject();
        JsonObject observations = dataSet.getAsJsonObject("observations");

        JsonObject structure = root.getAsJsonObject("structure");
        JsonArray obsDims = structure.getAsJsonObject("dimensions").getAsJsonArray("observation");

        JsonArray timePeriodValues = null;
        for (int i = 0; i < obsDims.size(); i++) {
            JsonObject dim = obsDims.get(i).getAsJsonObject();
            if ("TIME_PERIOD".equals(dim.get("id").getAsString())) {
                timePeriodValues = dim.getAsJsonArray("values");
                break;
            }
        }

        for (Map.Entry<String, JsonElement> entry : observations.entrySet()) {
            String key = entry.getKey();
            JsonArray values = entry.getValue().getAsJsonArray();
            if (values.size() > 0 && !values.get(0).isJsonNull()) {
                double value = values.get(0).getAsDouble();
                String[] dimensions = key.split(":");
                String featureName = buildFeatureName(dimensions);
                LocalDateTime timestamp;
                if (dimensions.length > 8 && timePeriodValues != null) {
                    int timeIdx = Integer.parseInt(dimensions[8]);
                    String year = timePeriodValues.get(timeIdx).getAsJsonObject().get("id").getAsString();
                    LocalDateTime periodStart = LocalDateTime.of(Integer.parseInt(year), 1, 1, 0, 0);
                    timestamp = ReleaseDate.applyLag(periodStart, Cadence.ANNUAL, RELEASE_LAG);
                } else {
                    timestamp = LocalDateTime.now();
                }
                if (!ReleaseFilter.isKnowableNow(timestamp)) continue;
                dataPoints.add(new DataPoint(timestamp, null, featureName, value));
            }
        }
    } catch (Exception e) {
        System.err.println("Error parsing SDMX-JSON data: " + e.getMessage());
        e.printStackTrace();
    }
    return dataPoints;
}
```

Then simplify `getDataPoints()` to:

```java
@Override
public Set<DataPoint> getDataPoints() {
    String rawData = WebHtmlGetter.get(URL);
    try (FileWriter writer = new FileWriter("sample_data.txt")) {
        writer.write(rawData);
    } catch (IOException e) {
        System.err.println("Warning: Could not write debug file: " + e.getMessage());
    }
    return parseObservations(rawData);
}
```

- [ ] **Step 5: Create minimal SDMX fixture JSON**

Save to `java/src/test/resources/lazic/sources/sdmx_labor_fixture.json`.

This fixture contains two observations:
- Both share `measure=3 (Unemployment), age=0 (Y15T24), sex=0 (M)`, time=2005
- But differ in `dim[1]` (0 = NZL national, 1 = Auckland sub-region) and `dim[3]` (0 = national level, 1 = regional)
- This reproduces the exact collision that the dim[1]/dim[3] suffix fix resolves

```json
{
  "structure": {
    "dimensions": {
      "observation": [
        {"id": "FREQ",             "values": [{"id": "A"}]},
        {"id": "REF_AREA",         "values": [{"id": "NZL"}, {"id": "NZL_AUK"}]},
        {"id": "GEO_SCOPE",        "values": [{"id": "NZL"}]},
        {"id": "TERRITORIAL_LEVEL","values": [{"id": "NAT"}, {"id": "REG"}]},
        {"id": "MEASURE",          "values": [{"id": "POP"}, {"id": "LF"}, {"id": "EMP"}, {"id": "UNE"}]},
        {"id": "AGE",              "values": [{"id": "Y15T24"}, {"id": "Y_GT15"}, {"id": "Y15T64"}]},
        {"id": "SEX",              "values": [{"id": "M"}, {"id": "F"}, {"id": "_T"}]},
        {"id": "UNIT_MEASURE",     "values": [{"id": "N"}]},
        {"id": "TIME_PERIOD",      "values": [{"id": "2005"}]}
      ]
    }
  },
  "dataSets": [{
    "observations": {
      "0:0:0:0:3:0:0:0:0": [11.3],
      "0:1:0:1:3:0:0:0:0": [4500.0]
    }
  }]
}
```

- [ ] **Step 6: Write NzLaborStatsTest.java**

```java
// java/src/test/java/lazic/sources/NzLaborStatsTest.java
package lazic.sources;

import lazic.utils.ingest.DataPoint;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

class NzLaborStatsTest {

    private String loadFixture() throws IOException {
        try (InputStream is = getClass().getResourceAsStream("/lazic/sources/sdmx_labor_fixture.json")) {
            assertNotNull(is, "Fixture file not found on classpath");
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    @Test
    void parseObservations_noConflictingFeatureNames() throws IOException {
        NzLaborStats source = new NzLaborStats();
        Set<DataPoint> points = source.parseObservations(loadFixture());

        // Group by (timestamp, feature) and assert each group has exactly 1 value
        Map<String, Long> countByKey = points.stream()
            .collect(Collectors.groupingBy(
                dp -> dp.getTimestamp() + "|" + dp.getFeatureName(),
                Collectors.counting()
            ));

        countByKey.forEach((key, count) ->
            assertEquals(1L, count,
                "Duplicate (timestamp, feature) found: " + key + " appeared " + count + " times")
        );
    }

    @Test
    void parseObservations_nationalAndRegionalHaveDistinctNames() throws IOException {
        NzLaborStats source = new NzLaborStats();
        Set<DataPoint> points = source.parseObservations(loadFixture());

        Set<String> featureNames = points.stream()
            .map(DataPoint::getFeatureName)
            .collect(Collectors.toSet());

        assertEquals(2, featureNames.size(),
            "National and Auckland regional observations must produce 2 distinct feature names, got: " + featureNames);
    }

    @Test
    void parseObservations_returnsNonEmptySet() throws IOException {
        NzLaborStats source = new NzLaborStats();
        Set<DataPoint> points = source.parseObservations(loadFixture());
        assertFalse(points.isEmpty(), "Should parse at least one observation from fixture");
    }
}
```

- [ ] **Step 7: Run the Java tests**

```
mvn test -f java/pom.xml
```

Expected: `BUILD SUCCESS`, all 6 tests pass (`DataPointTest` × 5, `NzLaborStatsTest` × 3).

If Maven is not on `$PATH`, use: `cd java; mvn test` or check for a wrapper (`./mvnw test`).

---

## Task 3: Size factor proxy using shares × close

**Files:**
- Modify: `python/src/skuld_research/data/prepared_panel.py` — add `market_cap_proxy` attribute built from `shares_outstanding × close` where available
- Modify: `python/src/skuld_research/factors/size.py` — fall back to `market_cap_proxy` when `market_cap` is NaN
- Modify: `python/tests/test_prepared_panel.py` — add test for proxy fallback
- Modify: `python/tests/test_factors.py` (or create if absent) — test size factor uses proxy

### Context
`market_cap` is 96.5% NaN pre-2022 because `shares_outstanding` was not in the source data until then. `close` price exists for the full history.

Before writing any code, we need to know whether `shares_outstanding` actually exists as a feature in the loaded panel. Run this first:

- [ ] **Step 1: Check whether shares_outstanding is present in the data**

```python
# Run interactively or as a one-liner:
uv run --project python python -c "
import pandas as pd
df = pd.read_csv('data/data_long.csv')
cols = df['feature'].unique()
matches = [c for c in cols if 'share' in c.lower() or 'outstanding' in c.lower() or 'shares' in c.lower()]
print('share-related features:', matches)
print('All ticker features (sample):', [c for c in cols if not c.startswith('nz_') and not c.startswith('eco') and not c.startswith('aqua')][:30])
"
```

- [ ] **Step 2 (conditional): If shares_outstanding exists — add market_cap_proxy to PreparedPanel**

If `shares_outstanding` is present as a feature, locate where `PreparedPanel` is constructed (in `prepared_panel.py`, find the `PreparedPanel` dataclass/namedtuple and its construction in `build_prepared_panel()`). Add:

```python
# In build_prepared_panel(), after market_cap is built:
shares = _pivot_feature(price_df, "shares_outstanding")  # adjust feature name to match
market_cap_proxy = shares.multiply(close, fill_value=np.nan)  # element-wise, aligned on index/columns
```

Add `market_cap_proxy: pd.DataFrame` to the `PreparedPanel` dataclass.

- [ ] **Step 3 (conditional): Update SizeFactor to fall back to proxy**

In `python/src/skuld_research/factors/size.py`, in the `score()` method, replace the single `panel.market_cap` lookup with:

```python
mc = panel.market_cap.loc[:t].dropna(how="all").iloc[-1] if hasattr(panel, 'market_cap') else pd.Series(dtype=float)
if mc.isna().all() and hasattr(panel, 'market_cap_proxy'):
    mc = panel.market_cap_proxy.loc[:t].dropna(how="all").iloc[-1]
proxy_used = mc.isna().mean() > 0.5  # majority from proxy
if proxy_used:
    import logging; logging.getLogger(__name__).debug("SizeFactor using market_cap_proxy at %s", t)
```

- [ ] **Step 4 (if shares_outstanding absent): Document and skip**

If `shares_outstanding` is not in the data, the proxy cannot be built from existing ingested data. In that case:
- Add a comment to `size.py` line 1: `# NOTE: market_cap is pre-2022 sparse (96.5% NaN). shares_outstanding not available in data. Proxy not possible without re-ingest.`
- No code changes needed.
- Skip steps 2 and 3.

- [ ] **Step 5: Write test for proxy fallback (if proxy was implemented)**

In `python/tests/test_prepared_panel.py`, add:

```python
def test_market_cap_proxy_fills_nan_market_cap(prepared_panel_fixture):
    """market_cap_proxy should be non-null where market_cap is null but close and shares exist."""
    panel = prepared_panel_fixture
    if not hasattr(panel, 'market_cap_proxy'):
        pytest.skip("market_cap_proxy not implemented (shares_outstanding absent from data)")
    # Where market_cap is NaN but close is available, proxy should be non-NaN if shares exist
    mc_nan_mask = panel.market_cap.isna()
    proxy_at_mc_nan = panel.market_cap_proxy[mc_nan_mask]
    assert proxy_at_mc_nan.notna().any().any(), "Proxy should fill at least some NaN market_cap cells"
```

- [ ] **Step 6: Run tests**

```
uv run --project python pytest python/tests/test_prepared_panel.py python/tests/test_factors.py -v
```

Expected: all pass.

---

## Task 4: Remove duplicate wiki article entries

**Files:**
- Modify: `java/src/main/resources/lazic/sources/config/wikimedia_pages.csv`

### Context
`economy_of_australia` appears on lines 212 and 286. `economy_of_china` appears on lines 220 and 285. Both are duplicate entries introduced during an audit pass on 2026-01-14. The `DataPoint.equals()` fix (already applied) will prevent duplicate values from entering the `HashSet`, but the duplicate HTTP requests still fire (wasting ~8 network calls per ingest run). Removing the duplicates from the CSV is the cleanest fix.

- [ ] **Step 1: Verify duplicates in CSV**

```
uv run --project python python -c "
from collections import Counter
import csv
with open('java/src/main/resources/lazic/sources/config/wikimedia_pages.csv') as f:
    rows = [r for r in csv.reader(f) if r and not r[0].startswith('#')]
articles = [r[1] for r in rows if len(r) > 1]
dups = {a: c for a, c in Counter(articles).items() if c > 1}
print('Duplicates:', dups)
"
```

Expected output: `Duplicates: {'Economy_of_Australia': 2, 'Economy_of_China': 2}` (and possibly others).

- [ ] **Step 2: Remove the duplicate lines from the 2026-01-14 audit section**

Open `java/src/main/resources/lazic/sources/config/wikimedia_pages.csv`. The 2026-01-14 audit section starts around line 280. Remove the lines that duplicate earlier entries. Keep the original entries (lines ~212 and ~220) and delete only the duplicate lines in the audit section.

After editing, re-run the verification command from Step 1. Expected: `Duplicates: {}`.

- [ ] **Step 3: Run Python tests to confirm no regressions**

```
uv run --project python pytest python/ -q
```

Expected: 277 passed, 1 pre-existing failure (`test_m8_mom_spec_hash_matches_expected`).

---

## Execution Order

Tasks are independent. Suggested order for a single session:
1. **Task 4** first — trivial CSV edit, 2 minutes
2. **Task 2** — Java tests, moderate effort
3. **Task 1** — IC audit script, moderate effort
4. **Task 3** — size proxy, conditional on data availability check in Step 1

---

## Verification Checklist

Before declaring done:
- [ ] `uv run --project python pytest python/ -q` → 277 passed, 1 pre-existing failure
- [ ] `mvn test -f java/pom.xml` → BUILD SUCCESS (if Java tests implemented)
- [ ] `python scripts/factor_ic_audit.py` → prints IC table and backtest summary without error
- [ ] Wiki CSV has no duplicate articles (Step 4 verification command shows `{}`)
