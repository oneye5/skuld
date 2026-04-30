# Data Sources Reference

## Timestamp convention (knowledge-time)

Every `DataPoint.timestamp` produced by an ingestion source is the **public-release date** of that observation, not the period start. This is the timestamp at which an investor could first have known the value, and it is the value Python research code treats as the knowledge-time index.

For sources whose raw API stamps an observation at the *period start* (most OECD/SDMX macro feeds), the source class shifts the timestamp to **end-of-period + publication lag** via `lazic.utils.ingest.ReleaseDate.applyLag(periodStart, cadence, lag)`. Each affected source declares a `RELEASE_LAG` constant with a comment citing the publication-calendar source.

Sources whose raw API already returns a release-date timestamp (e.g. `YfPrices` daily bars, finalised at end of trading day) do not need a lag wrap.

See `docs/specs/2026-04-30-lookahead-bias-remediation.md` for the lag table and rationale.

## Market Data Sources

### YfPrices (Yahoo Finance)

Daily OHLCV price data for all tickers.

| Feature | Description |
|---------|-------------|
| `open`, `high`, `low`, `close` | OHLC prices, raw (in the share units that prevailed at the bar's date — *not* adjusted for splits) |
| `adj_close` | Yahoo's dividend/split-adjusted close. Known to be unreliable on tickers with dramatic reverse splits (e.g. negative values on MPG.NZ post-2026 1-for-40); the Python adjustments layer (`docs/ADJUSTMENTS.md`) is the SSOT for adjusted prices in research code. |
| `volume` | Trading volume |
| `dividend` | Per-share cash dividend, **back-adjusted** to the share unit of its ex_date by multiplying by the cumulative product of every split ratio with `split_date > ex_date`. This keeps `dividend` and raw `close` on the same per-share scale at every point in history. Yahoo's source field is in *current* share-equivalent units, which would otherwise mismatch the raw `close` for any ticker that has had a split. |
| `split` | Stock split events. Ratio = numerator/denominator (e.g. 4-for-1 forward = 4.0; 1-for-40 reverse = 0.025). |

**API endpoint:**
```
https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}?interval=1d&period1=0&period2=99999999999&includeAdjustedClose=true&events=div,split
```

**Implementation:** `sources/YfPrices.java`

### YfFinances (Yahoo Finance Fundamentals)

Company financial metrics: market cap, P/E, EPS, book value, revenue, and ~130 other fields.

**Implementation:** `sources/YfFinances.java`

## Ticker Configuration

NZX tickers (~146) are defined in `sources/config/Tickers.java`. International tickers that produced data in the current run:

| Ticker | Description |
|--------|-------------|
| `%5EFTSE` | FTSE 100 (UK) |
| `%5ETNX` | 10Y US Treasury Yield |
| `ZS=F` | Soybean futures |

## Macroeconomic and Alternative Sources

| Source class | File | External source | Description |
|---|---|---|---|
| `NzGdp` | `NzGdp.java` | Stats NZ | Quarterly GDP, growth rate, components |
| `NzRatesFx` | `NzRatesFx.java` | RBNZ | OCR, government bond yields, exchange rates |
| `NzBusinessConfidence` | `NzBusinessConfidence.java` | ANZ Business Outlook | Overall confidence, activity outlook, employment intentions |
| `NzLaborStats` | `NzLaborStats.java` | Stats NZ | Unemployment rate, employment growth, LFP rate |
| `NzVehicleRegistrations` | `NzVehicleRegistrations.java` | Motor Industry Association | New and commercial vehicle registrations |
| `NzBalanceOfPayments` | `NzBalanceOfPayments.java` | Stats NZ | Current account, trade balance, foreign investment flows |
| `NzTaxRevenue` | `NzTaxRevenue.java` | IRD/Treasury | Total tax, GST, company tax |
| `NzPensions` | `NzPensions.java` | — | Pension/superannuation data |
| `NzLaborTaxation` | `NzLaborTaxation.java` | — | Employment-related taxation |
| `NzRoadFatalities` | `NzRoadFatalities.java` | — | Traffic fatalities as economic/sentiment proxy |
| `GlobalAquacultureProduction` | `GlobalAquacultureProduction.java` | — | Global aquaculture data; NZ has significant salmon/mussel exports |
| `WikimediaPageviews` | `WikimediaPageviews.java` | Wikimedia | Wikipedia pageviews for NZX companies; 1.17M rows, signal unvalidated |

## Data Format

Macroeconomic rows have an **empty `ticker` field**. Use empty string `""` — not a `MACRO_` prefix — or the Python loader will not recognise the row.

```csv
timestamp,ticker,feature,value
1705276800000,AIR.NZ,close,0.65
1705276800000,AIR.NZ,volume,1234567
1705276800000,,gdp_growth,0.023
```

## Adding a New Source

### 1. Create source class

```java
// sources/MyNewSource.java
package lazic.sources;

import lazic.utils.ingest.*;
import java.util.*;

public class MyNewSource extends DataSourceBase {

    private final String API_URL = "https://api.example.com/data";

    @Override
    public Set<DataPoint> getDataPoints() {
        Set<DataPoint> points = new HashSet<>();
        try {
            String json = WebHtmlGetter.get(API_URL);
            // parse and create DataPoints
            points.add(new DataPoint(
                timestamp,
                "",          // empty string for macro; ticker symbol (e.g. "AIR.NZ") for equities
                "feature_name",
                value
            ));
        } catch (Exception e) {
            System.err.println("Failed to fetch: " + e.getMessage());
        }
        return points;
    }
}
```

### 2. Register in Main.java

```java
new MyNewSource();
IngestManager.INSTANCE.fetchDataFromSources();
```

### 3. Update this file

Add a row to the sources table above.

## Related Documentation

- [Architecture](ARCHITECTURE.md) — System design, core components, build/run instructions
- [Data Analysis](../../docs/DATA_ANALYSIS.md) — Long-CSV schema and contents
