# Data Sources Reference

## Market Data Sources

### YfPrices (Yahoo Finance)

Daily OHLCV price data for all tickers.

| Feature | Description |
|---------|-------------|
| `open`, `high`, `low`, `close` | OHLC prices |
| `adj_close` | Dividend/split-adjusted close |
| `volume` | Trading volume |
| `dividend` | Dividend events |
| `split` | Stock split events |

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
