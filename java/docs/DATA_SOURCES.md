# Data Sources Reference

> **Navigation:** [Main README](../../README.md) | [Pipeline Guide](../../docs/RANKING_PIPELINE_GUIDE.md) | [Java Architecture](ARCHITECTURE.md) | [Java Data Sources](DATA_SOURCES.md) | [TODO](../../docs/TODO.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Market Data Sources](#market-data-sources)
3. [Ticker Configuration](#ticker-configuration)
4. [Macroeconomic Sources](#macroeconomic-sources)
5. [Alternative Data Sources](#alternative-data-sources)
6. [Data Format](#data-format)
7. [Adding a New Source](#adding-a-new-source)
8. [Error Handling](#error-handling)
9. [Related Documentation](#related-documentation)

---

## Overview

The Java ingestion module fetches data from multiple sources to provide comprehensive market data for the ML pipeline.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCE CATEGORIES                              │
└────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  MARKET DATA     │    │  MACROECONOMIC   │    │  ALTERNATIVE     │
│                  │    │                  │    │                  │
│ • YfPrices       │    │ • NzGdp          │    │ • RoadFatalities │
│ • YfFinances     │    │ • NzRatesFx      │    │ • Aquaculture    │
│                  │    │ • NzBusinessConf │    │                  │
│ OHLCV, dividends │    │ • NzLaborStats   │    │ Sentiment proxy  │
│ splits, volume   │    │ • NzTaxRevenue   │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Market Data Sources

### YfPrices (Yahoo Finance)

**Purpose:** Daily OHLCV price data for all tickers.

**Features Fetched:**
| Feature | Description |
|---------|-------------|
| `Open` | Opening price |
| `High` | Daily high |
| `Low` | Daily low |
| `Close` | Closing price |
| `AdjClose` | Adjusted close (dividend/split adjusted) |
| `Volume` | Trading volume |
| `Dividend` | Dividend events |
| `Split` | Stock split events |

**API Endpoint:**
```
https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}
  ?interval=1d
  &period1=0
  &period2=99999999999
  &includeAdjustedClose=true
  &events=div,split
```

**Implementation:** `sources/YfPrices.java`

### YfFinances (Yahoo Finance Fundamentals)

**Purpose:** Company financial metrics.

**Features Fetched:**
- Market capitalization
- P/E ratio
- EPS
- Book value
- Revenue
- Other fundamental data

**Implementation:** `sources/YfFinances.java`

## Ticker Configuration

### NZX Stocks

Main trading stocks on New Zealand Stock Exchange:

```java
// sources/config/Tickers.java
String[] TICKERS = {
    "ANZ.NZ", "AIR.NZ", "FPH.NZ", "SPK.NZ", ...
    // ~139 NZX tickers
};
```

### International Indices

For market context:

| Ticker | Description |
|--------|-------------|
| `%5EGSP` | S&P 500 (US) |
| `%5EFTSE` | FTSE 100 (UK) |
| `%5EN225` | Nikkei 225 (Japan) |
| `000001.SS` | Shanghai Composite (China) |

### Commodities

| Ticker | Description |
|--------|-------------|
| `CL=F` | Crude Oil |
| `NG=F` | Natural Gas |
| `GC=F` | Gold |
| `SI=F` | Silver |
| `HG=F` | Copper |
| `ZC=F` | Corn |
| `ZW=F` | Wheat |
| `UX=F` | Uranium |

### Forex

| Ticker | Description |
|--------|-------------|
| `NZDUSD=X` | NZD/USD |
| `NZDAUD=X` | NZD/AUD |
| `NZDEUR=X` | NZD/EUR |
| `NZDJPY=X` | NZD/JPY |

### Other

| Ticker | Description |
|--------|-------------|
| `%5ETNX` | 10Y US Treasury Yield |

## Macroeconomic Sources

### NzGdp

**Purpose:** New Zealand GDP data.

**Source:** Stats NZ

**Features:**
- Quarterly GDP
- GDP growth rate
- GDP components

**Implementation:** `sources/NzGdp.java`

### NzRatesFx

**Purpose:** NZ interest rates and FX data.

**Source:** RBNZ

**Features:**
- Official Cash Rate (OCR)
- Government bond yields
- Exchange rates

**Implementation:** `sources/NzRatesFx.java`

### NzBusinessConfidence

**Purpose:** Business sentiment indicators.

**Source:** ANZ Business Outlook Survey

**Features:**
- Overall confidence
- Activity outlook
- Employment intentions

**Implementation:** `sources/NzBusinessConfidence.java`

### NzLaborStats

**Purpose:** Labor market data.

**Source:** Stats NZ

**Features:**
- Unemployment rate
- Employment growth
- Labor force participation

**Implementation:** `sources/NzLaborStats.java`

### NzVehicleRegistrations

**Purpose:** Economic activity proxy.

**Source:** Motor Industry Association

**Features:**
- New vehicle registrations
- Commercial vehicle registrations

**Implementation:** `sources/NzVehicleRegistrations.java`

### NzBalanceOfPayments

**Purpose:** External account data.

**Source:** Stats NZ

**Features:**
- Current account balance
- Trade balance
- Foreign investment flows

**Implementation:** `sources/NzBalanceOfPayments.java`

### NzTaxRevenue

**Purpose:** Government revenue data.

**Source:** IRD/Treasury

**Features:**
- Total tax revenue
- GST revenue
- Company tax

**Implementation:** `sources/NzTaxRevenue.java`

### NzPensions

**Purpose:** Pension/superannuation data.

**Implementation:** `sources/NzPensions.java`

### NzLaborTaxation

**Purpose:** Employment-related taxation data.

**Implementation:** `sources/NzLaborTaxation.java`

## Alternative Data Sources

### NzRoadFatalities

**Purpose:** Traffic fatality data as sentiment/economic proxy.

**Rationale:** Road fatalities correlate with:
- Economic activity (more driving)
- Consumer confidence
- Tourism activity

**Implementation:** `sources/NzRoadFatalities.java`

### GlobalAquacultureProduction

**Purpose:** Global aquaculture/fishery data.

**Rationale:** NZ has significant aquaculture exports (salmon, mussels).

**Implementation:** `sources/GlobalAquacultureProduction.java`

## Data Format

### Output Format (Long CSV)

```csv
timestamp,ticker,feature,value
1705276800000,AIR.NZ,Close,0.65
1705276800000,AIR.NZ,Volume,1234567
1705276800000,AIR.NZ,High,0.67
1705276800000,MACRO_NZ_GDP,GDP_Growth,0.023
...
```

### Macroeconomic Prefix

Macroeconomic data uses `MACRO_` prefix to distinguish from ticker data:

```
MACRO_NZ_GDP          → NZ GDP data
MACRO_NZ_OCR          → Official Cash Rate
MACRO_NZ_UNEMPLOYMENT → Unemployment rate
```

This prefix is handled by the Python pipeline's `add_macro_prefix()` function.

## Adding a New Source

### 1. Create Source Class

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
            // Parse JSON
            // Create DataPoints
            points.add(new DataPoint(
                timestamp,
                "MACRO_MY_DATA",  // or ticker for stock data
                "FeatureName",
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
public static void main(String[] args) {
    // ... existing sources ...
    new MyNewSource();
    
    IngestManager.INSTANCE.fetchDataFromSources();
    // ...
}
```

### 3. Update Documentation

Add source to this file with:
- Purpose
- Data source
- Features fetched
- Implementation file

## Error Handling

### Network Failures

```java
try {
    String data = WebHtmlGetter.get(url);
} catch (Exception e) {
    System.err.println("Failed to fetch " + ticker + ": " + e.getMessage());
    return new HashSet<>();  // Return empty, don't crash
}
```

### Invalid Data

- Null values are filtered by IngestManager
- Invalid JSON is logged and skipped
- Missing fields result in null values (filtered out)

## Related Documentation

- [Architecture](ARCHITECTURE.md) — System design
- [Main README](../../README.md) — Project overview
- [Python Pipeline](../../docs/RANKING_PIPELINE_GUIDE.md) — Data consumer
