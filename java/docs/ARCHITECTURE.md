# Java Data Ingestion Architecture

> **Navigation:** [Main README](../../README.md) | [Data Sources](DATA_SOURCES.md) | [Python Pipeline](../../docs/RANKING_PIPELINE_GUIDE.md)

---

## Overview

The Java module fetches and consolidates financial data from multiple sources into a unified long-format CSV for the Python ML pipeline.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      JAVA INGESTION ARCHITECTURE                            │
└────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────┐
                         │     Main.java       │
                         │  (Entry Point)      │
                         └──────────┬──────────┘
                                    │
                                    │ Registers data sources
                                    ▼
                         ┌─────────────────────┐
                         │   IngestManager     │
                         │   (Singleton)       │
                         │                     │
                         │ • sources: Set      │
                         │ • data: Set         │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │  YfPrices    │ │  YfFinances  │ │  NzGdp       │
          │              │ │              │ │              │
          │ Yahoo OHLCV  │ │ Yahoo Fin.   │ │ Stats NZ     │
          └──────────────┘ └──────────────┘ └──────────────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    │
                                    │ Parallel fetch
                                    ▼
                         ┌─────────────────────┐
                         │ Set<DataPoint>      │
                         │                     │
                         │ (timestamp, ticker, │
                         │  feature, value)    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   CsvLongParser     │
                         │                     │
                         │ saveCsv(path)       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  data_long.csv      │
                         │                     │
                         │ timestamp,ticker,   │
                         │ feature,value       │
                         └─────────────────────┘
```

## Core Components

### DataPoint

The fundamental data structure:

```java
public class DataPoint {
    LocalDateTime timestamp;
    String ticker;
    String featureName;
    Double value;
}
```

### DataSourceBase

Abstract base class for all data sources:

```java
public abstract class DataSourceBase {
    public abstract Set<DataPoint> getDataPoints();
    
    public DataSourceBase() {
        IngestManager.INSTANCE.sources.add(this);
    }
}
```

### IngestManager

Singleton that orchestrates data collection:

```java
public enum IngestManager {
    INSTANCE;
    
    public Set<DataSourceBase> sources = new HashSet<>();
    public Set<DataPoint> data = new HashSet<>();
    
    public void fetchDataFromSources() {
        data.clear();
        sources.parallelStream().forEach(source -> {
            var dataPoints = source.getDataPoints();
            dataPoints = dataPoints.stream()
                .filter(dp -> dp.getValue() != null)
                .collect(Collectors.toSet());
            this.data.addAll(dataPoints);
        });
    }
}
```

### CsvLongParser

Outputs data to CSV:

```java
public class CsvLongParser {
    public static void saveCsv(String path) {
        // Sort by timestamp, ticker, feature
        // Write: timestamp,ticker,feature,value
    }
}
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. REGISTRATION                                                        │
│     ─────────────                                                       │
│     new YfPrices();     // Registers with IngestManager                 │
│     new NzGdp();        // Each source self-registers                   │
│     ...                                                                  │
│                                                                          │
│  2. PARALLEL FETCH                                                      │
│     ──────────────                                                      │
│     IngestManager.fetchDataFromSources()                                │
│     → sources.parallelStream() → source.getDataPoints()                 │
│                                                                          │
│  3. DATA POINTS                                                         │
│     ────────────                                                        │
│     {                                                                    │
│       timestamp: 2024-01-15 00:00:00,                                   │
│       ticker: "AIR.NZ",                                                 │
│       feature: "Close",                                                 │
│       value: 0.65                                                       │
│     }                                                                    │
│                                                                          │
│  4. CSV OUTPUT                                                          │
│     ──────────                                                          │
│     timestamp,ticker,feature,value                                      │
│     1705276800000,AIR.NZ,Close,0.65                                     │
│     1705276800000,AIR.NZ,Volume,1234567                                 │
│     ...                                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
java/
├── pom.xml                           # Maven config
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   └── DATA_SOURCES.md              # Source documentation
└── src/main/java/lazic/
    ├── Main.java                    # Entry point
    ├── sources/
    │   ├── config/
    │   │   └── Tickers.java         # Ticker configuration
    │   ├── YfPrices.java            # Yahoo Finance prices
    │   ├── YfFinances.java          # Yahoo Finance financials
    │   ├── NzGdp.java               # NZ GDP data
    │   ├── NzBusinessConfidence.java
    │   ├── NzRatesFx.java           # NZ rates & FX
    │   ├── NzVehicleRegistrations.java
    │   ├── NzLaborStats.java
    │   ├── NzRoadFatalities.java
    │   ├── NzBalanceOfPayments.java
    │   ├── NzTaxRevenue.java
    │   ├── NzPensions.java
    │   ├── NzLaborTaxation.java
    │   └── GlobalAquacultureProduction.java
    └── utils/
        ├── ingest/
        │   ├── DataPoint.java       # Data structure
        │   ├── DataSourceBase.java  # Base class
        │   ├── IngestManager.java   # Orchestrator
        │   ├── CsvLongParser.java   # CSV output
        │   └── WebHtmlGetter.java   # HTTP client
        └── db/
            └── ...                   # Database utilities
```

## Building and Running

### Prerequisites

- Java 17+
- Maven 3.6+

### Build

```bash
cd java
mvn clean compile
```

### Run

```bash
mvn exec:java -Dexec.mainClass="lazic.Main"
```

Or from IDE: Run `Main.java`

### Output

Data is written to `data/data_long.csv` (relative to project root).

## Adding a New Data Source

### Step 1: Create Source Class

```java
package lazic.sources;

import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import java.util.*;

public class MyNewSource extends DataSourceBase {
    
    @Override
    public Set<DataPoint> getDataPoints() {
        Set<DataPoint> dataPoints = new HashSet<>();
        
        // Fetch data from API/file
        String rawData = WebHtmlGetter.get(url);
        
        // Parse and create DataPoints
        for (var item : parseData(rawData)) {
            dataPoints.add(new DataPoint(
                item.getDate(),
                item.getTicker(),  // or "MACRO_" + featureName for macro data
                item.getFeature(),
                item.getValue()
            ));
        }
        
        return dataPoints;
    }
}
```

### Step 2: Register in Main.java

```java
public static void main(String[] args) {
    // Existing sources
    new YfPrices();
    new NzGdp();
    
    // Add your source
    new MyNewSource();
    
    // Run ingestion
    IngestManager.INSTANCE.fetchDataFromSources();
    CsvLongParser.saveCsv(outputPath);
}
```

## Error Handling

- **Null values:** Filtered out by IngestManager
- **Failed fetches:** Logged to stderr, source skipped
- **Invalid data:** Source-specific validation

## Dependencies

```xml
<!-- pom.xml -->
<dependencies>
    <dependency>
        <groupId>com.google.code.gson</groupId>
        <artifactId>gson</artifactId>
        <version>2.11.0</version>
    </dependency>
</dependencies>
```

## Related Documentation

- [Data Sources](DATA_SOURCES.md) — Detailed source configuration
- [Main README](../../README.md) — Project overview
- [Python Pipeline](../../docs/RANKING_PIPELINE_GUIDE.md) — Data consumer
