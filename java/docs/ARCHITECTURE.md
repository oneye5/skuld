# Java Data Ingestion Architecture

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
                         │ • data: List        │
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
    
    public final Set<DataSourceBase> sources = new HashSet<>();
    public final List<DataPoint> data = Collections.synchronizedList(new ArrayList<>());
    
    public void fetchDataFromSources() {
        data.clear();
        sources.parallelStream().forEach(source -> {
            var dataPoints = source.getDataPoints();
            String sourceName = source.getSourceName();
            dataPoints.stream()
                .filter(dp -> dp.getValue() != null)
                .forEach(dp -> {
                    dp.setSource(sourceName);
                    data.add(dp);
                });
        });
    }
}
```

> **Note on self-registration:** `DataSourceBase`'s constructor registers `this` into `IngestManager.INSTANCE.sources`. This means every `new SomeSource()` call in `Main.java` automatically wires the source into the manager with no separate registration step, which keeps `Main.java` compact. The trade-off is that construction has a side effect: an unconstructed or partially-constructed source can be added if a subclass constructor throws. Keep source constructors trivial (no I/O) to avoid this.

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
- [Data Analysis](../../docs/DATA_ANALYSIS.md) — Long-CSV schema and contents
