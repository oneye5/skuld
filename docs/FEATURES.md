# Feature Engineering Reference

> **Navigation:** [Main README](../README.md) | [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) | [Features](FEATURES.md) | [Clustering](CLUSTERING.md) | [Testing](TESTING.md) | [Annual Statistics](ANNUAL_STATISTICS.md) | [Data Leakage](DATA_LEAKAGE.md) | [TODO](TODO.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Categories](#feature-categories)
3. [Leakage Prevention](#leakage-prevention)
4. [Feature Selection](#feature-selection)
5. [Performance Optimization](#performance-optimization)
6. [Adding New Features](#adding-new-features)
7. [Related Documentation](#related-documentation)

---

## Overview

The pipeline computes features from raw OHLCV data using research-backed methodologies. Features are organized into categories and computed per-ticker to avoid cross-asset leakage.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING FLOW                             │
└────────────────────────────────────────────────────────────────────────────┘

Raw OHLCV Data
      │
      ├─────────────────────┬─────────────────────┬─────────────────────┐
      │                     │                     │                     │
      ▼                     ▼                     ▼                     ▼
┌───────────┐        ┌───────────┐        ┌───────────┐        ┌───────────┐
│ TECHNICAL │        │  ALPHA    │        │  RATIOS   │        │ CLUSTER   │
│           │        │  FACTORS  │        │           │        │ FEATURES  │
│ • RSI     │        │ • Reversal│        │ • P/E     │        │ • Rel Rank│
│ • MACD    │        │ • Momentum│        │ • Volume  │        │ • Peer    │
│ • ATR     │        │   Quality │        │   Ratios  │        │   Compare │
│ • Vol     │        │ • IdioVol │        │           │        │           │
│ • MA Dist │        │ • InfoDisc│        │           │        │           │
└───────────┘        └───────────┘        └───────────┘        └───────────┘
      │                     │                     │                     │
      └─────────────────────┴─────────────────────┴─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ CROSS-SECTIONAL     │
                         │ (per timestamp)     │
                         │                     │
                         │ Rank_RSI, Rank_Vol  │
                         │ Rank_ROC, etc.      │
                         └─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Final Feature Set   │
                         │ (~80-100 features)  │
                         └─────────────────────┘
```

## Feature Categories

### 1. Technical Features (`features/technical.py`)

Standard momentum, volatility, and trend indicators.

| Feature | Formula | Window | Description |
|---------|---------|--------|-------------|
| `RSI_14` | Relative Strength Index | 14 days | Overbought/oversold indicator |
| `MACD_Line` | EMA(12) - EMA(26) | 12/26 days | Momentum direction |
| `MACD_Signal` | EMA(MACD, 9) | 9 days | Signal line |
| `MACD_Hist` | MACD - Signal | - | Momentum acceleration |
| `ROC_10` | `(P_t - P_{t-10}) / P_{t-10}` | 10 days | Short-term momentum |
| `ROC_252` | `(P_t - P_{t-252}) / P_{t-252}` | 252 days | Annual momentum |
| `ATR_14` | Average True Range | 14 days | Volatility measure |
| `NATR_14` | ATR / Close | 14 days | Normalized volatility |
| `Vol_20` | `std(returns) × √252` | 20 days | Short-term volatility |
| `Vol_252` | `std(returns) × √252` | 252 days | Annual volatility |
| `BB_Width_20` | `(Upper - Lower) / MA` | 20 days | Bollinger Band width |
| `Dist_MA_20` | `(Close - MA) / MA` | 20 days | Distance from MA |
| `Dist_MA_200` | `(Close - MA) / MA` | 200 days | Long-term trend |
| `Pos_52w_Range` | `(Close - Low) / (High - Low)` | 252 days | Position in range |
| `Lag_Return_*` | `r_{t-n}` | 1,2,3,5 days | Lagged returns |

### 2. Alpha Factors (`features/alpha_factors.py`)

Research-backed factors from academic finance literature.

#### Short-term Reversal (Jegadeesh 1990)

```
┌───────────────────────────────────────────────────────────────┐
│  REVERSAL EFFECT                                              │
│                                                               │
│  Stock drops sharply     →   Tends to bounce back            │
│  Stock rises sharply     →   Tends to pull back              │
│                                                               │
│  Features:                                                    │
│  • Rev_5d:  5-day return (classic reversal)                  │
│  • Rev_10d: 10-day return (extended reversal)                │
│  • Rev_5d_Skip1: 5-day return skipping most recent day       │
└───────────────────────────────────────────────────────────────┘
```

#### Momentum Quality

High R² trend = consistent, reliable momentum.

| Feature | Formula | Description |
|---------|---------|-------------|
| `Trend_RSq_20` | R² of price trend | 20-day trend quality |
| `Trend_RSq_60` | R² of price trend | 60-day trend quality |
| `QualMom_60` | Momentum × R² | Quality-adjusted momentum |

#### Idiosyncratic Volatility (Ang et al. 2006)

```
┌───────────────────────────────────────────────────────────────┐
│  IDIOSYNCRATIC VOLATILITY                                     │
│                                                               │
│  Total Vol = Market Vol + Idiosyncratic Vol                  │
│                                                               │
│  Low idiosyncratic vol stocks → tend to outperform           │
│  (volatility anomaly)                                         │
│                                                               │
│  Features:                                                    │
│  • IdioVol_20:  20-day residual volatility                   │
│  • IdioVol_60:  60-day residual volatility                   │
└───────────────────────────────────────────────────────────────┘
```

#### Information Discreteness (Da, Gurun, Warachka 2014)

```
Continuous momentum (many small moves)  → More persistent
Discrete momentum (few large moves)     → Less persistent

InfoDisc = sign(sum(returns)) × (# up days - # down days) / total days
```

| Feature | Window | Description |
|---------|--------|-------------|
| `InfoDisc_21` | 21 days | Monthly discreteness |
| `InfoDisc_63` | 63 days | Quarterly discreteness |

#### Maximum Returns (Bali et al. 2011)

Lottery-like stocks (extreme recent returns) tend to underperform.

| Feature | Description |
|---------|-------------|
| `MAX_21d` | Maximum single-day return in 21 days |
| `MaxMinSpread_21d` | Max return - Min return |

#### Higher Moments

| Feature | Description |
|---------|-------------|
| `Skew_60d` | Return skewness (asymmetry) |
| `Kurt_60d` | Return kurtosis (tail risk) |
| `DownVol_60d` | Downside volatility only |

#### Volume Features

| Feature | Description |
|---------|-------------|
| `RelVol_20d` | Current volume / 20-day average |
| `Amihud_21d` | |Return| / Volume (illiquidity) |

#### Momentum Acceleration

| Feature | Description |
|---------|-------------|
| `MomAccel_21_63` | Short-term mom - Long-term mom |
| `Near52wHigh` | Distance from 52-week high |

### 3. Cross-Sectional Features (`features/cross_sectional.py`)

Rank features computed **per timestamp** after train/test split.

```
┌───────────────────────────────────────────────────────────────┐
│  CROSS-SECTIONAL RANKING                                      │
│                                                               │
│  At timestamp T, rank all stocks:                            │
│                                                               │
│  Stock │ RSI_14 │ Rank_RSI_14                                │
│  ──────┼────────┼────────────                                │
│  A     │ 75     │ 0.90  (top 10%)                            │
│  B     │ 45     │ 0.45  (middle)                             │
│  C     │ 30     │ 0.15  (bottom 15%)                         │
│                                                               │
│  Ranks are percentiles: 0.0 to 1.0                           │
└───────────────────────────────────────────────────────────────┘
```

Features ranked:
- `Rank_RSI_14`, `Rank_ROC_252`, `Rank_Vol_252`
- `Rank_Rev_5d`, `Rank_Rev_10d`
- `Rank_IdioVol_20`, `Rank_IdioVol_60`
- `Rank_InfoDisc_21`, `Rank_MAX_21d`
- And more...

### 4. Cluster Features (`features/cluster_fast.py`)

See [Clustering Documentation](CLUSTERING.md) for details.

| Feature | Description |
|---------|-------------|
| `cluster_id` | Cluster assignment (0-9) |
| `Cluster_Return_Rank` | Rank within cluster |
| `Cluster_Vol_Rank` | Volatility rank within cluster |

## Leakage Prevention

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE COMPUTATION ORDER                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. TICKER-LEVEL FEATURES (before split)                                │
│     ─────────────────────────────────────                               │
│     RSI, MACD, ROC, Vol, Alpha factors                                  │
│     → Computed per-ticker using only that ticker's history              │
│     → Safe because they only use past data for each row                 │
│                                                                          │
│  2. TRAIN/TEST SPLIT                                                    │
│     ───────────────────                                                 │
│     Strict temporal boundary: test timestamps > train timestamps        │
│                                                                          │
│  3. CROSS-SECTIONAL FEATURES (after split, per-timestamp)              │
│     ─────────────────────────────────────────────────────              │
│     Rank_RSI, Rank_Vol, etc.                                            │
│     → Computed separately for train and test                            │
│     → Train ranks use only train data                                   │
│     → Test ranks use only test data                                     │
│                                                                          │
│  4. CLUSTER FEATURES (after split, train → test)                       │
│     ───────────────────────────────────────────                         │
│     Clusters computed on train only                                     │
│     Same cluster_map applied to test                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Feature Selection

### Excluded Columns

Raw price and event columns are excluded from model features:

```python
excluded = {
    TIMESTAMP, TICKER, TARGET, FORWARD_RETURN, CLOSE,
    'Open', 'High', 'Low', 'Volume',
    'AdjClose',   # Raw price level
    'Dividend',   # Point-in-time event
    'Split',      # Point-in-time event
}
```

### Feature Count

Typical feature count after engineering:

| Category | Count | Examples |
|----------|-------|----------|
| Technical | ~25 | RSI, MACD, ATR, Vol |
| Alpha Factors | ~20 | Reversal, IdioVol, InfoDisc |
| Cross-Sectional | ~25 | Rank_* versions |
| Cluster | ~5 | cluster_id, Cluster_*_Rank |
| **Total** | **~75-100** | |

## Performance Optimization

### Vectorized Implementation

```python
# Vectorized groupby (fast)
df['RSI_14'] = df.groupby(TICKER)[CLOSE].transform(
    lambda x: compute_rsi(x, 14)
)

# vs. Loop-based (slow)
for ticker in df[TICKER].unique():
    mask = df[TICKER] == ticker
    df.loc[mask, 'RSI_14'] = compute_rsi(df.loc[mask, CLOSE], 14)
```

### Caching

Feature computation is cached:

```python
from core.data_cache import load_cached_wide_data_with_features

# First call: computes features (~30s)
df = load_cached_wide_data_with_features()

# Subsequent calls: loads from cache (~2s)
df = load_cached_wide_data_with_features()
```

## Adding New Features

### Step-by-Step

1. **Add computation** in appropriate module (`technical.py`, `alpha_factors.py`, etc.)
2. **Add to cross-sectional** if needed in `cross_sectional.py`
3. **Write tests** in corresponding test file
4. **Clear cache** to regenerate features

### Example: Adding a New Alpha Factor

```python
# In features/alpha_factors.py

def _add_my_new_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Add my new research-backed factor.
    
    Reference: Paper (Year) - "Title"
    """
    if CLOSE not in df.columns:
        return df
    
    # Compute factor
    df['MyFactor_20'] = df[CLOSE].pct_change(20) ** 2
    
    return df

# Add to main function
def add_alpha_factors(df):
    ...
    ticker_df = _add_my_new_factor(ticker_df)
    ...
```

## Related Documentation

- [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) — How features are used
- [Clustering](CLUSTERING.md) — Cluster feature details
- [Data Leakage](DATA_LEAKAGE.md) — Leakage prevention
- [Testing](TESTING.md) — Feature tests
