# Clustering Methodology

> **Navigation:** [Main README](../README.md) | [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) | [Features](FEATURES.md) | [Testing](TESTING.md)

---

## Overview

The pipeline implements **statistical sector clustering** for NZX stocks since official sector classifications are sparse. Clusters are based on return correlation patterns and risk/return characteristics.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        CLUSTERING ARCHITECTURE                              │
└────────────────────────────────────────────────────────────────────────────┘

                    Per Rolling Window (Leakage-Safe)
                    ─────────────────────────────────
                                   │
                    ┌──────────────┴──────────────┐
                    │     TRAINING DATA ONLY      │
                    │                             │
                    │  ┌───────────────────────┐  │
                    │  │ Filter to .NZ tickers │  │
                    │  └───────────┬───────────┘  │
                    │              │              │
                    │  ┌───────────▼───────────┐  │
                    │  │ Compute Daily Returns │  │
                    │  │ (clip at ±100%)       │  │
                    │  └───────────┬───────────┘  │
                    │              │              │
                    │  ┌───────────▼───────────┐  │
                    │  │ Extract Features:     │  │
                    │  │ • Volatility          │  │
                    │  │ • Mean Return         │  │
                    │  │ • Skewness            │  │
                    │  │ • Kurtosis            │  │
                    │  │ • Autocorrelation     │  │
                    │  │ • Positive Day %      │  │
                    │  │ • Vol of Vol          │  │
                    │  └───────────┬───────────┘  │
                    │              │              │
                    │  ┌───────────▼───────────┐  │
                    │  │ Standardize Features  │  │
                    │  └───────────┬───────────┘  │
                    │              │              │
                    │  ┌───────────▼───────────┐  │
                    │  │ K-Means Clustering    │  │
                    │  │ (n_clusters=10)       │  │
                    │  └───────────┬───────────┘  │
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ cluster_map: {ticker: id}   │
                    │                             │
                    │ STOCK01.NZ → Cluster 0      │
                    │ STOCK02.NZ → Cluster 3      │
                    │ STOCK03.NZ → Cluster 0      │
                    │ ...                         │
                    └─────────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ CLUSTER-RELATIVE FEATURES   │
                    │                             │
                    │ For each stock at time t:   │
                    │ • Cluster_Return_Rank       │
                    │ • Cluster_Vol_Rank          │
                    │ • Cluster_Momentum_Rank     │
                    │                             │
                    │ Compare stock to its        │
                    │ cluster peers               │
                    └─────────────────────────────┘
```

## Why Clustering?

### Problem: Sparse Sector Data

NZX stocks often lack detailed sector classifications. Clustering provides:
- **Statistical sectors** based on actual behavior
- **Peer comparison** for relative features
- **Regime detection** (stocks moving together)

### Clustering vs Industry

```
┌─────────────────────────────────────────────────────────────────────────┐
│              INDUSTRY SECTOR ≠ STATISTICAL CLUSTER                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Industry (GICS):           Statistical Cluster:                        │
│  ─────────────────          ────────────────────                        │
│  Based on business          Based on price behavior                     │
│  activities                                                              │
│                                                                          │
│  Example:                   Example:                                    │
│  MFT (Manufacturing)        MFT and MOV have                            │
│  MOV (Movies)               -0.04 correlation!                          │
│  Same "Consumer" sector     Different clusters                          │
│                                                                          │
│  → Industry != Statistical behavior                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Core Function: `compute_clusters_fast`

```python
from features.cluster_fast import compute_clusters_fast

cluster_map = compute_clusters_fast(
    wide_df,
    n_clusters=10,          # Number of clusters
    lookback_days=500,      # Historical data to use
    max_daily_return=1.0,   # Clip extreme returns
    min_obs=100,            # Minimum observations per ticker
)
# Returns: {"STOCK.NZ": 0, "ABC.NZ": 3, ...}
```

### Feature Extraction

For each ticker, compute:

| Feature | Formula | Purpose |
|---------|---------|---------|
| Volatility | `std(returns) × √252` | Risk level |
| Mean Return | `mean(returns) × 252` | Drift |
| Skewness | `skew(returns)` | Asymmetry |
| Kurtosis | `kurtosis(returns)` | Tail risk |
| Autocorrelation | `corr(r_t, r_{t-1})` | Mean reversion |
| Positive Days % | `mean(returns > 0)` | Win rate |
| Vol of Vol | `std(rolling_vol)` | Regime switching |

### Clustering Algorithm

**K-Means** is used for balanced clusters:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Cluster
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
```

Why K-Means over Hierarchical?
- **More balanced** clusters (hierarchical tends to create one giant cluster)
- **Faster** for repeated computation per window
- **Deterministic** with fixed random seed

## Leakage Prevention

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LEAKAGE-SAFE CLUSTER WORKFLOW                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WRONG (Leakage):                                                       │
│  ─────────────────                                                      │
│  1. Compute clusters on ALL data                                        │
│  2. Split train/test                                                    │
│  3. Use clusters for features                                           │
│  → Clusters "know" future behavior!                                     │
│                                                                          │
│  CORRECT (Our approach):                                                │
│  ───────────────────────                                                │
│  1. Split train/test FIRST                                              │
│  2. Compute clusters on TRAIN data only                                 │
│  3. Apply same cluster_map to test data                                 │
│  → Clusters only use historical information                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Per-Window Recomputation

Clusters are recomputed for each rolling window:

```python
# In run_single_ranking_window():
train_df = wide_slice[wide_slice[TIMESTAMP] < train_end_ts]
cluster_map = compute_clusters_fast(train_df, n_clusters=10)

# Apply to both train and test
train_df = add_cluster_features_fast(train_df, cluster_map)
test_df = add_cluster_features_fast(test_df, cluster_map)
```

## Cluster Features

### Features Added

| Feature | Description |
|---------|-------------|
| `cluster_id` | Cluster assignment (integer 0-9) |
| `Cluster_Return_Rank` | Stock's return rank within cluster |
| `Cluster_Vol_Rank` | Stock's volatility rank within cluster |
| `Cluster_Momentum_Rank` | Stock's momentum rank within cluster |

### Usage in Model

Cluster features help the model:
1. **Normalize** for market conditions
2. **Identify** relative outperformers within peer groups
3. **Capture** sector rotation effects

## Configuration

### Optimal Settings (NZX)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_clusters` | 10-12 | ~139 NZX stocks, ~10-14 per cluster |
| `lookback_days` | 500 | ~2 years of history |
| `min_obs` | 100 | Ensure statistical significance |
| `max_daily_return` | 1.0 | Filter data errors (100% moves) |

### Cluster Size Distribution

Typical cluster sizes after K-Means:

```
Cluster 0: 18 tickers  ████████████████████
Cluster 1: 15 tickers  █████████████████
Cluster 2: 12 tickers  ██████████████
Cluster 3: 14 tickers  ████████████████
...
```

## Validation

### Benchmarking Results

```
┌────────────────────────────────────────────────────────────────────┐
│  METHOD              │ TIME    │ MAX CLUSTER │ KNOWN PAIRS FOUND │
├──────────────────────┼─────────┼─────────────┼───────────────────┤
│  Pure Correlation    │ 0.84ms  │ 90%         │ 5/7               │
│  Hierarchical        │ 1.2ms   │ 85%         │ 4/7               │
│  K-Means (balanced)  │ 1.5ms   │ 20%         │ 4/7               │
└──────────────────────┴─────────┴─────────────┴───────────────────┘

Selected: K-Means for cluster balance despite slightly fewer "known pairs"
```

### Testing

```bash
# Run cluster-specific tests
cd python/ml-pipeline
uv run pytest tests/test_cluster_fast.py tests/test_cluster_leakage.py -v
```

Test coverage:
- Cluster assignment determinism
- Leakage prevention (different cutoffs → different clusters)
- Edge cases (empty data, single ticker)
- Feature computation correctness

## Related Documentation

- [Features Guide](FEATURES.md) — All feature engineering
- [Data Leakage Guide](DATA_LEAKAGE.md) — Leakage prevention
- [Testing Guide](TESTING.md) — Test coverage
- `features/cluster_fast.py` — Implementation
- `tests/test_cluster_leakage.py` — Leakage tests
