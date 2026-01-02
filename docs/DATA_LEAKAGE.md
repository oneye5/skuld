# Data Leakage Prevention Guide

> **Navigation:** [Main README](../README.md) | [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) | [Testing](TESTING.md) | [Features](FEATURES.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Prevention Strategies](#prevention-strategies)
3. [Validation Framework](#validation-framework)
4. [Testing for Leakage](#testing-for-leakage)
5. [Warning Signs](#warning-signs)
6. [Related Documentation](#related-documentation)

---

## Overview

Data leakage occurs when information from the future (test set) bleeds into training. This guide documents the pipeline's leakage prevention strategies.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          LEAKAGE TYPES                                      │
└────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ TEMPORAL LEAKAGE │    │ FEATURE LEAKAGE  │    │ SCALER LEAKAGE   │
│                  │    │                  │    │                  │
│ Future data used │    │ Features encode  │    │ Normalizing with │
│ to predict past  │    │ future returns   │    │ future statistics│
│                  │    │                  │    │                  │
│ Example:         │    │ Example:         │    │ Example:         │
│ Forward fill     │    │ Using target as  │    │ fit() on train+  │
│ from future      │    │ a feature        │    │ test combined    │
└──────────────────┘    └──────────────────┘    └──────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ CROSS-SECTIONAL  │    │ CLUSTER LEAKAGE  │    │ TARGET LEAKAGE   │
│ LEAKAGE          │    │                  │    │                  │
│                  │    │ Clusters include │    │ Forward returns  │
│ Ranks computed   │    │ future behavior  │    │ computed wrong   │
│ across train+test│    │                  │    │                  │
│                  │    │ Example:         │    │ Example:         │
│ Example:         │    │ Clustering on    │    │ Using future     │
│ percentile(all)  │    │ full dataset     │    │ prices for past  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## Prevention Strategies

### 1. Temporal Ordering

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STRICT TEMPORAL ORDERING                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Timeline: ─────────────────────────────────────────────────────►       │
│                                                                          │
│            │◄─────── TRAIN ───────►│◄─── TEST ────►│                   │
│            │                        │               │                   │
│            t_start            train_end_ts    test_end_ts              │
│                                                                          │
│  Rules:                                                                  │
│  1. train.timestamp.max() < test.timestamp.min()                        │
│  2. No overlap between train and test timestamps                        │
│  3. Forward returns computed with future price data excluded            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# core/splitter.py
def split_by_timestamp(df, train_end_ts, test_end_ts):
    train = df[df[TIMESTAMP] < train_end_ts]
    test = df[(df[TIMESTAMP] >= train_end_ts) & (df[TIMESTAMP] < test_end_ts)]
    return train, test

# Validation
def validate_no_lookahead(train_df, test_df):
    train_max = train_df[TIMESTAMP].max()
    test_min = test_df[TIMESTAMP].min()
    if train_max >= test_min:
        raise ValidationError("Temporal overlap detected!")
```

### 2. Forward Fill Safety

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FORWARD FILL ORDERING                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WRONG:                                                                  │
│  Data: [(t=3, val=30), (t=1, val=NaN), (t=2, val=20)]                  │
│  Forward fill → t=1 gets val=30 (FUTURE VALUE!)                        │
│                                                                          │
│  CORRECT:                                                               │
│  1. Sort by [TICKER, TIMESTAMP]                                         │
│  2. Then forward fill                                                   │
│  Data after sort: [(t=1, val=NaN), (t=2, val=20), (t=3, val=30)]       │
│  Forward fill → t=1 gets val=0 (no prior value)                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# core/preprocessor.py
def preprocess_data(df):
    # Sort FIRST
    df = df.sort_values([TICKER, TIMESTAMP])
    
    # Then forward fill
    for col in numeric_cols:
        df[col] = df.groupby(TICKER)[col].ffill()
        df[col] = df[col].fillna(0)  # No prior value
    
    return df
```

### 3. Scaler Isolation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SCALER FIT ISOLATION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WRONG:                                                                  │
│  scaler.fit(all_data)          # Sees test statistics                  │
│  train_scaled = scaler.transform(train)                                │
│  test_scaled = scaler.transform(test)                                  │
│                                                                          │
│  CORRECT:                                                               │
│  scaler.fit(train_only)        # Only train statistics                 │
│  train_scaled = scaler.transform(train)                                │
│  test_scaled = scaler.transform(test)   # Using train params           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# core/scaler.py
def fit_scaler(df: pd.DataFrame) -> ScalerSet:
    """Fit on training data ONLY."""
    scaler = RobustScaler()
    scaler.fit(df[continuous_cols])
    return ScalerSet(scaler, continuous_cols, ...)

def transform_data(df: pd.DataFrame, scaler_set: ScalerSet):
    """Transform using pre-fitted scaler."""
    result = df.copy()
    result[cols] = scaler_set.scaler.transform(df[cols])
    return result

# In pipeline:
scaler_set = fit_scaler(train_df)  # Train only!
train_scaled = transform_data(train_df, scaler_set)
test_scaled = transform_data(test_df, scaler_set)
```

### 4. Cross-Sectional Feature Safety

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CROSS-SECTIONAL RANKING                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WRONG (Global ranking):                                                │
│  df['Rank_RSI'] = df.groupby(TIMESTAMP)['RSI'].rank(pct=True)          │
│  # If df contains train+test, test data affects train ranks!           │
│                                                                          │
│  CORRECT (Per-split ranking):                                           │
│  1. Split train/test FIRST                                              │
│  2. Rank train independently                                            │
│  3. Rank test independently                                             │
│                                                                          │
│  train = add_cross_sectional_features(train)                           │
│  test = add_cross_sectional_features(test)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# features/cross_sectional.py
def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-timestamp rank features.
    
    IMPORTANT: Call AFTER train/test split, on each set separately.
    """
    result = df.copy()
    for col in features_to_rank:
        # Rank within each timestamp (not across all data)
        result[f'Rank_{col}'] = result.groupby(TIMESTAMP)[col].rank(pct=True)
    return result

# In pipeline:
train = add_cross_sectional_features(train)  # Separate
test = add_cross_sectional_features(test)    # Separate
```

### 5. Cluster Safety

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLUSTER COMPUTATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  WRONG:                                                                  │
│  cluster_map = compute_clusters(all_data)                              │
│  # Clusters "know" how stocks behave in future!                        │
│                                                                          │
│  CORRECT:                                                               │
│  cluster_map = compute_clusters(train_data)   # Train only             │
│  train = add_cluster_features(train, cluster_map)                      │
│  test = add_cluster_features(test, cluster_map)  # Same map            │
│                                                                          │
│  Why: Cluster membership is based only on historical behavior          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# In pipeline/ranking_pipeline.py
# After train/test split
cluster_map = compute_clusters_fast(train_df, n_clusters=10)

# Apply same cluster map to both
train_df = add_cluster_features_fast(train_df, cluster_map)
test_df = add_cluster_features_fast(test_df, cluster_map)
```

### 6. Forward Return Computation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FORWARD RETURN SAFETY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each row at time t:                                                │
│  forward_return = (P_{t+365} - P_t) / P_t                              │
│                                                                          │
│  The forward return IS future information (it's the target).           │
│  But it must be computed correctly:                                     │
│                                                                          │
│  1. Target is computed using price data that exists                    │
│  2. If future price doesn't exist → NaN (drop row)                     │
│  3. Features must NOT include forward_return                           │
│                                                                          │
│  In test set:                                                           │
│  - We need forward_return for evaluation                               │
│  - But model predictions must NOT see it                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# core/target_builder.py
def compute_forward_returns(df, lookahead_days=365):
    """Compute forward returns as target variable."""
    result = df.copy()
    
    # For each row, look up price at t+lookahead
    # If no price exists → NaN
    result['forward_return'] = compute_return(
        current_prices=df['Close'],
        future_prices=get_future_prices(df, lookahead_days)
    )
    
    return result

# Feature exclusion
def get_feature_columns(df):
    """Get columns for model training."""
    excluded = {TIMESTAMP, TICKER, 'forward_return', ...}
    return [c for c in df.columns if c not in excluded]
```

## Validation Framework

### Decorators

```python
from core.validation import validate_dataframe, validate_no_nan

@validate_dataframe(required_cols=[TIMESTAMP, TICKER])
def process(df):
    ...

@validate_no_nan(columns=['Close', 'Volume'])
def compute_features(df):
    ...
```

### Runtime Checks

```python
from core.validation import validate_no_lookahead

def run_pipeline():
    train, test = split(df)
    
    # Validate no temporal overlap
    validate_no_lookahead(train, test)  # Raises if violated
    
    ...
```

## Testing for Leakage

### Test Categories

| Test File | Purpose |
|-----------|---------|
| `test_comprehensive_leakage.py` | All leakage types |
| `test_cluster_leakage.py` | Cluster-specific |
| `test_leakage_investigation.py` | Deep analysis |

### Key Tests

```python
def test_forward_fill_respects_temporal_order():
    """Forward fill must not propagate future values."""
    df = pd.DataFrame({
        TIMESTAMP: [3000, 1000, 2000],  # Out of order!
        TICKER: ["A", "A", "A"],
        "feature": [30.0, np.nan, 20.0],
    })
    
    result = preprocess_data(df)
    
    # t=1000 should be 0.0 (no prior), NOT 20.0 or 30.0
    assert result[result[TIMESTAMP] == 1000]["feature"].values[0] == 0.0

def test_scaler_not_fit_on_test():
    """Scaler must not see test data."""
    ...

def test_clusters_use_only_train_data():
    """Clusters must be computed on training data only."""
    ...
```

## Warning Signs

### Suspiciously Good Results

| Metric | Normal | Suspect Leakage |
|--------|--------|-----------------|
| IC | 0.03 - 0.10 | > 0.15 |
| Sharpe | 0.5 - 2.0 | > 3.0 |
| Hit Rate | 52% - 60% | > 70% |

### Debug Steps

1. **Check temporal ordering:** `assert train.max_ts < test.min_ts`
2. **Verify scaler isolation:** Print scaler fit statistics
3. **Audit features:** Remove suspicious features one by one
4. **Test with random data:** IC should be ~0 with random features

## Related Documentation

- [Testing Guide](TESTING.md) — Test implementation
- [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) — Pipeline architecture
- [Features Guide](FEATURES.md) — Feature engineering
- `core/validation.py` — Validation implementation
- `tests/test_comprehensive_leakage.py` — Leakage tests
