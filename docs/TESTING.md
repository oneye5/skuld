# Testing Documentation

> **Navigation:** [Main README](../README.md) | [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) | [Features](FEATURES.md) | [Data Leakage](DATA_LEAKAGE.md)

---

## Overview

The pipeline includes **300+ comprehensive tests** covering data leakage prevention, feature correctness, pipeline integration, and metric calculations.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           TEST ARCHITECTURE                                 │
└────────────────────────────────────────────────────────────────────────────┘

                         ┌────────────────────────┐
                         │     TEST CATEGORIES    │
                         └────────────┬───────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
       ▼                              ▼                              ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│   LEAKAGE    │            │  CORRECTNESS │            │ INTEGRATION  │
│   TESTS      │            │    TESTS     │            │   TESTS      │
│              │            │              │            │              │
│ • Temporal   │            │ • Features   │            │ • Pipeline   │
│ • Feature    │            │ • Metrics    │            │ • Consistency│
│ • Scaler     │            │ • Returns    │            │ • End-to-End │
│ • Cluster    │            │ • Portfolio  │            │              │
└──────────────┘            └──────────────┘            └──────────────┘
      │                           │                           │
      └───────────────────────────┴───────────────────────────┘
                                      │
                                      ▼
                              300+ Total Tests
```

## Running Tests

### All Tests

```bash
cd python/ml-pipeline
uv run pytest                         # Full suite (~2-3 minutes)
uv run pytest -q                      # Quiet mode
uv run pytest -x                      # Stop on first failure
uv run pytest -v                      # Verbose output
```

### Specific Categories

```bash
# Leakage tests only
uv run pytest tests/test_comprehensive_leakage.py tests/test_cluster_leakage.py

# Feature tests
uv run pytest tests/test_alpha_factors.py tests/test_technical.py

# Integration tests
uv run pytest tests/test_pipeline_integration.py
```

### Single Test

```bash
uv run pytest tests/test_scaler.py::test_fit_transform_separation -v
```

### Test Coverage

```bash
uv run pytest --cov=. --cov-report=html
# Open htmlcov/index.html
```

## Test Categories

### 1. Leakage Tests

**Purpose:** Ensure no future information bleeds into historical data.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LEAKAGE TEST SCENARIOS                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  TEMPORAL LEAKAGE                                                       │
│  ────────────────                                                       │
│  Test: Forward fill doesn't propagate future values to past            │
│  │ t=1000: NaN                                                          │
│  │ t=2000: 20.0                                                         │
│  │ t=3000: 30.0                                                         │
│  After forward fill, t=1000 should be 0.0, NOT 20.0 or 30.0            │
│                                                                          │
│  SCALER LEAKAGE                                                         │
│  ──────────────                                                         │
│  Test: Scaler fit only on training data                                │
│  ┌──────────┐  ┌──────────┐                                            │
│  │  Train   │  │   Test   │                                            │
│  │ fit(X)   │  │ NOT fit  │                                            │
│  │transform │  │transform │                                            │
│  └──────────┘  └──────────┘                                            │
│                                                                          │
│  CROSS-SECTIONAL LEAKAGE                                                │
│  ───────────────────────                                                │
│  Test: Ranks computed per-timestamp, not across train+test             │
│  Train ranks use only train data at each timestamp                     │
│  Test ranks use only test data at each timestamp                       │
│                                                                          │
│  CLUSTER LEAKAGE                                                        │
│  ──────────────                                                         │
│  Test: Clusters computed on training data only                         │
│  Different cutoff dates should produce different clusters              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Test Files:**
- `test_comprehensive_leakage.py` — Comprehensive leakage tests
- `test_cluster_leakage.py` — Cluster-specific leakage tests
- `test_leakage_investigation.py` — Deep leakage analysis

### 2. Feature Tests

**Purpose:** Verify feature calculations are mathematically correct.

| Test File | Coverage |
|-----------|----------|
| `test_alpha_factors.py` | Reversal, momentum quality, idiosyncratic vol |
| `test_technical.py` | RSI, MACD, ATR, Bollinger Bands |
| `test_cross_sectional.py` | Per-timestamp ranking |
| `test_cluster_fast.py` | Clustering correctness |
| `test_ratios.py` | Financial ratios |

**Example Test:**

```python
def test_rsi_bounds(self):
    """RSI should always be between 0 and 100."""
    df = generate_test_data()
    result = add_technical_features(df)
    
    rsi = result['RSI_14'].dropna()
    assert rsi.min() >= 0
    assert rsi.max() <= 100
```

### 3. Metric Tests

**Purpose:** Ensure evaluation metrics are calculated correctly.

| Test File | Coverage |
|-----------|----------|
| `test_ranking_metrics.py` | IC, ICIR, hit rate, quintiles |
| `test_metric_correctness.py` | Edge cases, NaN handling |
| `test_portfolio.py` | Backtest, Sharpe, drawdown |
| `test_annual_statistics.py` | Annual return statistics |

**Example Test:**

```python
def test_ic_with_perfect_correlation(self):
    """IC should be 1.0 when predictions perfectly match returns."""
    predictions = pd.Series([1, 2, 3, 4, 5])
    returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    
    ic = compute_ic(predictions, returns)
    assert abs(ic - 1.0) < 0.001
```

### 4. Integration Tests

**Purpose:** Verify end-to-end pipeline behavior.

| Test File | Coverage |
|-----------|----------|
| `test_pipeline_integration.py` | Full pipeline with synthetic data |
| `test_pipeline_consistency.py` | Determinism, reproducibility |
| `test_prediction_pipeline.py` | Prediction-only mode |

**Synthetic Data Tests:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SYNTHETIC DATA SCENARIOS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PERFECT SIGNAL                                                         │
│  ──────────────                                                         │
│  Feature = Future Return                                                │
│  Expected: IC ≈ 1.0, Sharpe >> 0                                       │
│                                                                          │
│  RANDOM SIGNAL                                                          │
│  ─────────────                                                          │
│  Feature = Random noise                                                 │
│  Expected: IC ≈ 0.0, Sharpe ≈ 0                                        │
│                                                                          │
│  INVERSE SIGNAL                                                         │
│  ──────────────                                                         │
│  Feature = -Future Return                                               │
│  Expected: IC ≈ -1.0, Sharpe < 0                                       │
│                                                                          │
│  MODERATE SIGNAL                                                        │
│  ───────────────                                                        │
│  Feature = 0.3 × Future Return + 0.7 × Noise                           │
│  Expected: IC ≈ 0.3, Sharpe > 0                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5. Component Tests

**Purpose:** Test individual components in isolation.

| Test File | Coverage |
|-----------|----------|
| `test_scaler.py` | RobustScaler wrapper |
| `test_splitter.py` | Train/test splitting |
| `test_target_builder.py` | Forward return computation |
| `test_long_to_wide.py` | Data format conversion |
| `test_validation.py` | Input validation |
| `test_anomaly_detection.py` | Price anomaly filtering |

## Test Patterns

### Fixture Pattern

```python
@pytest.fixture
def realistic_wide_df():
    """Create realistic test data."""
    np.random.seed(42)
    # ... generate data
    return df

def test_something(realistic_wide_df):
    result = function_under_test(realistic_wide_df)
    assert ...
```

### Parametrized Tests

```python
@pytest.mark.parametrize("window,expected_min", [
    (14, 0),
    (20, 0),
    (50, 0),
])
def test_rsi_windows(window, expected_min):
    ...
```

### Edge Case Testing

```python
def test_empty_dataframe():
    """Handle empty input gracefully."""
    result = function(pd.DataFrame())
    assert result.empty

def test_single_ticker():
    """Handle single ticker without error."""
    ...

def test_all_nan_column():
    """Handle all-NaN columns."""
    ...
```

## Test Data Generation

### Synthetic Data Helper

```python
def generate_synthetic_wide_data(
    n_tickers: int = 30,
    n_days: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic stock data for testing."""
    np.random.seed(seed)
    
    tickers = [f"STOCK_{i:02d}" for i in range(n_tickers)]
    timestamps = [start_ts + i * MS_PER_DAY for i in range(n_days)]
    
    rows = []
    for ticker in tickers:
        log_returns = np.random.normal(0.0002, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(log_returns))
        # ... add OHLCV
    
    return pd.DataFrame(rows)
```

### Predictive Feature Generator

```python
def add_predictive_feature(
    df: pd.DataFrame,
    correlation: float = 1.0,
    forward_days: int = 365,
) -> pd.DataFrame:
    """Add feature with known correlation to future returns."""
    # Compute actual forward returns
    forward_returns = compute_forward_returns(df, forward_days)
    
    # Add noise based on desired correlation
    signal = correlation * forward_returns
    noise = (1 - correlation) * np.random.randn(len(df))
    
    df['predictive_feature'] = signal + noise
    return df
```

## Continuous Integration

### Pre-commit Checks

Before committing:

```bash
# Run fast tests
uv run pytest tests/ -x -q --tb=line

# Check leakage tests specifically
uv run pytest tests/test_comprehensive_leakage.py -v
```

### Full Suite

```bash
# Run all tests with coverage
uv run pytest tests/ --cov=. --cov-report=term-missing
```

## Adding New Tests

### Guidelines

1. **Test one thing** per test function
2. **Use descriptive names** that explain what's being tested
3. **Include docstrings** explaining the test purpose
4. **Use fixtures** for common setup
5. **Test edge cases** (empty, single item, NaN, etc.)

### Example Test

```python
class TestMyNewFeature:
    """Tests for my_new_feature function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ['A', 'A', 'A'],
            CLOSE: [100, 110, 105],
        })
    
    def test_basic_computation(self, sample_data):
        """Test basic feature computation."""
        result = my_new_feature(sample_data)
        
        assert 'MyFeature' in result.columns
        assert not result['MyFeature'].isna().all()
    
    def test_no_lookahead(self, sample_data):
        """Ensure feature doesn't use future data."""
        result = my_new_feature(sample_data)
        
        # First row should only use data up to that point
        first_row = result.iloc[0]
        # ... verify no future information used
```

## Related Documentation

- [Data Leakage Guide](DATA_LEAKAGE.md) — Detailed leakage prevention
- [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) — Pipeline architecture
- [Features Guide](FEATURES.md) — Feature engineering
- `tests/conftest.py` — Shared fixtures
