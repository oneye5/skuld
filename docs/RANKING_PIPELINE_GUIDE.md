# Ranking Pipeline User Guide

**Project**: Skuld - Time Series Forecasting Framework  
**Pipeline**: Ranking-Based Cross-Sectional Stock Prediction  
**Last Updated**: 2025-12-29

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Running the Evaluation Pipeline](#running-the-evaluation-pipeline)
4. [Running Predictions on Future Data](#running-predictions-on-future-data)
5. [Configuration Options](#configuration-options)
6. [Understanding Results](#understanding-results)
7. [Data Validation & Quality](#data-validation--quality)
8. [Experiment Tracking](#experiment-tracking)
9. [Logging](#logging)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

---

## Overview

The ranking pipeline uses **Learning-to-Rank (LTR)** with LightGBM's `LGBMRanker` to predict relative stock performance. Instead of predicting absolute returns or binary labels, the model learns to rank stocks by their expected forward returns at each timestamp.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Forward Returns** | Target variable: 365-day simple returns `(P_{t+365} - P_t) / P_t` |
| **Cross-Sectional Ranking** | At each timestamp, rank all stocks by predicted score |
| **IC (Information Coefficient)** | Correlation between predicted ranks and actual returns |
| **Long-Short Portfolio** | Go long top-N, short bottom-N ranked stocks |

### Pipeline Flow

```
Raw Data (data_long.csv)
    ↓
Wide Format Conversion
    ↓
Data Validation (timestamps, duplicates, prices)
    ↓
Forward Return Calculation (365-day)
    ↓
Feature Engineering (technical, ratios, cross-sectional, alpha factors)
    ↓
Rolling Window Train/Test Split
    ↓
Lookahead Validation (ensure no data leakage)
    ↓
LGBMRanker Training & Prediction
    ↓
Evaluation (IC, ICIR, Quintile Returns)
    ↓
Portfolio Backtest (Sharpe, Drawdown)
    ↓
Experiment Manifest (reproducibility)
```

### Key Safeguards Against Data Leakage

The pipeline includes multiple safeguards to prevent data leakage:

1. **Temporal Validation**: `validate_no_lookahead()` ensures test data comes strictly after training data
2. **Scaler Isolation**: Scalers are fit on training data only, then applied to test data
3. **Cross-Sectional Features**: Computed per-timestamp after train/test split
4. **Forward Fill Safety**: Data is sorted by `[TICKER, TIMESTAMP]` before forward fill to prevent future→past leakage

---

## Quick Start

### Prerequisites

```bash
cd python/ml-pipeline
uv sync  # Install dependencies
```

### Run Debug Script (Quick Validation)

For fast iteration and code validation (~30 seconds):

```bash
uv run python scripts/debug_ranking_quick.py
```

### Run Full Pipeline

For complete evaluation with rolling windows (~5-15 minutes):

```bash
uv run python scripts/run_model_evaluation.py
```

---

## Running the Evaluation Pipeline

### Basic Usage

The main entry point is `run_model_evaluation.py`:

```bash
cd python/ml-pipeline
uv run python scripts/run_model_evaluation.py
```

This runs the full pipeline with default settings:
- 365-day forward returns
- 5 rolling windows
- Long top-10, short bottom-10 stocks
- 10 bps transaction costs

### Common Command-Line Options

```bash
# Change forward return horizon
uv run python scripts/run_model_evaluation.py --forward-days 20

# Adjust portfolio size
uv run python scripts/run_model_evaluation.py --top-n 5 --bottom-n 5

# Reduce windows for faster testing
uv run python scripts/run_model_evaluation.py --num-windows 2

# Long-only strategy (no shorting)
uv run python scripts/run_model_evaluation.py --bottom-n 0

# Higher transaction costs
uv run python scripts/run_model_evaluation.py --cost-bps 20

# Combine options
uv run python scripts/run_model_evaluation.py --forward-days 10 --top-n 5 --num-windows 3
```

### Full Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-windows` | 5 | Number of rolling windows |
| `--window-movement` | 0.5 | Years between window starts |
| `--test-period` | 0.5 | Test period length in years |
| `--forward-days` | 365 | Forward return horizon (days) |
| `--return-type` | simple | Return type: `simple` or `log` |
| `--no-winsorize` | False | Disable return winsorization |
| `--top-n` | 10 | Long portfolio size |
| `--bottom-n` | 10 | Short portfolio size |
| `--cost-bps` | 10 | Transaction cost (basis points) |
| `--min-stocks` | 10 | Min stocks per timestamp |
| `--no-save` | False | Don't save results to disk |

### Output Location

Results are saved to:
```
python/ml-pipeline/output/runs/ranking_YYYYMMDD_HHMMSS/
├── config.json              # Configuration used
├── metrics.json             # IC, ICIR, Sharpe, etc.
├── predictions.csv          # All predictions with actuals
├── quintile_returns.csv     # Returns by quintile
├── window_summaries.json    # Per-window statistics
└── plots/
    ├── quintile_returns.png
    ├── ic_series.png
    └── cumulative_returns.png
```

---

## Running Predictions on Future Data

### Option 1: Using the Debug Script as Template

For quick predictions on new data, modify `debug_ranking_quick.py`:

```python
"""predict_future.py - Generate predictions for new/future data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from core.preprocessor import preprocess_data
from core.scaler import fit_scaler, transform_data
from features.technical import add_technical_features
from features.cross_sectional import add_cross_sectional_features
from features.ratios import add_financial_ratios
from learner.ranking import LightGBMRankerWrapper, build_group_from_timestamps, RankerConfig
from config.columns import TIMESTAMP, TICKER

def predict_future(
    historical_data_path: str = None,
    future_data_path: str = None,  # Optional: separate file with new data
):
    """Generate rankings for the most recent timestamp(s).
    
    Args:
        historical_data_path: Path to historical data (for training)
        future_data_path: Path to new data (for prediction), or None to use latest in historical
    """
    # Load historical data for training
    print("Loading historical data...")
    long_df = load_long_data(historical_data_path)  # Uses default if None
    
    # Convert to wide format
    wide_df = long_to_wide(
        add_macro_prefix(clean_and_classify_tickers(long_df))
    )
    
    # Add features (no forward returns needed for prediction)
    wide_df = add_technical_features(wide_df)
    wide_df = add_financial_ratios(wide_df)
    wide_df = add_cross_sectional_features(wide_df)
    
    # Get feature columns
    excluded = {TIMESTAMP, TICKER, 'is_macro', 'Open', 'High', 'Low', 'Close', 'Volume'}
    feature_cols = [c for c in wide_df.columns 
                    if c not in excluded 
                    and pd.api.types.is_numeric_dtype(wide_df[c])]
    
    # Split: all but last timestamp for training, last timestamp for prediction
    timestamps = sorted(wide_df[TIMESTAMP].unique())
    train_ts = timestamps[:-1]  # All history
    predict_ts = timestamps[-1:]  # Latest timestamp
    
    train_df = wide_df[wide_df[TIMESTAMP].isin(train_ts)].copy()
    predict_df = wide_df[wide_df[TIMESTAMP].isin(predict_ts)].copy()
    
    # For training, we need forward returns (requires data from future)
    # IMPORTANT: Can only train on data where we have forward returns
    from core.target_builder import compute_forward_returns, FORWARD_RETURN
    train_df = compute_forward_returns(train_df, lookahead_days=5, drop_na=True)
    
    # Preprocess
    train_df = preprocess_data(train_df, add_missing_flags=False)
    predict_df = preprocess_data(predict_df, add_missing_flags=False)
    
    # Scale (fit on train)
    scaler = fit_scaler(train_df[feature_cols])
    train_df = transform_data(train_df, scaler)
    predict_df = transform_data(predict_df, scaler)
    
    # Train ranker
    print("Training model...")
    train_df = train_df.sort_values(TIMESTAMP)
    train_groups = build_group_from_timestamps(train_df, TIMESTAMP)
    
    ranker = LightGBMRankerWrapper(RankerConfig(n_estimators=100))
    ranker.fit(
        train_df[feature_cols], 
        train_df[FORWARD_RETURN], 
        train_groups
    )
    
    # Predict on latest timestamp
    print(f"Generating predictions for {len(predict_df)} stocks...")
    predictions = ranker.predict(predict_df[feature_cols])
    
    # Create output DataFrame
    result = pd.DataFrame({
        TIMESTAMP: predict_df[TIMESTAMP].values,
        TICKER: predict_df[TICKER].values,
        'predicted_score': predictions,
        'rank': pd.Series(predictions).rank(ascending=False).astype(int).values,
    })
    
    # Sort by predicted score (best first)
    result = result.sort_values('predicted_score', ascending=False)
    
    print("\n=== TOP 10 PREDICTED STOCKS ===")
    print(result.head(10).to_string(index=False))
    
    print("\n=== BOTTOM 10 PREDICTED STOCKS ===")
    print(result.tail(10).to_string(index=False))
    
    return result


if __name__ == "__main__":
    predictions = predict_future()
    predictions.to_csv("predictions_latest.csv", index=False)
    print("\nSaved to predictions_latest.csv")
```

### Option 2: Using Pipeline API Directly

```python
from pipeline.ranking_pipeline import (
    prepare_wide_data,
    add_all_features,
    get_feature_columns_for_ranking,
)
from learner.ranking import LightGBMRankerWrapper, RankerConfig
from core.data_loader import load_long_data

# Load and prepare data
long_df = load_long_data()
wide_df = prepare_wide_data(long_df)

# Add features
wide_df = add_all_features(wide_df)

# Get feature columns
feature_cols = get_feature_columns_for_ranking(wide_df)

# ... continue with training and prediction
```

### Option 3: Saving and Loading Models

To save a trained model for later use:

```python
import pickle
from pathlib import Path

# After training
model_path = Path("output/models/ranker_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump({
        'ranker': ranker,
        'scaler': scaler,
        'feature_cols': feature_cols,
    }, f)

# Loading later
with open(model_path, 'rb') as f:
    saved = pickle.load(f)
    
ranker = saved['ranker']
scaler = saved['scaler']
feature_cols = saved['feature_cols']

# Predict on new data
predictions = ranker.predict(new_data[feature_cols])
```

---

## Configuration Options

### Editing Default Configuration

Configuration lives in `config/ranking_config.py`:

```python
# Target settings
FORWARD_RETURN_DAYS = 365     # Change to 1, 10, 20 for different horizons
RETURN_TYPE = "simple"        # "simple" or "log"
WINSORIZE_LIMITS = (-0.5, 0.5)  # Clip extreme returns

# Model settings
RANKER_N_ESTIMATORS = 100     # More = slower but potentially better
RANKER_LEARNING_RATE = 0.05   # Lower = more regularization

# Portfolio settings
PORTFOLIO_TOP_N = 10          # Long positions
PORTFOLIO_BOTTOM_N = 10       # Short positions
TRANSACTION_COST_BPS = 10.0   # 10 bps = 0.1%
```

### Key Configuration Trade-offs

| Setting | Lower Value | Higher Value |
|---------|-------------|--------------|
| `FORWARD_RETURN_DAYS` | More signal, higher turnover | Less signal, lower turnover |
| `RANKER_N_ESTIMATORS` | Faster, underfit risk | Slower, overfit risk |
| `PORTFOLIO_TOP_N` | More concentrated, higher risk | More diversified, lower alpha |
| `TRANSACTION_COST_BPS` | Optimistic returns | Conservative returns |

---

## Understanding Results

### Key Metrics Explained

| Metric | What It Measures | Good Value | Warning Sign |
|--------|------------------|------------|--------------|
| **Mean IC** | Avg correlation between predictions and returns | > 0.03 | < 0 or > 0.15 |
| **ICIR** | Consistency of IC (annualized) | > 0.5 | < 0.2 |
| **Quintile Spread** | Q5 return - Q1 return | > 0 | < 0 |
| **Sharpe Ratio** | Risk-adjusted portfolio return | > 0.5 | < 0 or > 3.0 |
| **Hit Rate** | % of top picks with positive returns | > 55% | < 50% |

### Reading the Output

```
=== RANKING METRICS ===
Mean IC:      0.0326    ← Positive = model has some predictive power
Std IC:       0.0564
ICIR:         4.21      ← Annualized, stable IC
Mean Rank IC: 0.0312    ← Spearman correlation (robust)
Hit Rate:     0.5789    ← 58% of top picks are positive
Quintile Spread: 0.0245 ← Long-short spread of 2.45%

=== QUINTILE RETURNS ===
Q1 (bottom): -0.0123   ← Worst predicted stocks
Q2:          -0.0045
Q3:           0.0012
Q4:           0.0078
Q5 (top):     0.0122   ← Best predicted stocks ✓ Monotonic!

=== BACKTEST RESULTS ===
Sharpe Ratio:   0.68   ← Decent risk-adjusted return
Total Return:   5.65%
Max Drawdown:  -8.2%
Avg Turnover:   45%
```

### Warning Signs to Watch For

| Issue | Symptom | Likely Cause |
|-------|---------|--------------|
| **Data Leakage** | IC > 0.15, Sharpe > 3.0 | Forward-looking features, overlapping returns |
| **Model Not Learning** | IC < 0, non-monotonic quintiles | Insufficient data, wrong features |
| **Overfitting** | High train IC, low test IC | Too many trees, not enough data |
| **Wrong Annualization** | Sharpe seems too high | Mismatch between return horizon and annualization |

---

## Data Validation & Quality

The pipeline includes comprehensive validation utilities in `core/validation.py` to catch data issues early.

### Automatic Validation

The pipeline automatically validates:
- **No lookahead bias**: Test timestamps must come strictly after train timestamps
- **Group consistency**: LGBMRanker group sizes must match data size
- **Data quality**: Warnings for missing values, infinities, invalid prices

### Manual Validation

You can run validation checks manually:

```python
from core.validation import (
    validate_wide_data,
    validate_no_lookahead,
    check_data_quality_report,
)

# Validate wide format data
issues = validate_wide_data(df, raise_on_error=False)
if issues:
    print("Data issues found:", issues)

# Check for lookahead bias
validate_no_lookahead(train_df, test_df)  # Raises ValidationError if violated

# Generate quality report
report = check_data_quality_report(df)
print(f"Rows: {report['n_rows']}, Columns: {report['n_columns']}")
print(f"Columns with missing: {report['columns_with_missing']}")
print(f"Columns with infinities: {report['columns_with_infinities']}")
```

### Validation Decorators

Use decorators to validate function inputs:

```python
from core.validation import validate_dataframe, validate_no_nan

@validate_dataframe(required_cols=['timestamp', 'ticker'], min_rows=10)
def process_data(df):
    # df is guaranteed to have required columns and minimum rows
    ...

@validate_no_nan(columns=['Close', 'Volume'])
def compute_features(df):
    # df is guaranteed to have no NaN in specified columns
    ...
```

### Data Quality Checks

| Check | Description | Action on Failure |
|-------|-------------|-------------------|
| Missing columns | Required columns not present | Raises `ValidationError` |
| Negative timestamps | Timestamps < 0 | Reports issue |
| Invalid Close prices | Close <= 0 or NaN | Warns (may be macro data) |
| Duplicate pairs | Same (timestamp, ticker) | Reports issue |
| Unsorted timestamps | Timestamps not monotonic | Reports issue |

---

## Experiment Tracking

The pipeline supports experiment tracking for reproducibility via `core/experiment_tracking.py`.

### Experiment Manifest

Each experiment can generate a manifest containing:
- Git commit hash and branch
- Whether there were uncommitted changes
- Configuration parameters used
- Feature columns
- Performance metrics
- Data hash for verification

```python
from core.experiment_tracking import create_experiment_manifest, compute_data_hash

# Create manifest
manifest = create_experiment_manifest(
    config={
        "forward_days": 365,
        "top_n": 10,
        "num_windows": 5,
    },
    feature_columns=feature_cols,
    metrics={
        "mean_ic": 0.05,
        "sharpe": 1.2,
    },
    data_hash=compute_data_hash(df),
    notes="Baseline experiment with alpha factors",
)

# Save to output directory
manifest.save(output_dir)
```

### Comparing Experiments

```python
from core.experiment_tracking import (
    ExperimentManifest,
    compare_experiments,
    find_best_experiment,
)

# Load two experiments
manifest_a = ExperimentManifest.load(Path("output/runs/exp_a/manifest.json"))
manifest_b = ExperimentManifest.load(Path("output/runs/exp_b/manifest.json"))

# Compare them
diff = compare_experiments(manifest_a, manifest_b)
print(f"Same git commit: {diff['same_git_commit']}")
print(f"Config changes: {diff['config_differences']}")
print(f"Metric changes: {diff['metric_differences']}")

# Find best experiment by metric
best = find_best_experiment(
    Path("output/runs"),
    metric="sharpe_ratio",
    higher_is_better=True,
)
print(f"Best experiment: {best.experiment_id}")
```

---

## Logging

The pipeline uses structured logging via `core/logging_config.py`.

### Setting Up Logging

```python
from core.logging_config import setup_logging, get_logger

# Set up logging (call once at startup)
setup_logging(
    level=logging.INFO,
    log_file=Path("output/pipeline.log"),
    console=True,
)

# Get module-specific logger
logger = get_logger(__name__)
logger.info("Starting pipeline...")
```

### Timing Operations

```python
from core.logging_config import log_timing, timed

# Context manager for timing
with log_timing("feature engineering"):
    df = add_features(df)
# Logs: "Starting: feature engineering"
# Logs: "Completed: feature engineering (2.34s)"

# Decorator for timing functions
@timed
def slow_function():
    ...
# Logs execution time automatically
```

### Structured Logging Helpers

```python
from core.logging_config import log_dataframe_info, log_metrics

# Log DataFrame info
log_dataframe_info(df, "Training data")
# Logs: "Training data: 10,000 rows × 50 cols (12.3 MB)"

# Log metrics
log_metrics({
    "mean_ic": 0.05,
    "sharpe": 1.2,
}, prefix="Final")
# Logs formatted metrics with prefix
```

### Progress Logging

```python
from core.logging_config import ProgressLogger

progress = ProgressLogger(total=100, desc="Processing windows")
for i in range(100):
    process_window(i)
    progress.update()
progress.finish()
# Logs: "Processing windows: 10% (10/100) [1.2s elapsed, ETA 10.8s]"
# Logs: "Processing windows: Complete! 100 items in 12.0s"
```

---

## Troubleshooting

### Common Errors

#### "DataFrame must be sorted by timestamp for LGBMRanker"

The ranking model requires data sorted by timestamp (group ID).

**Solution:**
```python
df = df.sort_values(TIMESTAMP).reset_index(drop=True)
```

#### "sum(group) != len(X)"

Group sizes don't match data size.

**Solution:**
```python
# Rebuild groups after filtering
groups = build_group_from_timestamps(df, TIMESTAMP)
assert sum(groups) == len(df)
```

#### "No features available after preprocessing"

Features were dropped during preprocessing.

**Solution:**
```python
# Check which columns remain
print(f"Columns: {df.columns.tolist()}")
# Reduce sparsity threshold
wide_df = drop_sparse_columns(wide_df, threshold=0.98)  # More permissive
```

#### "Not enough data after applying gap"

Train/test split with gap leaves no test data.

**Solution:**
- Use more data (longer time range)
- Reduce `forward_return_days`
- Reduce train/test gap

### Performance Issues

#### Pipeline Too Slow

1. **Reduce data size** for debugging:
   ```python
   # Take recent 2 years only
   max_ts = df[TIMESTAMP].max()
   two_years_ms = 2 * 365 * 86400 * 1000
   df = df[df[TIMESTAMP] > (max_ts - two_years_ms)]
   ```

2. **Reduce model complexity**:
   ```bash
   uv run python scripts/run_model_evaluation.py --num-windows 2
   ```

3. **Reduce estimators**:
   ```python
   config = RankerConfig(n_estimators=50)  # Default is 100
   ```

#### Memory Issues

1. Use float32 instead of float64:
   ```python
   for col in df.columns:
       if df[col].dtype == 'float64':
           df[col] = df[col].astype('float32')
   ```

2. Process windows sequentially (default behavior)

3. Delete intermediate DataFrames:
   ```python
   del intermediate_df
   gc.collect()
   ```

---

## API Reference

### Core Classes

#### `LightGBMRankerWrapper`

```python
from learner.ranking import LightGBMRankerWrapper, RankerConfig

config = RankerConfig(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
)
ranker = LightGBMRankerWrapper(config)

# Fit (data MUST be sorted by timestamp)
ranker.fit(X_train, y_train, group=train_groups)

# Predict
scores = ranker.predict(X_test)
```

#### `RankingMetrics`

```python
from evaluation.ranking_metrics import RankingMetrics

metrics = RankingMetrics.from_predictions(
    predictions_df,            # Must have: timestamp, ticker, predicted_score, actual_return
    timestamp_col="timestamp",
    predicted_col="predicted_score",
    actual_col="actual_return",
    min_stocks=5,
)

print(metrics.summary())
print(f"IC: {metrics.mean_ic:.4f}")
print(f"ICIR: {metrics.icir:.4f}")
```

#### `PortfolioConfig` & `BacktestResult`

```python
from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig

config = PortfolioConfig(
    top_n=10,
    bottom_n=10,
    transaction_cost_bps=10.0,
)

backtest = run_portfolio_backtest(
    predictions_df,
    config,
    timestamp_col="timestamp",
    score_col="predicted_score",
    return_col="actual_return",
)

print(f"Sharpe: {backtest.sharpe_ratio:.2f}")
print(f"Total Return: {backtest.total_return:.2%}")
```

### Utility Functions

```python
from learner.ranking import build_group_from_timestamps

# Build group parameter for LGBMRanker
groups = build_group_from_timestamps(df, timestamp_col="timestamp")
# Returns: [10, 12, 11, ...] where each element is stocks-per-timestamp


from core.target_builder import compute_forward_returns

# Add forward returns to DataFrame
df = compute_forward_returns(
    df,
    lookahead_days=5,
    return_type="simple",
    winsorize_limits=(-0.5, 0.5),
    drop_na=True,
)
```

---

## See Also

- [config/settings.py](../python/ml-pipeline/config/settings.py) - All configuration options
- [core/validation.py](../python/ml-pipeline/core/validation.py) - Data validation utilities
- [core/experiment_tracking.py](../python/ml-pipeline/core/experiment_tracking.py) - Experiment tracking
- [core/logging_config.py](../python/ml-pipeline/core/logging_config.py) - Logging utilities
- [tests/](../python/ml-pipeline/tests/) - Unit tests for all components
