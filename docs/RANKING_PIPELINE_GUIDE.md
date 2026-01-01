# Ranking Pipeline User Guide

> **Navigation:** [Main README](../README.md) | [Annual Statistics](ANNUAL_STATISTICS.md) | [Features](FEATURES.md) | [Testing](TESTING.md) | [Clustering](CLUSTERING.md)

---

**Project**: Skuld - Time Series Forecasting Framework  
**Pipeline**: Ranking-Based Cross-Sectional Stock Prediction  
**Last Updated**: January 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Running the Pipeline](#running-the-pipeline)
5. [Configuration](#configuration)
6. [Understanding Results](#understanding-results)
7. [Data Validation](#data-validation)
8. [Experiment Tracking](#experiment-tracking)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Overview

The ranking pipeline uses **Learning-to-Rank (LTR)** with LightGBM's `LGBMRanker` to predict relative stock performance. Instead of predicting absolute returns, the model learns to **rank** stocks by expected forward returns at each timestamp.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Forward Returns** | Target: 365-day simple returns `(P_{t+365} - P_t) / P_t` |
| **Cross-Sectional Ranking** | At each timestamp, rank all stocks by predicted score |
| **IC (Information Coefficient)** | Correlation between predicted ranks and actual returns |
| **Long-Short Portfolio** | Go long top-N, short bottom-N ranked stocks |

### Why Ranking?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION VS RANKING                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CLASSIFICATION (Binary):         RANKING (Continuous):                  │
│  ┌──────────┐                     ┌──────────────────────┐              │
│  │ Stock A  │ → Up/Down           │ Stock A: Score 0.85  │ → Rank 1    │
│  │ Stock B  │ → Up/Down           │ Stock B: Score 0.72  │ → Rank 2    │
│  │ Stock C  │ → Up/Down           │ Stock C: Score 0.45  │ → Rank 3    │
│  └──────────┘                     │ Stock D: Score 0.31  │ → Rank 4    │
│                                   └──────────────────────┘              │
│  Problem: Ignores magnitude        Advantage: Captures relative         │
│  of returns                        performance across all stocks        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

```bash
cd python/ml-pipeline
uv sync  # Install dependencies
```

### Run Full Pipeline (~5-15 minutes)

```bash
uv run python scripts/run_model_evaluation.py
```

### Debug Script (~30 seconds)

For fast iteration during development:

```bash
uv run python scripts/debug_ranking_quick.py
```

---

## Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE PIPELINE FLOW                             │
└────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │ data_long.csv   │
                              │ (Java output)   │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA PREPARATION                                                 │
│                                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │ Long→Wide   │───▶│ Clean &     │───▶│ Anomaly    │                   │
│  │ Conversion  │    │ Classify    │    │ Detection  │                   │
│  └─────────────┘    │ Tickers     │    │ & Filter   │                   │
│                     └─────────────┘    └─────────────┘                   │
│                                                                           │
│  • Pivot: (timestamp, ticker, feature) → (timestamp, ticker, f1, f2...)  │
│  • MACRO_ prefix for macroeconomic indicators                            │
│  • Detect price jumps (>200%) indicating splits/errors                   │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: FEATURE ENGINEERING                                              │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                      FEATURE CATEGORIES                          │     │
│  ├─────────────────────────────────────────────────────────────────┤     │
│  │                                                                   │     │
│  │  TECHNICAL (per ticker):     RATIOS:          ALPHA FACTORS:     │     │
│  │  • RSI, MACD, ROC            • P/E ratios     • Reversal (5d)    │     │
│  │  • Bollinger Bands           • Volume ratios  • Momentum Quality │     │
│  │  • ATR, Volatility           • Price ratios   • Idio Volatility  │     │
│  │  • 52-week High/Low                           • Information Disc │     │
│  │  • Moving Averages                            • Max Returns      │     │
│  │                                               • Skew/Kurtosis    │     │
│  │                                                                   │     │
│  │  CROSS-SECTIONAL (per timestamp, AFTER train/test split):        │     │
│  │  • Rank_RSI, Rank_ROC, Rank_Vol, etc.                           │     │
│  │  • Percentile position within timestamp peers                    │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: TARGET COMPUTATION                                               │
│                                                                           │
│  Forward Return = (Price_{t+365} - Price_t) / Price_t                    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ Options:                                                         │     │
│  │ • return_type: "simple" or "log"                                │     │
│  │ • winsorize_limits: (-0.5, 0.5) clips to [-50%, +50%]          │     │
│  │ • tolerance_days: allows slight date mismatch in lookups        │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: ROLLING WINDOW EVALUATION                                        │
│                                                                           │
│   Window 1        Window 2        Window 3            Window N           │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐        ┌─────────┐         │
│  │ Train   │     │ Train   │     │ Train   │  ...   │ Train   │         │
│  │ ─────── │     │ ─────── │     │ ─────── │        │ ─────── │         │
│  │ Test    │     │ Test    │     │ Test    │        │ Test    │         │
│  └─────────┘     └─────────┘     └─────────┘        └─────────┘         │
│       │               │               │                  │               │
│       └───────────────┴───────────────┴──────────────────┘               │
│                               │                                          │
│                               ▼                                          │
│                    ┌─────────────────────┐                              │
│                    │ Aggregate Predictions│                              │
│                    │ from all windows     │                              │
│                    └─────────────────────┘                              │
│                                                                           │
│  Per Window:                                                             │
│  1. Split train/test by timestamp (strict temporal ordering)            │
│  2. Fit scaler on train only                                            │
│  3. Compute clusters on train only (if enabled)                         │
│  4. Add cross-sectional features (per timestamp)                        │
│  5. Train LGBMRanker                                                     │
│  6. Predict on test set                                                  │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: EVALUATION                                                       │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ RANKING METRICS  │  │ PORTFOLIO METRICS│  │ STABILITY METRICS│       │
│  │                  │  │                  │  │                  │       │
│  │ • Mean IC        │  │ • Sharpe Ratio   │  │ • IC Stability   │       │
│  │ • Rank IC        │  │ • Total Return   │  │ • Quintile       │       │
│  │ • ICIR           │  │ • Max Drawdown   │  │   Monotonicity   │       │
│  │ • Hit Rate       │  │ • Turnover       │  │ • Annual Stats   │       │
│  │ • Quintile Spread│  │ • Calmar Ratio   │  │                  │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         ┌──────────────────────┐
                         │ OUTPUT               │
                         │ • metrics.json       │
                         │ • predictions.csv    │
                         │ • quintile_returns   │
                         │ • plots/             │
                         └──────────────────────┘
```

---

## Running the Pipeline

### Basic Usage

```bash
cd python/ml-pipeline
uv run python scripts/run_model_evaluation.py
```

### Command-Line Options

```bash
# Change forward return horizon
uv run python scripts/run_model_evaluation.py --forward-days 20

# Adjust portfolio size
uv run python scripts/run_model_evaluation.py --top-n 5 --bottom-n 5

# Reduce windows for faster testing
uv run python scripts/run_model_evaluation.py --num-windows 2

# Long-only strategy
uv run python scripts/run_model_evaluation.py --bottom-n 0

# Higher transaction costs
uv run python scripts/run_model_evaluation.py --cost-bps 20
```

### Full Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-windows` | 20 | Number of rolling windows |
| `--window-movement` | 1.0 | Years between window starts |
| `--test-period` | 0.1 | Test period length in years |
| `--forward-days` | 365 | Forward return horizon |
| `--return-type` | simple | `simple` or `log` |
| `--no-winsorize` | False | Disable return clipping |
| `--top-n` | 10 | Long portfolio size |
| `--bottom-n` | 10 | Short portfolio size |
| `--cost-bps` | 10 | Transaction cost (basis points) |
| `--min-stocks` | 10 | Min stocks per timestamp |
| `--no-save` | False | Don't save results |

### Output Location

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

## Configuration

### Key Settings (`config/settings.py`)

```python
# =============================================================================
# TARGET DEFINITION
# =============================================================================
FORWARD_RETURN_DAYS = 365           # Prediction horizon
RETURN_TYPE = "simple"              # "simple" or "log"
WINSORIZE_LIMITS = (-0.5, 0.5)     # Clip extreme returns
RETURN_PRICE_COLUMN = "AdjClose"    # Use adjusted close for total return

# =============================================================================
# ROLLING WINDOWS
# =============================================================================
NUM_ROLLING_WINDOWS = 20            # Number of evaluation windows
ROLLING_WINDOW_MOVEMENT_YEARS = 1   # Years between windows
TEST_PERIOD_YEARS = 0.1             # Test period length

# =============================================================================
# LIGHTGBM RANKER
# =============================================================================
RANKER_N_ESTIMATORS = 150           # Boosting iterations
RANKER_NUM_LEAVES = 127             # Tree complexity
RANKER_LEARNING_RATE = 0.05         # Shrinkage
RANKER_DEVICE = "gpu"               # "cpu" or "gpu"

# =============================================================================
# PORTFOLIO
# =============================================================================
PORTFOLIO_TOP_N = 10                # Long positions
PORTFOLIO_BOTTOM_N = 10             # Short positions
TRANSACTION_COST_BPS = 10.0         # 10 bps = 0.1%
```

### Configuration Trade-offs

| Setting | Lower Value | Higher Value |
|---------|-------------|--------------|
| `FORWARD_RETURN_DAYS` | More signal, higher turnover | Less signal, lower turnover |
| `RANKER_N_ESTIMATORS` | Faster, underfit risk | Slower, overfit risk |
| `RANKER_NUM_LEAVES` | Simpler model, more stable | Complex model, more capacity |
| `PORTFOLIO_TOP_N` | Concentrated, higher risk | Diversified, lower alpha |

---

## Understanding Results

### Key Metrics

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         METRIC INTERPRETATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  IC (Information Coefficient)                                           │
│  ────────────────────────────                                           │
│  Correlation between predictions and actual returns                     │
│                                                                          │
│  │ -0.10 │  0.00 │  0.03 │  0.10 │  0.15 │                             │
│  │   ▼   │   ▼   │   ▼   │   ▼   │   ▼   │                             │
│  │ Bad   │ Noise │ Good  │ Great │ Check │                             │
│  │       │       │       │       │ for   │                             │
│  │       │       │       │       │ leak  │                             │
│                                                                          │
│  ICIR (Information Coefficient Information Ratio)                       │
│  ─────────────────────────────────────────────────                      │
│  IC consistency: mean(IC) / std(IC) × sqrt(periods_per_year)           │
│                                                                          │
│  │ 0.00  │  0.50 │  1.00 │  2.00 │  4.00 │                             │
│  │   ▼   │   ▼   │   ▼   │   ▼   │   ▼   │                             │
│  │ Noise │ Weak  │ Good  │ Strong│ Check │                             │
│                                                                          │
│  Quintile Spread                                                        │
│  ───────────────                                                        │
│  Return difference: Q5 (top) - Q1 (bottom)                              │
│  Should be POSITIVE and MONOTONIC                                       │
│                                                                          │
│    Q1      Q2      Q3      Q4      Q5                                   │
│    ─2%     0%     +1%     +3%     +5%    ← Monotonic (good)            │
│    ─2%    +4%     ─1%     +3%     +5%    ← Non-monotonic (suspicious)  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reading Output

```
=== RANKING METRICS ===
Mean IC:      0.0326    ← Positive = model has predictive power
Std IC:       0.0564
ICIR:         4.21      ← Annualized, stable IC
Mean Rank IC: 0.0312    ← Spearman (robust to outliers)
Hit Rate:     0.5789    ← 58% of top picks are positive
Quintile Spread: 0.0245 ← 2.45% long-short spread

=== QUINTILE RETURNS ===
Q1 (bottom): -0.0123   ← Worst predicted
Q2:          -0.0045
Q3:           0.0012
Q4:           0.0078
Q5 (top):     0.0122   ← Best predicted ✓ Monotonic

=== BACKTEST RESULTS ===
Sharpe Ratio:   0.68   ← Decent risk-adjusted return
Total Return:   5.65%
Max Drawdown:  -8.2%
Avg Turnover:   45%
```

### Warning Signs

| Issue | Symptom | Likely Cause |
|-------|---------|--------------|
| **Data Leakage** | IC > 0.15, Sharpe > 3.0 | Forward-looking features |
| **Model Not Learning** | IC < 0, non-monotonic quintiles | Wrong features, too little data |
| **Overfitting** | High train IC, low test IC | Too many trees, not regularized |

---

## Data Validation

The pipeline includes comprehensive validation. See [Data Leakage Guide](DATA_LEAKAGE.md) for details.

### Automatic Checks

```python
from core.validation import validate_wide_data, validate_no_lookahead

# Validate DataFrame structure
issues = validate_wide_data(df, raise_on_error=False)

# Ensure no lookahead bias
validate_no_lookahead(train_df, test_df)  # Raises if violated
```

### Validation Decorators

```python
from core.validation import validate_dataframe, validate_no_nan

@validate_dataframe(required_cols=['timestamp', 'ticker'], min_rows=10)
def process_data(df):
    ...

@validate_no_nan(columns=['Close', 'Volume'])
def compute_features(df):
    ...
```

---

## Experiment Tracking

Track experiments for reproducibility via `core/experiment_tracking.py`.

```python
from core.experiment_tracking import create_experiment_manifest, compute_data_hash

manifest = create_experiment_manifest(
    config={"forward_days": 365, "top_n": 10},
    feature_columns=feature_cols,
    metrics={"mean_ic": 0.05, "sharpe": 1.2},
    data_hash=compute_data_hash(df),
    notes="Baseline with alpha factors",
)
manifest.save(output_dir)
```

### Compare Experiments

```python
from core.experiment_tracking import compare_experiments, find_best_experiment

diff = compare_experiments(manifest_a, manifest_b)
print(f"Config changes: {diff['config_differences']}")

best = find_best_experiment(Path("output/runs"), metric="sharpe_ratio")
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `sum(group) != len(X)` | Group sizes mismatch | Rebuild groups after filtering |
| `No features available` | All columns dropped | Check sparsity threshold |
| `Not enough data after gap` | Short time range | Reduce `forward_return_days` |

### Performance Issues

**Pipeline Too Slow:**
```bash
# Reduce windows
uv run python scripts/run_model_evaluation.py --num-windows 2

# Or use debug script
uv run python scripts/debug_ranking_quick.py
```

**Memory Issues:**
```python
# Use float32
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
```

---

## API Reference

### Core Classes

```python
# Ranker
from learner.ranking import LightGBMRankerWrapper, RankerConfig

config = RankerConfig(n_estimators=100)
ranker = LightGBMRankerWrapper(config)
ranker.fit(X_train, y_train, group=train_groups)
scores = ranker.predict(X_test)

# Metrics
from evaluation.ranking_metrics import RankingMetrics

metrics = RankingMetrics.from_predictions(predictions_df)
print(f"IC: {metrics.mean_ic:.4f}")

# Portfolio
from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig

config = PortfolioConfig(top_n=10, bottom_n=10)
backtest = run_portfolio_backtest(predictions_df, config)
print(f"Sharpe: {backtest.sharpe_ratio:.2f}")
```

### Utility Functions

```python
# Build groups for LGBMRanker
from learner.ranking import build_group_from_timestamps
groups = build_group_from_timestamps(df, timestamp_col="timestamp")

# Compute forward returns
from core.target_builder import compute_forward_returns
df = compute_forward_returns(df, lookahead_days=365)
```

---

## Related Documentation

- [Main README](../README.md) — Project overview
- [Annual Statistics](ANNUAL_STATISTICS.md) — Performance metrics
- [Clustering](CLUSTERING.md) — Statistical sector clustering
- [Features](FEATURES.md) — Feature engineering reference
- [Testing](TESTING.md) — Test coverage
- [Data Leakage](DATA_LEAKAGE.md) — Leakage prevention
