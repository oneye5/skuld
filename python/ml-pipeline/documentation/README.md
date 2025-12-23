# Skuld ML Pipeline

Binary classifier predicting whether NZX ticker prices will gain ≥X% within Y days.

## Quick Start

```bash
cd python/ml-pipeline
uv sync                    # Install dependencies
uv run main.py             # Run rolling window backtest
uv run pytest              # Run tests
```

## Configuration

All settings are in [config/model_config.py](../config/model_config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `LOOKAHEAD_DAYS` | 365 | Days ahead to predict |
| `GAIN_THRESHOLD_PCT` | 13.0 | % gain threshold for positive class |
| `PREDICTION_THRESHOLD` | 0.55 | Probability cutoff for buy signal |
| `NUM_ROLLING_WINDOWS` | 5 | Number of backtest windows |
| `TEST_PERIOD_YEARS` | 2.0 | Test set duration per window |

**Changing the model:** Edit `create_model()` in model_config.py. Any sklearn-compatible classifier with `fit()` and `predict_proba()` works.

---

## Data Format

Raw data in `skuld/data/data_long.csv` (long format):

| timestamp | ticker | feature | value |
|-----------|--------|---------|-------|
| 1609459200000 | ANZ.NZ | Close | 25.50 |
| 1609459200000 | | GDP_NZ | 320000 |

- **ticker**: Stock symbol (e.g., `ANZ.NZ`). Empty = macro data.
- **feature**: Data point name (`Close`, `Volume`, `GDP_NZ`, etc.)
- **timestamp**: Unix milliseconds

---

## Pipeline Flow

```
data_long.csv
     │
     ▼
┌─────────────────────────────────────────────────┐
│  1. add_macro_prefix()                          │  Prefix empty-ticker rows with MACRO_
│  2. long_to_wide()                              │  Pivot to wide format (anchored on Close)
│  3. add_technical_features()                    │  RSI, MACD, momentum, etc.
└─────────────────────────────────────────────────┘
     │
     ▼ (for each rolling window)
┌─────────────────────────────────────────────────┐
│  4. split_by_timestamp()                        │  Train/test split
│  5. create_labels()                             │  Binary target: gained ≥X% in Y days?
│  6. convert_prices_to_returns()                 │  Prevent "high price" leakage
│  7. impute_data()                               │  Fill NaN (fit on train only)
│  8. add_cyclical_time_features()                │  Day-of-year, day-of-week sin/cos
│  9. one_hot_encode_tickers()                    │  Ticker identity features
│ 10. fit_scalers() / transform_data()            │  StandardScaler (fit on train only)
│ 11. select_features()                           │  Drop low-variance features
│ 12. train_model()                               │  Fit model from create_model()
│ 13. predict()                                   │  Probability predictions
└─────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────┐
│ 14. Evaluation (combined across all windows)    │
│     - Classification metrics (precision, etc.)  │
│     - Trading simulation (Sharpe, returns)      │
│     - Baseline comparison (buy everything)      │
└─────────────────────────────────────────────────┘
```

---

## Directory Structure

```
python/ml-pipeline/
├── main.py                      # Entry point: runs rolling window backtest
├── config/
│   ├── model_config.py          # Model settings, create_model()
│   ├── column_names.py          # Column name constants
│   └── file_paths.py            # Output paths
├── data-preparation/            # Data transformation modules
│   ├── long-to-wide/            # converter.py
│   ├── data-splitting/train-test/ # splitter.py
│   ├── labeling/                # labeler.py
│   └── transformations/         # Feature engineering, scaling, etc.
├── learner/                     # Model training and prediction
│   ├── trainer.py               # train_model(), save/load
│   └── predictor.py             # predict()
├── evaluation/                  # Metrics and visualization
│   ├── model-evaluation/        # Classification metrics
│   └── trade-simulation/        # Trading sim
├── runnables/
│   ├── rolling_window_runner.py # Orchestrates backtest loop
│   └── single_window_pipeline.py # Runs one train/test cycle
├── tests/                       # Mirrors source structure
└── output/                      # Generated files (gitignored)
```

---

## Adding New Transformations

All transformation modules live in `data-preparation/transformations/`. Each module follows this pattern:

```python
# data-preparation/transformations/my_transform.py
import pandas as pd

def my_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transform description.
    
    Args:
        df: Wide-format DataFrame with TIMESTAMP, TICKER columns.
    
    Returns:
        DataFrame with new columns added.
    """
    result = df.copy()
    # Add your transformation
    result['new_feature'] = ...
    return result
```

**To integrate:**
1. Add the function to `transformations/__init__.py` exports
2. Import and call in `single_window_pipeline.py` at the appropriate step
3. Add tests in `tests/data-preparation/transformations/`

**Conventions:**
- Functions are pure: `DataFrame in → DataFrame out`
- No fitting on test data (fit stats on train, apply to both)
- Preserve TIMESTAMP and TICKER columns

---

## Leakage Prevention

Critical rules to avoid data leakage:

1. **Fit scalers/imputers on training data only**
   ```python
   stats = compute_imputation_stats(train_df)  # Fit on train
   train_imputed = impute_data(train_df, stats)
   test_imputed = impute_data(test_df, stats)   # Apply to test
   ```

2. **Labels use future data** — create labels before any transformations that depend on other rows

3. **Technical features** are computed per-ticker with their own history (safe before split)

4. **Macro prefix** identifies global vs ticker-specific scaling

---

## Output Files

After running `main.py`, outputs appear in `output/runs/<timestamp>/`:

```
output/runs/23-12-2025-1034/
├── config.json              # Config snapshot
├── predictions/             # Per-window predictions
├── evaluation/
│   ├── metrics.json         # Classification + trading metrics
│   ├── trades.csv           # All simulated trades
│   └── visualizations/      # Charts
└── scalers/                 # Fitted scalers for production
```

---

## Testing

```bash
uv run pytest                           # All tests
uv run pytest tests/data-preparation/ -v  # Specific module
uv run pytest -k labeler                # By keyword
```

Tests mirror source structure. Add tests before implementing new features (TDD).

---

## Key Modules Reference

| Module | Function | Purpose |
|--------|----------|---------|
| `converter.py` | `long_to_wide()` | Pivot long→wide, anchored on Close timestamp |
| `splitter.py` | `split_by_timestamp()` | Train/test split by time |
| `labeler.py` | `create_labels()` | Binary target from future price |
| `scaling.py` | `fit_scalers()`, `transform_data()` | StandardScaler, macro vs ticker |
| `imputation.py` | `compute_imputation_stats()`, `impute_data()` | Fill NaN |
| `trainer.py` | `train_model()` | Fit model from config |
| `predictor.py` | `predict()` | Probability predictions |
| `simulator.py` | `run_trading_simulation()` | Backtest trading strategy |

