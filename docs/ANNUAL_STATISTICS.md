# Annual Statistics for Real-World Implementation

## Overview

The ranking pipeline now includes **annual return distribution statistics** to help assess real-world implementation expectations and support Monte Carlo simulations.

## What's Included

When running the evaluation pipeline with `price_data` provided (for continuous daily returns), the system automatically computes:

### Return Distribution
- **Mean Annual Return**: Average annual return across all years
- **Median Annual Return**: Middle value of annual returns
- **Std Dev (Annual)**: Standard deviation of annual returns
- **Percentiles**: 5th, 25th, 75th, 95th percentiles
- **Range**: Best and worst year

### Win/Loss Profile
- **% Positive Years**: Percentage of years with positive returns
- **Avg Winning Year**: Average return in positive years
- **Avg Losing Year**: Average return in negative years

### Shape Statistics
- **Skewness**: Asymmetry of return distribution (positive = more upside)
- **Excess Kurtosis**: Tail heaviness (higher = more extreme outcomes)

### Risk-Adjusted
- **Avg Annual Sharpe**: Average Sharpe ratio computed per year

## How to Use

### 1. Run Pipeline with Price Data

The pipeline automatically computes annual statistics when it has access to daily price data:

```bash
uv run python scripts/run_model_evaluation.py
```

No additional flags needed - if the raw data includes daily prices, annual statistics will be computed automatically.

### 2. View Results

**Console Output:**
```
=== Annual Return Statistics ===
Years Covered:        3.5
Complete Years:       3

--- Return Distribution ---
Mean Annual Return:     8.50%
Median Annual Return:   9.00%
Std Dev (Annual):      12.00%

--- Percentiles ---
5th Percentile:        -5.00%
25th Percentile (Q1):   2.00%
75th Percentile (Q3):  15.00%
95th Percentile:       22.00%

--- Range ---
Best Year:             25.00%
Worst Year:            -8.00%

--- Win/Loss Profile ---
% Positive Years:      75.0%
Avg Winning Year:      12.00%
Avg Losing Year:       -6.00%

--- Shape ---
Skewness:              0.25
Excess Kurtosis:       1.50

--- Risk-Adjusted ---
Avg Annual Sharpe:     0.75
```

**JSON Output:**

Results are saved to `output/runs/ranking_<timestamp>/backtest_metrics.json`:

```json
{
  "annual_statistics": {
    "mean_annual_return": 0.085,
    "median_annual_return": 0.09,
    "std_annual_return": 0.12,
    "min_annual_return": -0.08,
    "max_annual_return": 0.25,
    "pct_5_annual_return": -0.05,
    "pct_25_annual_return": 0.02,
    "pct_75_annual_return": 0.15,
    "pct_95_annual_return": 0.22,
    "pct_positive_years": 0.75,
    "avg_positive_year": 0.12,
    "avg_negative_year": -0.06,
    "skewness_annual": 0.25,
    "kurtosis_annual": 1.5,
    "sharpe_annual_avg": 0.75,
    "num_years": 3,
    "years_sampled": 3.5
  }
}
```

### 3. Use for Monte Carlo Simulations

Export the statistics to your external risk modeling tools:

```python
import json

# Load the statistics
with open("output/runs/ranking_<timestamp>/backtest_metrics.json") as f:
    metrics = json.load(f)

annual_stats = metrics["annual_statistics"]

# Use in Monte Carlo simulation
# Example: Sample from normal distribution
import numpy as np

mean = annual_stats["mean_annual_return"]
std = annual_stats["std_annual_return"]
skew = annual_stats["skewness_annual"]

# Simple parametric simulation (assumes normal distribution)
simulated_returns = np.random.normal(mean, std, size=(1000, 10))  # 1000 paths, 10 years

# Or use the percentiles for non-parametric bootstrap
# percentiles = [
#     annual_stats["pct_5_annual_return"],
#     annual_stats["pct_25_annual_return"],
#     annual_stats["median_annual_return"],
#     annual_stats["pct_75_annual_return"],
#     annual_stats["pct_95_annual_return"],
# ]
```

## Understanding the Metrics

### Interpreting Skewness
- **Positive skewness** (> 0): More extreme positive returns than negative
  - Good for long strategies (occasional huge winners)
- **Negative skewness** (< 0): More extreme negative returns than positive
  - Warning sign (tail risk)
- **Near zero**: Symmetric distribution

### Interpreting Kurtosis
- **High kurtosis** (> 3): Fat tails, more extreme outcomes
  - Higher probability of outliers (good or bad)
- **Low kurtosis** (< 3): Thin tails, less extreme outcomes
  - More consistent, predictable returns

### Win Rate vs. Average Returns
- **High win rate + low avg winner**: "Picking up nickels"
- **Low win rate + high avg winner**: "Lottery ticket" strategy
- **Balanced (60-70% win, decent avg)**: Healthy profile

### Practical Use Cases

1. **Capital Requirements**: Use worst-year return to size positions
2. **Investor Expectations**: Show median and mean to set realistic goals
3. **Risk Budgeting**: Use std dev and drawdown for risk limits
4. **Strategy Comparison**: Compare Sharpe and skewness across strategies
5. **Monte Carlo**: Use full distribution for path-dependent simulations

## Requirements

- At least 252 days of continuous daily returns
- At least 2 complete calendar years (200+ trading days per year)
- Daily price data must be provided to the pipeline

## Limitations

- **Partial Years**: First/last years with < 200 days are excluded
- **Assumptions**: Calendar year grouping (Jan-Dec)
- **Sample Size**: With few years, statistics have high estimation error
- **Stationarity**: Assumes return distribution is stable over time

## See Also

- [RANKING_PIPELINE_GUIDE.md](RANKING_PIPELINE_GUIDE.md) - Full pipeline documentation
- `evaluation/portfolio_simulator.py` - Implementation details
- `tests/test_annual_statistics.py` - Unit tests and examples
