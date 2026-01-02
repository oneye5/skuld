# Annual Statistics for Real-World Implementation

> **Navigation:** [Main README](../README.md) | [Pipeline Guide](RANKING_PIPELINE_GUIDE.md) | [Features](FEATURES.md) | [Clustering](CLUSTERING.md) | [Testing](TESTING.md) | [Annual Statistics](ANNUAL_STATISTICS.md) | [Data Leakage](DATA_LEAKAGE.md) | [TODO](TODO.md)

---

## Table of Contents

1. [Overview](#overview)
2. [What's Computed](#whats-computed)
3. [Usage](#usage)
4. [Interpreting Results](#interpreting-results)
5. [Practical Applications](#practical-applications)
6. [Requirements](#requirements)
7. [Limitations](#limitations)
8. [Related Documentation](#related-documentation)

---

## Overview

The ranking pipeline computes **annual return distribution statistics** to support real-world implementation decisions and Monte Carlo simulations.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      ANNUAL STATISTICS COMPUTATION                          │
└────────────────────────────────────────────────────────────────────────────┘

Portfolio Daily Returns (from backtest)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Group by Calendar Year                                         │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│ │ Year 1  │ │ Year 2  │ │ Year 3  │ │ ...     │               │
│ │ Returns │ │ Returns │ │ Returns │ │         │               │
│ └────┬────┘ └────┬────┘ └────┬────┘ └─────────┘               │
│      │           │           │                                 │
│      ▼           ▼           ▼                                 │
│ ┌─────────────────────────────────┐                           │
│ │ Compound to Annual Returns      │                           │
│ │ R_annual = ∏(1 + r_daily) - 1   │                           │
│ └─────────────────────────────────┘                           │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ STATISTICS                                                     │
│                                                                │
│  Distribution:      │  Risk:              │  Shape:           │
│  • Mean             │  • Std Dev          │  • Skewness       │
│  • Median           │  • Percentiles      │  • Kurtosis       │
│  • Min/Max          │  • Win/Loss Profile │  • Annual Sharpe  │
└───────────────────────────────────────────────────────────────┘
```

## What's Computed

### Return Distribution
| Metric | Description |
|--------|-------------|
| **Mean Annual Return** | Average annual return across all years |
| **Median Annual Return** | Middle value (robust to outliers) |
| **Std Dev (Annual)** | Volatility of annual returns |
| **Percentiles** | 5th, 25th, 75th, 95th percentiles |
| **Range** | Best and worst year |

### Win/Loss Profile
| Metric | Description |
|--------|-------------|
| **% Positive Years** | Percentage of years with positive returns |
| **Avg Winning Year** | Average return in positive years |
| **Avg Losing Year** | Average return in negative years |

### Shape Statistics
| Metric | Description |
|--------|-------------|
| **Skewness** | Distribution asymmetry (positive = more upside) |
| **Excess Kurtosis** | Tail heaviness (higher = more extreme outcomes) |
| **Avg Annual Sharpe** | Risk-adjusted performance per year |

## Usage

### 1. Run the Pipeline

```bash
cd python/ml-pipeline
uv run python scripts/run_model_evaluation.py
```

Annual statistics are computed automatically when daily price data is available.

### 2. View Results

**Console Output:**
```
=== Annual Return Statistics ===
Years Covered:        5.0
Complete Years:       5

--- Return Distribution ---
Mean Annual Return:    12.50%
Median Annual Return:  11.00%
Std Dev (Annual):      18.00%

--- Percentiles ---
5th Percentile:        -8.00%
25th Percentile (Q1):   3.00%
75th Percentile (Q3):  22.00%
95th Percentile:       35.00%

--- Range ---
Best Year:             42.00%
Worst Year:           -12.00%

--- Win/Loss Profile ---
% Positive Years:      80.0%
Avg Winning Year:      18.00%
Avg Losing Year:       -8.00%

--- Shape ---
Skewness:              0.35
Excess Kurtosis:       0.80

--- Risk-Adjusted ---
Avg Annual Sharpe:     0.85
```

**JSON Output:** (in `output/runs/ranking_<timestamp>/backtest_metrics.json`)

```json
{
  "annual_statistics": {
    "mean_annual_return": 0.125,
    "median_annual_return": 0.11,
    "std_annual_return": 0.18,
    "min_annual_return": -0.12,
    "max_annual_return": 0.42,
    "pct_5_annual_return": -0.08,
    "pct_25_annual_return": 0.03,
    "pct_75_annual_return": 0.22,
    "pct_95_annual_return": 0.35,
    "pct_positive_years": 0.80,
    "avg_positive_year": 0.18,
    "avg_negative_year": -0.08,
    "skewness_annual": 0.35,
    "kurtosis_annual": 0.80,
    "sharpe_annual_avg": 0.85,
    "num_years": 5,
    "years_sampled": 5.0
  }
}
```

### 3. Monte Carlo Simulation

```python
import json
import numpy as np

# Load statistics
with open("output/runs/ranking_<timestamp>/backtest_metrics.json") as f:
    metrics = json.load(f)

stats = metrics["annual_statistics"]

# Parametric simulation (normal distribution)
mean = stats["mean_annual_return"]
std = stats["std_annual_return"]

# Simulate 1000 paths over 10 years
simulated = np.random.normal(mean, std, size=(1000, 10))

# Compute terminal wealth
wealth_paths = np.cumprod(1 + simulated, axis=1)
terminal_wealth = wealth_paths[:, -1]

print(f"Median 10-year wealth: {np.median(terminal_wealth):.2f}x")
print(f"5th percentile: {np.percentile(terminal_wealth, 5):.2f}x")
print(f"95th percentile: {np.percentile(terminal_wealth, 95):.2f}x")
```

## Interpreting Results

### Skewness Guide

```
Positive Skew (> 0)        │ Zero Skew (≈ 0)          │ Negative Skew (< 0)
                           │                          │
     ╱╲                    │      ╱╲                  │         ╱╲
    ╱  ╲___                │     ╱  ╲                 │     ___╱  ╲
   ╱      ╲__              │    ╱    ╲                │    __╱    ╲
──────────────►            │ ──────────►              │ ──────────────►
                           │                          │
More extreme positive      │ Symmetric                │ More extreme negative
returns (occasional        │ distribution             │ returns (tail risk)
big winners)               │                          │
```

### Kurtosis Guide

| Value | Interpretation |
|-------|----------------|
| < 3 | Thin tails, fewer extremes, more predictable |
| ≈ 3 | Normal distribution (baseline) |
| > 3 | Fat tails, more extremes, higher tail risk |

### Win Rate Analysis

| Pattern | Implication |
|---------|-------------|
| High win rate + low avg winner | "Picking up nickels" — steady but limited upside |
| Low win rate + high avg winner | "Lottery" — volatile, requires discipline |
| 60-70% win + decent avg both | Healthy, sustainable profile |

## Practical Applications

### Capital Requirements
Use **worst-year return** to size positions conservatively:
```python
max_loss_per_year = abs(stats["min_annual_return"])
capital_at_risk = portfolio_value * max_loss_per_year
```

### Risk Budgeting
Use **std dev** for position sizing:
```python
target_portfolio_vol = 0.15  # 15% target volatility
position_size = target_portfolio_vol / stats["std_annual_return"]
```

### Strategy Comparison
Compare multiple strategies:
```python
strategies = {"A": stats_a, "B": stats_b, "C": stats_c}
for name, s in strategies.items():
    print(f"{name}: Sharpe={s['sharpe_annual_avg']:.2f}, "
          f"Skew={s['skewness_annual']:.2f}")
```

## Requirements

- **Minimum data:** 252 days of continuous daily returns
- **Minimum years:** 2 complete calendar years (200+ trading days each)
- **Data source:** Daily price data from backtest

## Limitations

| Limitation | Description |
|------------|-------------|
| **Partial Years** | First/last years with <200 days excluded |
| **Calendar Grouping** | Assumes Jan-Dec years |
| **Sample Size** | Few years = high estimation error |
| **Stationarity** | Assumes return distribution is stable |

## Related Documentation

- [Ranking Pipeline Guide](RANKING_PIPELINE_GUIDE.md) — Full pipeline documentation
- [Testing](TESTING.md) — Test coverage including annual statistics tests
- `evaluation/portfolio_simulator.py` — Implementation details
- `tests/test_annual_statistics.py` — Unit tests and examples

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
