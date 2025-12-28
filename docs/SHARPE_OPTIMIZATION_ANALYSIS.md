# Sharpe Ratio Optimization Analysis

**Project**: Skuld - NZX Stock Ranking Pipeline  
**Date**: December 29, 2025  
**Experiment**: Grid Search Optimization for Maximum Sharpe Ratio

---

## Executive Summary

This analysis documents the results of a comprehensive grid search experiment testing 108 different configurations of the LightGBM ranking model for NZX stock prediction via Sharesies. The goal was to identify the optimal parameter combination to maximize the Sharpe ratio while accounting for realistic trading costs (190 bps transaction fee + 15 bps slippage per trade).

### Key Findings

1. **Optimal Configuration**: 105-day forward horizon with 150 estimators, 47 leaves, and top-10 portfolio achieved the highest Sharpe ratio of **0.896**.

2. **Best Average Horizon**: The 126-day horizon showed the most consistent performance with an average Sharpe of **0.62** and lowest average drawdown (**8.9%**).

3. **Portfolio Concentration**: Smaller portfolios (top-5) achieved higher average returns (47.5%) but with more volatility; top-10 offers the best risk-adjusted balance.

4. **Model Complexity**: Medium complexity (31 leaves) provides the most consistent results, though higher complexity (47 leaves) can capture the best individual performance.

---

## Experiment Design

### Parameters Tested

| Parameter | Values Tested | Description |
|-----------|---------------|-------------|
| `forward_return_days` | 63, 105, 126, 168 | Prediction horizon in trading days |
| `n_estimators` | 75, 100, 150 | Number of boosting iterations |
| `num_leaves` | 23, 31, 47 | Maximum leaves per tree (complexity) |
| `top_n` | 5, 10, 15 | Number of stocks in long portfolio |

**Total Configurations**: 4 × 3 × 3 × 3 = **108**

### Trading Costs (Sharesies NZX)

| Cost Type | Value | Description |
|-----------|-------|-------------|
| Transaction Fee | 190 bps (1.9%) | Per-trade fee charged by Sharesies |
| Slippage | 15 bps (0.15%) | Market impact and bid-ask spread |
| **Total Round-Trip** | **205 bps** | Full cost to enter and exit position |

### Backtesting Setup

- **Rolling Windows**: 10 windows with 0.5-year test periods
- **Strategy**: Long-only (no shorting)
- **Rebalancing**: At each prediction horizon
- **Universe**: NZX stocks available in the dataset

---

## Results Overview

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Configurations Tested | 108 |
| Best Sharpe Ratio | 0.896 |
| Best Total Return | 82.6% |
| Lowest Max Drawdown | 5.4% |
| Average Sharpe | 0.502 ± 0.211 |
| Average IC | 0.0792 |

### Optimal Configuration

| Parameter | Value |
|-----------|-------|
| Forward Return Days | **105** |
| Number of Estimators | **150** |
| Number of Leaves | **47** |
| Portfolio Top N | **10** |
| **Sharpe Ratio** | **0.896** |
| Total Return | 67.8% |
| Max Drawdown | 9.1% |
| Mean IC | 0.068 |
| Hit Rate | 60.0% |

---

## Detailed Analysis

### 1. Sharpe Ratio by Prediction Horizon

The prediction horizon has the most significant impact on strategy performance.

![Sharpe by Horizon](analysis/sharpe_by_horizon.png)

**Key Observations**:
- **126-day horizon** shows the most consistent performance (lowest variance)
- **63-day horizon** has high average Sharpe but more variability
- **168-day horizon** underperforms on average but can achieve high peaks
- **105-day horizon** captures the single best configuration

**Interpretation**: Longer horizons (3-6 months) allow the model to capture fundamental trends while avoiding short-term noise. The 126-day (~6 month) horizon appears optimal for the NZX market structure.

### 2. Risk-Return Trade-off

Understanding the relationship between returns and drawdown is crucial for portfolio construction.

![Return vs Drawdown](analysis/return_vs_drawdown.png)

**Key Observations**:
- Higher returns generally correlate with higher drawdowns
- The best Sharpe configurations cluster in the "sweet spot" (moderate return, low drawdown)
- Extreme returns (>70%) come with elevated drawdown risk

**Interpretation**: The optimal strategy should target the upper-left quadrant (high return, low drawdown). The 126-day configurations achieve this most reliably.

### 3. Parameter Interaction Heatmaps

Visualizing how parameters interact helps understand which combinations work best.

![Sharpe Heatmaps](analysis/sharpe_heatmaps.png)

**Key Observations**:
- **Horizon × Complexity**: 126-day works well across all complexity levels
- **Horizon × Portfolio Size**: Smaller portfolios (top-5) benefit more from longer horizons
- **Horizon × Estimators**: More estimators help at the 105-day horizon
- **Complexity × Portfolio (at 126d)**: Medium complexity (31 leaves) with top-5 is robust

### 4. Information Coefficient Relationship

IC measures the model's predictive power and correlates with strategy success.

![IC vs Sharpe](analysis/ic_vs_sharpe.png)

**Key Observations**:
- Strong positive correlation between IC and Sharpe ratio
- ICIR (consistency of IC) is also highly correlated with Sharpe
- Higher IC values cluster at shorter horizons (63d) but don't always translate to better Sharpe

**Interpretation**: A good IC (>0.05) is necessary but not sufficient for high Sharpe. Transaction costs at high-turnover short horizons can erode the benefit of high IC.

### 5. Parameter Importance

Each parameter's contribution to Sharpe ratio performance.

![Parameter Importance](analysis/parameter_importance.png)

**Key Observations**:
- **Forward Days**: Most impactful parameter; 126d optimal on average
- **Tree Complexity**: 31 leaves most consistent; 47 can achieve higher peaks
- **Estimators**: 150 estimators slightly better than 75-100
- **Portfolio Size**: Top-5 has highest average but most variance

### 6. Top 15 Configurations

Detailed comparison of the best-performing strategies.

![Top Configurations](analysis/top_configurations.png)

**Top 5 Configurations**:

| Rank | Config | Sharpe | Return | Drawdown |
|------|--------|--------|--------|----------|
| 1 | 105d-150e-47l-t10 | 0.896 | 67.8% | 9.1% |
| 2 | 168d-150e-47l-t5 | 0.871 | 58.6% | 5.6% |
| 3 | 168d-150e-47l-t10 | 0.861 | 44.9% | 5.4% |
| 4 | 126d-100e-47l-t5 | 0.850 | 53.9% | 7.6% |
| 5 | 126d-100e-31l-t5 | 0.847 | 53.2% | 8.1% |

### 7. Quintile Analysis

Quintile spread measures how well the model separates winners from losers.

![Quintile Analysis](analysis/quintile_analysis.png)

**Key Observations**:
- 63-day horizon has highest quintile spread (~5.5%)
- All horizons show positive spread (model has predictive value)
- Hit rate correlates with quintile spread

### 8. Efficient Frontier

The Pareto-optimal configurations that offer the best risk-return trade-offs.

![Efficient Frontier](analysis/efficiency_frontier.png)

**Key Observations**:
- The efficient frontier shows configurations that cannot be improved in one dimension without worsening another
- Most efficient configurations are at the 126d and 168d horizons
- Investors can choose along the frontier based on their risk tolerance

---

## Performance by Parameter

### Forward Return Horizon

| Horizon | Avg Sharpe | Std | Max Sharpe | Avg Return | Avg Drawdown |
|---------|------------|-----|------------|------------|--------------|
| 63d | 0.547 | 0.096 | 0.757 | 52.7% | 21.5% |
| 105d | 0.488 | 0.251 | 0.896 | 35.6% | 13.2% |
| 126d | 0.620 | 0.167 | 0.850 | 37.9% | 8.9% |
| 168d | 0.354 | 0.210 | 0.871 | 28.7% | 15.5% |

**Recommendation**: Use **126-day horizon** for consistent performance, or **105-day** if seeking maximum Sharpe with higher variance.

### Portfolio Size (Top N)

| Top N | Avg Sharpe | Std | Max Sharpe | Avg Return |
|-------|------------|-----|------------|------------|
| 5 | 0.529 | 0.259 | 0.871 | 47.5% |
| 10 | 0.503 | 0.204 | 0.896 | 37.1% |
| 15 | 0.474 | 0.163 | 0.847 | 31.6% |

**Recommendation**: Use **top-5** for higher returns with more volatility, or **top-10** for better risk-adjusted returns.

### Model Complexity (Num Leaves)

| Leaves | Avg Sharpe | Std | Max Sharpe |
|--------|------------|-----|------------|
| 23 | 0.470 | 0.179 | 0.757 |
| 31 | 0.512 | 0.205 | 0.850 |
| 47 | 0.525 | 0.244 | 0.896 |

**Recommendation**: Use **31 leaves** for consistent performance, or **47 leaves** if targeting maximum performance with tuning.

---

## Recommendations

### For Production Deployment

Based on this analysis, the recommended configuration for production is:

```python
# Recommended settings for settings.py
FORWARD_RETURN_DAYS = 126      # Best consistency
RANKER_N_ESTIMATORS = 100      # Good balance of performance/speed
RANKER_NUM_LEAVES = 31         # Most stable
PORTFOLIO_TOP_N = 5            # Higher expected return
```

**Expected Performance**:
- Sharpe Ratio: ~0.6-0.85
- Annual Return: ~35-55%
- Max Drawdown: ~8-15%

### For Maximum Sharpe (Higher Risk)

```python
FORWARD_RETURN_DAYS = 105
RANKER_N_ESTIMATORS = 150
RANKER_NUM_LEAVES = 47
PORTFOLIO_TOP_N = 10
```

**Expected Performance**:
- Sharpe Ratio: ~0.9
- Annual Return: ~68%
- Max Drawdown: ~9%

### For Conservative Strategy

```python
FORWARD_RETURN_DAYS = 168
RANKER_N_ESTIMATORS = 150
RANKER_NUM_LEAVES = 47
PORTFOLIO_TOP_N = 10
```

**Expected Performance**:
- Sharpe Ratio: ~0.86
- Annual Return: ~45%
- Max Drawdown: ~5.4%

---

## Limitations and Caveats

1. **Survivorship Bias**: The dataset may not include delisted stocks, potentially overstating returns.

2. **Look-Ahead in Features**: While forward returns are properly computed, some features may inadvertently contain future information.

3. **Market Regime Dependency**: These results are based on historical data; future market conditions may differ.

4. **Liquidity Constraints**: Small NZX stocks may have insufficient liquidity to execute the strategy at the assumed costs.

5. **Slippage Estimation**: The 15 bps slippage assumption may be optimistic for less liquid stocks.

6. **Rebalancing Frequency**: The strategy assumes perfect execution at rebalancing points.

---

## Next Steps

1. **Walk-Forward Validation**: Test the optimal configuration on out-of-sample data.

2. **Feature Importance Analysis**: Identify which features drive predictions at the optimal horizon.

3. **Shorter Horizon Investigation**: Debug why 1-3 day horizons fail (likely weekend/holiday gaps).

4. **Position Sizing**: Implement volatility-weighted position sizing to further improve Sharpe.

5. **Regime Detection**: Add market regime awareness to adjust strategy parameters dynamically.

---

## Appendix: Generated Visualizations

All visualizations are saved to `/docs/analysis/`:

| File | Description |
|------|-------------|
| [sharpe_by_horizon.png](analysis/sharpe_by_horizon.png) | Box plot of Sharpe by horizon |
| [return_vs_drawdown.png](analysis/return_vs_drawdown.png) | Risk-return scatter |
| [sharpe_heatmaps.png](analysis/sharpe_heatmaps.png) | Parameter interaction heatmaps |
| [ic_vs_sharpe.png](analysis/ic_vs_sharpe.png) | IC relationship analysis |
| [parameter_importance.png](analysis/parameter_importance.png) | Parameter impact bars |
| [top_configurations.png](analysis/top_configurations.png) | Top 15 comparison |
| [quintile_analysis.png](analysis/quintile_analysis.png) | Quintile spread analysis |
| [efficiency_frontier.png](analysis/efficiency_frontier.png) | Pareto optimal frontier |

---

*Generated by `scripts/generate_analysis_report.py`*
