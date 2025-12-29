# Implementation Plan: Ranking-Based Cross-Sectional Stock Prediction

**Project**: Skuld - Time Series Forecasting Framework  
**Owner**: oneye5  
**Created**: 2025-12-27  
**Updated**: 2025-12-28  
**Status**: Planning

## Executive Summary

This document outlines the comprehensive implementation plan for pivoting the Skuld framework from point prediction to ranking-based cross-sectional stock prediction. The new approach focuses on relative performance ranking across stocks within a universe at each time step, enabling portfolio construction and long-short strategies.

> **Note for AI Agents**: This plan follows the project's philosophy of *incremental changes* and *iteration speed*. Each phase can be implemented and tested independently. Write unit tests before implementing features, and use small debug scripts to validate transformations before running the full pipeline.

---

## Table of Contents

1. [Motivation and Objectives](#motivation-and-objectives)
2. [Research Foundation](#research-foundation)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Phases](#implementation-phases)
5. [Technical Specifications](#technical-specifications)
6. [Testing Strategy](#testing-strategy)
7. [Documentation Updates](#documentation-updates)
8. [Success Criteria](#success-criteria)
9. [Risk Mitigation](#risk-mitigation)
10. [Timeline and Milestones](#timeline-and-milestones)

---

## Motivation and Objectives

### Why Ranking-Based Prediction?

1. **Market Neutrality**: Ranking naturally removes market-wide effects (beta) and focuses on relative performance (alpha)
2. **Portfolio Construction**: Direct support for long-short strategies and quintile-based portfolios
3. **Robustness**: Rankings are more stable and less sensitive to outliers than point predictions (research shows rank-based loss functions are more robust to label noise)
4. **Practical Trading**: Aligns better with real-world portfolio management needs where capital allocation is the goal
5. **Better Metrics**: IC (Information Coefficient), Rank IC, and IR (Information Ratio) are industry-standard measures for alpha factor evaluation
6. **Reduced Overfitting**: Learning-to-rank objectives focus on relative ordering, not absolute values, reducing sensitivity to scale

### Key Objectives

- Transform prediction task from binary classification to cross-sectional ranking
- Implement cross-sectional evaluation metrics (IC, Rank IC, IR, Hit Rate, Quintile Spread)
- Support portfolio-based backtesting with realistic transaction costs
- Maintain backward compatibility with existing feature engineering pipeline
- Provide clear migration path for existing classification pipeline

---

## Research Foundation

### Academic Background

The ranking-based approach draws from **Learning-to-Rank (LTR)** literature and quantitative finance research:

1. **LambdaMART/LambdaRank** (Burges et al., 2010): The core ranking algorithm used by LightGBM. It optimizes NDCG directly via gradient approximation, making it suitable for stock ranking where we care about the ordering of returns.

2. **Information Coefficient in Finance**: IC measures Pearson correlation between predicted scores and realized returns. An IC of 0.05-0.10 is considered good in practice (Grinold & Kahn, "Active Portfolio Management").

3. **The Fundamental Law of Active Management**: `IR ≈ IC × √BR` where IR is Information Ratio, IC is Information Coefficient, and BR is breadth (number of independent bets). This motivates maximizing IC across many timestamps.

4. **Cross-Sectional Momentum** (Jegadeesh & Titman, 1993): Sorting stocks by past returns and going long winners/short losers generates alpha. Our ranking approach naturally captures this.

### Key Research Insights to Apply

| Insight | Application in This Project |
|---------|----------------------------|
| **IC Decay**: IC typically decays over holding periods | Track IC at multiple horizons (1d, 5d, 20d) |
| **IC Stability > Magnitude**: Consistent small IC beats volatile high IC | Optimize for IR = Mean(IC)/Std(IC) not just Mean(IC) |
| **Quintile Spread**: Top-bottom quintile return spread is key metric | Report quintile performance as primary backtest output |
| **Transaction Costs Matter**: Frequent rebalancing erodes alpha | Implement realistic cost models; test weekly/monthly rebalancing |
| **Group Size Matters for LTR**: LGBMRanker performs better with ≥10 items per group | Ensure minimum stocks per timestamp; document constraints |

### Recommended Readings (for human implementers)

- Grinold, R. & Kahn, R. "Active Portfolio Management" - Chapters on IC and the Fundamental Law
- de Prado, M. L. "Advances in Financial Machine Learning" - Chapter 10 on Bet Sizing
- Burges, C. "From RankNet to LambdaRank to LambdaMART" (Microsoft Research, 2010)

---

## Architecture Overview

### Core Components

The implementation will integrate into the existing `python/ml-pipeline` structure:

```
python/ml-pipeline/
├── config/
│   ├── settings.py              # Add ranking-specific settings
│   ├── columns.py               # Add ranking column names
│   └── ranking_config.py        # NEW: Ranking-specific constants (keeps settings.py clean)
├── core/
│   ├── data_loader.py           # Existing loader (compatible)
│   ├── long_to_wide.py          # Existing converter (compatible)
│   └── target_builder.py        # NEW: Continuous target (forward returns) vs binary labels
├── learner/
│   ├── ranking.py               # NEW: Ranking model wrappers (LGBMRanker, XGBRanker)
│   ├── trainer.py               # Modify to support ranking models with group parameter
│   └── predictor.py             # Existing predictor (compatible)
├── evaluation/
│   ├── metrics.py               # Existing classification metrics (keep for comparison)
│   ├── ranking_metrics.py       # NEW: IC, Rank IC, Hit Rate, Quintile analysis
│   ├── portfolio_simulator.py   # NEW: Portfolio construction & backtest
│   ├── simulator.py             # Existing trade simulator (keep for comparison)
│   └── visualization.py         # NEW: Ranking-specific plots
├── pipeline/
│   ├── rolling_window.py        # Existing (keep for classification)
│   └── ranking_pipeline.py      # NEW: End-to-end ranking pipeline
├── tests/
│   ├── test_ranking_metrics.py  # NEW: Unit tests for IC/RankIC calculations
│   ├── test_ranking_model.py    # NEW: Unit tests for LGBMRanker wrapper
│   └── test_portfolio.py        # NEW: Unit tests for portfolio construction
└── runnables/
    ├── main_ranking.py          # NEW: Entry point for ranking pipeline
    └── debug_ranking.py         # NEW: Quick validation script for development
```

### Design Principles

1. **Parallel Pipelines**: Keep classification and ranking pipelines separate initially. This allows A/B comparison and gradual migration.

2. **Shared Components**: Reuse `data_loader.py`, `long_to_wide.py`, `preprocessor.py`, `scaler.py`, and feature engineering modules. Only the target definition, model, and evaluation differ.

3. **Config Isolation**: New `ranking_config.py` avoids polluting the existing `settings.py` while keeping ranking parameters centralized.

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

> **AI Agent Note**: Start here. Write unit tests first, then implement. Verify with small test DataFrames before integrating.

#### 1.1 Target Builder (Continuous Returns)

**Task**: Create a module to compute forward returns (continuous target) instead of binary labels.

**File**: `python/ml-pipeline/core/target_builder.py`

**Why**: The ranking model needs continuous return values, not binary 0/1 labels. We still want to keep `labeler.py` for the classification pipeline.

```python
def compute_forward_returns(
    df: pd.DataFrame,
    lookahead_days: int = 5,  # Common horizons: 1, 5, 20 days
    return_type: str = "simple",  # "simple" or "log"
) -> pd.DataFrame:
    """Compute forward returns for each ticker.
    
    Args:
        df: Wide format DataFrame with timestamp, ticker, Close columns.
        lookahead_days: Number of days to compute forward return.
        return_type: "simple" for (P_t+n - P_t) / P_t, "log" for ln(P_t+n / P_t).
    
    Returns:
        DataFrame with 'forward_return' column added.
    """
    # Implementation: merge_asof for forward price lookup (similar to labeler.py)
```

**Test First** (`tests/test_target_builder.py`):
```python
def test_forward_returns_basic():
    """5-day forward return: if price goes from 100 to 110, return = 0.10"""
    # Create synthetic data, verify calculation
```

#### 1.2 Metrics & Evaluation

**Task**: Implement cross-sectional evaluation metrics.

**File**: `python/ml-pipeline/evaluation/ranking_metrics.py`

**Core Metrics** (in order of importance):

| Metric | Formula | Why It Matters |
|--------|---------|----------------|
| **Rank IC** | `scipy.stats.spearmanr(predicted_scores, actual_returns)` per timestamp | Primary metric. Robust to outliers. |
| **IC** | `scipy.stats.pearsonr(predicted_scores, actual_returns)` per timestamp | Standard metric, but sensitive to outliers. |
| **ICIR** | `mean(IC_series) / std(IC_series) * sqrt(252)` | Measures IC consistency (annualized). |
| **Hit Rate** | `% of top-N predictions with positive returns` | Intuitive for trading. |
| **Quintile Spread** | `mean_return(Q5) - mean_return(Q1)` | Shows long-short profitability. |

**Implementation Notes**:
- Compute IC/RankIC **per timestamp** first, then aggregate (mean, std, IR).
- Use `scipy.stats.spearmanr` and `scipy.stats.pearsonr` for correlation.
- Handle edge cases: timestamps with <5 stocks should be skipped or flagged.

```python
@dataclass
class RankingMetrics:
    mean_ic: float
    std_ic: float
    icir: float  # annualized
    mean_rank_ic: float
    std_rank_ic: float
    rank_icir: float
    hit_rate_top_n: float
    quintile_returns: dict[int, float]  # {1: -0.02, 2: -0.01, ..., 5: 0.03}
    quintile_spread: float  # Q5 - Q1
    num_timestamps: int
    avg_stocks_per_timestamp: float
```

**Test First** (`tests/test_ranking_metrics.py`):
```python
def test_perfect_ranking_ic():
    """If predicted scores exactly equal actual returns, IC = 1.0"""

def test_random_ranking_ic_near_zero():
    """Random predictions should have IC ≈ 0"""

def test_quintile_spread_calculation():
    """Top quintile should have highest returns for good model"""
```

#### 1.3 Configuration Updates

**Task**: Create ranking-specific config file.

**File**: `python/ml-pipeline/config/ranking_config.py`

```python
"""Ranking pipeline configuration.

Separate from settings.py to avoid polluting the classification pipeline.
"""

# =============================================================================
# TARGET DEFINITION (for ranking)
# =============================================================================
FORWARD_RETURN_DAYS: int = 5
"""Number of days to compute forward return. Common values: 1, 5, 20."""

RETURN_TYPE: str = "simple"
"""'simple' for arithmetic returns, 'log' for logarithmic returns."""

# =============================================================================
# RANKING MODEL
# =============================================================================
RANKING_MODEL: str = "lightgbm"
"""Ranking model to use: 'lightgbm' (LGBMRanker) or 'xgboost' (XGBRanker)."""

RANKING_OBJECTIVE: str = "lambdarank"
"""Objective function: 'lambdarank' (listwise), 'rank_xendcg', or 'pairwise'."""

MIN_STOCKS_PER_TIMESTAMP: int = 10
"""Minimum stocks required per timestamp for valid ranking. LGBMRanker needs sufficient group size."""

# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================
PORTFOLIO_TOP_N: int = 10
"""Number of top-ranked stocks for long portfolio."""

PORTFOLIO_BOTTOM_N: int = 10
"""Number of bottom-ranked stocks for short portfolio."""

REBALANCE_FREQUENCY: str = "daily"
"""Rebalancing frequency: 'daily', 'weekly', 'monthly'."""

WEIGHTING_SCHEME: str = "equal"
"""Portfolio weighting: 'equal', 'score_weighted', 'inverse_volatility'."""

TRANSACTION_COST_BPS: float = 10.0
"""Round-trip transaction cost in basis points (10 bps = 0.1%)."""
```

---

### Phase 2: Model Implementation (Week 2)

> **AI Agent Note**: LGBMRanker has specific requirements. Read the LightGBM docs carefully. Data MUST be sorted by group (timestamp) before training.

#### 2.1 Ranking Model Wrapper

**Task**: Create a unified interface for ranking models.

**File**: `python/ml-pipeline/learner/ranking.py`

**Critical Implementation Details for LGBMRanker**:

1. **Data Sorting**: LGBMRanker requires data to be sorted by group (timestamp). The `group` parameter is a list where each element is the number of samples in that group.

2. **Group Construction**: 
   ```python
   # Example: 3 timestamps with 10, 15, 12 stocks each
   group = [10, 15, 12]  # sum(group) must equal len(X)
   ```

3. **Objective Options**:
   - `lambdarank`: Best for NDCG optimization (recommended)
   - `rank_xendcg`: Alternative listwise loss
   - `pairwise`: Simpler, but often worse performance

4. **Label Treatment**: LGBMRanker expects higher labels = better. For returns, this is natural (higher return = better). For losses, negate them.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


@dataclass
class RankerConfig:
    """Configuration for ranking models."""
    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1  # -1 = no limit
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: tuple[int, ...] = (5, 10, 20)
    random_state: int = 42


class BaseRanker(ABC):
    """Abstract base class for ranking models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, group: list[int]) -> "BaseRanker":
        """Fit ranking model.
        
        Args:
            X: Feature DataFrame, MUST be sorted by timestamp.
            y: Target Series (forward returns), same order as X.
            group: List of group sizes. sum(group) == len(X).
        
        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores (higher = better predicted rank)."""
        pass


class LightGBMRankerWrapper(BaseRanker):
    """Wrapper for LightGBM LGBMRanker with sklearn-like interface."""
    
    def __init__(self, config: RankerConfig | None = None):
        self.config = config or RankerConfig()
        self.model: LGBMRanker | None = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, group: list[int]) -> "LightGBMRankerWrapper":
        # Validate group sizes
        if sum(group) != len(X):
            raise ValueError(f"sum(group)={sum(group)} != len(X)={len(X)}")
        
        self.model = LGBMRanker(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            objective=self.config.objective,
            metric=self.config.metric,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        
        self.model.fit(
            X, y, 
            group=group,
            eval_at=self.config.eval_at,
        )
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
```

**Test First** (`tests/test_ranking_model.py`):
```python
def test_lgbm_ranker_fit_predict():
    """Basic fit/predict smoke test with synthetic data."""
    
def test_lgbm_ranker_group_validation():
    """Should raise ValueError if sum(group) != len(X)."""

def test_lgbm_ranker_sorted_data_requirement():
    """Verify model trains successfully with properly sorted data."""
```

#### 2.2 Group Builder Utility

**Task**: Create a utility to build the `group` parameter from sorted data.

**File**: `python/ml-pipeline/learner/ranking.py` (add to same file)

```python
def build_group_from_timestamps(df: pd.DataFrame, timestamp_col: str = "timestamp") -> list[int]:
    """Build group sizes from a DataFrame sorted by timestamp.
    
    Args:
        df: DataFrame sorted by timestamp_col.
        timestamp_col: Name of timestamp column.
    
    Returns:
        List of group sizes (number of stocks per timestamp).
    
    Raises:
        ValueError: If DataFrame is not sorted by timestamp.
    """
    # Verify sorted
    timestamps = df[timestamp_col].values
    if not np.all(timestamps[:-1] <= timestamps[1:]):
        raise ValueError("DataFrame must be sorted by timestamp for LGBMRanker")
    
    # Count samples per timestamp
    return df.groupby(timestamp_col).size().tolist()
```

---

### Phase 3: Portfolio & Backtesting (Week 3)

> **AI Agent Note**: Portfolio simulation is where we measure real-world performance. Start simple (equal-weight long-short), then add complexity.

#### 3.1 Portfolio Simulator

**Task**: Implement portfolio construction and backtesting.

**File**: `python/ml-pipeline/evaluation/portfolio_simulator.py`

**Design Philosophy**: 
- Each timestamp is a rebalancing point.
- At each rebalance: rank stocks by predicted score, go long top-N, short bottom-N.
- Track daily returns, cumulative returns, turnover.

**Core Data Structures**:
```python
@dataclass
class PortfolioConfig:
    top_n: int = 10
    bottom_n: int = 10
    weighting: str = "equal"  # "equal", "score_weighted"
    transaction_cost_bps: float = 10.0
    long_only: bool = False  # If True, ignore bottom_n

@dataclass  
class BacktestResult:
    daily_returns: pd.Series  # Index = timestamp, values = daily portfolio return
    cumulative_returns: pd.Series
    sharpe_ratio: float  # Annualized
    max_drawdown: float
    total_return: float
    avg_turnover: float  # Average portfolio turnover per rebalance
    quintile_returns: pd.DataFrame  # Columns = Q1..Q5, rows = timestamps
    holdings_history: pd.DataFrame  # For debugging: timestamp, ticker, weight
```

**Key Functions**:
```python
def run_portfolio_backtest(
    predictions_df: pd.DataFrame,  # timestamp, ticker, predicted_score
    returns_df: pd.DataFrame,      # timestamp, ticker, actual_return
    config: PortfolioConfig,
) -> BacktestResult:
    """Run long-short portfolio backtest.
    
    At each timestamp:
    1. Rank stocks by predicted_score.
    2. Long top-N, short bottom-N (equal weight or score-weighted).
    3. Compute portfolio return = mean(long returns) - mean(short returns).
    4. Apply transaction costs based on turnover.
    """

def compute_quintile_returns(
    predictions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute average return for each quintile at each timestamp.
    
    Returns DataFrame with columns [Q1, Q2, Q3, Q4, Q5] and index = timestamps.
    Q5 = top quintile (highest predicted scores).
    """
```

**Test First** (`tests/test_portfolio.py`):
```python
def test_equal_weight_long_short():
    """Long 2, short 2 from 10 stocks. Verify weights sum correctly."""

def test_transaction_cost_applied():
    """Returns should be reduced by transaction costs on turnover."""

def test_quintile_returns_ordering():
    """If model is perfect, Q5 > Q4 > Q3 > Q2 > Q1."""
```

#### 3.2 Visualization

**Task**: Create plots for ranking analysis.

**File**: `python/ml-pipeline/evaluation/visualization.py`

**Priority Plots** (implement in this order):

1. **Quintile Bar Chart**: Average return by quintile (most important - shows if model separates winners from losers).

2. **IC Time Series**: Plot IC over time with rolling mean. Shows stability.

3. **Cumulative Returns**: Equity curve of Long-Short strategy vs. market benchmark.

4. **Turnover Histogram**: Distribution of portfolio turnover (useful for cost analysis).

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_quintile_returns(quintile_df: pd.DataFrame, save_path: str | None = None):
    """Bar chart of average returns by quintile."""
    avg_returns = quintile_df.mean()
    colors = ['red', 'orange', 'gray', 'lightgreen', 'green']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(avg_returns.index, avg_returns.values, color=colors)
    ax.set_xlabel("Quintile")
    ax.set_ylabel("Average Return")
    ax.set_title("Return by Predicted Quintile")
    ax.axhline(0, color='black', linewidth=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_ic_series(ic_series: pd.Series, rolling_window: int = 20, save_path: str | None = None):
    """Plot IC over time with rolling average."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ic_series.index, ic_series.values, alpha=0.5, label="Daily IC")
    ax.plot(ic_series.index, ic_series.rolling(rolling_window).mean(), 
            color='red', linewidth=2, label=f"{rolling_window}-day Rolling Mean")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    ax.set_title("Information Coefficient Over Time")
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
```

---

### Phase 4: Integration & Pipeline (Week 4)

> **AI Agent Note**: This is the integration phase. Reuse existing components where possible. The ranking pipeline should feel familiar to anyone who's used the classification pipeline.

#### 4.1 Ranking Pipeline

**Task**: Create an end-to-end pipeline for ranking.

**File**: `python/ml-pipeline/pipeline/ranking_pipeline.py`

**Pipeline Steps** (mirrors `rolling_window.py` structure):

```python
def run_ranking_pipeline(
    long_df: pd.DataFrame | None = None,
    num_windows: int = NUM_ROLLING_WINDOWS,
    # ... other params from ranking_config.py
) -> RankingPipelineResult:
    """Run the ranking pipeline across rolling windows.
    
    Steps:
    1. Load data (reuse data_loader.py)
    2. Convert to wide format (reuse long_to_wide.py)
    3. Compute forward returns (new: target_builder.py)
    4. For each rolling window:
       a. Split train/test (reuse splitter.py logic)
       b. Apply feature engineering (reuse existing features/)
       c. Scale features (reuse scaler.py - fit on train only!)
       d. Build group parameter for LGBMRanker
       e. Train ranker
       f. Predict scores on test
       g. Compute IC, RankIC for this window
    5. Aggregate metrics across windows
    6. Run portfolio backtest
    7. Generate visualizations and report
    """
```

**Key Difference from Classification Pipeline**:
- Target is continuous (forward returns) not binary
- Model is LGBMRanker, not classifier
- Evaluation uses IC/RankIC, not precision/recall
- Output includes quintile analysis and portfolio backtest

#### 4.2 Runnable Entry Point

**Task**: Create a runnable script.

**File**: `python/ml-pipeline/runnables/main_ranking.py`

```python
"""Entry point for ranking pipeline.

Usage:
    uv run python runnables/main_ranking.py
    uv run python runnables/main_ranking.py --forward-days 20 --top-n 5
"""

import argparse
from pipeline.ranking_pipeline import run_ranking_pipeline
from config.ranking_config import *

def main():
    parser = argparse.ArgumentParser(description="Run ranking-based stock prediction pipeline")
    parser.add_argument("--forward-days", type=int, default=FORWARD_RETURN_DAYS,
                        help="Forward return horizon in days")
    parser.add_argument("--top-n", type=int, default=PORTFOLIO_TOP_N,
                        help="Number of stocks for long portfolio")
    parser.add_argument("--bottom-n", type=int, default=PORTFOLIO_BOTTOM_N,
                        help="Number of stocks for short portfolio")
    # ... more args
    
    args = parser.parse_args()
    result = run_ranking_pipeline(
        forward_return_days=args.forward_days,
        portfolio_top_n=args.top_n,
        portfolio_bottom_n=args.bottom_n,
    )
    
    print(f"\n=== Results ===")
    print(f"Mean IC: {result.mean_ic:.4f}")
    print(f"ICIR: {result.icir:.4f}")
    print(f"Quintile Spread: {result.quintile_spread:.4f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")

if __name__ == "__main__":
    main()
```

#### 4.3 Debug Script (Fast Iteration)

**Task**: Create a quick validation script for development.

**File**: `python/ml-pipeline/runnables/debug_ranking.py`

```python
"""Quick debug script for ranking development.

Runs on a small subset of data to verify code correctness.
Takes ~30 seconds instead of 15 minutes.

Usage:
    uv run python runnables/debug_ranking.py
"""

from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide
from core.target_builder import compute_forward_returns
from learner.ranking import LightGBMRankerWrapper, build_group_from_timestamps, RankerConfig
from evaluation.ranking_metrics import compute_ranking_metrics

def main():
    print("Loading data (subset)...")
    long_df = load_long_data()
    
    # Take only recent 2 years for speed
    max_ts = long_df["timestamp"].max()
    two_years_ms = 2 * 365 * 86400 * 1000
    long_df = long_df[long_df["timestamp"] > (max_ts - two_years_ms)]
    print(f"Subset: {len(long_df):,} rows")
    
    # Convert and prepare
    wide_df = long_to_wide(long_df)
    wide_df = compute_forward_returns(wide_df, lookahead_days=5)
    
    # Simple train/test split (last 20% = test)
    # ... quick validation logic
    
    print("Debug run complete!")

if __name__ == "__main__":
    main()
```

---

## Technical Specifications

### Data Format for Ranking

LGBMRanker requires data to be sorted by group (Query ID). In our case, the "Query" is the Timestamp.

| Requirement | Description |
|-------------|-------------|
| **Sorting** | DataFrame MUST be sorted by timestamp before training |
| **X (Features)** | DataFrame with numeric feature columns |
| **y (Targets)** | Series of forward returns (higher = better) |
| **group** | List of integers: `[n_stocks_ts1, n_stocks_ts2, ...]` where `sum(group) == len(X)` |

**Example**:
```python
# Data with 3 timestamps, variable stocks per timestamp
# Timestamp 1: AAPL, GOOGL, MSFT (3 stocks)
# Timestamp 2: AAPL, GOOGL (2 stocks)  
# Timestamp 3: AAPL, GOOGL, MSFT, AMZN (4 stocks)

group = [3, 2, 4]  # sum = 9 = len(X)
```

### Target Definition

| Target Type | Formula | When to Use |
|-------------|---------|-------------|
| **Simple Return** | `(P_t+n - P_t) / P_t` | Default. Works well for short horizons. |
| **Log Return** | `ln(P_t+n / P_t)` | Better for longer horizons. Additive property. |
| **Excess Return** | `R_stock - R_market` | For market-neutral ranking. |
| **Winsorized Return** | Clip returns to [-0.5, 0.5] | Reduces outlier impact on training. |

**Recommendation**: Start with simple returns, winsorized at ±50%. This is robust and interpretable.

### Minimum Data Requirements

| Constraint | Minimum | Recommended |
|------------|---------|-------------|
| Stocks per timestamp | 10 | 30+ |
| Training timestamps | 50 | 252+ (1 year daily) |
| Test timestamps | 20 | 63+ (1 quarter daily) |

---

## Testing Strategy

### Unit Tests

> **AI Agent Note**: Write these tests BEFORE implementing the corresponding module. Tests are your specification.

| Test File | What It Tests | Priority |
|-----------|---------------|----------|
| `tests/test_target_builder.py` | Forward return calculation | P0 (First) |
| `tests/test_ranking_metrics.py` | IC, RankIC, quintile calculations | P0 |
| `tests/test_ranking_model.py` | LGBMRanker wrapper fit/predict | P0 |
| `tests/test_portfolio.py` | Portfolio construction, weights, costs | P1 |
| `tests/test_visualization.py` | Plot generation (smoke test) | P2 |

**Test Data Strategy**:
- Create `tests/fixtures/synthetic_data.py` with helper functions to generate test DataFrames.
- Use deterministic random seeds for reproducibility.
- Include edge cases: single stock, all same returns, missing data.

### Integration Tests

| Test | Purpose |
|------|---------|
| `tests/test_ranking_pipeline_e2e.py` | Full pipeline with tiny synthetic data (~100 rows) |
| `tests/test_ranking_vs_classification.py` | Verify both pipelines can run on same data |

### Performance Benchmarks

After implementation, record baseline metrics on the full dataset:
- Pipeline runtime (target: <5 min for single window)
- Memory usage peak
- IC/ICIR values for comparison with future changes

---

## Documentation Updates

| Document | Updates Needed |
|----------|----------------|
| `python/ml-pipeline/README.md` | Add "Ranking Pipeline" section with quick start |
| `docs/RANKING_METHODOLOGY.md` | NEW: Detailed explanation of IC, ranking, portfolio construction |
| `docs/IMPLEMENTATION_PLAN_RANKING.md` | This document - keep updated as implementation progresses |
| `.github/copilot-instructions.md` | Add ranking pipeline commands and conventions |

---

## Success Criteria

### Minimum Viable Product (MVP)

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| **Pipeline Runs** | No errors on full dataset | `uv run python runnables/main_ranking.py` completes |
| **Positive IC** | Mean IC > 0.02 | Reported in output metrics |
| **Monotonic Quintiles** | Q5 return > Q1 return | Quintile bar chart shows upward slope |
| **Test Coverage** | >80% for new modules | `uv run pytest --cov` |

### Stretch Goals

| Criterion | Target | Notes |
|-----------|--------|-------|
| **ICIR > 0.5** | Annualized | Indicates consistent skill |
| **Sharpe Ratio > 0.5** | Long-short portfolio | After transaction costs |
| **Runtime** | <5 min per rolling window | For iteration speed |

### Red Flags (Stop and Investigate)

- Mean IC < 0 (model is worse than random)
- Quintile returns are not monotonic (model doesn't separate winners/losers)
- Very high turnover (>50% per rebalance) without corresponding performance
- Large discrepancy between IC and portfolio returns (implementation bug)

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Insufficient stocks per timestamp** | Medium | High (LGBMRanker fails) | Filter timestamps with <10 stocks; document limitation |
| **Lookahead bias in features** | Medium | Critical (inflated metrics) | Code review; ensure features use only past data |
| **Scaler leakage** | Medium | High (inflated metrics) | Fit scaler on train only; add validation check |
| **LGBMRanker version incompatibility** | Low | Medium | Pin lightgbm version in pyproject.toml |
| **Memory issues on full data** | Medium | Medium | Process windows sequentially; use chunked operations |
| **Transaction costs underestimated** | Medium | Medium (overstated returns) | Use conservative 10-20 bps; report gross and net returns |

### Leakage Prevention Checklist

Before running the full pipeline, verify:

- [ ] Forward returns computed using only future prices (no current-day price in calculation)
- [ ] Features use only past data (no future information)
- [ ] Scaler fitted on training data only
- [ ] Train/test split respects time ordering (no random shuffle)
- [ ] No ticker information leaks into test predictions inappropriately

---

## Timeline and Milestones

### Accelerated 4-Week Plan

| Week | Phase | Deliverables | Validation |
|------|-------|--------------|------------|
| **1** | Foundation | `target_builder.py`, `ranking_metrics.py`, `ranking_config.py` | Unit tests pass |
| **2** | Model | `ranking.py` (LGBMRanker wrapper), group builder | Model trains on synthetic data |
| **3** | Backtest | `portfolio_simulator.py`, `visualization.py` | Backtest runs, plots generated |
| **4** | Integration | `ranking_pipeline.py`, `main_ranking.py`, docs | Full pipeline runs on real data |

### Milestone Checkpoints

**End of Week 1**: 
- Can compute forward returns from wide DataFrame
- Can compute IC between two Series (predicted, actual)
- Unit tests for both pass

**End of Week 2**:
- LGBMRanker wrapper can fit/predict
- Can build `group` parameter from sorted DataFrame
- Debug script runs on 2-year subset

**End of Week 3**:
- Portfolio backtest produces Sharpe ratio
- Quintile bar chart generated
- Can compare long-short vs benchmark

**End of Week 4**:
- Full pipeline runs on complete dataset
- Results saved to `output/runs/`
- README updated with ranking pipeline instructions

---

## Appendix: Quick Reference

### Running the Ranking Pipeline

```bash
cd python/ml-pipeline

# Install dependencies (if not done)
uv sync

# Run full ranking pipeline
uv run python runnables/main_ranking.py

# Run with custom parameters
uv run python runnables/main_ranking.py --forward-days 20 --top-n 5

# Quick debug (fast, small subset)
uv run python runnables/debug_ranking.py

# Run tests
uv run pytest tests/test_ranking*.py -v
```

### Key Imports

```python
# Config
from config.ranking_config import FORWARD_RETURN_DAYS, PORTFOLIO_TOP_N

# Core
from core.target_builder import compute_forward_returns

# Model
from learner.ranking import LightGBMRankerWrapper, RankerConfig, build_group_from_timestamps

# Evaluation
from evaluation.ranking_metrics import compute_ranking_metrics, RankingMetrics
from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig

# Visualization
from evaluation.visualization import plot_quintile_returns, plot_ic_series
```

### Interpreting Results

| Metric | Good | Bad | Interpretation |
|--------|------|-----|----------------|
| Mean IC | > 0.03 | < 0 | Positive = model has skill |
| ICIR | > 0.5 | < 0.2 | Higher = more consistent |
| Quintile Spread | > 0% | < 0% | Positive = long-short works |
| Hit Rate (Top-10) | > 55% | < 50% | > 50% = better than random |
| Sharpe (after costs) | > 0.5 | < 0 | Standard risk-adjusted return |

---
