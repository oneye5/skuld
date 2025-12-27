# Implementation Plan: Ranking-Based Cross-Sectional Stock Prediction

**Project**: Skuld - Time Series Forecasting Framework  
**Owner**: oneye5  
**Created**: 2025-12-27  
**Status**: Planning

## Executive Summary

This document outlines the comprehensive implementation plan for pivoting the Skuld framework from point prediction to ranking-based cross-sectional stock prediction. The new approach focuses on relative performance ranking across stocks within a universe at each time step, enabling portfolio construction and long-short strategies.

---

## Table of Contents

1. [Motivation and Objectives](#motivation-and-objectives)
2. [Architecture Overview](#architecture-overview)
3. [Implementation Phases](#implementation-phases)
4. [Technical Specifications](#technical-specifications)
5. [Testing Strategy](#testing-strategy)
6. [Documentation Updates](#documentation-updates)
7. [Success Criteria](#success-criteria)
8. [Risk Mitigation](#risk-mitigation)
9. [Timeline and Milestones](#timeline-and-milestones)

---

## Motivation and Objectives

### Why Ranking-Based Prediction?

1. **Market Neutrality**: Ranking naturally removes market-wide effects and focuses on relative performance
2. **Portfolio Construction**: Direct support for long-short strategies and quintile-based portfolios
3. **Robustness**: Rankings are more stable and less sensitive to outliers than point predictions
4. **Practical Trading**: Aligns better with real-world portfolio management needs
5. **Better Metrics**: IC, Rank IC, and information ratio are more meaningful for equity selection

### Key Objectives

- Transform prediction task from regression to ranking
- Implement cross-sectional evaluation metrics (IC, Rank IC, IR)
- Support portfolio-based backtesting and performance analysis
- Maintain backward compatibility where possible
- Provide clear migration path for existing users

---

## Architecture Overview

### Core Components

```
skuld/
├── models/
│   ├── ranking/
│   │   ├── __init__.py
│   │   ├── base.py              # Base ranking model interface
│   │   ├── pairwise.py          # Pairwise ranking models
│   │   ├── listwise.py          # Listwise ranking models
│   │   └── pointwise.py         # Pointwise-to-rank conversion
│   └── ...
├── evaluation/
│   ├── ranking_metrics.py       # IC, Rank IC, Hit Rate
│   ├── portfolio_metrics.py     # Sharpe, turnover, returns
│   └── cross_sectional.py       # Cross-sectional analysis
├── data/
│   ├── cross_sectional_loader.py # Load cross-sectional data
│   └── universe.py              # Stock universe management
├── backtesting/
│   ├── portfolio.py             # Portfolio construction
│   ├── rebalance.py             # Rebalancing strategies
│   └── transaction_costs.py     # Cost modeling
└── visualization/
    ├── ranking_plots.py         # Ranking-specific visualizations
    └── portfolio_plots.py       # Portfolio performance plots
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

#### 1.1 Data Infrastructure

**Task**: Implement cross-sectional data structures and loaders

**Files to Create/Modify**:
- `skuld/data/cross_sectional_loader.py`
- `skuld/data/universe.py`
- `skuld/data/datasets.py` (modify)

**Key Features**:
```python
class CrossSectionalDataset:
    """Dataset for cross-sectional stock data"""
    
    def __init__(self, symbols, features, targets, dates):
        """
        Parameters
        ----------
        symbols : List[str]
            List of stock symbols
        features : pd.DataFrame
            Features with MultiIndex (date, symbol)
        targets : pd.Series
            Forward returns with MultiIndex (date, symbol)
        dates : pd.DatetimeIndex
            Trading dates
        """
        self.symbols = symbols
        self.features = features
        self.targets = targets
        self.dates = dates
    
    def get_cross_section(self, date):
        """Get all stocks at a specific date"""
        return self.features.xs(date, level=0)
    
    def get_time_series(self, symbol):
        """Get time series for a specific stock"""
        return self.features.xs(symbol, level=1)
```

**Universe Management**:
```python
class Universe:
    """Manage stock universe with filtering"""
    
    def __init__(self, constituents, filters=None):
        """
        Parameters
        ----------
        constituents : Dict[date, List[str]]
            Universe constituents over time
        filters : List[callable]
            Filtering functions (liquidity, market cap, etc.)
        """
        self.constituents = constituents
        self.filters = filters or []
    
    def get_universe(self, date):
        """Get filtered universe at date"""
        stocks = self.constituents.get(date, [])
        for filter_fn in self.filters:
            stocks = filter_fn(stocks, date)
        return stocks
```

#### 1.2 Ranking Metrics

**Task**: Implement cross-sectional evaluation metrics

**File**: `skuld/evaluation/ranking_metrics.py`

**Core Metrics**:
```python
def information_coefficient(predictions, actuals, method='pearson'):
    """
    Calculate Information Coefficient
    
    Parameters
    ----------
    predictions : pd.Series
        Predicted rankings/scores with (date, symbol) index
    actuals : pd.Series
        Actual forward returns with (date, symbol) index
    method : str
        'pearson' or 'spearman'
    
    Returns
    -------
    pd.Series
        IC at each date
    """
    ic_by_date = predictions.groupby(level=0).apply(
        lambda x: x.corr(actuals.loc[x.index], method=method)
    )
    return ic_by_date

def rank_information_coefficient(predictions, actuals):
    """Calculate Rank IC (Spearman correlation)"""
    return information_coefficient(predictions, actuals, method='spearman')

def information_ratio(ic_series):
    """Calculate Information Ratio from IC series"""
    return ic_series.mean() / ic_series.std() * np.sqrt(252)

def hit_rate(predictions, actuals, top_pct=0.2):
    """
    Calculate hit rate for top predictions
    
    Parameters
    ----------
    predictions : pd.Series
        Predicted rankings
    actuals : pd.Series
        Actual returns
    top_pct : float
        Percentage of top predictions to evaluate
    
    Returns
    -------
    float
        Proportion of top predictions with positive returns
    """
    def calc_hit(group):
        pred = group[0]
        actual = group[1]
        n_top = int(len(pred) * top_pct)
        top_idx = pred.nlargest(n_top).index
        return (actual.loc[top_idx] > 0).mean()
    
    return predictions.groupby(level=0).apply(
        lambda x: calc_hit((x, actuals.loc[x.index]))
    ).mean()
```

**Additional Metrics**:
```python
def quintile_returns(predictions, actuals, n_quantiles=5):
    """Calculate returns by prediction quintile"""
    def calc_quantile_rets(group):
        pred = group[0]
        actual = group[1]
        quantiles = pd.qcut(pred, n_quantiles, labels=False)
        return actual.groupby(quantiles).mean()
    
    return predictions.groupby(level=0).apply(
        lambda x: calc_quantile_rets((x, actuals.loc[x.index]))
    ).groupby(level=1).mean()

def top_bottom_spread(predictions, actuals, top_pct=0.2):
    """Calculate return spread between top and bottom predictions"""
    quintile_rets = quintile_returns(predictions, actuals, n_quantiles=5)
    return quintile_rets.iloc[-1] - quintile_rets.iloc[0]
```

---

### Phase 2: Model Implementation (Weeks 3-4)

#### 2.1 Base Ranking Model Interface

**File**: `skuld/models/ranking/base.py`

```python
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseRankingModel(ABC):
    """Base class for ranking models"""
    
    @abstractmethod
    def fit(self, features, targets, universe=None):
        """
        Fit ranking model
        
        Parameters
        ----------
        features : pd.DataFrame
            Features with MultiIndex (date, symbol)
        targets : pd.Series
            Target returns with MultiIndex (date, symbol)
        universe : Universe, optional
            Stock universe constraints
        """
        pass
    
    @abstractmethod
    def predict_scores(self, features):
        """
        Predict ranking scores
        
        Parameters
        ----------
        features : pd.DataFrame
            Features with MultiIndex (date, symbol)
        
        Returns
        -------
        pd.Series
            Ranking scores (higher is better)
        """
        pass
    
    def predict_ranks(self, features):
        """
        Predict ranks (1 = best)
        
        Returns
        -------
        pd.Series
            Ranks at each date
        """
        scores = self.predict_scores(features)
        ranks = scores.groupby(level=0).rank(ascending=False)
        return ranks
    
    def evaluate(self, features, actuals, metrics=None):
        """
        Evaluate model performance
        
        Parameters
        ----------
        features : pd.DataFrame
            Test features
        actuals : pd.Series
            Actual returns
        metrics : List[str], optional
            Metrics to calculate
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        if metrics is None:
            metrics = ['ic', 'rank_ic', 'ir', 'hit_rate']
        
        predictions = self.predict_scores(features)
        results = {}
        
        if 'ic' in metrics:
            ic = information_coefficient(predictions, actuals, method='pearson')
            results['ic_mean'] = ic.mean()
            results['ic_std'] = ic.std()
            results['ic_series'] = ic
        
        if 'rank_ic' in metrics:
            rank_ic = rank_information_coefficient(predictions, actuals)
            results['rank_ic_mean'] = rank_ic.mean()
            results['rank_ic_std'] = rank_ic.std()
            results['rank_ic_series'] = rank_ic
        
        if 'ir' in metrics:
            ic = results.get('ic_series') or information_coefficient(predictions, actuals)
            results['ir'] = information_ratio(ic)
        
        if 'hit_rate' in metrics:
            results['hit_rate'] = hit_rate(predictions, actuals)
        
        return results
```

#### 2.2 Pointwise Ranking Models

**File**: `skuld/models/ranking/pointwise.py`

```python
from sklearn.base import BaseEstimator
import lightgbm as lgb
import xgboost as xgb

class PointwiseRanker(BaseRankingModel):
    """Convert regression model to ranking via scores"""
    
    def __init__(self, base_model, objective='regression'):
        """
        Parameters
        ----------
        base_model : estimator
            Underlying regression model
        objective : str
            Training objective ('regression', 'lambdarank')
        """
        self.base_model = base_model
        self.objective = objective
    
    def fit(self, features, targets, universe=None):
        """Fit pointwise model"""
        # Flatten cross-sectional structure for training
        X = features.reset_index(drop=True)
        y = targets.reset_index(drop=True)
        
        # Handle universe filtering if provided
        if universe is not None:
            mask = self._filter_by_universe(features.index, universe)
            X = X[mask]
            y = y[mask]
        
        self.base_model.fit(X, y)
        return self
    
    def predict_scores(self, features):
        """Predict scores using base model"""
        X = features.reset_index(drop=True)
        scores = self.base_model.predict(X)
        return pd.Series(scores, index=features.index)

class LightGBMRanker(PointwiseRanker):
    """LightGBM-based ranking model"""
    
    def __init__(self, **lgb_params):
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            **lgb_params
        }
        model = lgb.LGBMRegressor(**params)
        super().__init__(model)
```

#### 2.3 Pairwise Ranking Models

**File**: `skuld/models/ranking/pairwise.py`

```python
class PairwiseRanker(BaseRankingModel):
    """Pairwise ranking using preference learning"""
    
    def __init__(self, base_model):
        self.base_model = base_model
    
    def _create_pairs(self, features, targets, n_pairs_per_date=100):
        """Create pairwise preference data"""
        pairs_X = []
        pairs_y = []
        
        for date in features.index.get_level_values(0).unique():
            # Get cross-section
            X_date = features.xs(date, level=0)
            y_date = targets.xs(date, level=0)
            
            # Sample pairs
            n_stocks = len(X_date)
            for _ in range(min(n_pairs_per_date, n_stocks * (n_stocks - 1) // 2)):
                i, j = np.random.choice(n_stocks, 2, replace=False)
                
                # Feature difference
                x_diff = X_date.iloc[i] - X_date.iloc[j]
                pairs_X.append(x_diff)
                
                # Preference label (1 if i > j, 0 otherwise)
                pairs_y.append(1 if y_date.iloc[i] > y_date.iloc[j] else 0)
        
        return np.array(pairs_X), np.array(pairs_y)
    
    def fit(self, features, targets, universe=None):
        """Fit pairwise ranking model"""
        X_pairs, y_pairs = self._create_pairs(features, targets)
        self.base_model.fit(X_pairs, y_pairs)
        return self
    
    def predict_scores(self, features):
        """Predict scores (not directly supported, use approximation)"""
        # Use model to score each stock independently
        X = features.reset_index(drop=True)
        scores = self.base_model.predict_proba(X)[:, 1]
        return pd.Series(scores, index=features.index)
```

#### 2.4 Listwise Ranking Models

**File**: `skuld/models/ranking/listwise.py`

```python
import torch
import torch.nn as nn

class ListwiseRanker(BaseRankingModel):
    """Listwise ranking using neural networks"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        self.input_dim = input_dim
        self.model = self._build_model(input_dim, hidden_dims, dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def _build_model(self, input_dim, hidden_dims, dropout):
        """Build neural ranking model"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Output score
        return nn.Sequential(*layers)
    
    def _listwise_loss(self, scores, targets):
        """ListMLE or ListNet loss"""
        # ListMLE: Maximize likelihood of correct ranking
        # Sort by targets (descending)
        sorted_idx = torch.argsort(targets, descending=True)
        sorted_scores = scores[sorted_idx]
        
        # Calculate ListMLE loss
        loss = 0
        for i in range(len(sorted_scores) - 1):
            # Log probability of selecting item i from remaining items
            log_prob = sorted_scores[i] - torch.logsumexp(sorted_scores[i:], dim=0)
            loss -= log_prob
        
        return loss / len(sorted_scores)
    
    def fit(self, features, targets, universe=None, epochs=100, batch_size=32):
        """Fit listwise model"""
        self.model.train()
        
        dates = features.index.get_level_values(0).unique()
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            for date in dates:
                # Get cross-section
                X_date = torch.FloatTensor(features.xs(date, level=0).values)
                y_date = torch.FloatTensor(targets.xs(date, level=0).values)
                
                # Forward pass
                scores = self.model(X_date).squeeze()
                loss = self._listwise_loss(scores, y_date)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict_scores(self, features):
        """Predict ranking scores"""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features.reset_index(drop=True).values)
            scores = self.model(X).squeeze().numpy()
        return pd.Series(scores, index=features.index)
```

---

### Phase 3: Portfolio & Backtesting (Weeks 5-6)

#### 3.1 Portfolio Construction

**File**: `skuld/backtesting/portfolio.py`

```python
class Portfolio:
    """Portfolio construction from rankings"""
    
    def __init__(self, strategy='long_short', n_long=20, n_short=20, 
                 weight_method='equal'):
        """
        Parameters
        ----------
        strategy : str
            'long_short', 'long_only', 'quantile'
        n_long : int
            Number of long positions
        n_short : int
            Number of short positions
        weight_method : str
            'equal', 'score_weighted', 'risk_parity'
        """
        self.strategy = strategy
        self.n_long = n_long
        self.n_short = n_short
        self.weight_method = weight_method
    
    def construct(self, scores, date, prices=None, constraints=None):
        """
        Construct portfolio from scores
        
        Parameters
        ----------
        scores : pd.Series
            Stock scores at date
        date : datetime
            Portfolio date
        prices : pd.Series, optional
            Current prices for weighting
        constraints : dict, optional
            Position constraints
        
        Returns
        -------
        pd.Series
            Portfolio weights (sum to 1 for long, -1 for short)
        """
        if self.strategy == 'long_short':
            return self._long_short_portfolio(scores, date, prices, constraints)
        elif self.strategy == 'long_only':
            return self._long_only_portfolio(scores, date, prices, constraints)
        elif self.strategy == 'quantile':
            return self._quantile_portfolio(scores, date, prices, constraints)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _long_short_portfolio(self, scores, date, prices, constraints):
        """Construct long-short portfolio"""
        # Select top long and bottom short
        long_stocks = scores.nlargest(self.n_long)
        short_stocks = scores.nsmallest(self.n_short)
        
        # Calculate weights
        if self.weight_method == 'equal':
            long_weights = pd.Series(1.0 / self.n_long, index=long_stocks.index)
            short_weights = pd.Series(-1.0 / self.n_short, index=short_stocks.index)
        elif self.weight_method == 'score_weighted':
            long_weights = long_stocks / long_stocks.sum()
            short_weights = -short_stocks / short_stocks.sum()
        else:
            raise ValueError(f"Unknown weight method: {self.weight_method}")
        
        # Combine
        weights = pd.concat([long_weights, short_weights])
        return weights
    
    def _long_only_portfolio(self, scores, date, prices, constraints):
        """Construct long-only portfolio"""
        long_stocks = scores.nlargest(self.n_long)
        
        if self.weight_method == 'equal':
            weights = pd.Series(1.0 / self.n_long, index=long_stocks.index)
        elif self.weight_method == 'score_weighted':
            weights = long_stocks / long_stocks.sum()
        
        return weights
    
    def _quantile_portfolio(self, scores, date, prices, constraints):
        """Construct quintile portfolios"""
        # Create 5 quintile portfolios
        quantiles = pd.qcut(scores, 5, labels=False)
        portfolios = {}
        
        for q in range(5):
            mask = (quantiles == q)
            stocks = scores[mask]
            if len(stocks) > 0:
                portfolios[q] = pd.Series(1.0 / len(stocks), index=stocks.index)
        
        return portfolios
```

#### 3.2 Backtesting Engine

**File**: `skuld/backtesting/engine.py`

```python
class RankingBacktest:
    """Backtest ranking-based strategies"""
    
    def __init__(self, model, portfolio, rebalance_freq='monthly', 
                 transaction_cost=0.001):
        """
        Parameters
        ----------
        model : BaseRankingModel
            Ranking model
        portfolio : Portfolio
            Portfolio construction strategy
        rebalance_freq : str
            'daily', 'weekly', 'monthly'
        transaction_cost : float
            Transaction cost as fraction of trade value
        """
        self.model = model
        self.portfolio = portfolio
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
    
    def run(self, features, returns, prices, start_date, end_date):
        """
        Run backtest
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature data with (date, symbol) index
        returns : pd.Series
            Forward returns with (date, symbol) index
        prices : pd.Series
            Stock prices with (date, symbol) index
        start_date : datetime
            Backtest start
        end_date : datetime
            Backtest end
        
        Returns
        -------
        BacktestResults
            Backtest results object
        """
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(
            start_date, end_date, self.rebalance_freq
        )
        
        # Initialize tracking
        portfolio_weights = {}
        portfolio_returns = []
        turnover = []
        holdings_history = []
        
        prev_weights = None
        
        for date in rebalance_dates:
            # Get features and predict scores
            features_date = features.xs(date, level=0)
            scores = self.model.predict_scores(
                features_date.to_frame().T.set_index([[date] * len(features_date), features_date.index])
            )
            
            # Construct portfolio
            weights = self.portfolio.construct(
                scores.droplevel(0), date, prices.xs(date, level=0)
            )
            portfolio_weights[date] = weights
            
            # Calculate turnover
            if prev_weights is not None:
                turnover_date = self._calculate_turnover(prev_weights, weights)
                turnover.append(turnover_date)
            
            # Calculate returns until next rebalance
            next_date_idx = rebalance_dates.index(date) + 1
            if next_date_idx < len(rebalance_dates):
                next_date = rebalance_dates[next_date_idx]
                period_returns = self._calculate_period_returns(
                    weights, returns, prices, date, next_date
                )
                portfolio_returns.extend(period_returns)
            
            holdings_history.append({
                'date': date,
                'weights': weights,
                'scores': scores
            })
            
            prev_weights = weights
        
        # Create results object
        results = BacktestResults(
            returns=pd.Series(portfolio_returns),
            turnover=pd.Series(turnover),
            holdings=holdings_history,
            portfolio_weights=portfolio_weights
        )
        
        return results
    
    def _get_rebalance_dates(self, start, end, freq):
        """Get rebalancing dates"""
        dates = pd.date_range(start, end, freq='D')
        
        if freq == 'daily':
            return dates
        elif freq == 'weekly':
            return dates[dates.dayofweek == 0]  # Mondays
        elif freq == 'monthly':
            return dates[dates.is_month_start]
        else:
            raise ValueError(f"Unknown frequency: {freq}")
    
    def _calculate_turnover(self, prev_weights, curr_weights):
        """Calculate portfolio turnover"""
        # Align indices
        all_stocks = prev_weights.index.union(curr_weights.index)
        prev = prev_weights.reindex(all_stocks, fill_value=0)
        curr = curr_weights.reindex(all_stocks, fill_value=0)
        
        return (prev - curr).abs().sum() / 2
    
    def _calculate_period_returns(self, weights, returns, prices, start, end):
        """Calculate portfolio returns for period"""
        # Simplified: daily rebalancing within period
        dates = pd.date_range(start, end, freq='D')[1:]  # Exclude start
        period_returns = []
        
        for date in dates:
            if date in returns.index.get_level_values(0):
                # Get returns for holdings
                rets = returns.xs(date, level=0)
                # Calculate weighted return
                port_ret = (weights * rets.reindex(weights.index, fill_value=0)).sum()
                # Subtract transaction costs
                port_ret -= self.transaction_cost * self._calculate_turnover({}, weights)
                period_returns.append(port_ret)
        
        return period_returns

class BacktestResults:
    """Container for backtest results"""
    
    def __init__(self, returns, turnover, holdings, portfolio_weights):
        self.returns = returns
        self.turnover = turnover
        self.holdings = holdings
        self.portfolio_weights = portfolio_weights
    
    def summary(self):
        """Generate performance summary"""
        return {
            'total_return': (1 + self.returns).prod() - 1,
            'annualized_return': self.returns.mean() * 252,
            'annualized_volatility': self.returns.std() * np.sqrt(252),
            'sharpe_ratio': self.returns.mean() / self.returns.std() * np.sqrt(252),
            'max_drawdown': self._max_drawdown(),
            'avg_turnover': self.turnover.mean(),
            'win_rate': (self.returns > 0).mean()
        }
    
    def _max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def plot(self):
        """Plot backtest results"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Cumulative returns
        cumulative = (1 + self.returns).cumprod()
        axes[0].plot(cumulative)
        axes[0].set_title('Cumulative Returns')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].grid(True)
        
        # Rolling Sharpe
        rolling_sharpe = (
            self.returns.rolling(60).mean() / 
            self.returns.rolling(60).std() * 
            np.sqrt(252)
        )
        axes[1].plot(rolling_sharpe)
        axes[1].set_title('Rolling 60-Day Sharpe Ratio')
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].grid(True)
        
        # Turnover
        axes[2].plot(self.turnover)
        axes[2].set_title('Portfolio Turnover')
        axes[2].set_ylabel('Turnover')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig
```

---

### Phase 4: Visualization & Analysis (Week 7)

#### 4.1 Ranking Visualizations

**File**: `skuld/visualization/ranking_plots.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ic_decay(ic_series, max_lag=20):
    """Plot IC decay over time lags"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags = range(1, max_lag + 1)
    ic_by_lag = [ic_series.shift(lag).corr(ic_series) for lag in lags]
    
    ax.plot(lags, ic_by_lag, marker='o')
    ax.set_xlabel('Lag (days)')
    ax.set_ylabel('IC Autocorrelation')
    ax.set_title('Information Coefficient Decay')
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    return fig

def plot_ic_distribution(ic_series):
    """Plot IC distribution and time series"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series
    axes[0].plot(ic_series.index, ic_series.values, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].axhline(y=ic_series.mean(), color='g', linestyle='--', 
                    label=f'Mean: {ic_series.mean():.4f}')
    axes[0].set_title('IC Time Series')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('IC')
    axes[0].legend()
    axes[0].grid(True)
    
    # Distribution
    axes[1].hist(ic_series, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1].axvline(x=ic_series.mean(), color='g', linestyle='--',
                    label=f'Mean: {ic_series.mean():.4f}')
    axes[1].set_title('IC Distribution')
    axes[1].set_xlabel('IC')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_quintile_performance(predictions, actuals, n_quantiles=5):
    """Plot performance by prediction quintile"""
    def calc_quantile_rets(group):
        pred = group[0]
        actual = group[1]
        quantiles = pd.qcut(pred, n_quantiles, labels=False, duplicates='drop')
        return actual.groupby(quantiles).mean()
    
    quintile_rets = predictions.groupby(level=0).apply(
        lambda x: calc_quantile_rets((x, actuals.loc[x.index]))
    )
    
    mean_rets = quintile_rets.groupby(level=1).mean()
    std_rets = quintile_rets.groupby(level=1).std()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(n_quantiles)
    ax.bar(x, mean_rets.values, yerr=std_rets.values, capsize=5, alpha=0.7)
    ax.set_xlabel('Quintile (0=Lowest Prediction, 4=Highest)')
    ax.set_ylabel('Average Return')
    ax.set_title('Average Returns by Prediction Quintile')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_top_bottom_spread(predictions, actuals, window=60):
    """Plot rolling spread between top and bottom predictions"""
    def calc_spread(group):
        pred = group[0]
        actual = group[1]
        top = actual[pred >= pred.quantile(0.8)].mean()
        bottom = actual[pred <= pred.quantile(0.2)].mean()
        return top - bottom
    
    spread = predictions.groupby(level=0).apply(
        lambda x: calc_spread((x, actuals.loc[x.index]))
    )
    
    rolling_spread = spread.rolling(window).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(spread.index, spread.values, alpha=0.3, label='Daily')
    ax.plot(rolling_spread.index, rolling_spread.values, linewidth=2, 
            label=f'{window}-day MA')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Return Spread')
    ax.set_title('Top 20% vs Bottom 20% Return Spread')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_feature_importance_by_ic(model, features, actuals):
    """Plot feature importance based on individual feature IC"""
    feature_ics = {}
    
    for col in features.columns:
        ic = information_coefficient(features[col], actuals)
        feature_ics[col] = ic.mean()
    
    feature_ics = pd.Series(feature_ics).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_ics) * 0.3)))
    
    colors = ['g' if x > 0 else 'r' for x in feature_ics.values]
    ax.barh(range(len(feature_ics)), feature_ics.values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(feature_ics)))
    ax.set_yticklabels(feature_ics.index)
    ax.set_xlabel('Average IC')
    ax.set_title('Feature Importance by Information Coefficient')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    
    return fig
```

---

### Phase 5: Integration & Testing (Week 8)

#### 5.1 End-to-End Example

**File**: `examples/ranking_stock_prediction.py`

```python
"""
Complete example of ranking-based stock prediction
"""

import pandas as pd
import numpy as np
from skuld.data.cross_sectional_loader import CrossSectionalDataset
from skuld.data.universe import Universe
from skuld.models.ranking import LightGBMRanker
from skuld.backtesting import Portfolio, RankingBacktest
from skuld.evaluation.ranking_metrics import *
from skuld.visualization.ranking_plots import *

# 1. Load data
print("Loading data...")
features = pd.read_parquet('data/features.parquet')  # (date, symbol) index
returns = pd.read_parquet('data/returns.parquet')    # (date, symbol) index
prices = pd.read_parquet('data/prices.parquet')      # (date, symbol) index

# 2. Define universe
print("Setting up universe...")
universe = Universe(
    constituents=get_sp500_constituents(),  # Custom function
    filters=[
        lambda stocks, date: filter_by_liquidity(stocks, date, min_volume=1e6),
        lambda stocks, date: filter_by_market_cap(stocks, date, min_cap=1e9)
    ]
)

# 3. Create dataset
dataset = CrossSectionalDataset(
    symbols=features.index.get_level_values(1).unique(),
    features=features,
    targets=returns,
    dates=features.index.get_level_values(0).unique()
)

# 4. Split data
train_end = '2020-12-31'
test_start = '2021-01-01'
test_end = '2023-12-31'

train_features = features.loc[:train_end]
train_returns = returns.loc[:train_end]
test_features = features.loc[test_start:test_end]
test_returns = returns.loc[test_start:test_end]

# 5. Train ranking model
print("Training model...")
model = LightGBMRanker(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    objective='regression'
)

model.fit(train_features, train_returns, universe=universe)

# 6. Evaluate model
print("Evaluating model...")
eval_results = model.evaluate(
    test_features, 
    test_returns,
    metrics=['ic', 'rank_ic', 'ir', 'hit_rate']
)

print(f"\nModel Performance:")
print(f"  IC Mean: {eval_results['ic_mean']:.4f}")
print(f"  IC Std: {eval_results['ic_std']:.4f}")
print(f"  Rank IC Mean: {eval_results['rank_ic_mean']:.4f}")
print(f"  Information Ratio: {eval_results['ir']:.4f}")
print(f"  Hit Rate: {eval_results['hit_rate']:.4f}")

# 7. Visualize IC
fig_ic = plot_ic_distribution(eval_results['ic_series'])
fig_ic.savefig('output/ic_distribution.png')

fig_quintile = plot_quintile_performance(
    model.predict_scores(test_features),
    test_returns
)
fig_quintile.savefig('output/quintile_performance.png')

# 8. Run backtest
print("\nRunning backtest...")
portfolio = Portfolio(
    strategy='long_short',
    n_long=20,
    n_short=20,
    weight_method='equal'
)

backtest = RankingBacktest(
    model=model,
    portfolio=portfolio,
    rebalance_freq='monthly',
    transaction_cost=0.001
)

results = backtest.run(
    features=test_features,
    returns=test_returns,
    prices=prices.loc[test_start:test_end],
    start_date=test_start,
    end_date=test_end
)

# 9. Analyze results
print("\nBacktest Results:")
summary = results.summary()
for key, value in summary.items():
    print(f"  {key}: {value:.4f}")

# 10. Visualize backtest
fig_backtest = results.plot()
fig_backtest.savefig('output/backtest_results.png')

print("\nDone! Results saved to output/")
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_ranking_metrics.py`

```python
import pytest
import pandas as pd
import numpy as np
from skuld.evaluation.ranking_metrics import *

def test_information_coefficient():
    """Test IC calculation"""
    dates = pd.date_range('2020-01-01', periods=10)
    symbols = ['A', 'B', 'C', 'D', 'E']
    
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # Perfect positive correlation
    predictions = pd.Series(range(len(index)), index=index)
    actuals = pd.Series(range(len(index)), index=index)
    
    ic = information_coefficient(predictions, actuals, method='pearson')
    assert ic.mean() == pytest.approx(1.0, abs=1e-10)
    
    # Perfect negative correlation
    actuals_neg = pd.Series(range(len(index))[::-1], index=index)
    ic_neg = information_coefficient(predictions, actuals_neg, method='pearson')
    assert ic_neg.mean() == pytest.approx(-1.0, abs=1e-10)

def test_rank_ic():
    """Test Rank IC calculation"""
    dates = pd.date_range('2020-01-01', periods=5)
    symbols = ['A', 'B', 'C']
    
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # Monotonic predictions and actuals
    predictions = pd.Series([1, 2, 3] * 5, index=index)
    actuals = pd.Series([0.1, 0.2, 0.3] * 5, index=index)
    
    rank_ic = rank_information_coefficient(predictions, actuals)
    assert rank_ic.mean() == pytest.approx(1.0, abs=1e-10)

def test_information_ratio():
    """Test IR calculation"""
    ic_series = pd.Series([0.05, 0.03, 0.08, -0.02, 0.06])
    ir = information_ratio(ic_series)
    
    expected_ir = ic_series.mean() / ic_series.std() * np.sqrt(252)
    assert ir == pytest.approx(expected_ir, abs=1e-10)

def test_hit_rate():
    """Test hit rate calculation"""
    dates = pd.date_range('2020-01-01', periods=3)
    symbols = ['A', 'B', 'C', 'D', 'E']
    
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # Top predictions have positive returns
    predictions = pd.Series([5, 4, 3, 2, 1] * 3, index=index)
    actuals = pd.Series([0.1, 0.05, 0.0, -0.05, -0.1] * 3, index=index)
    
    hr = hit_rate(predictions, actuals, top_pct=0.2)
    assert hr == 1.0  # Top 20% (1 stock) always has positive return
```

**File**: `tests/test_ranking_models.py`

```python
import pytest
import pandas as pd
import numpy as np
from skuld.models.ranking import LightGBMRanker, PointwiseRanker

@pytest.fixture
def sample_data():
    """Create sample cross-sectional data"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100)
    symbols = [f'STOCK_{i}' for i in range(50)]
    
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # Random features
    features = pd.DataFrame(
        np.random.randn(len(index), 10),
        index=index,
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    # Returns correlated with features
    returns = (features.sum(axis=1) * 0.01 + np.random.randn(len(index)) * 0.02)
    
    return features, returns

def test_lightgbm_ranker(sample_data):
    """Test LightGBM ranking model"""
    features, returns = sample_data
    
    # Split data
    train_size = 80
    train_dates = features.index.get_level_values(0).unique()[:train_size]
    test_dates = features.index.get_level_values(0).unique()[train_size:]
    
    train_features = features.loc[train_dates]
    train_returns = returns.loc[train_dates]
    test_features = features.loc[test_dates]
    test_returns = returns.loc[test_dates]
    
    # Train model
    model = LightGBMRanker(n_estimators=10, num_leaves=15)
    model.fit(train_features, train_returns)
    
    # Predict
    predictions = model.predict_scores(test_features)
    
    assert len(predictions) == len(test_features)
    assert predictions.index.equals(test_features.index)
    
    # Check IC is positive (model learned something)
    ic = information_coefficient(predictions, test_returns)
    assert ic.mean() > 0

def test_ranking_evaluation(sample_data):
    """Test model evaluation"""
    features, returns = sample_data
    
    train_dates = features.index.get_level_values(0).unique()[:80]
    test_dates = features.index.get_level_values(0).unique()[80:]
    
    train_features = features.loc[train_dates]
    train_returns = returns.loc[train_dates]
    test_features = features.loc[test_dates]
    test_returns = returns.loc[test_dates]
    
    model = LightGBMRanker(n_estimators=10)
    model.fit(train_features, train_returns)
    
    # Evaluate
    results = model.evaluate(test_features, test_returns)
    
    assert 'ic_mean' in results
    assert 'rank_ic_mean' in results
    assert 'ir' in results
    assert 'hit_rate' in results
    
    assert -1 <= results['ic_mean'] <= 1
    assert -1 <= results['rank_ic_mean'] <= 1
    assert 0 <= results['hit_rate'] <= 1
```

**File**: `tests/test_portfolio.py`

```python
import pytest
import pandas as pd
import numpy as np
from skuld.backtesting import Portfolio

def test_long_short_portfolio():
    """Test long-short portfolio construction"""
    scores = pd.Series(
        [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
        index=['A', 'B', 'C', 'D', 'E', 'F']
    )
    
    portfolio = Portfolio(strategy='long_short', n_long=2, n_short=2)
    weights = portfolio.construct(scores, pd.Timestamp('2020-01-01'))
    
    # Check long positions
    assert weights['A'] > 0
    assert weights['B'] > 0
    
    # Check short positions
    assert weights['E'] < 0
    assert weights['F'] < 0
    
    # Check weights sum correctly
    assert abs(weights[weights > 0].sum() - 1.0) < 1e-10
    assert abs(weights[weights < 0].sum() + 1.0) < 1e-10

def test_long_only_portfolio():
    """Test long-only portfolio construction"""
    scores = pd.Series(
        [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
        index=['A', 'B', 'C', 'D', 'E', 'F']
    )
    
    portfolio = Portfolio(strategy='long_only', n_long=3)
    weights = portfolio.construct(scores, pd.Timestamp('2020-01-01'))
    
    # Check only long positions
    assert (weights >= 0).all()
    
    # Check weights sum to 1
    assert abs(weights.sum() - 1.0) < 1e-10
    
    # Check top 3 stocks selected
    assert len(weights) == 3
    assert 'A' in weights.index
    assert 'B' in weights.index
    assert 'C' in weights.index
```

### Integration Tests

**File**: `tests/integration/test_end_to_end.py`

```python
import pytest
import pandas as pd
import numpy as np
from skuld.models.ranking import LightGBMRanker
from skuld.backtesting import Portfolio, RankingBacktest

@pytest.fixture
def synthetic_market_data():
    """Create synthetic market data"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252)  # 1 year
    symbols = [f'STOCK_{i}' for i in range(100)]
    
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # Features with some predictive power
    features = pd.DataFrame(
        np.random.randn(len(index), 20),
        index=index,
        columns=[f'feature_{i}' for i in range(20)]
    )
    
    # Returns correlated with features
    true_signal = features.iloc[:, :5].mean(axis=1) * 0.01
    noise = pd.Series(np.random.randn(len(index)) * 0.03, index=index)
    returns = true_signal + noise
    
    # Prices (cumulative returns)
    prices = returns.groupby(level=1).apply(lambda x: (1 + x).cumprod() * 100)
    
    return features, returns, prices

def test_full_workflow(synthetic_market_data):
    """Test complete workflow from training to backtest"""
    features, returns, prices = synthetic_market_data
    
    # Split data
    train_end = '2020-09-30'
    test_start = '2020-10-01'
    
    train_features = features.loc[:train_end]
    train_returns = returns.loc[:train_end]
    test_features = features.loc[test_start:]
    test_returns = returns.loc[test_start:]
    test_prices = prices.loc[test_start:]
    
    # Train model
    model = LightGBMRanker(n_estimators=20, num_leaves=15, learning_rate=0.1)
    model.fit(train_features, train_returns)
    
    # Evaluate
    eval_results = model.evaluate(test_features, test_returns)
    assert eval_results['ic_mean'] > 0  # Model has predictive power
    
    # Backtest
    portfolio = Portfolio(strategy='long_short', n_long=10, n_short=10)
    backtest = RankingBacktest(
        model=model,
        portfolio=portfolio,
        rebalance_freq='monthly',
        transaction_cost=0.001
    )
    
    results = backtest.run(
        features=test_features,
        returns=test_returns,
        prices=test_prices,
        start_date=pd.Timestamp(test_start),
        end_date=test_features.index.get_level_values(0).max()
    )
    
    # Check results
    summary = results.summary()
    assert 'sharpe_ratio' in summary
    assert 'total_return' in summary
    assert len(results.returns) > 0
    assert len(results.holdings) > 0
```

---

## Documentation Updates

### README.md Updates

Add the following section:

```markdown
## Ranking-Based Stock Prediction

Skuld now supports ranking-based cross-sectional stock prediction, designed for portfolio construction and long-short equity strategies.

### Quick Start

```python
from skuld.models.ranking import LightGBMRanker
from skuld.backtesting import Portfolio, RankingBacktest
from skuld.evaluation.ranking_metrics import information_coefficient

# Load cross-sectional data
features = pd.read_parquet('data/features.parquet')  # (date, symbol) index
returns = pd.read_parquet('data/returns.parquet')

# Train ranking model
model = LightGBMRanker(num_leaves=31, learning_rate=0.05)
model.fit(features, returns)

# Evaluate with IC
predictions = model.predict_scores(features)
ic = information_coefficient(predictions, returns)
print(f"Mean IC: {ic.mean():.4f}")

# Backtest portfolio strategy
portfolio = Portfolio(strategy='long_short', n_long=20, n_short=20)
backtest = RankingBacktest(model, portfolio, rebalance_freq='monthly')
results = backtest.run(features, returns, prices, start_date, end_date)

print(results.summary())
```

### Key Features

- **Cross-Sectional Models**: Ranking models optimized for relative performance prediction
- **IC Metrics**: Information Coefficient, Rank IC, and Information Ratio
- **Portfolio Construction**: Long-short, long-only, and quintile-based strategies
- **Comprehensive Backtesting**: Transaction costs, turnover, and realistic rebalancing
- **Rich Visualizations**: IC analysis, quintile performance, and portfolio analytics

### Documentation

- [Ranking Models Guide](docs/ranking_models.md)
- [Cross-Sectional Metrics](docs/ranking_metrics.md)
- [Portfolio Backtesting](docs/backtesting.md)
- [Examples](examples/ranking_stock_prediction.py)
```

### New Documentation Files

Create the following new documentation files:

1. **docs/ranking_models.md**: Complete guide to ranking models
2. **docs/ranking_metrics.md**: Explanation of IC, Rank IC, IR, and other metrics
3. **docs/backtesting.md**: Portfolio construction and backtesting guide
4. **docs/migration_guide.md**: Guide for migrating from point prediction to ranking

---

## Success Criteria

### Technical Criteria

1. **Model Performance**
   - Mean IC > 0.03 on validation data
   - Information Ratio > 1.0
   - Hit rate > 55% for top quintile

2. **Code Quality**
   - 90%+ test coverage for new modules
   - All tests passing in CI/CD
   - Code passes linting (flake8, black)
   - Comprehensive docstrings

3. **Performance**
   - Training time < 10 minutes for 100k samples
   - Prediction time < 1 second for 1000 stocks
   - Backtest runs at > 100 days/second

### Functional Criteria

1. **Features Complete**
   - All ranking models implemented
   - Portfolio construction working
   - Backtesting engine functional
   - Visualizations available

2. **Documentation**
   - All modules documented
   - 3+ complete examples
   - Migration guide available
   - API reference generated

3. **User Experience**
   - Simple API for common use cases
   - Clear error messages
   - Backward compatible where possible
   - Easy installation and setup

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Data loading performance | High | Medium | Implement efficient data structures, use Parquet |
| Model training time | Medium | High | Use incremental learning, optimize hyperparameters |
| Memory usage with large universes | High | Medium | Implement batch processing, lazy loading |
| Backtest accuracy | High | Low | Extensive validation, compare with known benchmarks |

### Implementation Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking changes to existing API | High | Medium | Maintain backward compatibility, versioning |
| Incomplete testing | Medium | Medium | Set coverage targets, automated testing |
| Poor documentation | Medium | High | Documentation sprints, peer review |
| Scope creep | Medium | Medium | Strict phase boundaries, MVP first |

---

## Timeline and Milestones

### Week 1-2: Foundation
- ✅ Cross-sectional data structures
- ✅ Universe management
- ✅ Basic ranking metrics (IC, Rank IC, IR)
- ✅ Unit tests for metrics

### Week 3-4: Models
- ✅ Base ranking model interface
- ✅ Pointwise rankers (LightGBM, XGBoost)
- ✅ Pairwise rankers
- ✅ Listwise rankers (neural)
- ✅ Model evaluation framework
- ✅ Model unit tests

### Week 5-6: Portfolio & Backtesting
- ✅ Portfolio construction strategies
- ✅ Backtesting engine
- ✅ Transaction cost modeling
- ✅ Performance metrics
- ✅ Integration tests

### Week 7: Visualization & Analysis
- ✅ IC visualizations
- ✅ Quintile analysis plots
- ✅ Portfolio performance plots
- ✅ Feature importance plots

### Week 8: Integration & Polish
- ✅ End-to-end examples
- ✅ Complete documentation
- ✅ Performance optimization
- ✅ Release preparation

### Post-Release
- Gather user feedback
- Address bugs and issues
- Plan additional features (e.g., ensemble ranking, advanced portfolio optimization)

---

## Appendix

### A. Example Data Format

**Features DataFrame**:
```
                          feature_0  feature_1  feature_2  ...
date       symbol                                          
2020-01-01 AAPL              0.523      1.234     -0.456  ...
           MSFT              1.234     -0.789      0.123  ...
           GOOGL            -0.456      0.567      1.789  ...
2020-01-02 AAPL              0.678      1.456     -0.234  ...
           ...                ...        ...        ...  ...
```

**Returns Series**:
```
date       symbol
2020-01-01 AAPL     0.012
           MSFT     0.008
           GOOGL   -0.003
2020-01-02 AAPL     0.015
           ...       ...
```

### B. Configuration Example

**config.yaml**:
```yaml
model:
  type: lightgbm_ranker
  params:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 100
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5

universe:
  name: sp500
  filters:
    - min_market_cap: 1e9
    - min_dollar_volume: 1e6
    - exclude_sectors: []

portfolio:
  strategy: long_short
  n_long: 20
  n_short: 20
  weight_method: equal
  rebalance_freq: monthly

backtest:
  start_date: "2021-01-01"
  end_date: "2023-12-31"
  transaction_cost: 0.001
  slippage: 0.0005
```

### C. Reference Papers

1. **Learning to Rank**:
   - Liu, T.Y. (2009). "Learning to Rank for Information Retrieval"
   - Burges et al. (2005). "Learning to Rank using Gradient Descent"

2. **Cross-Sectional Prediction**:
   - Gu, Kelly, Xiu (2020). "Empirical Asset Pricing via Machine Learning"
   - Chen & Zimmermann (2022). "Open Source Cross-Sectional Asset Pricing"

3. **Portfolio Construction**:
   - Brandt et al. (2009). "Parametric Portfolio Policies"
   - DeMiguel, Garlappi, Uppal (2009). "Optimal Versus Naive Diversification"

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-27 | 0.1.0 | Initial implementation plan created |

---

## Contributors

- oneye5 (Project Lead)

---

## License

This implementation plan is part of the Skuld project and follows the same license.

---

*End of Implementation Plan*
