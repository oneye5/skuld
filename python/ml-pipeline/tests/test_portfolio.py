"""Tests for the portfolio_simulator module - backtesting long-short strategies."""

import pandas as pd
import numpy as np
import pytest

from config.columns import TIMESTAMP, TICKER


class TestPortfolioConfig:
    """Tests for PortfolioConfig dataclass."""
    
    def test_default_values(self):
        """PortfolioConfig should have sensible defaults."""
        from evaluation.portfolio_simulator import PortfolioConfig
        
        config = PortfolioConfig()
        
        assert config.top_n == 10
        assert config.bottom_n == 10
        assert config.weighting == "equal"
        assert config.transaction_cost_bps == 10.0
    
    def test_long_only_mode(self):
        """Long-only mode should disable bottom_n."""
        from evaluation.portfolio_simulator import PortfolioConfig
        
        config = PortfolioConfig(long_only=True)
        
        assert config.long_only is True


class TestSelectTopBottomStocks:
    """Tests for stock selection functions."""
    
    def test_select_top_n_stocks(self):
        """Should select stocks with highest predicted scores."""
        from evaluation.portfolio_simulator import select_top_n_stocks
        
        df = pd.DataFrame({
            TICKER: ["A", "B", "C", "D", "E"],
            "predicted_score": [0.1, 0.5, 0.3, 0.8, 0.2],
        })
        
        top_2 = select_top_n_stocks(df, n=2, score_col="predicted_score")
        
        assert len(top_2) == 2
        assert set(top_2[TICKER].tolist()) == {"B", "D"}
    
    def test_select_bottom_n_stocks(self):
        """Should select stocks with lowest predicted scores."""
        from evaluation.portfolio_simulator import select_bottom_n_stocks
        
        df = pd.DataFrame({
            TICKER: ["A", "B", "C", "D", "E"],
            "predicted_score": [0.1, 0.5, 0.3, 0.8, 0.2],
        })
        
        bottom_2 = select_bottom_n_stocks(df, n=2, score_col="predicted_score")
        
        assert len(bottom_2) == 2
        assert set(bottom_2[TICKER].tolist()) == {"A", "E"}


class TestComputePortfolioReturn:
    """Tests for portfolio return calculations."""
    
    def test_equal_weight_long_only(self):
        """Equal-weight long portfolio return = average of stock returns."""
        from evaluation.portfolio_simulator import compute_portfolio_return
        
        long_returns = pd.Series([0.10, 0.05, 0.03])  # Mean = 0.06
        short_returns = pd.Series(dtype=float)  # Empty for long-only
        
        port_return = compute_portfolio_return(
            long_returns, short_returns, weighting="equal"
        )
        
        assert abs(port_return - 0.06) < 1e-6
    
    def test_equal_weight_long_short(self):
        """Long-short return = mean(long) - mean(short)."""
        from evaluation.portfolio_simulator import compute_portfolio_return
        
        long_returns = pd.Series([0.10, 0.05])    # Mean = 0.075
        short_returns = pd.Series([-0.05, -0.03])  # Mean = -0.04
        
        port_return = compute_portfolio_return(
            long_returns, short_returns, weighting="equal"
        )
        
        # 0.075 - (-0.04) = 0.115
        assert abs(port_return - 0.115) < 1e-6
    
    def test_transaction_cost_applied(self):
        """Returns should be reduced by transaction costs."""
        from evaluation.portfolio_simulator import apply_transaction_costs
        
        gross_return = 0.10
        turnover = 0.5  # 50% turnover
        cost_bps = 20.0  # 20 bps round-trip
        
        net_return = apply_transaction_costs(gross_return, turnover, cost_bps)
        
        # Cost = 0.5 * 20 / 10000 = 0.001
        expected = 0.10 - 0.001
        assert abs(net_return - expected) < 1e-6


class TestComputeTurnover:
    """Tests for portfolio turnover calculation."""
    
    def test_no_turnover_same_holdings(self):
        """Zero turnover if holdings are identical."""
        from evaluation.portfolio_simulator import compute_turnover
        
        prev_holdings = {"A": 0.5, "B": 0.5}
        curr_holdings = {"A": 0.5, "B": 0.5}
        
        turnover = compute_turnover(prev_holdings, curr_holdings)
        
        assert turnover == 0.0
    
    def test_full_turnover_different_holdings(self):
        """100% turnover if all holdings change."""
        from evaluation.portfolio_simulator import compute_turnover
        
        prev_holdings = {"A": 0.5, "B": 0.5}
        curr_holdings = {"C": 0.5, "D": 0.5}
        
        turnover = compute_turnover(prev_holdings, curr_holdings)
        
        assert turnover == 1.0
    
    def test_partial_turnover(self):
        """50% turnover if half the holdings change."""
        from evaluation.portfolio_simulator import compute_turnover
        
        prev_holdings = {"A": 0.5, "B": 0.5}
        curr_holdings = {"A": 0.5, "C": 0.5}
        
        turnover = compute_turnover(prev_holdings, curr_holdings)
        
        assert turnover == 0.5


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""
    
    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio = mean / std * sqrt(252)."""
        from evaluation.portfolio_simulator import compute_sharpe_ratio
        
        daily_returns = pd.Series([0.001] * 100)  # Constant positive return
        
        sharpe = compute_sharpe_ratio(daily_returns, periods_per_year=252)
        
        # With constant returns, std = 0, should return inf or very high
        assert sharpe > 10 or np.isinf(sharpe)
    
    def test_max_drawdown_calculation(self):
        """Max drawdown = largest peak-to-trough decline."""
        from evaluation.portfolio_simulator import compute_max_drawdown
        
        # Cumulative returns: go up to 10%, then down to 5%, then up to 8%
        cumulative_returns = pd.Series([0.0, 0.05, 0.10, 0.07, 0.05, 0.08])
        
        max_dd = compute_max_drawdown(cumulative_returns)
        
        # Peak was 0.10, trough was 0.05, drawdown = (1.10 - 1.05) / 1.10 ≈ 0.045
        assert max_dd > 0.04  # Approximate check


class TestRunPortfolioBacktest:
    """Integration tests for full portfolio backtest."""
    
    def test_basic_backtest(self):
        """Run a simple backtest with synthetic data."""
        from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig
        
        # Create 3 timestamps with 10 stocks each
        data = []
        for ts in range(3):
            for i in range(10):
                data.append({
                    TIMESTAMP: ts,
                    TICKER: f"STOCK_{i}",
                    "predicted_score": i / 10,  # Stock 9 has highest score
                    "actual_return": (i - 5) / 100,  # Stock 9 has +4% return
                })
        
        df = pd.DataFrame(data)
        
        config = PortfolioConfig(
            top_n=2,
            bottom_n=2,
            transaction_cost_bps=0,  # No costs for simplicity
        )
        
        result = run_portfolio_backtest(df, config)
        
        assert result is not None
        assert len(result.daily_returns) == 3
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
    
    def test_backtest_long_only(self):
        """Long-only backtest should not take short positions."""
        from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig
        
        data = []
        for ts in range(3):
            for i in range(10):
                data.append({
                    TIMESTAMP: ts,
                    TICKER: f"STOCK_{i}",
                    "predicted_score": i / 10,
                    "actual_return": (i - 5) / 100,
                })
        
        df = pd.DataFrame(data)
        
        config = PortfolioConfig(top_n=2, long_only=True)
        result = run_portfolio_backtest(df, config)
        
        # Long-only should have positive returns when top stocks perform well
        assert result is not None
    
    def test_perfect_model_positive_returns(self):
        """Perfect model (predicted = actual rank) should have positive returns."""
        from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig
        
        np.random.seed(42)
        data = []
        for ts in range(20):
            returns = np.linspace(-0.05, 0.05, 10)  # -5% to +5%
            for i, ret in enumerate(returns):
                data.append({
                    TIMESTAMP: ts,
                    TICKER: f"STOCK_{i}",
                    "predicted_score": ret,  # Perfect prediction
                    "actual_return": ret,
                })
        
        df = pd.DataFrame(data)
        
        config = PortfolioConfig(
            top_n=2,
            bottom_n=2,
            transaction_cost_bps=0,
        )
        
        result = run_portfolio_backtest(df, config)
        
        # Long top 2 (+3.3% to +5%) and short bottom 2 (-5% to -3.3%)
        # Should have consistent positive returns
        assert result.total_return > 0
        assert result.sharpe_ratio > 0


class TestQuintileBacktest:
    """Tests for quintile-based backtest analysis."""
    
    def test_compute_quintile_portfolio_returns(self):
        """Compute returns for each quintile portfolio."""
        from evaluation.portfolio_simulator import compute_quintile_portfolio_returns
        
        # Create data where higher quintiles have higher returns
        data = []
        for ts in range(5):
            for i in range(20):
                data.append({
                    TIMESTAMP: ts,
                    TICKER: f"STOCK_{i}",
                    "predicted_score": i,
                    "actual_return": i / 100,  # 0% to 19%
                })
        
        df = pd.DataFrame(data)
        
        quintile_returns = compute_quintile_portfolio_returns(df)
        
        # Q5 should have highest returns, Q1 lowest
        assert quintile_returns["Q5"].mean() > quintile_returns["Q1"].mean()
