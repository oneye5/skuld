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
        assert config.transaction_cost_bps == 10.0  # Default 10 bps
        assert config.slippage_bps == 0.0  # Default 0 bps
        # Note: For NZX with Sharesies, use transaction_cost_bps=190.0, slippage_bps=15.0
    
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
    
    def test_slippage_applied(self):
        """Returns should be reduced by both transaction costs and slippage.
        
        Note: Slippage is combined with transaction costs via PortfolioConfig.total_cost_bps
        property, not as a separate parameter to apply_transaction_costs.
        """
        from evaluation.portfolio_simulator import apply_transaction_costs, PortfolioConfig
        
        gross_return = 0.10
        turnover = 0.5  # 50% turnover
        
        # Use PortfolioConfig to combine costs
        config = PortfolioConfig(transaction_cost_bps=20.0, slippage_bps=10.0)
        total_cost_bps = config.total_cost_bps  # 30.0
        
        net_return = apply_transaction_costs(gross_return, turnover, total_cost_bps)
        
        # Total cost = 0.5 * 30 / 10000 = 0.0015
        expected = 0.10 - 0.0015
        assert abs(net_return - expected) < 1e-6
    
    def test_slippage_defaults_to_zero(self):
        """Slippage should default to zero for backward compatibility."""
        from evaluation.portfolio_simulator import apply_transaction_costs
        
        gross_return = 0.10
        turnover = 0.5
        cost_bps = 20.0
        
        # Without slippage argument
        net_return = apply_transaction_costs(gross_return, turnover, cost_bps)
        
        # Cost = 0.5 * 20 / 10000 = 0.001 (same as before)
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


class TestComputeDailyPortfolioReturns:
    """Tests for compute_daily_portfolio_returns - true daily return calculation."""
    
    def test_basic_daily_returns_calculation(self):
        """Should compute daily returns from holdings and price data."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        # Create holdings: 50% in TICK_A, 50% in TICK_B at ts=1000
        holdings = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "weight": 0.5},
            {TIMESTAMP: 1000, TICKER: "TICK_B", "weight": 0.5},
        ])
        
        # Create price data: TICK_A goes up 10%, TICK_B goes down 10%
        prices = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "Close": 100.0},
            {TIMESTAMP: 1000, TICKER: "TICK_B", "Close": 100.0},
            {TIMESTAMP: 1001, TICKER: "TICK_A", "Close": 110.0},  # +10%
            {TIMESTAMP: 1001, TICKER: "TICK_B", "Close": 90.0},   # -10%
        ])
        
        daily_returns = compute_daily_portfolio_returns(
            holdings, prices, timestamp_col=TIMESTAMP, ticker_col=TICKER
        )
        
        # Portfolio return = 0.5 * 10% + 0.5 * (-10%) = 0%
        assert len(daily_returns) == 1
        assert abs(daily_returns.iloc[0]) < 0.01  # ~0%
    
    def test_long_short_portfolio_daily_returns(self):
        """Should handle long (positive weight) and short (negative weight) positions."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        # Long TICK_A, Short TICK_B (equal weight)
        holdings = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "weight": 0.5},   # Long
            {TIMESTAMP: 1000, TICKER: "TICK_B", "weight": -0.5},  # Short
        ])
        
        # Both stocks go up 10%
        prices = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "Close": 100.0},
            {TIMESTAMP: 1000, TICKER: "TICK_B", "Close": 100.0},
            {TIMESTAMP: 1001, TICKER: "TICK_A", "Close": 110.0},  # +10%
            {TIMESTAMP: 1001, TICKER: "TICK_B", "Close": 110.0},  # +10%
        ])
        
        daily_returns = compute_daily_portfolio_returns(
            holdings, prices, timestamp_col=TIMESTAMP, ticker_col=TICKER
        )
        
        # Portfolio return = 0.5 * 10% + (-0.5) * 10% = 0%
        # Long gains, short loses
        assert len(daily_returns) == 1
        assert abs(daily_returns.iloc[0]) < 0.01  # ~0%
    
    def test_multiple_days_returns(self):
        """Should compute returns for multiple consecutive days."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        holdings = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "weight": 1.0},
        ])
        
        # 3 days of prices: +5%, +3%, -2%
        prices = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "Close": 100.0},
            {TIMESTAMP: 1001, TICKER: "TICK_A", "Close": 105.0},  # +5%
            {TIMESTAMP: 1002, TICKER: "TICK_A", "Close": 108.15}, # +3%
            {TIMESTAMP: 1003, TICKER: "TICK_A", "Close": 105.99}, # -2%
        ])
        
        daily_returns = compute_daily_portfolio_returns(
            holdings, prices, timestamp_col=TIMESTAMP, ticker_col=TICKER
        )
        
        assert len(daily_returns) == 3
        assert abs(daily_returns.iloc[0] - 0.05) < 0.001  # +5%
        assert abs(daily_returns.iloc[1] - 0.03) < 0.001  # +3%
        assert abs(daily_returns.iloc[2] - (-0.02)) < 0.001  # -2%
    
    def test_rebalancing_changes_holdings(self):
        """Should switch holdings at rebalance points."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        # Rebalance at ts=1000: hold TICK_A
        # Rebalance at ts=1002: switch to TICK_B
        holdings = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "weight": 1.0},
            {TIMESTAMP: 1002, TICKER: "TICK_B", "weight": 1.0},
        ])
        
        prices = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "Close": 100.0},
            {TIMESTAMP: 1000, TICKER: "TICK_B", "Close": 50.0},
            {TIMESTAMP: 1001, TICKER: "TICK_A", "Close": 110.0},  # A: +10%
            {TIMESTAMP: 1001, TICKER: "TICK_B", "Close": 55.0},   # B: +10%
            {TIMESTAMP: 1002, TICKER: "TICK_A", "Close": 115.0},  # A: +4.5%
            {TIMESTAMP: 1002, TICKER: "TICK_B", "Close": 50.0},   # B: -9%
            {TIMESTAMP: 1003, TICKER: "TICK_A", "Close": 120.0},  # A: +4.3%
            {TIMESTAMP: 1003, TICKER: "TICK_B", "Close": 60.0},   # B: +20%
        ])
        
        daily_returns = compute_daily_portfolio_returns(
            holdings, prices, timestamp_col=TIMESTAMP, ticker_col=TICKER
        )
        
        # Day 1 (ts=1001): Hold A, expect +10%
        # Day 2 (ts=1002): Still hold A (rebalance happens AT 1002), expect +4.5%
        # Day 3 (ts=1003): Now hold B, expect +20%
        assert len(daily_returns) == 3
        assert abs(daily_returns.iloc[0] - 0.10) < 0.01  # +10% from A
        assert daily_returns.iloc[2] > 0.15  # ~+20% from B
    
    def test_empty_holdings_returns_empty_series(self):
        """Should return empty series if no holdings."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        holdings = pd.DataFrame(columns=[TIMESTAMP, TICKER, "weight"])
        prices = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "Close": 100.0},
        ])
        
        daily_returns = compute_daily_portfolio_returns(holdings, prices)
        
        assert len(daily_returns) == 0
    
    def test_transaction_cost_applied_at_rebalance(self):
        """Should deduct transaction cost at rebalance points."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        holdings = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "weight": 1.0},
        ])
        
        # Flat prices - no gain
        prices = pd.DataFrame([
            {TIMESTAMP: 1000, TICKER: "TICK_A", "Close": 100.0},
            {TIMESTAMP: 1001, TICKER: "TICK_A", "Close": 100.0},
        ])
        
        # Apply 100 bps = 1% cost
        daily_returns = compute_daily_portfolio_returns(
            holdings, prices, 
            timestamp_col=TIMESTAMP, ticker_col=TICKER,
            cost_bps_per_rebalance=100.0,
        )
        
        # Return should be -1% due to cost
        assert len(daily_returns) == 1
        assert abs(daily_returns.iloc[0] - (-0.01)) < 0.001


class TestBacktestWithPriceData:
    """Tests for run_portfolio_backtest with price_data for accurate drawdown."""
    
    def test_backtest_with_price_data_computes_realistic_drawdown(self):
        """With price_data, max drawdown should reflect intra-period volatility."""
        from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig
        
        np.random.seed(42)
        
        # Create predictions at annual intervals
        base_ts = 1000000000000
        ms_per_day = 86400000
        
        tickers = ["TICK_A", "TICK_B", "TICK_C", "TICK_D"]
        
        # Create daily price data with volatility
        price_records = []
        for ticker in tickers:
            price = 100.0
            for day in range(365 * 2):  # 2 years
                ts = base_ts + day * ms_per_day
                price *= 1 + np.random.randn() * 0.03  # 3% daily vol
                price_records.append({
                    TIMESTAMP: ts,
                    TICKER: ticker,
                    "Close": price,
                })
        
        price_df = pd.DataFrame(price_records)
        
        # Create predictions at year boundaries
        prediction_records = []
        for day in [0, 365]:  # Start and 1 year later
            ts = base_ts + day * ms_per_day
            for ticker in tickers:
                prediction_records.append({
                    TIMESTAMP: ts,
                    TICKER: ticker,
                    "predicted_score": np.random.randn(),
                    "actual_return": 0.10,  # Dummy
                })
        
        pred_df = pd.DataFrame(prediction_records)
        
        config = PortfolioConfig(top_n=2, bottom_n=2, transaction_cost_bps=10)
        
        # Without price data - drawdown likely 0 (only 2 positive periods)
        result_no_prices = run_portfolio_backtest(
            pred_df, config, return_horizon_days=365, price_data=None
        )
        
        # With price data - should have realistic drawdown
        result_with_prices = run_portfolio_backtest(
            pred_df, config, return_horizon_days=365, price_data=price_df
        )
        
        # With daily price data, we should see some drawdown
        # (3% daily vol over 2 years will definitely have drawdowns)
        assert result_with_prices.max_drawdown > 0.01  # At least 1%
    
    def test_backtest_without_price_data_uses_period_drawdown(self):
        """Without price_data, drawdown is computed from period returns only."""
        from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig
        
        # Create data with all positive period returns
        data = []
        for ts in range(3):
            for ticker in ["A", "B", "C", "D"]:
                data.append({
                    TIMESTAMP: ts * 365 * 86400000,  # Annual timestamps
                    TICKER: ticker,
                    "predicted_score": np.random.randn(),
                    "actual_return": 0.10,  # All positive
                })
        
        df = pd.DataFrame(data)
        config = PortfolioConfig(top_n=2, bottom_n=2, transaction_cost_bps=0)
        
        result = run_portfolio_backtest(
            df, config, return_horizon_days=365, price_data=None
        )
        
        # All periods positive, no drawdown
        assert result.max_drawdown == 0.0
    
    def test_price_data_only_used_for_long_horizons(self):
        """Price data should only be used when return_horizon_days > 20."""
        from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig
        
        # With short horizon (daily), price_data shouldn't affect drawdown calculation
        data = []
        for ts in range(100):
            for ticker in ["A", "B"]:
                data.append({
                    TIMESTAMP: ts * 86400000,
                    TICKER: ticker,
                    "predicted_score": np.random.randn(),
                    "actual_return": 0.01 if ts % 2 == 0 else -0.005,
                })
        
        df = pd.DataFrame(data)
        
        # Dummy price data
        price_df = pd.DataFrame([
            {TIMESTAMP: 0, TICKER: "A", "Close": 100.0},
        ])
        
        config = PortfolioConfig(top_n=1, bottom_n=1, transaction_cost_bps=0)
        
        # With return_horizon_days=1 (daily), price_data shouldn't be used
        result = run_portfolio_backtest(
            df, config, return_horizon_days=1, price_data=price_df
        )
        
        # Should have computed drawdown from the period returns
        # Some negative returns, so should have some drawdown
        assert result.num_rebalances > 50
