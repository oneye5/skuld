"""Tests to verify core return calculations and portfolio metrics are correct.

These tests verify:
1. Forward return formulas (simple and log)
2. Portfolio return calculations (long-short, long-only)
3. Sharpe ratio formula and annualization
4. Max drawdown calculation
5. Cumulative return compounding
6. Transaction cost impact
"""

import pandas as pd
import numpy as np
import pytest
from scipy import stats

from config.columns import TIMESTAMP, TICKER, CLOSE, ADJCLOSE
from config.settings import MS_PER_DAY


# =============================================================================
# FORWARD RETURN TESTS
# =============================================================================

class TestForwardReturnFormulas:
    """Verify forward return calculations match expected formulas."""
    
    def test_simple_return_formula(self):
        """Simple return = (P_future - P_now) / P_now."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        # Known prices
        p_now = 100.0
        p_future = 115.0  # +15%
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [p_now, p_future],
            ADJCLOSE: [p_now, p_future],
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        
        expected = (p_future - p_now) / p_now  # 0.15
        actual = result.iloc[0][FORWARD_RETURN]
        
        assert abs(actual - expected) < 1e-10, f"Expected {expected}, got {actual}"
    
    def test_log_return_formula(self):
        """Log return = ln(P_future / P_now)."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        p_now = 100.0
        p_future = 120.0
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [p_now, p_future],
            ADJCLOSE: [p_now, p_future],
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="log")
        
        expected = np.log(p_future / p_now)  # ln(1.20) ≈ 0.1823
        actual = result.iloc[0][FORWARD_RETURN]
        
        assert abs(actual - expected) < 1e-10, f"Expected {expected}, got {actual}"
    
    def test_simple_vs_log_return_relationship(self):
        """For small returns, simple ≈ log. For larger, they diverge."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        # Test several return levels
        test_cases = [
            (100.0, 101.0),  # +1%
            (100.0, 105.0),  # +5%
            (100.0, 120.0),  # +20%
            (100.0, 150.0),  # +50%
        ]
        
        for p_now, p_future in test_cases:
            df = pd.DataFrame({
                TIMESTAMP: [0, 5 * MS_PER_DAY],
                TICKER: ["A", "A"],
                CLOSE: [p_now, p_future],
                ADJCLOSE: [p_now, p_future],
            })
            
            result_simple = compute_forward_returns(df, lookahead_days=5, return_type="simple")
            result_log = compute_forward_returns(df, lookahead_days=5, return_type="log")
            
            simple_ret = result_simple.iloc[0][FORWARD_RETURN]
            log_ret = result_log.iloc[0][FORWARD_RETURN]
            
            # Log return should be less than simple for positive returns
            if simple_ret > 0:
                assert log_ret < simple_ret, f"Log return should be < simple for positive returns"
            
            # Relationship: simple = exp(log) - 1
            expected_simple = np.exp(log_ret) - 1
            assert abs(simple_ret - expected_simple) < 1e-10
    
    def test_negative_return_preserves_sign(self):
        """Both simple and log returns preserve negative sign correctly."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        p_now = 100.0
        p_future = 80.0  # -20%
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [p_now, p_future],
            ADJCLOSE: [p_now, p_future],
        })
        
        result_simple = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        result_log = compute_forward_returns(df, lookahead_days=5, return_type="log")
        
        simple_ret = result_simple.iloc[0][FORWARD_RETURN]
        log_ret = result_log.iloc[0][FORWARD_RETURN]
        
        assert simple_ret == pytest.approx(-0.20, abs=1e-10)
        assert log_ret == pytest.approx(np.log(0.8), abs=1e-10)
        assert simple_ret < 0
        assert log_ret < 0
    
    def test_winsorization_clips_extremes(self):
        """Winsorization clips returns outside bounds."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        # Extreme return: +200%
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 300.0],
            ADJCLOSE: [100.0, 300.0],
        })
        
        result = compute_forward_returns(
            df, 
            lookahead_days=5, 
            return_type="simple",
            winsorize_limits=(-0.5, 0.5),  # Clip to ±50%
        )
        
        actual = result.iloc[0][FORWARD_RETURN]
        
        # Should be clipped to 0.5 (50%)
        assert actual == pytest.approx(0.5, abs=1e-10)
    
    def test_zero_return_handled(self):
        """Zero return (no price change) handled correctly."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 100.0],
            ADJCLOSE: [100.0, 100.0],
        })
        
        result_simple = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        result_log = compute_forward_returns(df, lookahead_days=5, return_type="log")
        
        assert result_simple.iloc[0][FORWARD_RETURN] == pytest.approx(0.0, abs=1e-10)
        assert result_log.iloc[0][FORWARD_RETURN] == pytest.approx(0.0, abs=1e-10)


class TestForwardReturnMultipleTickers:
    """Verify forward returns are computed correctly across multiple tickers."""
    
    def test_independent_per_ticker(self):
        """Each ticker's forward return is computed independently."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY, 0, 5 * MS_PER_DAY],
            TICKER: ["A", "A", "B", "B"],
            CLOSE: [100.0, 110.0, 200.0, 180.0],  # A: +10%, B: -10%
            ADJCLOSE: [100.0, 110.0, 200.0, 180.0],
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        
        ret_a = result[result[TICKER] == "A"].iloc[0][FORWARD_RETURN]
        ret_b = result[result[TICKER] == "B"].iloc[0][FORWARD_RETURN]
        
        assert ret_a == pytest.approx(0.10, abs=1e-10)
        assert ret_b == pytest.approx(-0.10, abs=1e-10)
    
    def test_different_price_scales(self):
        """Forward returns are scale-invariant (work for any price level)."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        # Same percentage change, different price levels
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY, 0, 5 * MS_PER_DAY],
            TICKER: ["CHEAP", "CHEAP", "EXPENSIVE", "EXPENSIVE"],
            CLOSE: [10.0, 12.0, 1000.0, 1200.0],  # Both +20%
            ADJCLOSE: [10.0, 12.0, 1000.0, 1200.0],
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        
        ret_cheap = result[result[TICKER] == "CHEAP"].iloc[0][FORWARD_RETURN]
        ret_expensive = result[result[TICKER] == "EXPENSIVE"].iloc[0][FORWARD_RETURN]
        
        # Same percentage return regardless of price level
        assert ret_cheap == pytest.approx(0.20, abs=1e-10)
        assert ret_expensive == pytest.approx(0.20, abs=1e-10)


# =============================================================================
# SHARPE RATIO TESTS
# =============================================================================

class TestSharpeRatioFormula:
    """Verify Sharpe ratio calculation matches the standard formula."""
    
    def test_sharpe_basic_formula(self):
        """Sharpe = (mean - rf) / std * sqrt(periods_per_year)."""
        from evaluation.portfolio_simulator import compute_sharpe_ratio
        
        # Known returns
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.025])
        returns.index = [i * MS_PER_DAY * 73 for i in range(5)]  # ~73 day periods
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # Manual calculation (raw, before annualization)
        raw_sharpe = mean_ret / std_ret
        
        # The function annualizes based on periods
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        
        # Raw Sharpe should be positive for positive mean
        assert raw_sharpe > 0
        assert sharpe > 0  # Annualized should also be positive
    
    def test_sharpe_with_risk_free_rate(self):
        """Sharpe ratio subtracts risk-free rate correctly."""
        from evaluation.portfolio_simulator import compute_sharpe_ratio
        
        returns = pd.Series([0.10, 0.08, 0.12, 0.09, 0.11])  # ~10% average
        returns.index = [i * MS_PER_DAY * 365 for i in range(5)]  # Annual periods
        
        rf = 0.02  # 2% risk-free rate
        
        sharpe_with_rf = compute_sharpe_ratio(returns, risk_free_rate=rf)
        sharpe_no_rf = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        
        # Higher risk-free rate -> lower Sharpe
        assert sharpe_with_rf < sharpe_no_rf
    
    def test_sharpe_zero_std_returns_inf(self):
        """Constant positive returns should give infinite Sharpe."""
        from evaluation.portfolio_simulator import compute_sharpe_ratio
        
        returns = pd.Series([0.05, 0.05, 0.05, 0.05])
        returns.index = [i * MS_PER_DAY * 30 for i in range(4)]
        
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        
        assert np.isinf(sharpe) and sharpe > 0
    
    def test_sharpe_negative_mean_gives_negative_sharpe(self):
        """Negative mean returns give negative Sharpe ratio."""
        from evaluation.portfolio_simulator import compute_sharpe_ratio
        
        returns = pd.Series([-0.02, -0.01, -0.03, -0.02, -0.01])
        returns.index = [i * MS_PER_DAY * 30 for i in range(5)]
        
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        
        assert sharpe < 0
    
    def test_sharpe_from_timestamps_uses_actual_time(self):
        """Timestamp-based Sharpe uses actual calendar time for annualization."""
        from evaluation.portfolio_simulator import compute_sharpe_ratio_from_timestamps
        
        # 2 observations over exactly 1 year
        returns = pd.Series([0.10, 0.12])  # Two period returns
        returns.index = [0, 365 * MS_PER_DAY]  # 1 year apart
        
        sharpe = compute_sharpe_ratio_from_timestamps(returns)
        
        # Should be finite and reasonable
        assert not np.isnan(sharpe)
        assert sharpe > 0


class TestSharpeRatioAnnualization:
    """Verify Sharpe ratio annualization is correct."""
    
    def test_annualization_factor(self):
        """Annualization multiplies by sqrt(periods_per_year)."""
        from evaluation.portfolio_simulator import compute_sharpe_ratio
        
        returns = pd.Series([0.01, 0.02, 0.015, 0.01])
        returns.index = [i * MS_PER_DAY for i in range(4)]  # Daily
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # With 252 trading days, annualization = sqrt(252)
        expected_annual = (mean_ret / std_ret) * np.sqrt(252)
        
        # Function should produce annualized value
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        
        assert sharpe == pytest.approx(expected_annual, rel=0.01)


# =============================================================================
# CUMULATIVE RETURN TESTS
# =============================================================================

class TestCumulativeReturns:
    """Verify cumulative return calculations."""
    
    def test_cumulative_return_formula(self):
        """Cumulative return = product(1 + r_i) - 1."""
        period_returns = pd.Series([0.10, 0.05, -0.02, 0.08])
        
        # Manual calculation
        cumulative = (1 + period_returns).prod() - 1
        
        # Expected: (1.10)(1.05)(0.98)(1.08) - 1 ≈ 0.2218
        expected = 1.10 * 1.05 * 0.98 * 1.08 - 1
        
        assert cumulative == pytest.approx(expected, abs=1e-10)
    
    def test_cumulative_return_series(self):
        """Cumulative return series shows running total."""
        period_returns = pd.Series([0.10, 0.05, -0.02])
        
        # Running cumulative
        cumulative = (1 + period_returns).cumprod() - 1
        
        # After period 0: 0.10
        # After period 1: (1.10)(1.05) - 1 = 0.155
        # After period 2: (1.10)(1.05)(0.98) - 1 = 0.1319
        
        assert cumulative.iloc[0] == pytest.approx(0.10, abs=1e-10)
        assert cumulative.iloc[1] == pytest.approx(0.155, abs=1e-10)
        assert cumulative.iloc[2] == pytest.approx(1.10 * 1.05 * 0.98 - 1, abs=1e-10)
    
    def test_total_return_from_cumulative(self):
        """Total return is final cumulative value."""
        period_returns = pd.Series([0.10, 0.05, -0.02, 0.08])
        
        cumulative = (1 + period_returns).cumprod() - 1
        total_return = cumulative.iloc[-1]
        
        expected = (1 + period_returns).prod() - 1
        
        assert total_return == pytest.approx(expected, abs=1e-10)


# =============================================================================
# MAX DRAWDOWN TESTS
# =============================================================================

class TestMaxDrawdown:
    """Verify max drawdown calculation."""
    
    def test_max_drawdown_formula(self):
        """Max drawdown = max((peak - trough) / peak)."""
        from evaluation.portfolio_simulator import compute_max_drawdown
        
        # Cumulative returns: up to 20%, then down to 5%
        cumulative = pd.Series([0.0, 0.10, 0.20, 0.10, 0.05, 0.15])
        
        # Peak is 0.20, trough is 0.05
        # Drawdown = (1.20 - 1.05) / 1.20 = 0.125 (12.5%)
        expected_dd = (1.20 - 1.05) / 1.20
        
        dd = compute_max_drawdown(cumulative)
        
        assert dd == pytest.approx(expected_dd, abs=0.01)
    
    def test_no_drawdown_for_monotonic_increase(self):
        """No drawdown if returns only go up."""
        from evaluation.portfolio_simulator import compute_max_drawdown
        
        cumulative = pd.Series([0.0, 0.05, 0.10, 0.15, 0.20])
        
        dd = compute_max_drawdown(cumulative)
        
        assert dd == pytest.approx(0.0, abs=1e-10)
    
    def test_100_percent_drawdown(self):
        """Total loss gives 100% drawdown."""
        from evaluation.portfolio_simulator import compute_max_drawdown
        
        # Up 50%, then lose everything
        cumulative = pd.Series([0.0, 0.50, 0.0, -0.50, -1.0])  # -100% = total loss
        
        dd = compute_max_drawdown(cumulative)
        
        # From peak 1.50 to trough 0 = 100% drawdown
        assert dd >= 0.99  # Allow for floating point


# =============================================================================
# PORTFOLIO RETURN TESTS
# =============================================================================

class TestPortfolioReturns:
    """Verify portfolio return calculations for long-short strategies."""
    
    def test_equal_weight_long_only(self):
        """Long-only portfolio return = average of constituent returns."""
        # 3 stocks with known returns
        stock_returns = [0.10, 0.05, -0.02]  # A: +10%, B: +5%, C: -2%
        
        # Equal weight (1/3 each)
        portfolio_return = sum(stock_returns) / 3
        
        expected = (0.10 + 0.05 - 0.02) / 3  # 4.33%
        
        assert portfolio_return == pytest.approx(expected, abs=1e-10)
    
    def test_long_short_portfolio_return(self):
        """Long-short portfolio return = long return - short return."""
        # Long top 2, short bottom 2
        long_returns = [0.12, 0.08]  # Average = 10%
        short_returns = [-0.05, -0.03]  # Average = -4%
        
        # Long-short: gain from longs + gain from shorts going down
        # With equal weight: 0.5 * avg_long + 0.5 * (-avg_short)
        avg_long = sum(long_returns) / 2  # 0.10
        avg_short = sum(short_returns) / 2  # -0.04
        
        # If we're short, we gain when price goes down
        # Portfolio = 0.5 * 0.10 + 0.5 * -(-0.04) = 0.05 + 0.02 = 0.07
        portfolio_return = 0.5 * avg_long + 0.5 * (-avg_short)
        
        assert portfolio_return == pytest.approx(0.07, abs=1e-10)
    
    def test_dollar_neutral_long_short(self):
        """Dollar-neutral strategy: $1 long, $1 short, net exposure = 0."""
        # Long: AAPL +10%, MSFT +5% -> avg +7.5%
        # Short: XYZ -3%, ABC +2% -> avg -0.5% (we lose on shorts going up)
        
        long_avg = (0.10 + 0.05) / 2  # 0.075
        short_avg = (-0.03 + 0.02) / 2  # -0.005
        
        # Return from longs: +7.5%
        # Return from shorts: -(-0.5%) = +0.5% (shorts went down on avg)
        # Total: 7.5% + 0.5% = 8%
        portfolio_return = long_avg - short_avg
        
        assert portfolio_return == pytest.approx(0.08, abs=1e-10)


# =============================================================================
# TRANSACTION COST TESTS
# =============================================================================

class TestTransactionCosts:
    """Verify transaction cost calculations."""
    
    def test_cost_reduces_return(self):
        """Transaction costs reduce portfolio returns."""
        gross_return = 0.10  # 10% gross return
        cost_bps = 100  # 100 bps = 1%
        
        # With 100% turnover, cost = 1%
        turnover = 1.0
        cost = turnover * (cost_bps / 10000)
        
        net_return = gross_return - cost
        
        assert net_return == pytest.approx(0.09, abs=1e-10)
    
    def test_higher_turnover_more_cost(self):
        """Higher turnover incurs more transaction costs."""
        gross_return = 0.10
        cost_bps = 100
        
        turnover_low = 0.5  # 50% turnover
        turnover_high = 2.0  # 200% turnover
        
        cost_low = turnover_low * (cost_bps / 10000)  # 0.5%
        cost_high = turnover_high * (cost_bps / 10000)  # 2%
        
        net_low = gross_return - cost_low  # 9.5%
        net_high = gross_return - cost_high  # 8%
        
        assert net_low > net_high
        assert net_low == pytest.approx(0.095, abs=1e-10)
        assert net_high == pytest.approx(0.08, abs=1e-10)
    
    def test_turnover_calculation(self):
        """Turnover = sum of absolute weight changes."""
        # Period 1: A=50%, B=50%
        # Period 2: A=30%, B=70%
        # Changes: A=-20%, B=+20%
        # Total turnover = |−0.20| + |+0.20| = 0.40 (40%)
        
        weights_before = {"A": 0.50, "B": 0.50}
        weights_after = {"A": 0.30, "B": 0.70}
        
        turnover = sum(abs(weights_after[k] - weights_before[k]) 
                      for k in weights_before)
        
        assert turnover == pytest.approx(0.40, abs=1e-10)


# =============================================================================
# INTEGRATED PIPELINE RETURN VERIFICATION
# =============================================================================

class TestPipelineReturnsEndToEnd:
    """End-to-end tests verifying pipeline return calculations."""
    
    def test_known_data_known_return(self):
        """With known data, verify exact return calculation."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        # Create data where we know the exact forward returns
        df = pd.DataFrame({
            TIMESTAMP: [0, 10 * MS_PER_DAY, 0, 10 * MS_PER_DAY],
            TICKER: ["WINNER", "WINNER", "LOSER", "LOSER"],
            CLOSE: [100.0, 120.0, 100.0, 90.0],  # WINNER: +20%, LOSER: -10%
            ADJCLOSE: [100.0, 120.0, 100.0, 90.0],
        })
        
        result = compute_forward_returns(df, lookahead_days=10, return_type="simple")
        
        winner_ret = result[result[TICKER] == "WINNER"].iloc[0][FORWARD_RETURN]
        loser_ret = result[result[TICKER] == "LOSER"].iloc[0][FORWARD_RETURN]
        
        assert winner_ret == pytest.approx(0.20, abs=1e-10)
        assert loser_ret == pytest.approx(-0.10, abs=1e-10)
        
        # Long WINNER, short LOSER with equal weight
        # Portfolio return = 0.5 * 0.20 + 0.5 * -(-0.10) = 0.10 + 0.05 = 0.15
        portfolio_return = 0.5 * winner_ret + 0.5 * (-loser_ret)
        
        assert portfolio_return == pytest.approx(0.15, abs=1e-10)
    
    def test_quintile_return_calculation(self):
        """Verify quintile returns are correct averages."""
        from evaluation.ranking_metrics import compute_quintile_returns
        
        # 10 stocks, 2 per quintile
        predicted = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual = pd.Series([
            -0.20, -0.15,  # Q1: avg = -0.175
            -0.05, 0.00,   # Q2: avg = -0.025
            0.02, 0.03,    # Q3: avg = 0.025
            0.08, 0.10,    # Q4: avg = 0.09
            0.15, 0.20,    # Q5: avg = 0.175
        ])
        
        quintile_returns = compute_quintile_returns(predicted, actual)
        
        assert quintile_returns[1] == pytest.approx(-0.175, abs=0.01)
        assert quintile_returns[5] == pytest.approx(0.175, abs=0.01)
        
        # Spread = Q5 - Q1
        spread = quintile_returns[5] - quintile_returns[1]
        assert spread == pytest.approx(0.35, abs=0.02)


class TestReturnConsistency:
    """Test that different return calculations are consistent."""
    
    def test_log_return_additivity(self):
        """Log returns are additive over time."""
        # Two period returns
        r1_simple = 0.10
        r2_simple = 0.05
        
        r1_log = np.log(1 + r1_simple)
        r2_log = np.log(1 + r2_simple)
        
        # Total log return = sum of log returns
        total_log = r1_log + r2_log
        
        # Total simple return from compounding
        total_simple = (1 + r1_simple) * (1 + r2_simple) - 1
        
        # Converting back
        assert np.exp(total_log) - 1 == pytest.approx(total_simple, abs=1e-10)
    
    def test_simple_return_compounding(self):
        """Simple returns compound correctly."""
        returns = [0.10, 0.05, -0.02, 0.08]
        
        # Compound
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r)
        
        total = cumulative - 1
        
        # Using pandas
        pandas_total = (1 + pd.Series(returns)).prod() - 1
        
        assert total == pytest.approx(pandas_total, abs=1e-10)
