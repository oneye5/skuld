"""Test implementation metrics calculation."""

import numpy as np
import pandas as pd
import pytest
from evaluation.portfolio_simulator import (
    run_portfolio_backtest,
    PortfolioConfig,
)


def test_implementation_metrics_calculation():
    """Test that implementation metrics are calculated correctly."""
    # Create synthetic backtest data
    # 10 timestamps, 20 stocks per timestamp
    timestamps = list(range(0, 10 * 86_400_000, 86_400_000))  # Daily for 10 days
    tickers = [f"STOCK_{i}" for i in range(20)]
    
    data = []
    for ts in timestamps:
        for i, ticker in enumerate(tickers):
            # Scores: top 10 positive, bottom 10 negative
            score = i - 10
            # Returns: correlate with scores + some noise to get drawdown
            return_val = 0.01 * score + np.random.normal(0, 0.05)  # Higher noise
            data.append({
                "timestamp": ts,
                "ticker": ticker,
                "predicted_score": score,
                "actual_return": return_val,
            })
    
    df = pd.DataFrame(data)
    
    # Run backtest
    config = PortfolioConfig(
        top_n=5,
        bottom_n=5,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )
    
    result = run_portfolio_backtest(
        df,
        config,
        timestamp_col="timestamp",
        ticker_col="ticker",
        score_col="predicted_score",
        return_col="actual_return",
        return_horizon_days=1,
    )
    
    # Check that implementation metrics are computed
    assert not np.isnan(result.annualized_return_post_fee)
    assert not np.isnan(result.annualized_return_pre_fee)
    assert not np.isnan(result.annualized_volatility)
    assert not np.isnan(result.total_cost_drag)
    assert not np.isnan(result.avg_cost_per_rebalance)
    assert not np.isnan(result.avg_holding_period_years)
    assert not np.isnan(result.return_per_unit_turnover)
    # Calmar ratio may be NaN if max drawdown is 0 (very unlikely with noise)
    
    # Check sanity of values
    assert result.num_rebalances == len(timestamps)
    assert result.annualized_return_pre_fee > result.annualized_return_post_fee  # Fees reduce returns
    assert result.total_cost_drag < 0  # Cost drag should be negative
    assert result.avg_holding_period_years > 0
    assert result.avg_holding_period_years < 1  # Should be less than a year for daily data
    
    # Check that cost metrics make sense
    assert abs(result.avg_cost_per_rebalance) < 0.01  # Should be small
    
    print("\n✓ All implementation metrics calculated correctly")
    print(f"  Annualized return (post-fee): {result.annualized_return_post_fee:.2%}")
    print(f"  Annualized return (pre-fee):  {result.annualized_return_pre_fee:.2%}")
    print(f"  Cost drag:                    {result.total_cost_drag:.2%}")
    print(f"  Calmar ratio:                 {result.calmar_ratio:.2f}" if not np.isnan(result.calmar_ratio) else "  Calmar ratio:                 N/A (no drawdown)")


def test_annualized_return_calculation():
    """Test that annualized returns are calculated correctly."""
    # Create data with known total return over known period
    # 2 years of data (730 days), but use 5-day return horizon so we don't sample down to 1 period
    timestamps = list(range(0, 730 * 86_400_000, 5 * 86_400_000))  # Every 5 days for 2 years
    tickers = [f"STOCK_{i}" for i in range(20)]
    
    data = []
    for ts in timestamps:
        for i, ticker in enumerate(tickers):
            score = i - 10
            # Fixed returns to get predictable total
            return_val = 0.02 if score > 0 else -0.02
            data.append({
                "timestamp": ts,
                "ticker": ticker,
                "predicted_score": score,
                "actual_return": return_val,
            })
    
    df = pd.DataFrame(data)
    
    config = PortfolioConfig(top_n=5, bottom_n=5, transaction_cost_bps=0)  # No costs for simplicity
    
    result = run_portfolio_backtest(
        df,
        config,
        timestamp_col="timestamp",
        ticker_col="ticker",
        score_col="predicted_score",
        return_col="actual_return",
        return_horizon_days=5,
    )
    
    # Calculate expected annualized return
    # If we have X periods over Y years, annualized = (1 + total_return)^(1/Y) - 1
    first_ts = result.daily_returns.index[0]
    last_ts = result.daily_returns.index[-1]
    total_years = (last_ts - first_ts) / (365.0 * 86_400_000)
    
    expected_annualized = (1 + result.total_return) ** (1 / total_years) - 1
    
    # Check that our calculation matches
    assert abs(result.annualized_return_post_fee - expected_annualized) < 0.01
    
    print("\n✓ Annualized return calculation verified")
    print(f"  Total return:       {result.total_return:.2%}")
    print(f"  Total years:        {total_years:.2f}")
    print(f"  Annualized return:  {result.annualized_return_post_fee:.2%}")
    print(f"  Expected:           {expected_annualized:.2%}")


if __name__ == "__main__":
    test_implementation_metrics_calculation()
    test_annualized_return_calculation()
    print("\n✓ All tests passed")
