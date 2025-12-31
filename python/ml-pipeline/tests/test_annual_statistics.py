"""Tests for annual statistics computation."""

import pytest
import pandas as pd
import numpy as np
from evaluation.portfolio_simulator import (
    compute_annual_statistics,
    AnnualStatistics,
)


class TestComputeAnnualStatistics:
    """Test annual statistics computation."""
    
    def test_basic_computation(self):
        """Test basic annual statistics from simulated daily returns."""
        # Create 3 full years of daily returns (252 days per year)
        np.random.seed(42)
        
        # Year 1: 10% return (2020)
        year1_returns = np.random.normal(0.10 / 252, 0.15 / np.sqrt(252), 252)
        
        # Year 2: -5% return (2021)
        year2_returns = np.random.normal(-0.05 / 252, 0.15 / np.sqrt(252), 252)
        
        # Year 3: 15% return (2022)
        year3_returns = np.random.normal(0.15 / 252, 0.15 / np.sqrt(252), 252)
        
        daily_returns = pd.Series(
            np.concatenate([year1_returns, year2_returns, year3_returns])
        )
        
        # Create timestamp index (milliseconds since epoch)
        # Use actual dates to ensure proper year grouping
        # Start at 2020-01-02 (first trading day)
        start_date = pd.Timestamp('2020-01-02')
        
        # Create a date range with business days
        dates = pd.date_range(start=start_date, periods=len(daily_returns), freq='B')
        
        # Convert to millisecond timestamps
        timestamps = (dates.astype(np.int64) // 1_000_000).tolist()
        daily_returns.index = timestamps
        
        # Compute annual statistics
        stats = compute_annual_statistics(daily_returns, risk_free_rate=0.0)
        
        # Basic validation
        assert stats is not None
        assert stats.num_years >= 2  # Should have at least 2 complete years
        assert stats.years_sampled >= 2.0  # Should cover ~3 years
        
        # Check that mean is reasonable
        assert -0.5 < stats.mean_annual_return < 0.5
        assert stats.std_annual_return > 0
        
        # Check that percentiles are ordered
        assert stats.min_annual_return <= stats.pct_5_annual_return
        assert stats.pct_5_annual_return <= stats.pct_25_annual_return
        assert stats.pct_25_annual_return <= stats.median_annual_return
        assert stats.median_annual_return <= stats.pct_75_annual_return
        assert stats.pct_75_annual_return <= stats.pct_95_annual_return
        assert stats.pct_95_annual_return <= stats.max_annual_return
        
        # Check win rate
        assert 0 <= stats.pct_positive_years <= 1
        
        # Since we have both positive and negative target returns,
        # we should have at least one of each (probabilistically)
        # But don't assert this strictly as randomness could give all positive
    
    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None."""
        # Only 100 days of data
        daily_returns = pd.Series(
            np.random.normal(0.001, 0.02, 100)
        )
        
        # Create timestamp index
        start_ts = pd.Timestamp('2020-01-01').value // 1_000_000
        daily_returns.index = range(start_ts, start_ts + 100 * 86_400_000, 86_400_000)
        
        stats = compute_annual_statistics(daily_returns)
        
        assert stats is None
    
    def test_all_positive_years(self):
        """Test statistics when all years are positive."""
        # Create 2 full years of positive returns using business days
        np.random.seed(42)
        year1_returns = np.random.normal(0.10 / 252, 0.10 / np.sqrt(252), 252)
        year2_returns = np.random.normal(0.15 / 252, 0.10 / np.sqrt(252), 252)
        
        daily_returns = pd.Series(np.concatenate([year1_returns, year2_returns]))
        
        # Create timestamp index using business days
        start_date = pd.Timestamp('2020-01-02')
        dates = pd.date_range(start=start_date, periods=len(daily_returns), freq='B')
        timestamps = (dates.astype(np.int64) // 1_000_000).tolist()
        daily_returns.index = timestamps
        
        stats = compute_annual_statistics(daily_returns)
        
        assert stats is not None
        assert stats.num_years >= 2
        assert stats.pct_positive_years > 0.5  # Most years should be positive (with noise, may not be 100%)
        assert stats.min_annual_return >= 0  # All years should be positive (modulo statistical noise)
    
    def test_to_dict_serialization(self):
        """Test that AnnualStatistics can be serialized to dict."""
        # Create minimal valid stats
        stats = AnnualStatistics(
            mean_annual_return=0.10,
            median_annual_return=0.09,
            std_annual_return=0.15,
            min_annual_return=-0.05,
            max_annual_return=0.25,
            pct_5_annual_return=0.02,
            pct_25_annual_return=0.05,
            pct_75_annual_return=0.15,
            pct_95_annual_return=0.22,
            pct_positive_years=0.75,
            avg_positive_year=0.12,
            avg_negative_year=-0.03,
            skewness_annual=0.2,
            kurtosis_annual=1.5,
            sharpe_annual_avg=0.8,
            num_years=4,
            years_sampled=4.0,
        )
        
        d = stats.to_dict()
        
        assert isinstance(d, dict)
        assert d['mean_annual_return'] == 0.10
        assert d['num_years'] == 4
        assert 'sharpe_annual_avg' in d
    
    def test_summary_string(self):
        """Test that summary() generates readable output."""
        stats = AnnualStatistics(
            mean_annual_return=0.10,
            median_annual_return=0.09,
            std_annual_return=0.15,
            min_annual_return=-0.05,
            max_annual_return=0.25,
            pct_5_annual_return=0.02,
            pct_25_annual_return=0.05,
            pct_75_annual_return=0.15,
            pct_95_annual_return=0.22,
            pct_positive_years=0.75,
            avg_positive_year=0.12,
            avg_negative_year=-0.03,
            skewness_annual=0.2,
            kurtosis_annual=1.5,
            sharpe_annual_avg=0.8,
            num_years=4,
            years_sampled=4.0,
        )
        
        summary = stats.summary()
        
        assert "Annual Return Statistics" in summary
        assert "Mean Annual Return" in summary
        assert "10.00%" in summary  # 0.10 formatted as percentage
        assert "Sharpe" in summary
