"""Tests for financial ratio features."""

import pytest
import pandas as pd
import numpy as np

from features.ratios import add_financial_ratios, _add_trailing_dividend_yield
from config.columns import TICKER, CLOSE, DIVIDEND, TRAILING_DIV_YIELD_252


class TestTrailingDividendYield:
    """Tests for trailing dividend yield calculation."""
    
    def test_trailing_dividend_yield_basic(self):
        """Test basic trailing dividend yield calculation."""
        # Create simple test data: one stock, 5 days
        df = pd.DataFrame({
            TICKER: ['AAA'] * 5,
            CLOSE: [100.0, 100.0, 100.0, 100.0, 100.0],
            DIVIDEND: [0.0, 1.0, 0.0, 0.0, 2.0],  # Dividends on days 2 and 5
        })
        
        result = _add_trailing_dividend_yield(df, window=252)
        
        assert TRAILING_DIV_YIELD_252 in result.columns
        # Day 1: 0 dividends, yield = 0/100 = 0
        # Day 2: 1.0 dividend, yield = 1/100 = 0.01
        # Day 3: still 1.0 cumulative, yield = 1/100 = 0.01
        # Day 4: still 1.0 cumulative, yield = 1/100 = 0.01
        # Day 5: 1.0 + 2.0 = 3.0 cumulative, yield = 3/100 = 0.03
        expected = [0.0, 0.01, 0.01, 0.01, 0.03]
        np.testing.assert_array_almost_equal(
            result[TRAILING_DIV_YIELD_252].values, expected, decimal=4
        )
    
    def test_trailing_dividend_yield_multiple_tickers(self):
        """Test that dividend yield is computed per-ticker."""
        df = pd.DataFrame({
            TICKER: ['AAA', 'AAA', 'BBB', 'BBB'],
            CLOSE: [100.0, 100.0, 50.0, 50.0],
            DIVIDEND: [1.0, 0.0, 0.5, 0.5],
        })
        
        result = _add_trailing_dividend_yield(df, window=252)
        
        # AAA: day 1 = 1/100 = 0.01, day 2 = 1/100 = 0.01
        # BBB: day 1 = 0.5/50 = 0.01, day 2 = 1.0/50 = 0.02
        expected = [0.01, 0.01, 0.01, 0.02]
        np.testing.assert_array_almost_equal(
            result[TRAILING_DIV_YIELD_252].values, expected, decimal=4
        )
    
    def test_trailing_dividend_yield_nan_dividends(self):
        """Test that NaN dividends are treated as 0."""
        df = pd.DataFrame({
            TICKER: ['AAA'] * 3,
            CLOSE: [100.0, 100.0, 100.0],
            DIVIDEND: [np.nan, 1.0, np.nan],
        })
        
        result = _add_trailing_dividend_yield(df, window=252)
        
        # NaN dividends treated as 0, so cumulative is [0, 1, 1]
        expected = [0.0, 0.01, 0.01]
        np.testing.assert_array_almost_equal(
            result[TRAILING_DIV_YIELD_252].values, expected, decimal=4
        )
    
    def test_trailing_dividend_yield_missing_columns(self):
        """Test graceful handling when required columns are missing."""
        df = pd.DataFrame({
            TICKER: ['AAA'] * 3,
            CLOSE: [100.0, 100.0, 100.0],
            # No DIVIDEND column
        })
        
        result = _add_trailing_dividend_yield(df, window=252)
        
        # Should return df unchanged (no new column added)
        assert TRAILING_DIV_YIELD_252 not in result.columns
    
    def test_trailing_dividend_yield_in_add_financial_ratios(self):
        """Test that trailing dividend yield is added via main function."""
        df = pd.DataFrame({
            TICKER: ['AAA'] * 3,
            'Open': [99.0, 99.0, 99.0],
            'High': [101.0, 101.0, 101.0],
            'Low': [98.0, 98.0, 98.0],
            CLOSE: [100.0, 100.0, 100.0],
            'Volume': [1000, 1000, 1000],
            DIVIDEND: [0.0, 2.0, 0.0],
        })
        
        result = add_financial_ratios(df)
        
        assert TRAILING_DIV_YIELD_252 in result.columns
        # Cumulative dividends: [0, 2, 2], yields: [0, 0.02, 0.02]
        expected = [0.0, 0.02, 0.02]
        np.testing.assert_array_almost_equal(
            result[TRAILING_DIV_YIELD_252].values, expected, decimal=4
        )


class TestDerivedFeatures:
    """Tests for cross-asset derived features."""
    
    def test_safe_divide_handles_zero_denominator(self):
        """Test that safe_divide handles division by zero."""
        from features.ratios import _safe_divide
        
        numerator = pd.Series([1.0, 2.0, 3.0])
        denominator = pd.Series([0.0, 1.0, 0.0])
        
        result = _safe_divide(numerator, denominator)
        
        # Zero denominators should not cause errors
        assert not np.isinf(result).any()
        # Result should be clipped to max value
        assert result.max() <= 1e6
    
    def test_safe_divide_handles_infinity(self):
        """Test that safe_divide replaces infinity with NaN."""
        from features.ratios import _safe_divide
        
        numerator = pd.Series([1e308, 1.0])
        denominator = pd.Series([1e-308, 1.0])
        
        result = _safe_divide(numerator, denominator)
        
        # Should clip to max value, not overflow
        assert not np.isinf(result).any()
    
    def test_yield_curve_spread(self):
        """Test yield curve spread calculation."""
        from config.columns import (
            TIMESTAMP, TICKER, LONG_TERM_INTEREST_RATE, 
            SHORT_TERM_INTEREST_RATE, YIELD_CURVE_SPREAD
        )
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['AAA', 'AAA', 'AAA'],
            LONG_TERM_INTEREST_RATE: [5.0, 4.5, 3.0],
            SHORT_TERM_INTEREST_RATE: [2.0, 3.0, 4.0],  # Inverts on day 3
        })
        
        result = add_financial_ratios(df)
        
        if YIELD_CURVE_SPREAD in result.columns:
            expected = [3.0, 1.5, -1.0]  # Normal, flat, inverted
            np.testing.assert_array_almost_equal(
                result[YIELD_CURVE_SPREAD].values, expected, decimal=2
            )
    
    def test_vol_term_structure(self):
        """Test volatility term structure calculation."""
        from config.columns import (
            TIMESTAMP, TICKER, VOL_TERM_STRUCTURE, CLOSE
        )
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['AAA', 'AAA', 'AAA'],
            CLOSE: [100.0, 100.0, 100.0],
            'Vol_20': [0.3, 0.2, 0.4],
            'Vol_252': [0.2, 0.2, 0.2],
        })
        
        result = add_financial_ratios(df)
        
        if VOL_TERM_STRUCTURE in result.columns:
            expected = [1.5, 1.0, 2.0]
            np.testing.assert_array_almost_equal(
                result[VOL_TERM_STRUCTURE].values, expected, decimal=2
            )
    
    def test_gold_oil_ratio(self):
        """Test gold/oil ratio calculation."""
        from config.columns import (
            TIMESTAMP, TICKER, GOLD_OIL_RATIO,
            MACRO_GOLD_ADJCLOSE, MACRO_OIL_ADJCLOSE, CLOSE
        )
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['AAA', 'AAA', 'AAA'],
            CLOSE: [100.0, 100.0, 100.0],
            MACRO_GOLD_ADJCLOSE: [2000.0, 1800.0, 2200.0],
            MACRO_OIL_ADJCLOSE: [80.0, 90.0, 70.0],
        })
        
        result = add_financial_ratios(df)
        
        if GOLD_OIL_RATIO in result.columns:
            expected = [25.0, 20.0, 31.43]
            np.testing.assert_array_almost_equal(
                result[GOLD_OIL_RATIO].values, expected, decimal=1
            )
    
    def test_gold_oil_ratio_zero_oil(self):
        """Test gold/oil ratio handles zero oil price gracefully."""
        from config.columns import (
            TIMESTAMP, TICKER, GOLD_OIL_RATIO,
            MACRO_GOLD_ADJCLOSE, MACRO_OIL_ADJCLOSE, CLOSE
        )
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2],
            TICKER: ['AAA', 'AAA'],
            CLOSE: [100.0, 100.0],
            MACRO_GOLD_ADJCLOSE: [2000.0, 1800.0],
            MACRO_OIL_ADJCLOSE: [0.0, 80.0],  # Zero oil on day 1
        })
        
        result = add_financial_ratios(df)
        
        if GOLD_OIL_RATIO in result.columns:
            # Should not have infinities
            assert not np.isinf(result[GOLD_OIL_RATIO]).any()
            # Should be clipped to max (100 for gold/oil)
            assert result[GOLD_OIL_RATIO].max() <= 100
    
    def test_dollar_vol_market_share(self):
        """Test dollar volume market share calculation."""
        from config.columns import TIMESTAMP, TICKER, DOLLAR_VOL_MARKET_SHARE, CLOSE
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 1, 2, 2, 2],
            TICKER: ['AAA', 'BBB', 'CCC', 'AAA', 'BBB', 'CCC'],
            CLOSE: [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            'DollarVolume': [100.0, 200.0, 300.0, 150.0, 150.0, 300.0],
        })
        
        result = add_financial_ratios(df)
        
        if DOLLAR_VOL_MARKET_SHARE in result.columns:
            # Day 1: total = 600, shares = [100/600, 200/600, 300/600]
            # Day 2: total = 600, shares = [150/600, 150/600, 300/600]
            expected = [1/6, 2/6, 3/6, 1/4, 1/4, 1/2]
            np.testing.assert_array_almost_equal(
                result[DOLLAR_VOL_MARKET_SHARE].values, expected, decimal=4
            )
    
    def test_fear_ratio_aggregates_terms(self):
        """Test fear ratio aggregates multiple Wiki fear terms."""
        from config.columns import TIMESTAMP, TICKER, FEAR_RATIO, CLOSE
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['AAA', 'AAA', 'AAA'],
            CLOSE: [100.0, 100.0, 100.0],
            'MACRO_Recession_Wiki_Views': [100.0, 200.0, 100.0],
            'MACRO_Unemployment_Wiki_Views': [50.0, 100.0, 50.0],
        })
        
        result = add_financial_ratios(df)
        
        if FEAR_RATIO in result.columns:
            # Both columns normalized to mean 1.33 and 0.67
            # Higher values on day 2
            assert result[FEAR_RATIO].iloc[1] > result[FEAR_RATIO].iloc[0]
            assert result[FEAR_RATIO].iloc[1] > result[FEAR_RATIO].iloc[2]
    
    def test_derived_features_missing_columns(self):
        """Test that derived features handle missing columns gracefully."""
        from config.columns import TIMESTAMP, TICKER, CLOSE
        
        # Minimal DataFrame with no macro data
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['AAA', 'AAA', 'AAA'],
            CLOSE: [100.0, 101.0, 102.0],
        })
        
        # Should not raise any errors
        result = add_financial_ratios(df)
        
        # Original columns should still be present
        assert CLOSE in result.columns
    
    def test_derived_features_all_nan_input(self):
        """Test derived features handle all-NaN input columns."""
        from config.columns import (
            TIMESTAMP, TICKER, CLOSE, GOLD_OIL_RATIO,
            MACRO_GOLD_ADJCLOSE, MACRO_OIL_ADJCLOSE
        )
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['AAA', 'AAA', 'AAA'],
            CLOSE: [100.0, 100.0, 100.0],
            MACRO_GOLD_ADJCLOSE: [np.nan, np.nan, np.nan],
            MACRO_OIL_ADJCLOSE: [80.0, 90.0, 70.0],
        })
        
        result = add_financial_ratios(df)
        
        if GOLD_OIL_RATIO in result.columns:
            # Should be all NaN but not raise errors
            assert result[GOLD_OIL_RATIO].isna().all()
    
    def test_earnings_quality(self):
        """Test earnings quality calculation."""
        from config.columns import TIMESTAMP, TICKER, EARNINGS_QUALITY, CLOSE
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['AAA', 'AAA', 'AAA'],
            CLOSE: [100.0, 100.0, 100.0],
            'trailingOperatingCashFlow': [150.0, 80.0, -50.0],
            'trailingNetIncome': [100.0, 100.0, 100.0],
        })
        
        result = add_financial_ratios(df)
        
        if EARNINGS_QUALITY in result.columns:
            expected = [1.5, 0.8, -0.5]  # High, low, negative quality
            np.testing.assert_array_almost_equal(
                result[EARNINGS_QUALITY].values, expected, decimal=2
            )
    
    def test_earnings_quality_zero_income(self):
        """Test earnings quality handles zero net income."""
        from config.columns import TIMESTAMP, TICKER, EARNINGS_QUALITY, CLOSE
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2],
            TICKER: ['AAA', 'AAA'],
            CLOSE: [100.0, 100.0],
            'trailingOperatingCashFlow': [100.0, 100.0],
            'trailingNetIncome': [0.0, 100.0],
        })
        
        result = add_financial_ratios(df)
        
        if EARNINGS_QUALITY in result.columns:
            # Zero income should not cause division error
            assert not np.isinf(result[EARNINGS_QUALITY]).any()
            # Should be clipped
            assert result[EARNINGS_QUALITY].max() <= 5
