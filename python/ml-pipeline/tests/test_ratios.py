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
