"""Tests for long-to-wide converter."""

import pandas as pd
import numpy as np
import pytest

from config.column_names import TIMESTAMP, TICKER, FEATURE, VALUE, CLOSE, OPEN, MACRO_PREFIX
from converter import long_to_wide


class TestLongToWide:
    """Tests for long_to_wide function."""
    
    def test_basic_conversion(self):
        """Should convert basic long format to wide format."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 1000, 2000, 2000],
            TICKER: ["ANZ.NZ", "ANZ.NZ", "ANZ.NZ", "ANZ.NZ"],
            FEATURE: [CLOSE, OPEN, CLOSE, OPEN],
            VALUE: [10.0, 9.5, 11.0, 10.5],
        })
        
        result = long_to_wide(df)
        
        assert len(result) == 2
        assert CLOSE in result.columns
        assert OPEN in result.columns
        assert result[result[TIMESTAMP] == 1000][CLOSE].values[0] == 10.0
    
    def test_includes_macro_data(self):
        """Should include macro data with forward-fill."""
        df = pd.DataFrame({
            TIMESTAMP: [500, 1000, 1000],
            TICKER: ["", "ANZ.NZ", "ANZ.NZ"],
            FEATURE: [f"{MACRO_PREFIX}GDP", CLOSE, OPEN],
            VALUE: [100.0, 10.0, 9.5],
        })
        
        result = long_to_wide(df)
        
        assert f"{MACRO_PREFIX}GDP" in result.columns
        # Macro data at 500 should be forward-filled to 1000
        assert result[f"{MACRO_PREFIX}GDP"].values[0] == 100.0
    
    def test_no_future_leakage_macro(self):
        """Should not include macro data from the future."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 1000, 2000],
            TICKER: ["ANZ.NZ", "ANZ.NZ", ""],
            FEATURE: [CLOSE, OPEN, f"{MACRO_PREFIX}GDP"],
            VALUE: [10.0, 9.5, 100.0],
        })
        
        result = long_to_wide(df)
        
        # At timestamp 1000, GDP from 2000 should not be available
        row_1000 = result[result[TIMESTAMP] == 1000]
        assert pd.isna(row_1000[f"{MACRO_PREFIX}GDP"].values[0])
    
    def test_multiple_tickers(self):
        """Should handle multiple tickers."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 1000, 1000, 1000],
            TICKER: ["ANZ.NZ", "ANZ.NZ", "BNZ.NZ", "BNZ.NZ"],
            FEATURE: [CLOSE, OPEN, CLOSE, OPEN],
            VALUE: [10.0, 9.5, 20.0, 19.5],
        })
        
        result = long_to_wide(df)
        
        assert len(result) == 2
        assert len(result[TICKER].unique()) == 2
    
    def test_requires_close_as_anchor(self):
        """Should only include rows where Close exists."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 1000, 2000],
            TICKER: ["ANZ.NZ", "ANZ.NZ", "ANZ.NZ"],
            FEATURE: [OPEN, "Volume", CLOSE],
            VALUE: [9.5, 1000, 10.0],
        })
        
        result = long_to_wide(df)
        
        # Only timestamp 2000 has Close
        assert len(result) == 1
        assert result[TIMESTAMP].values[0] == 2000
