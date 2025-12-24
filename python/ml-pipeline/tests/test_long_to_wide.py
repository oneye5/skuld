"""Tests for the long_to_wide converter."""

import pandas as pd
import numpy as np
import pytest

from config.columns import TIMESTAMP, TICKER, FEATURE, VALUE, CLOSE, MACRO_PREFIX
from core.long_to_wide import long_to_wide, add_macro_prefix


class TestAddMacroPrefix:
    """Tests for add_macro_prefix function."""
    
    def test_adds_prefix_to_empty_ticker(self):
        """Test that MACRO_ prefix is added for empty ticker rows."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2],
            TICKER: ["", "ANZ.NZ"],
            FEATURE: ["GDP", "Close"],
            VALUE: [100.0, 25.0],
        })
        
        result = add_macro_prefix(df)
        
        # Empty ticker row should have prefixed feature
        macro_row = result[result[TICKER] == ""]
        assert macro_row[FEATURE].iloc[0] == "MACRO_GDP"
        
        # Non-empty ticker row should be unchanged
        ticker_row = result[result[TICKER] == "ANZ.NZ"]
        assert ticker_row[FEATURE].iloc[0] == "Close"
    
    def test_does_not_modify_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            TIMESTAMP: [1],
            TICKER: [""],
            FEATURE: ["GDP"],
            VALUE: [100.0],
        })
        original_feature = df[FEATURE].iloc[0]
        
        add_macro_prefix(df)
        
        assert df[FEATURE].iloc[0] == original_feature


class TestLongToWide:
    """Tests for long_to_wide function."""
    
    def test_basic_conversion(self):
        """Test basic long to wide conversion."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 1],
            TICKER: ["A", "A", "A"],
            FEATURE: ["Close", "Open", "Volume"],
            VALUE: [100.0, 99.0, 1000.0],
        })
        
        result = long_to_wide(df)
        
        assert len(result) == 1
        assert "Close" in result.columns
        assert "Open" in result.columns
        assert "Volume" in result.columns
        assert result["Close"].iloc[0] == 100.0
    
    def test_multiple_tickers(self):
        """Test conversion with multiple tickers."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 1, 1],
            TICKER: ["A", "A", "B", "B"],
            FEATURE: ["Close", "Volume", "Close", "Volume"],
            VALUE: [100.0, 1000.0, 200.0, 2000.0],
        })
        
        result = long_to_wide(df)
        
        assert len(result) == 2
        assert set(result[TICKER].unique()) == {"A", "B"}
    
    def test_only_close_timestamps_used(self):
        """Test that only Close timestamps become rows."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 2],  # Close at ts=1, Volume at ts=1 and ts=2
            TICKER: ["A", "A", "A"],
            FEATURE: ["Close", "Volume", "Volume"],
            VALUE: [100.0, 1000.0, 2000.0],
        })
        
        result = long_to_wide(df)
        
        # Should only have 1 row (for Close at ts=1)
        assert len(result) == 1
        assert result[TIMESTAMP].iloc[0] == 1
    
    def test_macro_data_merged(self):
        """Test that macro data is merged correctly."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 1],
            TICKER: ["A", "A", ""],
            FEATURE: ["Close", "Volume", "GDP"],
            VALUE: [100.0, 1000.0, 500.0],
        })
        
        df = add_macro_prefix(df)
        result = long_to_wide(df)
        
        assert "MACRO_GDP" in result.columns
        assert result["MACRO_GDP"].iloc[0] == 500.0
    
    def test_empty_ticker_rows_excluded_from_anchors(self):
        """Test that rows with empty ticker don't become row anchors."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 1],
            TICKER: ["", "A"],
            FEATURE: ["Close", "Close"],
            VALUE: [999.0, 100.0],
        })
        
        result = long_to_wide(df)
        
        # Should only have 1 row (ticker A)
        assert len(result) == 1
        assert result[TICKER].iloc[0] == "A"
