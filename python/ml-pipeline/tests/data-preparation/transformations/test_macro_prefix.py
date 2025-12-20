"""Tests for transformations module."""

import pandas as pd
import numpy as np
import pytest

from config.column_names import TIMESTAMP, TICKER, FEATURE, VALUE, MACRO_PREFIX
from macro_prefix import add_macro_prefix


class TestMacroPrefix:
    """Tests for add_macro_prefix function."""
    
    def test_adds_prefix_to_empty_ticker(self):
        """Should add MACRO_ prefix to features with empty ticker."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["ANZ.NZ", "", "ANZ.NZ"],
            FEATURE: ["Close", "GDP", "Open"],
            VALUE: [10.0, 100.0, 11.0],
        })
        
        result = add_macro_prefix(df)
        
        assert result.loc[1, FEATURE] == f"{MACRO_PREFIX}GDP"
        assert result.loc[0, FEATURE] == "Close"
        assert result.loc[2, FEATURE] == "Open"
    
    def test_does_not_modify_non_macro(self):
        """Should not modify features with non-empty ticker."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000],
            TICKER: ["ANZ.NZ", "BNZ.NZ"],
            FEATURE: ["Close", "Close"],
            VALUE: [10.0, 20.0],
        })
        
        result = add_macro_prefix(df)
        
        assert result.loc[0, FEATURE] == "Close"
        assert result.loc[1, FEATURE] == "Close"
    
    def test_modifies_inplace(self):
        """Should modify DataFrame inplace for efficiency."""
        df = pd.DataFrame({
            TIMESTAMP: [1000],
            TICKER: [""],
            FEATURE: ["GDP"],
            VALUE: [100.0],
        })
        
        result = add_macro_prefix(df)
        
        # Should be same object (inplace modification)
        assert result is df
        # Feature should be modified
        assert df.loc[0, FEATURE] == f"{MACRO_PREFIX}GDP"
