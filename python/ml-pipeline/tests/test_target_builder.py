"""Tests for the target_builder module - forward return calculations for ranking."""

import pandas as pd
import numpy as np
import pytest

from config.columns import TIMESTAMP, TICKER, CLOSE
from config.settings import MS_PER_DAY


class TestComputeForwardReturns:
    """Tests for compute_forward_returns function."""
    
    def test_simple_return_basic(self):
        """5-day forward return: if price goes from 100 to 110, return = 0.10"""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY, 10 * MS_PER_DAY],
            TICKER: ["A", "A", "A"],
            CLOSE: [100.0, 110.0, 121.0],  # +10%, +10%
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        
        # Row 0: (110 - 100) / 100 = 0.10
        # Row 1: (121 - 110) / 110 = 0.10
        # Row 2: No future data - should be NaN or dropped
        assert len(result[result[FORWARD_RETURN].notna()]) == 2
        assert abs(result.iloc[0][FORWARD_RETURN] - 0.10) < 1e-6
        assert abs(result.iloc[1][FORWARD_RETURN] - 0.10) < 1e-6
    
    def test_log_return_basic(self):
        """Log return: ln(P_t+n / P_t)"""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 110.0],
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="log")
        
        # ln(110/100) ≈ 0.0953
        expected = np.log(110.0 / 100.0)
        assert abs(result.iloc[0][FORWARD_RETURN] - expected) < 1e-6
    
    def test_negative_return(self):
        """Forward return handles price decreases correctly."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 80.0],  # -20%
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        
        assert abs(result.iloc[0][FORWARD_RETURN] - (-0.20)) < 1e-6
    
    def test_multiple_tickers_independent(self):
        """Forward returns computed independently per ticker."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY, 0, 5 * MS_PER_DAY],
            TICKER: ["A", "A", "B", "B"],
            CLOSE: [100.0, 120.0, 100.0, 90.0],  # A: +20%, B: -10%
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        
        result_a = result[result[TICKER] == "A"]
        result_b = result[result[TICKER] == "B"]
        
        # First row of each ticker should have forward return
        assert abs(result_a.iloc[0][FORWARD_RETURN] - 0.20) < 1e-6
        assert abs(result_b.iloc[0][FORWARD_RETURN] - (-0.10)) < 1e-6
    
    def test_no_future_data_results_in_nan(self):
        """Rows without future data should have NaN forward return."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY, 10 * MS_PER_DAY],
            TICKER: ["A", "A", "A"],
            CLOSE: [100.0, 110.0, 121.0],
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        
        # Last row should have NaN (no future data)
        assert pd.isna(result.iloc[2][FORWARD_RETURN])
    
    def test_winsorization(self):
        """Extreme returns should be clipped when winsorize=True."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 300.0],  # +200% (extreme)
        })
        
        result = compute_forward_returns(
            df, lookahead_days=5, return_type="simple", 
            winsorize_limits=(-0.5, 0.5)
        )
        
        # Return of 2.0 should be clipped to 0.5
        assert result.iloc[0][FORWARD_RETURN] == 0.5
    
    def test_drop_na_option(self):
        """drop_na=True should remove rows with NaN forward returns."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY, 10 * MS_PER_DAY],
            TICKER: ["A", "A", "A"],
            CLOSE: [100.0, 110.0, 121.0],
        })
        
        result = compute_forward_returns(
            df, lookahead_days=5, return_type="simple", drop_na=True
        )
        
        # Last row should be dropped
        assert len(result) == 2
        assert FORWARD_RETURN in result.columns
        assert result[FORWARD_RETURN].isna().sum() == 0
    
    def test_different_lookahead_days(self):
        """Different lookahead periods should produce different results."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        # Price series: 100 -> 105 (day 5) -> 110 (day 10)
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY, 10 * MS_PER_DAY],
            TICKER: ["A", "A", "A"],
            CLOSE: [100.0, 105.0, 110.0],
        })
        
        result_5d = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        result_10d = compute_forward_returns(df, lookahead_days=10, return_type="simple")
        
        # 5-day return from day 0: (105 - 100) / 100 = 0.05
        # 10-day return from day 0: (110 - 100) / 100 = 0.10
        assert abs(result_5d.iloc[0][FORWARD_RETURN] - 0.05) < 1e-6
        assert abs(result_10d.iloc[0][FORWARD_RETURN] - 0.10) < 1e-6
    
    def test_preserves_other_columns(self):
        """Forward return computation should preserve all existing columns."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 110.0],
            "feature_1": [1.0, 2.0],
            "feature_2": [3.0, 4.0],
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple")
        
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert FORWARD_RETURN in result.columns


class TestGetMaxForwardTimestamp:
    """Tests for get_max_forward_timestamp helper."""
    
    def test_basic_max_timestamp(self):
        """Should return max timestamp that can have forward return."""
        from core.target_builder import get_max_forward_timestamp
        
        max_ts = 100 * MS_PER_DAY
        lookahead = 10
        
        result = get_max_forward_timestamp(max_ts, lookahead)
        
        # Max labelable = 100 - 10 = 90 days
        assert result == 90 * MS_PER_DAY
