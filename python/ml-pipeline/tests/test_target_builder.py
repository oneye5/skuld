"""Tests for the target_builder module - forward return calculations for ranking."""

import pandas as pd
import numpy as np
import pytest
import warnings

from config.columns import TIMESTAMP, TICKER, CLOSE, ADJCLOSE
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
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple", price_column=CLOSE)
        
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
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="log", price_column=CLOSE)
        
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
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple", price_column=CLOSE)
        
        assert abs(result.iloc[0][FORWARD_RETURN] - (-0.20)) < 1e-6
    
    def test_multiple_tickers_independent(self):
        """Forward returns computed independently per ticker."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY, 0, 5 * MS_PER_DAY],
            TICKER: ["A", "A", "B", "B"],
            CLOSE: [100.0, 120.0, 100.0, 90.0],  # A: +20%, B: -10%
        })
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple", price_column=CLOSE)
        
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
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple", price_column=CLOSE)
        
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
            winsorize_limits=(-0.5, 0.5), price_column=CLOSE
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
            df, lookahead_days=5, return_type="simple", drop_na=True, price_column=CLOSE
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
        
        result_5d = compute_forward_returns(df, lookahead_days=5, return_type="simple", price_column=CLOSE)
        result_10d = compute_forward_returns(df, lookahead_days=10, return_type="simple", price_column=CLOSE)
        
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
        
        result = compute_forward_returns(df, lookahead_days=5, return_type="simple", price_column=CLOSE)
        
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


class TestAdjustedCloseReturns:
    """Tests for AdjClose-based return calculations (total return with dividends)."""
    
    def test_adjclose_used_when_available(self):
        """When AdjClose is available, it should be used for return calculations."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        # Scenario: Stock pays a dividend, so AdjClose drops but return is different
        # Close: 100 -> 100 (price unchanged after dividend)
        # AdjClose: 100 -> 105 (reflects dividend reinvestment)
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 100.0],  # Price unchanged
            ADJCLOSE: [100.0, 105.0],  # +5% total return from dividend
        })
        
        result = compute_forward_returns(
            df, lookahead_days=5, return_type="simple", price_column=ADJCLOSE
        )
        
        # Should use AdjClose: (105 - 100) / 100 = 0.05
        assert abs(result.iloc[0][FORWARD_RETURN] - 0.05) < 1e-6
    
    def test_close_used_when_adjclose_not_available(self):
        """When AdjClose is missing, should fall back to Close with warning."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 110.0],  # No AdjClose column
        })
        
        # Should warn and fall back to Close
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_forward_returns(
                df, lookahead_days=5, return_type="simple", price_column=ADJCLOSE
            )
            
            # Check warning was issued
            assert len(w) == 1
            assert "AdjClose" in str(w[0].message)
            assert "falling back" in str(w[0].message)
        
        # Should still compute using Close
        assert abs(result.iloc[0][FORWARD_RETURN] - 0.10) < 1e-6
    
    def test_explicit_close_price_column(self):
        """Explicitly requesting Close should use Close even if AdjClose exists."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 110.0],
            ADJCLOSE: [100.0, 120.0],  # Different from Close
        })
        
        result = compute_forward_returns(
            df, lookahead_days=5, return_type="simple", price_column=CLOSE
        )
        
        # Should use Close: (110 - 100) / 100 = 0.10, not AdjClose
        assert abs(result.iloc[0][FORWARD_RETURN] - 0.10) < 1e-6
    
    def test_stock_split_scenario(self):
        """AdjClose should handle stock splits correctly.
        
        Example: 2-for-1 split, so Close halves but AdjClose is continuous.
        """
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        # 2-for-1 split scenario
        # Close: 100 -> 50 (looks like -50% loss)
        # AdjClose: 100 -> 100 (correctly shows 0% return)
        df = pd.DataFrame({
            TIMESTAMP: [0, 5 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 50.0],  # Raw price after 2:1 split
            ADJCLOSE: [100.0, 100.0],  # Adjusted - no actual loss
        })
        
        result_close = compute_forward_returns(
            df, lookahead_days=5, return_type="simple", price_column=CLOSE
        )
        result_adj = compute_forward_returns(
            df, lookahead_days=5, return_type="simple", price_column=ADJCLOSE
        )
        
        # Close would incorrectly show -50%
        assert abs(result_close.iloc[0][FORWARD_RETURN] - (-0.50)) < 1e-6
        # AdjClose correctly shows 0%
        assert abs(result_adj.iloc[0][FORWARD_RETURN] - 0.0) < 1e-6
