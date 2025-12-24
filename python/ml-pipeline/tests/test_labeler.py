"""Tests for the labeler module - especially automatic cutoff behavior."""

import pandas as pd
import numpy as np
import pytest

from config.columns import TIMESTAMP, TICKER, CLOSE, TARGET
from config.settings import MS_PER_DAY
from core.labeler import create_labels, get_max_labelable_timestamp


class TestCreateLabels:
    """Tests for create_labels function."""
    
    def test_basic_labeling(self):
        """Test that labels are created correctly."""
        # Create test data: prices that gain 15% after 10 days
        df = pd.DataFrame({
            TIMESTAMP: [0, 10 * MS_PER_DAY, 20 * MS_PER_DAY],
            TICKER: ["A", "A", "A"],
            CLOSE: [100.0, 115.0, 130.0],  # +15%, +13%
        })
        
        result = create_labels(df, lookahead_days=10, gain_threshold_pct=13.0)
        
        # First row should have label (can look ahead to second row)
        # Second row should have label (can look ahead to third row)
        # Third row should be dropped (no future data)
        assert len(result) == 2
        assert TARGET in result.columns
    
    def test_automatic_cutoff_drops_unlabelable_rows(self):
        """Test that rows without enough future data are automatically dropped."""
        # Create data where last 2 rows cannot be labeled (lookahead = 30 days)
        timestamps = [i * 10 * MS_PER_DAY for i in range(10)]  # 0, 10, 20, ... 90 days
        df = pd.DataFrame({
            TIMESTAMP: timestamps,
            TICKER: ["A"] * 10,
            CLOSE: [100.0] * 10,
        })
        
        # With 30-day lookahead, rows at day 70, 80, 90 cannot be labeled
        result = create_labels(df, lookahead_days=30, gain_threshold_pct=10.0)
        
        # Max timestamp that can be labeled: 90 - 30 = 60 days
        max_labelable_ts = 60 * MS_PER_DAY
        
        assert result[TIMESTAMP].max() <= max_labelable_ts
        assert len(result) == 7  # Days 0-60 inclusive (7 rows)
    
    def test_positive_label_above_threshold(self):
        """Test that positive labels are assigned when gain >= threshold."""
        df = pd.DataFrame({
            TIMESTAMP: [0, 10 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 120.0],  # +20% gain
        })
        
        result = create_labels(df, lookahead_days=10, gain_threshold_pct=15.0)
        
        assert len(result) == 1
        assert result.iloc[0][TARGET] == 1  # Gain of 20% >= 15%
    
    def test_negative_label_below_threshold(self):
        """Test that negative labels are assigned when gain < threshold."""
        df = pd.DataFrame({
            TIMESTAMP: [0, 10 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 105.0],  # +5% gain
        })
        
        result = create_labels(df, lookahead_days=10, gain_threshold_pct=15.0)
        
        assert len(result) == 1
        assert result.iloc[0][TARGET] == 0  # Gain of 5% < 15%
    
    def test_multiple_tickers_independent(self):
        """Test that labeling is done independently per ticker."""
        df = pd.DataFrame({
            TIMESTAMP: [0, 10 * MS_PER_DAY, 0, 10 * MS_PER_DAY],
            TICKER: ["A", "A", "B", "B"],
            CLOSE: [100.0, 120.0, 100.0, 90.0],  # A: +20%, B: -10%
        })
        
        result = create_labels(df, lookahead_days=10, gain_threshold_pct=15.0)
        
        assert len(result) == 2  # One labeled row per ticker
        
        result_a = result[result[TICKER] == "A"]
        result_b = result[result[TICKER] == "B"]
        
        assert result_a.iloc[0][TARGET] == 1  # A gained
        assert result_b.iloc[0][TARGET] == 0  # B lost
    
    def test_price_lookup_df_allows_future_data(self):
        """Test that price_lookup_df allows looking up prices beyond df."""
        # df only has first 2 rows
        df = pd.DataFrame({
            TIMESTAMP: [0, 10 * MS_PER_DAY],
            TICKER: ["A", "A"],
            CLOSE: [100.0, 110.0],
        })
        
        # lookup has future price
        lookup_df = pd.DataFrame({
            TIMESTAMP: [0, 10 * MS_PER_DAY, 20 * MS_PER_DAY],
            TICKER: ["A", "A", "A"],
            CLOSE: [100.0, 110.0, 130.0],
        })
        
        # Without lookup_df, second row can't be labeled
        result_no_lookup = create_labels(df, lookahead_days=10, gain_threshold_pct=10.0)
        assert len(result_no_lookup) == 1
        
        # With lookup_df, second row CAN be labeled
        result_with_lookup = create_labels(
            df, lookahead_days=10, gain_threshold_pct=10.0,
            price_lookup_df=lookup_df
        )
        assert len(result_with_lookup) == 2


class TestGetMaxLabelableTimestamp:
    """Tests for get_max_labelable_timestamp function."""
    
    def test_basic_calculation(self):
        """Test basic max labelable timestamp calculation."""
        max_ts = 100 * MS_PER_DAY
        lookahead = 30
        
        result = get_max_labelable_timestamp(max_ts, lookahead)
        
        expected = 70 * MS_PER_DAY
        assert result == expected
    
    def test_with_real_timestamps(self):
        """Test with realistic timestamp values."""
        # Simulate end of 2024
        max_ts = 1735689600000  # 2025-01-01 00:00:00 UTC
        lookahead = 365
        
        result = get_max_labelable_timestamp(max_ts, lookahead)
        
        # Should be ~1 year earlier
        expected = max_ts - (365 * MS_PER_DAY)
        assert result == expected
