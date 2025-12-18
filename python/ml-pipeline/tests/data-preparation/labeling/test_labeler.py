"""Tests for labeling module."""

import pandas as pd
import numpy as np
import pytest

from config.column_names import TIMESTAMP, TICKER, CLOSE, TARGET
from config.model_config import MS_PER_DAY
from labeler import create_labels


class TestCreateLabels:
    """Tests for create_labels function."""
    
    def test_positive_label_when_gain_exceeds_threshold(self):
        """Should label as 1 when price gain >= threshold."""
        # 10 -> 12 = 20% gain, threshold 2%
        df = pd.DataFrame({
            TIMESTAMP: [0, 365 * MS_PER_DAY],
            TICKER: ["ANZ.NZ", "ANZ.NZ"],
            CLOSE: [10.0, 12.0],
        })
        
        result = create_labels(df, lookahead_days=365, gain_threshold_pct=2.0)
        
        # First row should have target=1 (20% > 2%)
        assert result[result[TIMESTAMP] == 0][TARGET].values[0] == 1
    
    def test_negative_label_when_gain_below_threshold(self):
        """Should label as 0 when price gain < threshold."""
        # 10 -> 10.1 = 1% gain, threshold 2%
        df = pd.DataFrame({
            TIMESTAMP: [0, 365 * MS_PER_DAY],
            TICKER: ["ANZ.NZ", "ANZ.NZ"],
            CLOSE: [10.0, 10.1],
        })
        
        result = create_labels(df, lookahead_days=365, gain_threshold_pct=2.0)
        
        assert result[result[TIMESTAMP] == 0][TARGET].values[0] == 0
    
    def test_drops_rows_without_future_data(self):
        """Should drop rows where future price cannot be determined."""
        df = pd.DataFrame({
            TIMESTAMP: [0, 100 * MS_PER_DAY],  # Only 100 days apart
            TICKER: ["ANZ.NZ", "ANZ.NZ"],
            CLOSE: [10.0, 12.0],
        })
        
        result = create_labels(df, lookahead_days=365, gain_threshold_pct=2.0)
        
        # No rows should have valid targets (can't see 365 days ahead)
        assert len(result) == 0
    
    def test_handles_multiple_tickers(self):
        """Should process each ticker independently."""
        df = pd.DataFrame({
            TIMESTAMP: [0, 0, 365 * MS_PER_DAY, 365 * MS_PER_DAY],
            TICKER: ["ANZ.NZ", "BNZ.NZ", "ANZ.NZ", "BNZ.NZ"],
            CLOSE: [10.0, 20.0, 12.0, 19.0],  # ANZ +20%, BNZ -5%
        })
        
        result = create_labels(df, lookahead_days=365, gain_threshold_pct=2.0)
        
        anz_label = result[(result[TIMESTAMP] == 0) & (result[TICKER] == "ANZ.NZ")][TARGET].values[0]
        bnz_label = result[(result[TIMESTAMP] == 0) & (result[TICKER] == "BNZ.NZ")][TARGET].values[0]
        
        assert anz_label == 1  # 20% gain
        assert bnz_label == 0  # -5% loss
    
    def test_exact_threshold_is_positive(self):
        """Should label as 1 when gain equals exactly threshold."""
        # 10 -> 10.21 = 2.1% (slightly above to avoid floating point issues)
        df = pd.DataFrame({
            TIMESTAMP: [0, 365 * MS_PER_DAY],
            TICKER: ["ANZ.NZ", "ANZ.NZ"],
            CLOSE: [10.0, 10.21],
        })
        
        result = create_labels(df, lookahead_days=365, gain_threshold_pct=2.0)
        
        assert result[result[TIMESTAMP] == 0][TARGET].values[0] == 1
