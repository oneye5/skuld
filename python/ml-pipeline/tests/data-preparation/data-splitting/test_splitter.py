"""Tests for train-test splitting."""

import pandas as pd
import pytest

from config.column_names import TIMESTAMP, TICKER
from splitter import split_by_timestamp


class TestSplitByTimestamp:
    """Tests for split_by_timestamp function."""
    
    def test_splits_at_correct_timestamp(self):
        """Should split data at the specified timestamp."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000, 4000, 5000],
            TICKER: ["A"] * 5,
            "value": [1, 2, 3, 4, 5],
        })
        
        result = split_by_timestamp(df, train_end_ts=3000)
        
        # Train: 1000, 2000 (< 3000)
        assert len(result.train) == 2
        # Test: 3000, 4000, 5000 (>= 3000)
        assert len(result.test) == 3
    
    def test_train_contains_before_end_ts(self):
        """Train set should contain all data before train_end_ts."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["A"] * 3,
        })
        
        result = split_by_timestamp(df, train_end_ts=2500)
        
        assert (result.train[TIMESTAMP] < 2500).all()
    
    def test_test_contains_after_test_start_ts(self):
        """Test set should contain all data at or after test_start_ts."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["A"] * 3,
        })
        
        result = split_by_timestamp(df, train_end_ts=2000, test_start_ts=2000)
        
        assert (result.test[TIMESTAMP] >= 2000).all()
    
    def test_respects_test_end_ts(self):
        """Should exclude data after test_end_ts."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000, 4000],
            TICKER: ["A"] * 4,
        })
        
        result = split_by_timestamp(df, train_end_ts=2000, test_end_ts=3500)
        
        # Test should include 2000, 3000 but not 4000
        assert len(result.test) == 2
        assert 4000 not in result.test[TIMESTAMP].values
    
    def test_handles_empty_train(self):
        """Should handle case where train would be empty."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000],
            TICKER: ["A"] * 2,
        })
        
        result = split_by_timestamp(df, train_end_ts=500)
        
        assert len(result.train) == 0
        assert len(result.test) == 2
