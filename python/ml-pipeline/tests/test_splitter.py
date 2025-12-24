"""Tests for the splitter module."""

import pandas as pd
import pytest

from config.columns import TIMESTAMP, TICKER
from config.settings import MS_PER_DAY
from core.splitter import split_by_timestamp, calculate_window_timestamps


class TestSplitByTimestamp:
    """Tests for split_by_timestamp function."""
    
    def test_basic_split(self):
        """Test basic train/test split by timestamp."""
        df = pd.DataFrame({
            TIMESTAMP: [10, 20, 30, 40, 50],
            TICKER: ["A"] * 5,
            "value": [1, 2, 3, 4, 5],
        })
        
        result = split_by_timestamp(df, train_end_ts=30, test_end_ts=50)
        
        # Train: timestamps < 30 (10, 20)
        assert len(result.train) == 2
        assert result.train[TIMESTAMP].tolist() == [10, 20]
        
        # Test: timestamps >= 30 and < 50 (30, 40)
        assert len(result.test) == 2
        assert result.test[TIMESTAMP].tolist() == [30, 40]
    
    def test_split_metadata(self):
        """Test that split metadata is correct."""
        df = pd.DataFrame({
            TIMESTAMP: [10, 20, 30, 40, 50],
            TICKER: ["A"] * 5,
        })
        
        result = split_by_timestamp(df, train_end_ts=30, test_end_ts=50)
        
        assert result.train_start_ts == 10
        assert result.train_end_ts == 30
        assert result.test_start_ts == 30
        assert result.test_end_ts == 50
    
    def test_empty_train(self):
        """Test split when train would be empty."""
        df = pd.DataFrame({
            TIMESTAMP: [30, 40, 50],
            TICKER: ["A"] * 3,
        })
        
        result = split_by_timestamp(df, train_end_ts=30, test_end_ts=50)
        
        assert len(result.train) == 0
        assert len(result.test) == 2
    
    def test_empty_test(self):
        """Test split when test would be empty."""
        df = pd.DataFrame({
            TIMESTAMP: [10, 20, 25],
            TICKER: ["A"] * 3,
        })
        
        result = split_by_timestamp(df, train_end_ts=30, test_end_ts=50)
        
        assert len(result.train) == 3
        assert len(result.test) == 0


class TestCalculateWindowTimestamps:
    """Tests for calculate_window_timestamps function."""
    
    def test_single_window(self):
        """Test calculation for a single window."""
        data_max_ts = 100 * MS_PER_DAY
        
        result = calculate_window_timestamps(
            data_max_ts,
            num_windows=1,
            window_movement_years=1.0,
            lookahead_days=10,
            test_period_years=0.1,  # ~36.5 days
        )
        
        assert len(result) == 1
        train_end, test_end = result[0]
        
        # test_end should be data_max - lookahead
        expected_test_end = data_max_ts - (10 * MS_PER_DAY)
        assert test_end == expected_test_end
    
    def test_multiple_windows_move_backward(self):
        """Test that windows move backward in time."""
        data_max_ts = 1000 * MS_PER_DAY
        
        result = calculate_window_timestamps(
            data_max_ts,
            num_windows=3,
            window_movement_years=1.0,  # ~365 days
            lookahead_days=30,
            test_period_years=0.5,
        )
        
        assert len(result) == 3
        
        # Each window should have earlier test_end than the previous
        for i in range(1, len(result)):
            prev_train_end, prev_test_end = result[i - 1]
            curr_train_end, curr_test_end = result[i]
            
            assert curr_test_end < prev_test_end, "Windows should move backward"
            assert curr_train_end < prev_train_end, "Train end should also move back"
    
    def test_lookahead_respected(self):
        """Test that lookahead buffer is respected."""
        data_max_ts = 100 * MS_PER_DAY
        lookahead_days = 30
        
        result = calculate_window_timestamps(
            data_max_ts,
            num_windows=1,
            window_movement_years=1.0,
            lookahead_days=lookahead_days,
            test_period_years=0.1,
        )
        
        _, test_end = result[0]
        
        # test_end must leave room for lookahead
        assert test_end <= data_max_ts - (lookahead_days * MS_PER_DAY)
