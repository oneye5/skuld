"""Tests for price anomaly detection in preprocessor module."""

import pytest
import pandas as pd
import numpy as np

from core.preprocessor import (
    detect_price_anomalies,
    filter_anomalous_data,
    get_anomaly_summary,
)
from config.columns import TIMESTAMP, TICKER


class TestDetectPriceAnomalies:
    """Tests for detect_price_anomalies function."""
    
    def test_detects_extreme_positive_return(self):
        """Should flag returns exceeding threshold."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ['A'] * 5,
            'Close': [100, 102, 105, 350, 340],  # Day 4: +233% jump
        })
        
        result = detect_price_anomalies(df, return_threshold=2.0)
        
        assert '_is_anomaly' in result.columns
        assert '_daily_return' in result.columns
        
        # Day 4 (index 3) should be flagged
        anomalies = result[result['_is_anomaly']]
        assert len(anomalies) >= 1
        assert any(result.loc[result['Close'] == 350, '_is_anomaly'])
    
    def test_detects_extreme_negative_return(self):
        """Should flag extreme drops (potential reverse splits or errors)."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ['A'] * 5,
            'Close': [100, 102, 105, 20, 21],  # Day 4: -81% drop
        })
        
        # Use lower threshold to catch this
        result = detect_price_anomalies(df, return_threshold=0.5)
        
        anomalies = result[result['_is_anomaly']]
        assert len(anomalies) >= 1
        assert any(result.loc[result['Close'] == 20, '_is_anomaly'])
    
    def test_records_first_anomaly_timestamp(self):
        """Should record the timestamp of first anomaly per ticker."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ['A'] * 5,
            'Close': [100, 102, 350, 340, 1200],  # Two big jumps
        })
        
        result = detect_price_anomalies(df, return_threshold=2.0)
        
        # Should have _anomaly_timestamp column
        assert '_anomaly_timestamp' in result.columns
        
        # First anomaly is at timestamp 3 (the first big jump)
        assert result['_anomaly_timestamp'].iloc[0] == 3
    
    def test_handles_multiple_tickers_independently(self):
        """Should compute returns within each ticker, not across tickers."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 1, 2, 3],
            TICKER: ['A', 'A', 'A', 'B', 'B', 'B'],
            'Close': [100, 102, 105, 50, 51, 52],  # Normal moves for both
        })
        
        result = detect_price_anomalies(df, return_threshold=2.0)
        
        # No anomalies should be detected
        anomalies = result[result['_is_anomaly']]
        assert len(anomalies) == 0
    
    def test_preserves_all_original_columns(self):
        """Should keep all original columns plus new ones."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['A'] * 3,
            'Close': [100, 102, 105],
            'Volume': [1000, 1100, 1200],
            'custom_feature': [1.0, 2.0, 3.0],
        })
        
        result = detect_price_anomalies(df)
        
        assert all(col in result.columns for col in df.columns)
        assert '_is_anomaly' in result.columns
        assert '_daily_return' in result.columns
    
    def test_handles_empty_dataframe(self):
        """Should handle empty input gracefully."""
        df = pd.DataFrame(columns=[TIMESTAMP, TICKER, 'Close'])
        result = detect_price_anomalies(df)
        assert len(result) == 0
    
    def test_handles_missing_price_column(self):
        """Should return unchanged if price column doesn't exist."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['A'] * 3,
        })
        result = detect_price_anomalies(df, price_col='Close')
        assert '_is_anomaly' not in result.columns


class TestFilterAnomalousData:
    """Tests for filter_anomalous_data function."""
    
    def test_trims_data_before_anomaly(self):
        """Should remove all data BEFORE the anomaly, keeping newer series."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ['A'] * 5,
            'Close': [100, 102, 350, 340, 345],  # Anomaly at ts=3
        })
        df = detect_price_anomalies(df, return_threshold=2.0)
        
        filtered, removed = filter_anomalous_data(df, trim_before_anomaly=True)
        
        # Should keep ts=3,4,5 (from anomaly onwards)
        assert len(filtered) == 3
        assert filtered[TIMESTAMP].min() == 3
        
        # Should remove ts=1,2 (before anomaly)
        assert len(removed) == 2
        assert removed[TIMESTAMP].max() == 2
    
    def test_keeps_tickers_without_anomalies(self):
        """Should not affect tickers that have no anomalies."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 1, 2, 3],
            TICKER: ['GOOD', 'GOOD', 'GOOD', 'BAD', 'BAD', 'BAD'],
            'Close': [100, 102, 105, 10, 100, 98],  # BAD has 10x jump
        })
        df = detect_price_anomalies(df, return_threshold=2.0)
        
        filtered, removed = filter_anomalous_data(df, trim_before_anomaly=True)
        
        # GOOD ticker should be fully preserved
        good_data = filtered[filtered[TICKER] == 'GOOD']
        assert len(good_data) == 3
        
        # BAD ticker should have ts=1 trimmed
        bad_data = filtered[filtered[TICKER] == 'BAD']
        assert len(bad_data) == 2  # ts=2,3 kept
        assert bad_data[TIMESTAMP].min() == 2  # anomaly point onwards
    
    def test_simple_removal_mode(self):
        """With trim_before_anomaly=False, should just remove anomaly rows."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ['A'] * 5,
            'Close': [100, 102, 350, 340, 345],
        })
        df = detect_price_anomalies(df, return_threshold=2.0)
        
        filtered, removed = filter_anomalous_data(df, trim_before_anomaly=False)
        
        # Only anomaly row removed, others kept
        assert len(removed) == 1
        assert removed[TIMESTAMP].iloc[0] == 3
    
    def test_cleans_up_metadata_columns(self):
        """Should remove anomaly metadata columns from output."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ['A'] * 5,
            'Close': [100, 102, 350, 340, 345],
        })
        df = detect_price_anomalies(df, return_threshold=2.0)
        
        filtered, _ = filter_anomalous_data(df, trim_before_anomaly=True)
        
        # Metadata columns should be removed
        assert '_is_anomaly' not in filtered.columns
        assert '_anomaly_timestamp' not in filtered.columns
        assert '_daily_return' not in filtered.columns
    
    def test_warns_if_no_anomaly_column(self):
        """Should warn if _is_anomaly column is missing."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['A'] * 3,
            'Close': [100, 102, 105],
        })
        
        with pytest.warns(UserWarning, match="No _is_anomaly column found"):
            filter_anomalous_data(df)


class TestGetAnomalySummary:
    """Tests for get_anomaly_summary function."""
    
    def test_returns_correct_counts(self):
        """Should return accurate anomaly statistics."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ['A'] * 5,
            'Close': [100, 102, 350, 340, 345],
        })
        df = detect_price_anomalies(df, return_threshold=2.0)
        
        summary = get_anomaly_summary(df)
        
        assert summary['total_rows'] == 5
        assert summary['anomaly_rows'] == 1  # Just the jump
        assert summary['rows_to_trim'] == 2  # ts=1,2 would be trimmed
        assert summary['n_affected_tickers'] == 1
        assert 'A' in summary['affected_tickers']
    
    def test_reports_extreme_returns(self):
        """Should report max/min extreme returns."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ['A'] * 5,
            'Close': [100, 102, 350, 340, 345],
        })
        df = detect_price_anomalies(df, return_threshold=2.0)
        
        summary = get_anomaly_summary(df)
        
        assert 'max_return' in summary
        assert summary['max_return'] > 2.0  # The 233% jump
    
    def test_handles_no_anomalies(self):
        """Should handle case with no anomalies."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ['A'] * 3,
            'Close': [100, 102, 105],
        })
        df = detect_price_anomalies(df, return_threshold=2.0)
        
        summary = get_anomaly_summary(df)
        
        assert summary['anomaly_rows'] == 0
        assert summary['rows_to_trim'] == 0
        assert summary['n_affected_tickers'] == 0


class TestRealWorldScenarios:
    """Integration tests simulating real data quality issues."""
    
    def test_ticker_recycling_trims_old_company(self):
        """Should trim old company data when ticker is recycled."""
        # Simulating: old company delisted at $54, new company listed at $815
        df = pd.DataFrame({
            TIMESTAMP: list(range(1, 11)),
            TICKER: ['RECYCLE'] * 10,
            'Close': [50, 51, 52, 53, 54, 815, 820, 825, 830, 835],
        })
        
        result = detect_price_anomalies(df, return_threshold=2.0)
        filtered, removed = filter_anomalous_data(result, trim_before_anomaly=True)
        
        # Should keep only ts >= 6 (the new company)
        assert len(filtered) == 5
        assert filtered[TIMESTAMP].min() == 6
        assert filtered['Close'].min() == 815
        
        # Old company data (ts 1-5) should be removed
        assert len(removed) == 5
        assert removed['Close'].max() == 54
    
    def test_stock_split_trims_unadjusted_period(self):
        """Should trim pre-split data if prices weren't adjusted."""
        # Simulating a 10:1 split where prices weren't adjusted
        df = pd.DataFrame({
            TIMESTAMP: list(range(1, 11)),
            TICKER: ['SPLIT'] * 10,
            'Close': [1000, 1010, 1020, 1030, 103, 104, 105, 106, 107, 108],
        })
        
        result = detect_price_anomalies(df, return_threshold=0.5)
        filtered, removed = filter_anomalous_data(result, trim_before_anomaly=True)
        
        # Should keep only post-split data
        assert filtered[TIMESTAMP].min() == 5  # Split happened at ts=5
        assert filtered['Close'].max() < 200  # All post-split prices
    
    def test_pipeline_integration_mixed_tickers(self):
        """Test full pipeline with mix of good and bad tickers."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            TICKER: ['GOOD']*5 + ['RECYCLE']*5 + ['SPLIT']*5,
            'Close': [
                100, 102, 104, 106, 108,           # GOOD: normal
                50, 52, 54, 815, 820,              # RECYCLE: ticker recycling at ts=4 (1409% jump)
                100, 101, 505, 510, 515,           # SPLIT: 5x jump at ts=3 (400% return)
            ],
        })
        
        # Step 1: Detect
        df = detect_price_anomalies(df, return_threshold=2.0)
        
        # Step 2: Summarize
        summary = get_anomaly_summary(df)
        assert summary['n_affected_tickers'] == 2  # RECYCLE and SPLIT
        
        # Step 3: Filter
        filtered, removed = filter_anomalous_data(df, trim_before_anomaly=True)
        
        # GOOD should be intact (5 rows)
        good = filtered[filtered[TICKER] == 'GOOD']
        assert len(good) == 5
        
        # RECYCLE: keep ts >= 4 (2 rows)
        recycle = filtered[filtered[TICKER] == 'RECYCLE']
        assert len(recycle) == 2
        assert recycle[TIMESTAMP].min() == 4
        
        # SPLIT: keep ts >= 3 (3 rows)
        split = filtered[filtered[TICKER] == 'SPLIT']
        assert len(split) == 3
        assert split[TIMESTAMP].min() == 3
