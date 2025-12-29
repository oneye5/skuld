"""Tests for validation utilities."""

import pytest
import pandas as pd
import numpy as np

from core.validation import (
    ValidationError,
    DataQualityWarning,
    validate_dataframe,
    validate_no_nan,
    validate_wide_data,
    validate_no_lookahead,
    validate_feature_columns,
    validate_groups_match_data,
    validate_ranking_config,
    check_data_quality_report,
)
from config.columns import TIMESTAMP, TICKER, CLOSE


class TestValidateDataframe:
    """Tests for the validate_dataframe decorator."""
    
    def test_missing_required_columns_raises(self):
        """Should raise when required columns are missing."""
        @validate_dataframe(required_cols=[TIMESTAMP, TICKER, "missing_col"])
        def process(df):
            return df
        
        df = pd.DataFrame({TIMESTAMP: [1, 2], TICKER: ["A", "B"]})
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            process(df)
    
    def test_valid_dataframe_passes(self):
        """Should pass when all required columns present."""
        @validate_dataframe(required_cols=[TIMESTAMP, TICKER])
        def process(df):
            return df
        
        df = pd.DataFrame({TIMESTAMP: [1, 2], TICKER: ["A", "B"]})
        result = process(df)
        
        assert len(result) == 2
    
    def test_min_rows_check(self):
        """Should raise when DataFrame has too few rows."""
        @validate_dataframe(min_rows=10)
        def process(df):
            return df
        
        df = pd.DataFrame({TIMESTAMP: [1, 2, 3]})
        
        with pytest.raises(ValidationError, match="has 3 rows, minimum is 10"):
            process(df)
    
    def test_duplicate_check(self):
        """Should raise when duplicates found on specified columns."""
        @validate_dataframe(check_no_duplicates=[TIMESTAMP, TICKER])
        def process(df):
            return df
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 2],
            TICKER: ["A", "A", "B"],  # Duplicate (1, A)
        })
        
        with pytest.raises(ValidationError, match="duplicate rows"):
            process(df)


class TestValidateNoNan:
    """Tests for the validate_no_nan decorator."""
    
    def test_nan_values_raise(self):
        """Should raise when NaN values found in specified columns."""
        @validate_no_nan(columns=["value"])
        def process(df):
            return df
        
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
        
        with pytest.raises(ValidationError, match="has 1 NaN values"):
            process(df)
    
    def test_no_nan_passes(self):
        """Should pass when no NaN values."""
        @validate_no_nan(columns=["value"])
        def process(df):
            return df
        
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = process(df)
        
        assert len(result) == 3


class TestValidateWideData:
    """Tests for validate_wide_data function."""
    
    def test_missing_required_columns(self):
        """Should raise when timestamp/ticker missing."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_wide_data(df, raise_on_error=True)
    
    def test_valid_wide_data(self):
        """Should pass for valid wide format data."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["A", "B", "C"],
            CLOSE: [100.0, 101.0, 102.0],
        })
        
        issues = validate_wide_data(df, raise_on_error=False)
        assert len(issues) == 0
    
    def test_negative_timestamps(self):
        """Should report negative timestamps."""
        df = pd.DataFrame({
            TIMESTAMP: [-1000, 2000, 3000],
            TICKER: ["A", "B", "C"],
        })
        
        issues = validate_wide_data(df, raise_on_error=False)
        assert any("negative timestamps" in issue for issue in issues)
    
    def test_duplicate_pairs(self):
        """Should report duplicate (timestamp, ticker) pairs."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 1000, 2000],
            TICKER: ["A", "A", "B"],  # Duplicate (1000, A)
        })
        
        issues = validate_wide_data(df, raise_on_error=False)
        assert any("duplicate" in issue.lower() for issue in issues)
    
    def test_invalid_close_prices_warns(self):
        """Should warn about invalid Close prices."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["A", "B", "C"],
            CLOSE: [100.0, -50.0, 0.0],  # Invalid
        })
        
        with pytest.warns(DataQualityWarning, match="invalid Close price"):
            validate_wide_data(df, raise_on_error=False)


class TestValidateNoLookahead:
    """Tests for validate_no_lookahead function."""
    
    def test_valid_split_passes(self):
        """Should pass when test comes strictly after train."""
        train_df = pd.DataFrame({TIMESTAMP: [1000, 2000, 3000]})
        test_df = pd.DataFrame({TIMESTAMP: [4000, 5000, 6000]})
        
        result = validate_no_lookahead(train_df, test_df)
        assert result is True
    
    def test_overlapping_raises(self):
        """Should raise when test overlaps with train."""
        train_df = pd.DataFrame({TIMESTAMP: [1000, 2000, 3000]})
        test_df = pd.DataFrame({TIMESTAMP: [3000, 4000, 5000]})  # 3000 overlaps
        
        with pytest.raises(ValidationError, match="Lookahead bias"):
            validate_no_lookahead(train_df, test_df)
    
    def test_test_before_train_raises(self):
        """Should raise when test comes before train."""
        train_df = pd.DataFrame({TIMESTAMP: [4000, 5000, 6000]})
        test_df = pd.DataFrame({TIMESTAMP: [1000, 2000, 3000]})
        
        with pytest.raises(ValidationError, match="Lookahead bias"):
            validate_no_lookahead(train_df, test_df)


class TestValidateGroupsMatchData:
    """Tests for validate_groups_match_data function."""
    
    def test_matching_groups_pass(self):
        """Should pass when sum(groups) == len(X)."""
        X = pd.DataFrame({"a": range(100)})
        groups = [20, 30, 50]  # Sum = 100
        
        result = validate_groups_match_data(X, groups)
        assert result is True
    
    def test_mismatched_groups_raise(self):
        """Should raise when groups don't match data size."""
        X = pd.DataFrame({"a": range(100)})
        groups = [20, 30, 40]  # Sum = 90, not 100
        
        with pytest.raises(ValidationError, match="don't match data"):
            validate_groups_match_data(X, groups)


class TestValidateRankingConfig:
    """Tests for validate_ranking_config function."""
    
    def test_valid_config_no_warnings(self):
        """Should return empty list for valid config."""
        warnings = validate_ranking_config(
            forward_days=20,
            top_n=5,
            bottom_n=5,
            min_stocks=15,
        )
        assert len(warnings) == 0
    
    def test_invalid_forward_days_raises(self):
        """Should raise for non-positive forward_days."""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_ranking_config(forward_days=0, top_n=5, bottom_n=5, min_stocks=10)
    
    def test_long_horizon_warns(self):
        """Should warn for very long forward horizons."""
        warnings = validate_ranking_config(
            forward_days=500,  # > 365
            top_n=5,
            bottom_n=5,
            min_stocks=15,
        )
        assert any("long" in w.lower() for w in warnings)
    
    def test_insufficient_min_stocks_warns(self):
        """Should warn when min_stocks < top_n + bottom_n."""
        warnings = validate_ranking_config(
            forward_days=20,
            top_n=10,
            bottom_n=10,
            min_stocks=15,  # < 10 + 10
        )
        assert any("insufficient" in w.lower() for w in warnings)


class TestCheckDataQualityReport:
    """Tests for check_data_quality_report function."""
    
    def test_basic_report_generation(self):
        """Should generate report with expected keys."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["A", "B", "C"],
            "value": [1.0, 2.0, 3.0],
        })
        
        report = check_data_quality_report(df)
        
        assert "n_rows" in report
        assert report["n_rows"] == 3
        assert "n_columns" in report
        assert "memory_mb" in report
        assert "n_unique_timestamps" in report
        assert report["n_unique_timestamps"] == 3
        assert "n_unique_tickers" in report
        assert report["n_unique_tickers"] == 3
    
    def test_detects_missing_values(self):
        """Should detect columns with missing values."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["A", "B", "C"],
            "value": [1.0, np.nan, 3.0],
        })
        
        report = check_data_quality_report(df)
        
        assert report["columns_with_missing"] > 0
    
    def test_detects_infinities(self):
        """Should detect columns with infinity values."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["A", "B", "C"],
            "value": [1.0, np.inf, 3.0],
        })
        
        report = check_data_quality_report(df)
        
        assert "value" in report["columns_with_infinities"]
