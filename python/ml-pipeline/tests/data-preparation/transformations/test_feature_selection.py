"""Tests for feature selection module."""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add paths for imports
_ml_pipeline = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "feature-selection"))

from config.column_names import TIMESTAMP, TICKER, TARGET
from filter_selection import (
    select_features,
    get_feature_columns,
    _drop_high_missing,
    _drop_low_variance,
    _drop_high_correlation,
    compute_feature_stats,
)


@pytest.fixture
def sample_train_data():
    """Create sample training data for tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        TIMESTAMP: range(n),
        TICKER: ["TEST.NZ"] * n,
        "feature_good": np.random.randn(n),  # Good variance
        "feature_low_var": [1.0] * n,  # Zero variance
        "feature_missing": [np.nan] * 96 + [1.0] * 4,  # 96% missing
        "feature_corr_a": np.random.randn(n),  # Will correlate with corr_b
        TARGET: np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_test_data():
    """Create sample test data matching train structure."""
    np.random.seed(123)
    n = 50
    return pd.DataFrame({
        TIMESTAMP: range(n),
        TICKER: ["TEST.NZ"] * n,
        "feature_good": np.random.randn(n),
        "feature_low_var": [1.0] * n,
        "feature_missing": [np.nan] * 48 + [1.0] * 2,
        "feature_corr_a": np.random.randn(n),
        TARGET: np.random.randint(0, 2, n),
    })


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""
    
    def test_excludes_metadata(self, sample_train_data):
        """Test that metadata columns are excluded."""
        feature_cols = get_feature_columns(sample_train_data)
        
        assert TIMESTAMP not in feature_cols
        assert TICKER not in feature_cols
        assert TARGET not in feature_cols
    
    def test_includes_features(self, sample_train_data):
        """Test that feature columns are included."""
        feature_cols = get_feature_columns(sample_train_data)
        
        assert "feature_good" in feature_cols
        assert "feature_low_var" in feature_cols


class TestDropHighMissing:
    """Tests for _drop_high_missing function."""
    
    def test_drops_high_missing(self, sample_train_data):
        """Test that features with high missing ratio are dropped."""
        feature_cols = ["feature_good", "feature_missing"]
        kept, dropped = _drop_high_missing(sample_train_data, feature_cols, 0.95)
        
        assert "feature_good" in kept
        assert "feature_missing" in dropped
    
    def test_keeps_low_missing(self, sample_train_data):
        """Test that features with low missing ratio are kept."""
        feature_cols = ["feature_good"]
        kept, dropped = _drop_high_missing(sample_train_data, feature_cols, 0.95)
        
        assert "feature_good" in kept
        assert len(dropped) == 0


class TestDropLowVariance:
    """Tests for _drop_low_variance function."""
    
    def test_drops_zero_variance(self, sample_train_data):
        """Test that zero variance features are dropped."""
        feature_cols = ["feature_good", "feature_low_var"]
        kept, dropped = _drop_low_variance(sample_train_data, feature_cols, 0.01)
        
        assert "feature_good" in kept
        assert "feature_low_var" in dropped
    
    def test_keeps_sufficient_variance(self):
        """Test that features with sufficient variance are kept."""
        df = pd.DataFrame({
            "high_var": np.random.randn(100) * 10,
            "low_var": np.ones(100) + np.random.randn(100) * 0.001,
        })
        
        kept, dropped = _drop_low_variance(df, ["high_var", "low_var"], 0.01)
        
        assert "high_var" in kept


class TestDropHighCorrelation:
    """Tests for _drop_high_correlation function."""
    
    def test_drops_highly_correlated(self):
        """Test that highly correlated features are dropped."""
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame({
            "original": base,
            "correlated": base + np.random.randn(100) * 0.01,  # Almost identical
            "uncorrelated": np.random.randn(100),
        })
        
        kept, dropped = _drop_high_correlation(
            df, ["original", "correlated", "uncorrelated"], 0.95
        )
        
        # Original should be kept, correlated dropped
        assert "original" in kept
        assert "correlated" in dropped
        assert "uncorrelated" in kept
    
    def test_keeps_first_of_correlated_pair(self):
        """Test that the first feature in a correlated pair is kept."""
        np.random.seed(42)
        base = np.random.randn(100)
        df = pd.DataFrame({
            "first": base,
            "second": base * 1.0,  # Perfectly correlated
        })
        
        kept, dropped = _drop_high_correlation(df, ["first", "second"], 0.95)
        
        assert "first" in kept
        assert "second" in dropped


class TestSelectFeatures:
    """Tests for the main select_features function."""
    
    def test_returns_filtered_dataframes(self, sample_train_data, sample_test_data):
        """Test that filtered DataFrames are returned."""
        train_out, test_out, dropped = select_features(
            sample_train_data, sample_test_data
        )
        
        assert isinstance(train_out, pd.DataFrame)
        assert isinstance(test_out, pd.DataFrame)
        assert isinstance(dropped, list)
    
    def test_drops_low_variance_features(self, sample_train_data, sample_test_data):
        """Test that low variance features are dropped."""
        train_out, test_out, dropped = select_features(
            sample_train_data, sample_test_data,
            variance_threshold=0.01
        )
        
        assert "feature_low_var" in dropped
        assert "feature_low_var" not in train_out.columns
        assert "feature_low_var" not in test_out.columns
    
    def test_drops_high_missing_features(self, sample_train_data, sample_test_data):
        """Test that features with high missing are dropped."""
        train_out, test_out, dropped = select_features(
            sample_train_data, sample_test_data,
            missing_threshold=0.95
        )
        
        assert "feature_missing" in dropped
    
    def test_preserves_metadata_and_target(self, sample_train_data, sample_test_data):
        """Test that metadata and target columns are preserved."""
        train_out, test_out, dropped = select_features(
            sample_train_data, sample_test_data
        )
        
        assert TIMESTAMP in train_out.columns
        assert TICKER in train_out.columns
        assert TARGET in train_out.columns
    
    def test_same_columns_in_train_and_test(self, sample_train_data, sample_test_data):
        """Test that train and test have the same columns after selection."""
        train_out, test_out, dropped = select_features(
            sample_train_data, sample_test_data
        )
        
        assert list(train_out.columns) == list(test_out.columns)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_train = pd.DataFrame(columns=[TIMESTAMP, TICKER, "feature", TARGET])
        empty_test = pd.DataFrame(columns=[TIMESTAMP, TICKER, "feature", TARGET])
        
        train_out, test_out, dropped = select_features(empty_train, empty_test)
        
        assert train_out.empty
        assert test_out.empty


class TestComputeFeatureStats:
    """Tests for compute_feature_stats function."""
    
    def test_returns_stats_dataframe(self, sample_train_data):
        """Test that stats DataFrame is returned."""
        stats = compute_feature_stats(sample_train_data)
        
        assert isinstance(stats, pd.DataFrame)
        assert "feature" in stats.columns
        assert "variance" in stats.columns
        assert "missing_ratio" in stats.columns
    
    def test_stats_for_all_features(self, sample_train_data):
        """Test that stats are computed for all features."""
        stats = compute_feature_stats(sample_train_data)
        features = stats["feature"].tolist()
        
        assert "feature_good" in features
        assert "feature_low_var" in features
