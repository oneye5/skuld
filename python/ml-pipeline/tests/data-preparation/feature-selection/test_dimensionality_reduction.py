"""Tests for dimensionality reduction module."""

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
from dimensionality_reduction import PCATransformer, add_pca_features, reduce_with_pca


@pytest.fixture
def sample_data():
    """Create sample data for tests."""
    np.random.seed(42)
    n = 100
    
    # Create correlated features (good for PCA)
    base = np.random.randn(n)
    data = {
        TIMESTAMP: range(n),
        TICKER: ["TEST"] * n,
        "feature_1": base + np.random.randn(n) * 0.1,
        "feature_2": base * 2 + np.random.randn(n) * 0.1,
        "feature_3": base * -1 + np.random.randn(n) * 0.1,
        "feature_4": np.random.randn(n),  # Independent
        "feature_5": np.random.randn(n),  # Independent
        TARGET: np.random.randint(0, 2, n),
    }
    return pd.DataFrame(data)


class TestPCATransformer:
    """Tests for PCATransformer class."""
    
    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self for chaining."""
        transformer = PCATransformer(n_components=2)
        result = transformer.fit(sample_data)
        assert result is transformer
    
    def test_augment_mode_keeps_originals(self, sample_data):
        """Test that augment mode preserves original features."""
        transformer = PCATransformer(n_components=2, augment=True)
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        # Should have original features plus PCA
        assert "feature_1" in result.columns
        assert "PCA_1" in result.columns
        assert "PCA_2" in result.columns
    
    def test_replace_mode_removes_originals(self, sample_data):
        """Test that replace mode removes original features."""
        transformer = PCATransformer(n_components=2, augment=False)
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)
        
        # Should have only metadata, PCA, and target
        assert "feature_1" not in result.columns
        assert "PCA_1" in result.columns
        assert TIMESTAMP in result.columns
        assert TARGET in result.columns
    
    def test_get_result_returns_variance_explained(self, sample_data):
        """Test that result includes variance explained."""
        transformer = PCATransformer(n_components=2)
        transformer.fit(sample_data)
        result = transformer.get_result()
        
        assert result.n_components == 2
        assert len(result.explained_variance_ratio) == 2
        assert result.total_variance_explained > 0
        assert result.total_variance_explained <= 1
    
    def test_fit_on_train_transform_on_test(self, sample_data):
        """Test that fitting on train works for transforming test."""
        train = sample_data.iloc[:80]
        test = sample_data.iloc[80:]
        
        transformer = PCATransformer(n_components=2)
        transformer.fit(train)
        
        train_out = transformer.transform(train)
        test_out = transformer.transform(test)
        
        assert train_out.shape[1] == test_out.shape[1]
        assert "PCA_1" in test_out.columns


class TestAddPCAFeatures:
    """Tests for add_pca_features convenience function."""
    
    def test_adds_pca_to_both_sets(self, sample_data):
        """Test that PCA features are added to train and test."""
        train = sample_data.iloc[:80]
        test = sample_data.iloc[80:]
        
        train_out, test_out, result = add_pca_features(train, test, n_components=3)
        
        assert "PCA_1" in train_out.columns
        assert "PCA_1" in test_out.columns
        assert result.n_components == 3


class TestReduceWithPCA:
    """Tests for reduce_with_pca convenience function."""
    
    def test_replaces_features(self, sample_data):
        """Test that features are replaced with PCA."""
        train = sample_data.iloc[:80]
        test = sample_data.iloc[80:]
        
        train_out, test_out, result = reduce_with_pca(train, test, variance_to_explain=0.95)
        
        assert "feature_1" not in train_out.columns
        assert "PCA_1" in train_out.columns
        assert result.total_variance_explained >= 0.95 or result.n_components == 5
