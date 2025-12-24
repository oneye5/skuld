"""Tests for the scaler module."""

import pandas as pd
import numpy as np
import pytest

from config.columns import TIMESTAMP, TICKER, TARGET
from core.scaler import fit_scaler, transform_data


class TestFitScaler:
    """Tests for fit_scaler function."""
    
    def test_identifies_binary_columns(self):
        """Test that binary columns are identified correctly."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "B", "C"],
            "binary_col": [0, 1, 0],
            "continuous_col": [1.5, 2.5, 3.5],
        })
        
        scaler_set = fit_scaler(df)
        
        assert "binary_col" in scaler_set.binary_cols
        assert "continuous_col" in scaler_set.continuous_cols
    
    def test_excludes_metadata_columns(self):
        """Test that timestamp, ticker, target are excluded from scaling."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "B", "C"],
            TARGET: [0, 1, 0],
            "feature": [1.5, 2.5, 3.5],
        })
        
        scaler_set = fit_scaler(df)
        
        assert TIMESTAMP in scaler_set.excluded_cols
        assert TARGET in scaler_set.excluded_cols
        assert "feature" in scaler_set.continuous_cols


class TestTransformData:
    """Tests for transform_data function."""
    
    def test_transforms_continuous_columns(self):
        """Test that continuous columns are scaled."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "B", "C"],
            "feature": [10.0, 20.0, 30.0],
        })
        
        scaler_set = fit_scaler(df)
        result = transform_data(df, scaler_set)
        
        # Values should be different after scaling
        assert not np.allclose(result["feature"].values, df["feature"].values)
    
    def test_preserves_binary_columns(self):
        """Test that binary columns are not scaled."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "B", "C"],
            "binary": [0, 1, 1],
            "continuous": [10.0, 20.0, 30.0],
        })
        
        scaler_set = fit_scaler(df)
        result = transform_data(df, scaler_set)
        
        # Binary column should be unchanged
        np.testing.assert_array_equal(result["binary"].values, df["binary"].values)
    
    def test_preserves_excluded_columns(self):
        """Test that excluded columns are not modified."""
        df = pd.DataFrame({
            TIMESTAMP: [1000, 2000, 3000],
            TICKER: ["A", "B", "C"],
            "feature": [10.0, 20.0, 30.0],
        })
        
        scaler_set = fit_scaler(df)
        result = transform_data(df, scaler_set)
        
        # Timestamp should be unchanged
        np.testing.assert_array_equal(result[TIMESTAMP].values, df[TIMESTAMP].values)
    
    def test_robust_scaler_handles_outliers(self):
        """Test that RobustScaler handles outliers well."""
        # Data with an outlier
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3, 4, 5],
            TICKER: ["A"] * 5,
            "feature": [10.0, 11.0, 12.0, 13.0, 1000.0],  # 1000 is outlier
        })
        
        scaler_set = fit_scaler(df)
        result = transform_data(df, scaler_set)
        
        # Non-outlier values should be relatively close to each other
        non_outlier_values = result["feature"].values[:4]
        assert np.std(non_outlier_values) < 1.0  # Small std after scaling
