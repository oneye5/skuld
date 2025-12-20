"""Tests for lag features module."""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add paths for imports
_ml_pipeline = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "transformations"))

from config.column_names import TIMESTAMP, TICKER, MACRO_PREFIX
from lag_features import add_macro_lag_features, add_ticker_lag_features


@pytest.fixture
def sample_data_with_macro() -> pd.DataFrame:
    """Create sample data with macro features."""
    n_rows = 200
    base_ts = 1600000000000
    
    # Create timestamps (daily)
    timestamps = [base_ts + i * 86400000 for i in range(n_rows)]
    
    return pd.DataFrame({
        TIMESTAMP: timestamps,
        TICKER: "TEST.NZ",
        "Close": np.random.uniform(90, 110, n_rows),
        f"{MACRO_PREFIX}GDP": np.linspace(100, 120, n_rows),  # Steadily increasing
        f"{MACRO_PREFIX}CPI": np.linspace(2, 4, n_rows),  # Steadily increasing
    })


@pytest.fixture
def sample_data_with_technical() -> pd.DataFrame:
    """Create sample data with technical features."""
    n_rows = 100
    base_ts = 1600000000000
    
    timestamps = [base_ts + i * 86400000 for i in range(n_rows)]
    
    return pd.DataFrame({
        TIMESTAMP: timestamps,
        TICKER: "TEST.NZ",
        "Close": np.random.uniform(90, 110, n_rows),
        "return_1d": np.random.normal(0, 2, n_rows),
        "return_5d": np.random.normal(0, 5, n_rows),
        "return_20d": np.random.normal(0, 10, n_rows),
        "rsi_14": np.random.uniform(30, 70, n_rows),
        "price_to_sma_20": np.random.normal(0, 5, n_rows),
        "volatility_20d": np.random.uniform(1, 5, n_rows),
    })


class TestAddMacroLagFeatures:
    """Tests for macro lag feature creation."""
    
    def test_creates_lag_columns(self, sample_data_with_macro):
        """Test that lag columns are created for macro features."""
        result = add_macro_lag_features(sample_data_with_macro)
        
        # Check default lags (30, 90 days)
        for lag in [30, 90]:
            assert f"{MACRO_PREFIX}GDP_change_{lag}d" in result.columns
            assert f"{MACRO_PREFIX}CPI_change_{lag}d" in result.columns
    
    def test_custom_lags(self, sample_data_with_macro):
        """Test with custom lag periods."""
        result = add_macro_lag_features(sample_data_with_macro, lags=[10, 20])
        
        assert f"{MACRO_PREFIX}GDP_change_10d" in result.columns
        assert f"{MACRO_PREFIX}GDP_change_20d" in result.columns
        # Default lags should not be present
        assert f"{MACRO_PREFIX}GDP_change_30d" not in result.columns
    
    def test_lag_values_correct(self, sample_data_with_macro):
        """Test that lag values are calculated correctly."""
        result = add_macro_lag_features(sample_data_with_macro, lags=[1])
        
        # For steadily increasing GDP (100 to 120 over 200 rows)
        # The 1-day change should be roughly constant
        gdp_change = result[f"{MACRO_PREFIX}GDP_change_1d"].dropna()
        
        # Should have non-NaN values after first row
        assert len(gdp_change) > 0
        
        # All changes should be positive (increasing GDP)
        assert (gdp_change > 0).all()
    
    def test_preserves_original_columns(self, sample_data_with_macro):
        """Test that original columns are preserved."""
        original_cols = list(sample_data_with_macro.columns)
        result = add_macro_lag_features(sample_data_with_macro)
        
        for col in original_cols:
            assert col in result.columns
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=[TIMESTAMP, TICKER])
        result = add_macro_lag_features(empty_df)
        
        assert result.empty
    
    def test_no_macro_columns(self):
        """Test handling when no macro columns exist."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "A", "A"],
            "Close": [100, 101, 102],
        })
        result = add_macro_lag_features(df)
        
        # Should return same DataFrame
        assert len(result.columns) == len(df.columns)
    
    def test_float32_dtype(self, sample_data_with_macro):
        """Test that new columns are float32."""
        result = add_macro_lag_features(sample_data_with_macro)
        
        for col in result.columns:
            if "_change_" in col:
                assert result[col].dtype == np.float32


class TestAddTickerLagFeatures:
    """Tests for ticker lag feature creation."""
    
    def test_creates_lag_columns(self, sample_data_with_technical):
        """Test that lag columns are created for technical features."""
        result = add_ticker_lag_features(sample_data_with_technical)
        
        # Should create 20d lags for selected features
        assert "return_5d_lag_20d" in result.columns
        assert "rsi_14_lag_20d" in result.columns
    
    def test_lag_values_shifted(self, sample_data_with_technical):
        """Test that lag values are properly shifted."""
        result = add_ticker_lag_features(sample_data_with_technical)
        result = result.sort_values(TIMESTAMP).reset_index(drop=True)
        
        # The 20-day lag at row 25 should equal original value at row 5
        original_val = sample_data_with_technical.sort_values(TIMESTAMP).iloc[5]["return_5d"]
        lagged_val = result.iloc[25]["return_5d_lag_20d"]
        
        assert np.isclose(original_val, lagged_val, equal_nan=True)
    
    def test_handles_multiple_tickers(self):
        """Test that lags are calculated per ticker."""
        df1 = pd.DataFrame({
            TIMESTAMP: list(range(1, 31)),
            TICKER: "AAA",
            "return_5d": list(range(1, 31)),
        })
        df2 = pd.DataFrame({
            TIMESTAMP: list(range(1, 31)),
            TICKER: "BBB",
            "return_5d": list(range(100, 130)),
        })
        df = pd.concat([df1, df2], ignore_index=True)
        
        result = add_ticker_lag_features(df)
        
        # Check that AAA's lag is from AAA (not BBB)
        aaa_result = result[result[TICKER] == "AAA"].sort_values(TIMESTAMP)
        bbb_result = result[result[TICKER] == "BBB"].sort_values(TIMESTAMP)
        
        # The lag_20d at position 20 should equal value at position 0
        assert aaa_result.iloc[20]["return_5d_lag_20d"] == 1  # AAA value
        assert bbb_result.iloc[20]["return_5d_lag_20d"] == 100  # BBB value
    
    def test_preserves_row_count(self, sample_data_with_technical):
        """Test that row count is preserved."""
        result = add_ticker_lag_features(sample_data_with_technical)
        assert len(result) == len(sample_data_with_technical)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=[TIMESTAMP, TICKER, "return_1d"])
        result = add_ticker_lag_features(empty_df)
        
        assert result.empty
    
    def test_missing_columns(self):
        """Test handling when expected columns are missing."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "A", "A"],
            "Close": [100, 101, 102],
        })
        result = add_ticker_lag_features(df)
        
        # Should return same DataFrame (no lag columns to create)
        assert len(result.columns) == len(df.columns)


class TestDataIntegrity:
    """Tests for data integrity after lag feature engineering."""
    
    def test_no_future_leakage_macro(self, sample_data_with_macro):
        """Test that macro lags don't leak future information."""
        result = add_macro_lag_features(sample_data_with_macro, lags=[30])
        
        # First 30 rows should have NaN for 30-day lag
        # (can't look back 30 days from rows 0-29)
        first_30 = result.iloc[:30][f"{MACRO_PREFIX}GDP_change_30d"]
        assert first_30.isna().all()
    
    def test_no_future_leakage_ticker(self, sample_data_with_technical):
        """Test that ticker lags don't leak future information."""
        result = add_ticker_lag_features(sample_data_with_technical)
        
        # First 20 rows should have NaN for 20-day lag
        result_sorted = result.sort_values(TIMESTAMP)
        first_20 = result_sorted.iloc[:20]["return_5d_lag_20d"]
        assert first_20.isna().all()
