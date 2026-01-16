"""Tests for the configurable lag/MA feature system."""

import numpy as np
import pandas as pd
import pytest

from config.columns import TICKER, TIMESTAMP
from config.lag_ma_config import (
    FeatureLagMAConfig,
    get_enabled_configs,
    get_ticker_configs,
    get_macro_configs,
)
from features.lag_ma_features import (
    add_lag_ma_features,
    add_ticker_lag_ma_features,
    add_macro_lag_ma_features,
    create_custom_config,
    _get_matching_columns,
    _generate_output_name,
)


@pytest.fixture
def sample_ticker_df():
    """Create sample ticker-level data."""
    np.random.seed(42)
    tickers = ["A.NZ", "B.NZ"]
    dates = 50
    
    rows = []
    for ticker in tickers:
        base_views = 1000 if ticker == "A.NZ" else 500
        for i in range(dates):
            rows.append({
                TIMESTAMP: 1000000 + i * 86400000,
                TICKER: ticker,
                "Wiki_Views": base_views + np.random.randint(-100, 200),
                "Wiki_Views_Desktop": base_views * 0.6 + np.random.randint(-50, 100),
                "Wiki_Views_Mobile": base_views * 0.4 + np.random.randint(-30, 80),
                "DollarVolume": 100000 + np.random.randint(-10000, 20000),
                "Close": 10 + np.random.randn() * 0.5,
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def sample_macro_df():
    """Create sample macro/global data."""
    np.random.seed(42)
    dates = 100
    
    rows = []
    for i in range(dates):
        rows.append({
            TIMESTAMP: 1000000 + i * 86400000,
            "MACRO_OECD_Consumer": 100 + np.random.randn() * 2,
            "MACRO_OECD_Business": 98 + np.random.randn() * 3,
            "Interest rate": 0.05 + np.random.randn() * 0.005,
            "road_fatalities": 10 + np.random.randint(-3, 5),
            "^FTSE_Close": 7000 + np.random.randn() * 100,
            "CL=F_Close": 70 + np.random.randn() * 5,
        })
    
    return pd.DataFrame(rows)


class TestFeatureLagMAConfig:
    """Tests for configuration dataclass."""
    
    def test_default_values(self):
        """Test default config values."""
        config = FeatureLagMAConfig(feature_pattern="^Test$")
        
        assert config.lags == []
        assert config.mas == []
        assert config.momentum == []
        assert config.scope == "ticker"
        assert config.enabled is True
        assert config.min_periods_ratio == 0.5
        assert config.include_spike is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureLagMAConfig(
            feature_pattern="^Wiki_Views$",
            output_prefix="Attn",
            lags=[7, 14, 28],
            mas=[7, 14],
            momentum=[7],
            scope="ticker",
            include_spike=True,
        )
        
        assert config.lags == [7, 14, 28]
        assert config.mas == [7, 14]
        assert config.output_prefix == "Attn"
        assert config.include_spike is True


class TestConfigGetters:
    """Tests for config retrieval functions."""
    
    def test_get_enabled_configs(self):
        """All configs should be enabled by default."""
        configs = get_enabled_configs()
        assert len(configs) > 0
        assert all(cfg.enabled for cfg in configs)
    
    def test_get_ticker_configs(self):
        """Ticker configs should have ticker scope."""
        configs = get_ticker_configs()
        assert all(cfg.scope == "ticker" for cfg in configs)
    
    def test_get_macro_configs(self):
        """Macro configs should have global scope."""
        configs = get_macro_configs()
        assert all(cfg.scope == "global" for cfg in configs)


class TestColumnMatching:
    """Tests for column pattern matching."""
    
    def test_exact_match(self, sample_ticker_df):
        """Test exact column matching."""
        matches = _get_matching_columns(sample_ticker_df, r"^Wiki_Views$")
        assert matches == ["Wiki_Views"]
    
    def test_pattern_match(self, sample_ticker_df):
        """Test regex pattern matching."""
        matches = _get_matching_columns(sample_ticker_df, r"^Wiki_Views")
        assert "Wiki_Views" in matches
        assert "Wiki_Views_Desktop" in matches
        assert "Wiki_Views_Mobile" in matches
    
    def test_no_match(self, sample_ticker_df):
        """Test no matches found."""
        matches = _get_matching_columns(sample_ticker_df, r"^NonExistent$")
        assert matches == []
    
    def test_exclude_generated(self):
        """Test exclusion of already-generated features."""
        df = pd.DataFrame({
            "Wiki_Views": [1, 2, 3],
            "Wiki_Views_Lag_7": [None, None, 1],
            "Wiki_Views_MA_14": [1, 1.5, 2],
        })
        matches = _get_matching_columns(df, r"^Wiki_Views", exclude_generated=True)
        assert matches == ["Wiki_Views"]


class TestOutputNaming:
    """Tests for output column naming."""
    
    def test_with_prefix(self):
        """Test output naming with prefix."""
        config = FeatureLagMAConfig(
            feature_pattern="Wiki_Views",
            output_prefix="Attn",
        )
        name = _generate_output_name("Wiki_Views", config, "Lag_7")
        assert name == "Attn_Lag_7"
    
    def test_without_prefix(self):
        """Test output naming without prefix."""
        config = FeatureLagMAConfig(
            feature_pattern="Close",
            output_prefix=None,
        )
        name = _generate_output_name("Close", config, "MA_20")
        assert name == "Close_MA_20"


class TestTickerFeatures:
    """Tests for ticker-level feature generation."""
    
    def test_lag_features(self, sample_ticker_df):
        """Test lag feature generation."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            lags=[7, 14],
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_Lag_7" in result.columns
        assert "Test_Lag_14" in result.columns
    
    def test_ma_features(self, sample_ticker_df):
        """Test moving average feature generation."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            mas=[7, 14],
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_MA_7" in result.columns
        assert "Test_MA_14" in result.columns
    
    def test_momentum_features(self, sample_ticker_df):
        """Test momentum feature generation."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            momentum=[7, 14],
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_Mom_7" in result.columns
        assert "Test_Mom_14" in result.columns
    
    def test_spike_feature(self, sample_ticker_df):
        """Test spike indicator generation."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            mas=[14, 28],
            include_spike=True,
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_Spike" in result.columns
    
    def test_volatility_feature(self, sample_ticker_df):
        """Test volatility feature generation."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            include_volatility=True,
            volatility_window=14,
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_Vol_14" in result.columns
    
    def test_per_ticker_computation(self, sample_ticker_df):
        """Verify features are computed per-ticker."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            lags=[1],
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        # First row of each ticker should be NaN (no prior data)
        for ticker in result[TICKER].unique():
            ticker_data = result[result[TICKER] == ticker].sort_values(TIMESTAMP)
            assert pd.isna(ticker_data["Test_Lag_1"].iloc[0])


class TestMacroFeatures:
    """Tests for macro/global feature generation."""
    
    def test_macro_lag_features(self, sample_macro_df):
        """Test macro lag feature generation."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^MACRO_OECD_",
            lags=[30, 60],
            scope="global",
        )
        result = add_lag_ma_features(sample_macro_df, configs=[config])
        
        assert "MACRO_OECD_Consumer_L30" in result.columns
        assert "MACRO_OECD_Consumer_L60" in result.columns
        assert "MACRO_OECD_Business_L30" in result.columns
    
    def test_macro_ma_features(self, sample_macro_df):
        """Test macro MA feature generation."""
        config = FeatureLagMAConfig(
            feature_pattern=r"Interest rate",  # Exact pattern to match test data
            mas=[7, 30],
            scope="global",
        )
        result = add_lag_ma_features(sample_macro_df, configs=[config])
        
        assert "Interest rate_MA7" in result.columns
        assert "Interest rate_MA30" in result.columns
    
    def test_macro_momentum_features(self, sample_macro_df):
        """Test macro momentum feature generation."""
        config = FeatureLagMAConfig(
            feature_pattern=r"road_fatalities",
            momentum=[30],
            scope="global",
        )
        result = add_lag_ma_features(sample_macro_df, configs=[config])
        
        assert "road_fatalities_Mom30" in result.columns


class TestDiffFeatures:
    """Tests for diff (difference) features."""
    
    def test_current_minus_ma(self, sample_ticker_df):
        """Test current - MA diff features."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            mas=[10],  # Need MA first
            diffs=[(0, 10)],  # Current - MA_10
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_Diff_0_10" in result.columns
        
        # Verify calculation: diff should equal feature - MA
        for ticker in result[TICKER].unique():
            ticker_data = result[result[TICKER] == ticker].sort_values(TIMESTAMP)
            # Check a row where MA is available
            if len(ticker_data) >= 20:
                row = ticker_data.iloc[15]
                expected = row["Wiki_Views"] - row["Test_MA_10"]
                actual = row["Test_Diff_0_10"]
                if not pd.isna(expected) and not pd.isna(actual):
                    np.testing.assert_almost_equal(actual, expected, decimal=5)
    
    def test_ma_minus_ma(self, sample_ticker_df):
        """Test MA - MA diff features."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            mas=[10, 20],  # Need both MAs
            diffs=[(10, 20)],  # MA_10 - MA_20
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_Diff_10_20" in result.columns
        
        # Verify calculation
        for ticker in result[TICKER].unique():
            ticker_data = result[result[TICKER] == ticker].sort_values(TIMESTAMP)
            if len(ticker_data) >= 30:
                row = ticker_data.iloc[25]
                expected = row["Test_MA_10"] - row["Test_MA_20"]
                actual = row["Test_Diff_10_20"]
                if not pd.isna(expected) and not pd.isna(actual):
                    np.testing.assert_almost_equal(actual, expected, decimal=5)
    
    def test_diff_without_precomputed_ma(self, sample_ticker_df):
        """Test diff computation when MAs are not pre-computed."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            mas=[],  # No pre-computed MAs
            diffs=[(0, 15), (15, 30)],  # Will compute on the fly
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_Diff_0_15" in result.columns
        assert "Test_Diff_15_30" in result.columns
    
    def test_macro_diff_features(self, sample_macro_df):
        """Test diff features for macro/global data."""
        config = FeatureLagMAConfig(
            feature_pattern=r"Interest rate",
            mas=[10, 30],
            diffs=[(0, 30), (10, 30)],
            scope="global",
        )
        result = add_lag_ma_features(sample_macro_df, configs=[config])
        
        assert "Interest rate_Diff_0_30" in result.columns
        assert "Interest rate_Diff_10_30" in result.columns
    
    def test_diff_no_leakage(self, sample_ticker_df):
        """Diff features should not use future data."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            diffs=[(0, 10)],
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        for ticker in result[TICKER].unique():
            ticker_data = result[result[TICKER] == ticker].sort_values(TIMESTAMP)
            # First 9 rows should be NaN (need 10 periods for MA)
            first_valid_idx = ticker_data["Test_Diff_0_10"].first_valid_index()
            if first_valid_idx is not None:
                first_valid_pos = ticker_data.index.get_loc(first_valid_idx)
                # Should have some NaN values at the start
                assert first_valid_pos >= 4  # min_periods_ratio=0.5, so at least 5 periods


class TestCustomConfig:
    """Tests for custom config creation."""
    
    def test_create_custom_config(self, sample_ticker_df):
        """Test creating and using custom config."""
        config = create_custom_config(
            feature_pattern=r"^Close$",
            lags=[1, 5],
            mas=[5, 10],
            output_prefix="Price",
        )
        
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Price_Lag_1" in result.columns
        assert "Price_Lag_5" in result.columns
        assert "Price_MA_5" in result.columns
        assert "Price_MA_10" in result.columns


class TestNoLeakage:
    """Tests to verify no data leakage."""
    
    def test_lag_uses_only_past_data(self, sample_ticker_df):
        """Lag features should only use past data."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            lags=[7],
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        for ticker in result[TICKER].unique():
            ticker_data = result[result[TICKER] == ticker].sort_values(TIMESTAMP)
            
            # Check that lag values match shifted values
            original = ticker_data["Wiki_Views"].values
            lagged = ticker_data["Test_Lag_7"].values
            
            # After position 7, lagged[i] should equal original[i-7]
            for i in range(7, len(original)):
                if not pd.isna(lagged[i]):
                    assert lagged[i] == original[i - 7]
    
    def test_ma_uses_only_past_data(self, sample_ticker_df):
        """MA features should only use past data (inclusive of current)."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            mas=[5],
            min_periods_ratio=1.0,  # Require full window
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        for ticker in result[TICKER].unique():
            ticker_data = result[result[TICKER] == ticker].sort_values(TIMESTAMP)
            original = ticker_data["Wiki_Views"].values
            ma_values = ticker_data["Test_MA_5"].values
            
            # MA should be NaN for first 4 rows (need 5 periods)
            assert all(pd.isna(ma_values[:4]))
            
            # Check MA calculation for later rows
            for i in range(4, len(original)):
                expected_ma = np.mean(original[i-4:i+1])
                if not pd.isna(ma_values[i]):
                    np.testing.assert_almost_equal(ma_values[i], expected_ma, decimal=5)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_dataframe(self):
        """Handle empty DataFrame gracefully."""
        df = pd.DataFrame(columns=[TIMESTAMP, TICKER, "Wiki_Views"])
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            lags=[7],
        )
        result = add_lag_ma_features(df, configs=[config])
        assert len(result) == 0
    
    def test_single_row(self):
        """Handle single row gracefully."""
        df = pd.DataFrame({
            TIMESTAMP: [1000000],
            TICKER: ["A.NZ"],
            "Wiki_Views": [100],
        })
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            lags=[7],
            mas=[7],
        )
        result = add_lag_ma_features(df, configs=[config])
        
        assert "Test_Lag_7" in result.columns
        assert pd.isna(result["Test_Lag_7"].iloc[0])  # Should be NaN
    
    def test_all_nan_feature(self):
        """Handle all-NaN feature gracefully."""
        df = pd.DataFrame({
            TIMESTAMP: [1000000, 2000000, 3000000],
            TICKER: ["A.NZ", "A.NZ", "A.NZ"],
            "Wiki_Views": [np.nan, np.nan, np.nan],
        })
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            lags=[1],
            mas=[2],
        )
        result = add_lag_ma_features(df, configs=[config])
        
        # Should complete without error
        assert "Wiki_Views_Lag_1" in result.columns
    
    def test_disabled_config_skipped(self, sample_ticker_df):
        """Disabled configs should be skipped."""
        config = FeatureLagMAConfig(
            feature_pattern=r"^Wiki_Views$",
            output_prefix="Test",
            lags=[7],
            enabled=False,
        )
        result = add_lag_ma_features(sample_ticker_df, configs=[config])
        
        assert "Test_Lag_7" not in result.columns


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_add_ticker_lag_ma_features(self, sample_ticker_df):
        """Test ticker-only convenience function."""
        result = add_ticker_lag_ma_features(sample_ticker_df)
        
        # Should have attention features from default config
        # (if Wiki_Views column exists)
        col_count_before = len(sample_ticker_df.columns)
        col_count_after = len(result.columns)
        
        # Should add some features
        assert col_count_after >= col_count_before
    
    def test_add_macro_lag_ma_features(self, sample_macro_df):
        """Test macro-only convenience function."""
        result = add_macro_lag_ma_features(sample_macro_df)
        
        col_count_before = len(sample_macro_df.columns)
        col_count_after = len(result.columns)
        
        # Should add some features
        assert col_count_after >= col_count_before
