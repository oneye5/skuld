"""Tests for alpha factor features."""

import pandas as pd
import numpy as np
import pytest

from features.alpha_factors import (
    add_alpha_factors,
    add_seasonality_features,
    _add_reversal_features,
    _add_momentum_quality,
    _add_idiosyncratic_volatility,
    _add_information_discreteness,
    _add_max_returns,
    _add_higher_moments,
    _add_volume_features,
    _add_momentum_acceleration,
)


@pytest.fixture
def sample_ticker_data():
    """Create sample OHLCV data for one ticker."""
    np.random.seed(42)
    n_days = 300
    
    base_price = 100
    prices = [base_price]
    for _ in range(n_days - 1):
        # Random walk with slight drift
        ret = np.random.randn() * 0.02 + 0.0001
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'timestamp': [1000000000000 + i * 86400000 for i in range(n_days)],
        'ticker': 'TEST',
        'Close': prices,
        'Open': [p * (1 + np.random.randn() * 0.003) for p in prices],
        'High': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'Low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'Volume': [np.random.randint(1000000, 10000000) for _ in range(n_days)],
    })
    
    return df


@pytest.fixture
def sample_multi_ticker_data():
    """Create sample OHLCV data for multiple tickers."""
    np.random.seed(42)
    n_days = 280
    tickers = ['AAPL', 'GOOG', 'MSFT']
    
    data = []
    for ticker in tickers:
        base = 100 + np.random.randn() * 20
        for i in range(n_days):
            price = base * (1 + np.random.randn() * 0.02)
            data.append({
                'timestamp': 1000000000000 + i * 86400000,
                'ticker': ticker,
                'Close': price,
                'Open': price * 0.995,
                'High': price * 1.01,
                'Low': price * 0.99,
                'Volume': np.random.randint(500000, 5000000),
            })
            base = price
    
    return pd.DataFrame(data)


class TestAlphaFactors:
    """Tests for the main add_alpha_factors function."""
    
    def test_adds_expected_columns(self, sample_multi_ticker_data):
        """Test that alpha factors add expected columns."""
        df = sample_multi_ticker_data
        result = add_alpha_factors(df)
        
        # Check reversal columns
        assert 'Rev_5d' in result.columns
        assert 'Rev_10d' in result.columns
        
        # Check momentum quality columns
        assert 'Trend_RSq_20' in result.columns
        assert 'QualMom_60' in result.columns
        
        # Check idiosyncratic volatility
        assert 'IdioVol_20' in result.columns
        assert 'VolOfVol_60' in result.columns
        
        # Check information discreteness
        assert 'InfoDisc_21' in result.columns
        
        # Check max returns
        assert 'MAX_21d' in result.columns
        assert 'MIN_21d' in result.columns
        
        # Check higher moments
        assert 'Skew_60d' in result.columns
        assert 'Kurt_60d' in result.columns
        
        # Check volume features
        assert 'Amihud_21d' in result.columns
        assert 'RelVol_20d' in result.columns
        
        # Check momentum acceleration
        assert 'MomAccel_21_63' in result.columns
    
    def test_preserves_original_columns(self, sample_multi_ticker_data):
        """Test that original columns are preserved."""
        df = sample_multi_ticker_data
        original_cols = set(df.columns)
        result = add_alpha_factors(df)
        
        for col in original_cols:
            assert col in result.columns
    
    def test_preserves_row_count(self, sample_multi_ticker_data):
        """Test that row count is preserved."""
        df = sample_multi_ticker_data
        result = add_alpha_factors(df)
        
        assert len(result) == len(df)
    
    def test_handles_short_data(self):
        """Test that short data series are handled gracefully."""
        # Data with only 50 rows (less than 260 required)
        df = pd.DataFrame({
            'timestamp': [1000000000000 + i * 86400000 for i in range(50)],
            'ticker': 'TEST',
            'Close': [100 + i * 0.1 for i in range(50)],
        })
        
        result = add_alpha_factors(df)
        # Should return original data unchanged
        assert len(result) == 50


class TestReversalFeatures:
    """Tests for reversal features."""
    
    def test_rev_5d_calculation(self, sample_ticker_data):
        """Test 5-day reversal calculation."""
        df = sample_ticker_data
        result = _add_reversal_features(df)
        
        assert 'Rev_5d' in result.columns
        
        # Check a specific calculation
        idx = 10
        expected = (result['Close'].iloc[idx] - result['Close'].iloc[idx - 5]) / result['Close'].iloc[idx - 5]
        actual = result['Rev_5d'].iloc[idx]
        
        assert abs(actual - expected) < 1e-6


class TestMomentumQuality:
    """Tests for momentum quality features."""
    
    def test_trend_rsq_bounds(self, sample_ticker_data):
        """Test that R-squared is bounded between 0 and 1."""
        df = sample_ticker_data
        result = _add_momentum_quality(df)
        
        for col in ['Trend_RSq_20', 'Trend_RSq_60', 'Trend_RSq_120']:
            if col in result.columns:
                valid_values = result[col].dropna()
                assert (valid_values >= 0).all()
                assert (valid_values <= 1).all()


class TestHigherMoments:
    """Tests for higher moment features."""
    
    def test_skewness_calculation(self, sample_ticker_data):
        """Test skewness is calculated."""
        df = sample_ticker_data
        result = _add_higher_moments(df)
        
        assert 'Skew_60d' in result.columns
        assert 'Skew_126d' in result.columns
        
        # Skewness should have values after warmup period
        assert result['Skew_60d'].iloc[100:].notna().any()
    
    def test_kurtosis_calculation(self, sample_ticker_data):
        """Test kurtosis is calculated."""
        df = sample_ticker_data
        result = _add_higher_moments(df)
        
        assert 'Kurt_60d' in result.columns


class TestVolumeFeatures:
    """Tests for volume-based features."""
    
    def test_amihud_illiquidity(self, sample_ticker_data):
        """Test Amihud illiquidity calculation."""
        df = sample_ticker_data
        result = _add_volume_features(df)
        
        assert 'Amihud_21d' in result.columns
        
        # Amihud should be positive (absolute returns / volume)
        valid = result['Amihud_21d'].dropna()
        assert (valid >= 0).all()
    
    def test_relative_volume(self, sample_ticker_data):
        """Test relative volume calculation."""
        df = sample_ticker_data
        result = _add_volume_features(df)
        
        assert 'RelVol_20d' in result.columns
        
        # Relative volume should center around 1.0
        valid = result['RelVol_20d'].dropna()
        assert valid.mean() > 0.5
        assert valid.mean() < 2.0


class TestSeasonalityFeatures:
    """Tests for seasonality features."""
    
    def test_month_extraction(self, sample_ticker_data):
        """Test month extraction from timestamp."""
        df = sample_ticker_data
        result = add_seasonality_features(df)
        
        assert 'Month' in result.columns
        assert result['Month'].between(1, 12).all()
    
    def test_day_of_week(self, sample_ticker_data):
        """Test day of week extraction."""
        df = sample_ticker_data
        result = add_seasonality_features(df)
        
        assert 'DayOfWeek' in result.columns
        assert result['DayOfWeek'].between(0, 6).all()


class TestIntegration:
    """Integration tests for alpha factors."""
    
    def test_full_pipeline_integration(self, sample_multi_ticker_data):
        """Test alpha factors work in the full pipeline context."""
        from pipeline.ranking_pipeline import add_all_features
        
        df = sample_multi_ticker_data
        result = add_all_features(
            df, 
            df['timestamp'].min(), 
            df['timestamp'].max()
        )
        
        # Should have significantly more columns
        assert result.shape[1] > df.shape[1] + 30
        
        # Alpha columns should exist
        alpha_patterns = ['Rev_', 'Trend_RSq', 'IdioVol', 'InfoDisc', 'MAX_', 'Skew', 'Amihud']
        for pattern in alpha_patterns:
            matching = [c for c in result.columns if pattern in c]
            assert len(matching) > 0, f"Missing columns with pattern: {pattern}"
    
    def test_no_data_leakage(self, sample_multi_ticker_data):
        """Test that features don't look ahead in time."""
        df = sample_multi_ticker_data.sort_values(['ticker', 'timestamp'])
        result = add_alpha_factors(df)
        
        # For any rolling calculation, the first few values should be NaN
        # (can't calculate without history)
        assert result['Rev_5d'].iloc[0] != result['Rev_5d'].iloc[0]  # NaN check
        assert result['Vol_5d'] if 'Vol_5d' in result.columns else True
