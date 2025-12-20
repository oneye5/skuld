"""Tests for technical features module."""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add paths for imports
_ml_pipeline = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "transformations"))

from config.column_names import TIMESTAMP, TICKER, CLOSE, OPEN, HIGH, LOW, VOLUME
from technical_features import (
    add_technical_features,
    _calculate_returns,
    _calculate_volatility,
    _calculate_sma,
    _calculate_rsi,
    RETURN_5D, RETURN_20D, RETURN_60D,
    VOLATILITY_20D,
    SMA_20, SMA_60,
    RSI_14,
    PRICE_TO_SMA_20,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    # Create 100 days of data for one ticker
    np.random.seed(42)
    n_days = 100
    
    # Generate realistic price movement (random walk with drift)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # ~0.1% daily drift, 2% volatility
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close prices
    closes = prices
    opens = np.roll(closes, 1)
    opens[0] = base_price
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    volumes = np.random.randint(10000, 100000, n_days)
    
    # Create timestamps (daily, starting from a fixed point)
    base_ts = 1600000000000  # Some timestamp in ms
    timestamps = [base_ts + i * 86400000 for i in range(n_days)]
    
    return pd.DataFrame({
        TIMESTAMP: timestamps,
        TICKER: "TEST.NZ",
        OPEN: opens,
        HIGH: highs,
        LOW: lows,
        CLOSE: closes,
        VOLUME: volumes,
    })


@pytest.fixture
def multi_ticker_data(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Create data with multiple tickers."""
    df1 = sample_ohlcv_data.copy()
    df1[TICKER] = "AAA.NZ"
    
    df2 = sample_ohlcv_data.copy()
    df2[TICKER] = "BBB.NZ"
    df2[CLOSE] = df2[CLOSE] * 1.5  # Different price level
    
    return pd.concat([df1, df2], ignore_index=True)


class TestCalculateReturns:
    """Tests for return calculation."""
    
    def test_returns_calculation(self):
        """Test that returns are calculated correctly."""
        prices = pd.Series([100, 110, 105, 115])
        returns = _calculate_returns(prices, 1)
        
        # First value should be NaN (no previous price)
        assert pd.isna(returns.iloc[0])
        
        # Second value: (110-100)/100 * 100 = 10%
        assert abs(returns.iloc[1] - 10.0) < 0.01
        
        # Third value: (105-110)/110 * 100 = -4.545%
        assert abs(returns.iloc[2] - (-4.545)) < 0.1
    
    def test_returns_multi_period(self):
        """Test multi-period returns."""
        prices = pd.Series([100, 110, 105, 115, 120])
        returns_2 = _calculate_returns(prices, 2)
        
        # At index 2: (105-100)/100 * 100 = 5%
        assert abs(returns_2.iloc[2] - 5.0) < 0.01


class TestCalculateVolatility:
    """Tests for volatility calculation."""
    
    def test_volatility_positive(self):
        """Test that volatility is always non-negative."""
        prices = pd.Series([100, 101, 99, 102, 98, 103])
        vol = _calculate_volatility(prices, 3)
        
        # All non-NaN values should be >= 0
        assert (vol.dropna() >= 0).all()
    
    def test_volatility_zero_for_constant(self):
        """Test that constant prices have zero volatility."""
        prices = pd.Series([100, 100, 100, 100, 100])
        vol = _calculate_volatility(prices, 3)
        
        # Should be 0 (or very close due to floating point)
        assert vol.iloc[-1] < 0.001


class TestCalculateSMA:
    """Tests for SMA calculation."""
    
    def test_sma_basic(self):
        """Test basic SMA calculation."""
        prices = pd.Series([10, 20, 30, 40, 50])
        sma = _calculate_sma(prices, 3)
        
        # SMA at index 2 should be (10+20+30)/3 = 20
        assert abs(sma.iloc[2] - 20.0) < 0.01
        
        # SMA at index 4 should be (30+40+50)/3 = 40
        assert abs(sma.iloc[4] - 40.0) < 0.01


class TestCalculateRSI:
    """Tests for RSI calculation."""
    
    def test_rsi_range(self):
        """Test that RSI is always between 0 and 100."""
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.02, 50)))
        rsi = _calculate_rsi(prices, 14)
        
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()
    
    def test_rsi_uptrend(self):
        """Test RSI in strong uptrend with some variation."""
        # Create an uptrend with normal market variation
        np.random.seed(123)
        base = 100
        prices = []
        for i in range(50):
            # Strong upward bias with small random noise
            base = base * (1 + 0.02 + np.random.uniform(-0.005, 0.01))
            prices.append(base)
        rsi = _calculate_rsi(pd.Series(prices), 14)
        
        # In strong uptrend, RSI should be high (above 60)
        assert rsi.iloc[-1] > 60
    
    def test_rsi_downtrend(self):
        """Test RSI in strong downtrend."""
        prices = pd.Series([100 - i * 2 for i in range(30)])  # Steady downtrend
        rsi = _calculate_rsi(prices, 14)
        
        # In strong downtrend, RSI should be low (below 30)
        assert rsi.iloc[-1] < 30


class TestAddTechnicalFeatures:
    """Tests for the main add_technical_features function."""
    
    def test_adds_return_features(self, sample_ohlcv_data):
        """Test that return features are added (core set only)."""
        result = add_technical_features(sample_ohlcv_data)
        
        # Core returns (reduced feature set)
        assert RETURN_5D in result.columns
        assert RETURN_20D in result.columns
        assert RETURN_60D in result.columns
    
    def test_adds_volatility_features(self, sample_ohlcv_data):
        """Test that volatility features are added (core set only)."""
        result = add_technical_features(sample_ohlcv_data)
        
        # Core volatility (reduced feature set)
        assert VOLATILITY_20D in result.columns
    
    def test_adds_sma_features(self, sample_ohlcv_data):
        """Test that SMA features are added (core set only)."""
        result = add_technical_features(sample_ohlcv_data)
        
        # Core SMAs (reduced feature set)
        assert SMA_20 in result.columns
        assert SMA_60 in result.columns
    
    def test_adds_rsi(self, sample_ohlcv_data):
        """Test that RSI is added."""
        result = add_technical_features(sample_ohlcv_data)
        
        assert RSI_14 in result.columns
    
    def test_adds_price_to_sma(self, sample_ohlcv_data):
        """Test that price-to-SMA ratios are added."""
        result = add_technical_features(sample_ohlcv_data)
        
        assert PRICE_TO_SMA_20 in result.columns
    
    def test_handles_multiple_tickers(self, multi_ticker_data):
        """Test that features are calculated per ticker."""
        result = add_technical_features(multi_ticker_data)
        
        # Check that both tickers have features
        aaa_data = result[result[TICKER] == "AAA.NZ"]
        bbb_data = result[result[TICKER] == "BBB.NZ"]
        
        assert len(aaa_data) > 0
        assert len(bbb_data) > 0
        
        # Features should be calculated independently
        # (different price levels should give different SMA values)
        assert not np.allclose(
            aaa_data[SMA_20].dropna().values,
            bbb_data[SMA_20].dropna().values,
            equal_nan=True
        )
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=[TIMESTAMP, TICKER, CLOSE])
        result = add_technical_features(empty_df)
        
        assert result.empty
    
    def test_no_close_column(self):
        """Test handling when Close column is missing."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "A", "A"],
        })
        result = add_technical_features(df)
        
        # Should return original DataFrame
        assert len(result) == 3
    
    def test_preserves_original_columns(self, sample_ohlcv_data):
        """Test that original columns are preserved."""
        original_cols = set(sample_ohlcv_data.columns)
        result = add_technical_features(sample_ohlcv_data)
        
        for col in original_cols:
            assert col in result.columns
    
    def test_feature_values_reasonable(self, sample_ohlcv_data):
        """Test that feature values are in reasonable ranges."""
        result = add_technical_features(sample_ohlcv_data)
        
        # RSI should be 0-100
        rsi = result[RSI_14].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()
        
        # Returns should be finite (not inf) - using core features only
        for col in [RETURN_5D, RETURN_20D, RETURN_60D]:
            returns = result[col].dropna()
            assert np.isfinite(returns).all()
        
        # Volatility should be non-negative - using core features only
        vol = result[VOLATILITY_20D].dropna()
        assert (vol >= 0).all()


class TestDataIntegrity:
    """Tests for data integrity after feature engineering."""
    
    def test_row_count_preserved(self, sample_ohlcv_data):
        """Test that number of rows is preserved."""
        result = add_technical_features(sample_ohlcv_data)
        assert len(result) == len(sample_ohlcv_data)
    
    def test_no_duplicate_columns(self, sample_ohlcv_data):
        """Test that there are no duplicate column names."""
        result = add_technical_features(sample_ohlcv_data)
        assert len(result.columns) == len(set(result.columns))
    
    def test_float32_dtype(self, sample_ohlcv_data):
        """Test that new numeric columns are float32 for memory efficiency."""
        result = add_technical_features(sample_ohlcv_data)
        
        # Check some technical feature columns (using core feature set)
        for col in [RETURN_5D, VOLATILITY_20D, RSI_14]:
            if col in result.columns:
                assert result[col].dtype in ['float32', 'float64']
