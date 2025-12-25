"""Tests for technical features."""

import pandas as pd
import numpy as np
import pytest

from config.columns import CLOSE, OPEN, HIGH, LOW, VOLUME, TICKER, TIMESTAMP
from features.technical import add_technical_features


def test_add_technical_features():
    """Test that technical features are added correctly."""
    # Create sample data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    timestamps = dates.astype(np.int64) // 10**6  # ms
    
    # Create a trend with some volatility
    t = np.linspace(0, 10, 100)
    price = 100 + t + np.sin(t) * 5
    
    df = pd.DataFrame({
        TIMESTAMP: timestamps,
        TICKER: "TEST.NZ",
        OPEN: price - 1,
        HIGH: price + 2,
        LOW: price - 2,
        CLOSE: price,
        VOLUME: 1000,
    })
    
    # Add features
    result = add_technical_features(df)
    
    # Check columns exist
    expected_cols = [
        "RSI_14", "MACD_Line", "MACD_Signal", "MACD_Hist",
        "ROC_10", "ATR_14", "NATR_14", "BB_Width_20",
        "Dist_SMA_20", "Dist_MA_50", "Vol_20",
        "Ret_Lag_1", "Ret_Lag_5"
    ]
    
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
        
    # Check RSI range
    assert result["RSI_14"].min() >= 0
    assert result["RSI_14"].max() <= 100
    
    # Check that we didn't lose rows (except maybe if we dropped them, but function doesn't drop)
    assert len(result) == len(df)
    
    # Check that NaNs are present at the start (before rolling window fills)
    # RSI needs 14 days
    assert pd.isna(result.loc[0, "ATR_14"])
    
    # Check that values are filled later
    assert not pd.isna(result.loc[50, "RSI_14"])


def test_add_technical_features_multiple_tickers():
    """Test with multiple tickers to ensure no cross-contamination."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    timestamps = dates.astype(np.int64) // 10**6
    
    df1 = pd.DataFrame({
        TIMESTAMP: timestamps,
        TICKER: "A.NZ",
        CLOSE: 100.0,
        OPEN: 100.0, HIGH: 101.0, LOW: 99.0, VOLUME: 100
    })
    
    df2 = pd.DataFrame({
        TIMESTAMP: timestamps,
        TICKER: "B.NZ",
        CLOSE: 200.0,
        OPEN: 200.0, HIGH: 201.0, LOW: 199.0, VOLUME: 100
    })
    
    df = pd.concat([df1, df2])
    
    result = add_technical_features(df)
    
    # Check that we have results for both
    assert len(result) == 200
    assert set(result[TICKER].unique()) == {"A.NZ", "B.NZ"}
    
    # Check that calculations are independent
    # Since prices are constant (mostly), RSI should be 50 or NaN
    # (Actually constant price -> diff=0 -> gain=0, loss=0 -> RS=NaN -> RSI=NaN or handled)
    # My implementation: gain / (loss + EPSILON). If loss=0, RS=large. RSI -> 100.
    # If gain=0, RS=0. RSI -> 0.
    # If both 0? 0/EPSILON = 0. RSI=0.
    
    # Let's check Lagged Return. Should be 0.
    # First row of B should NOT use last row of A.
    
    # Sort by ticker/timestamp to find first row of B
    result = result.sort_values([TICKER, TIMESTAMP])
    b_start_idx = result[result[TICKER] == "B.NZ"].index[0]
    
    # Lagged return for first row of B should be NaN, not (200-100)/100
    assert pd.isna(result.loc[b_start_idx, "Ret_Lag_1"])

