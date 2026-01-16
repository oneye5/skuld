"""Tests for attention-based aggregate features from Wikipedia pageviews.

Note: Ticker-level lag/MA/momentum features are now tested in test_lag_ma_features.py.
This file tests only the aggregate features (fear indicators, mobile ratio).
"""

import pytest
import pandas as pd
import numpy as np

from config.columns import TICKER, TIMESTAMP
from features.attention_features import (
    add_aggregate_attention_features,
    add_mobile_ratio_features,
)


class TestMobileRatioFeatures:
    """Tests for mobile vs desktop ratio features."""

    @pytest.fixture
    def sample_df(self):
        """Create sample data with Wiki columns."""
        np.random.seed(42)
        n_days = 60
        
        # Two tickers, 60 days each
        data = []
        for ticker in ["A.NZ", "B.NZ"]:
            base_views = 500 if ticker == "A.NZ" else 300
            for i in range(n_days):
                data.append({
                    TIMESTAMP: 1000000 + i * 86400000,  # daily
                    TICKER: ticker,
                    "Wiki_Views": base_views + np.random.randint(-50, 100),
                    "Wiki_Views_Desktop": int((base_views + np.random.randint(-30, 50)) * 0.6),
                    "Wiki_Views_Mobile": int((base_views + np.random.randint(-20, 40)) * 0.4),
                    "Close": 100 + i * 0.1,
                })
        
        return pd.DataFrame(data)

    def test_creates_mobile_ratio(self, sample_df):
        """Test that mobile ratio feature is created."""
        result = add_mobile_ratio_features(sample_df)
        
        assert "Attn_Mobile_Ratio" in result.columns
        # Ratio should be between 0 and 1
        ratios = result["Attn_Mobile_Ratio"].dropna()
        assert ratios.min() >= 0
        assert ratios.max() <= 1

    def test_creates_mobile_ratio_change(self, sample_df):
        """Test that mobile ratio change feature is created."""
        result = add_mobile_ratio_features(sample_df)
        
        assert "Attn_Mobile_Ratio_Change_14" in result.columns

    def test_no_wiki_columns_returns_unchanged(self):
        """Test that df without Wiki columns is returned unchanged."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "A", "A"],
            "Close": [100, 101, 102],
        })
        
        result = add_mobile_ratio_features(df)
        
        # Should have same columns (no features added)
        assert list(result.columns) == list(df.columns)


class TestAggregateAttentionFeatures:
    """Tests for aggregate attention features."""

    @pytest.fixture
    def sample_fear_df(self):
        """Create sample data with fear-related Wiki columns."""
        np.random.seed(42)
        n_days = 60
        
        data = []
        for i in range(n_days):
            data.append({
                TIMESTAMP: 1000000 + i * 86400000,
                TICKER: "A.NZ",
                "Recession_Wiki_Views": 1000 + np.random.randint(-100, 200),
                "Financial_crisis_Wiki_Views": 500 + np.random.randint(-50, 100),
                "Stock_market_crash_Wiki_Views": 300 + np.random.randint(-30, 60),
                "Wiki_Views_Desktop": 400 + np.random.randint(-40, 80),
                "Wiki_Views_Mobile": 200 + np.random.randint(-20, 40),
            })
        
        return pd.DataFrame(data)

    def test_creates_fear_aggregate(self, sample_fear_df):
        """Test that aggregate fear features are created."""
        result = add_aggregate_attention_features(sample_fear_df)
        
        assert "Attn_Fear_Total" in result.columns
        assert "Attn_Fear_Mean" in result.columns

    def test_creates_fear_momentum(self, sample_fear_df):
        """Test that fear momentum features are created."""
        result = add_aggregate_attention_features(sample_fear_df)
        
        assert "Attn_Fear_Mom_7" in result.columns
        assert "Attn_Fear_Mom_14" in result.columns

    def test_creates_fear_spike(self, sample_fear_df):
        """Test that fear spike feature is created."""
        result = add_aggregate_attention_features(sample_fear_df)
        
        assert "Attn_Fear_Spike" in result.columns

    def test_also_creates_mobile_ratio(self, sample_fear_df):
        """Test that mobile ratio is also created via aggregate function."""
        result = add_aggregate_attention_features(sample_fear_df)
        
        assert "Attn_Mobile_Ratio" in result.columns

    def test_no_fear_columns_still_creates_mobile_ratio(self):
        """Test that df without fear columns still gets mobile ratio."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "A", "A"],
            "Close": [100, 101, 102],
            "Wiki_Views_Desktop": [400, 410, 420],
            "Wiki_Views_Mobile": [200, 210, 220],
            "Random_Wiki_Views": [10, 20, 30],  # Not a fear indicator
        })
        
        result = add_aggregate_attention_features(df)
        
        # Should not have fear aggregate columns
        assert "Attn_Fear_Total" not in result.columns
        # But should have mobile ratio
        assert "Attn_Mobile_Ratio" in result.columns


class TestNoLeakage:
    """Tests to ensure aggregate features don't cause data leakage."""

    def test_fear_momentum_uses_past_data_only(self):
        """Test that fear momentum only uses past data."""
        data = {
            TIMESTAMP: [i for i in range(30)],
            TICKER: ["A"] * 30,
            "Recession_Wiki_Views": [100] * 15 + [1000] * 15,  # Jump at row 15
        }
        df = pd.DataFrame(data)
        
        result = add_aggregate_attention_features(df)
        
        # Fear momentum at row 14 should NOT be influenced by values from row 15+
        # pct_change(7) at row 14 looks at row 7 vs row 14
        # Both are 100, so momentum should be ~0
        mom_at_14 = result["Attn_Fear_Mom_7"].iloc[14]
        # Should be close to 0 since values before row 15 are all 100
        assert abs(mom_at_14) < 0.1  # Small tolerance for floating point
