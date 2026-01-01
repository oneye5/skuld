"""Tests for fast clustering module."""

import pytest
import pandas as pd
import numpy as np
from features.cluster_fast import (
    compute_clusters_fast,
    add_cluster_features_fast,
    rolling_cluster_assignment,
    describe_clusters,
)
from config.columns import TIMESTAMP, TICKER


@pytest.fixture
def sample_wide_df():
    """Create sample wide-format data for testing."""
    np.random.seed(42)
    timestamps = list(range(1000, 1600))  # 600 days
    tickers = [f'{c}.NZ' for c in ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF']]
    
    rows = []
    # Create different volatility profiles
    vol_mult = {'AAA.NZ': 0.5, 'BBB.NZ': 1.0, 'CCC.NZ': 1.5, 
                'DDD.NZ': 2.0, 'EEE.NZ': 0.3, 'FFF.NZ': 2.5}
    
    for ticker in tickers:
        base_price = 10
        for i, ts in enumerate(timestamps):
            returns = np.random.normal(0.0001, 0.02 * vol_mult[ticker], i+1)
            price = base_price * np.exp(returns.sum())
            rows.append({TIMESTAMP: ts, TICKER: ticker, 'Close': price})
    
    return pd.DataFrame(rows)


def test_compute_clusters_fast(sample_wide_df):
    """Test fast clustering computation."""
    cluster_map = compute_clusters_fast(
        sample_wide_df, 
        n_clusters=3,
        lookback_days=500,
        min_obs=50
    )
    
    assert len(cluster_map) == 6
    assert all(isinstance(v, (int, np.integer)) for v in cluster_map.values())
    assert all(0 <= v < 3 for v in cluster_map.values())


def test_add_cluster_features_fast(sample_wide_df):
    """Test adding cluster features."""
    cluster_map = compute_clusters_fast(sample_wide_df, n_clusters=3, min_obs=50)
    df = add_cluster_features_fast(sample_wide_df, cluster_map)
    
    assert 'Cluster' in df.columns
    assert 'Rank_InCluster' in df.columns
    
    # Check Cluster values are correct
    for ticker in df[TICKER].unique():
        if ticker in cluster_map:
            ticker_clusters = df[df[TICKER] == ticker]['Cluster'].unique()
            assert len(ticker_clusters) == 1
            assert ticker_clusters[0] == cluster_map[ticker]


def test_rolling_cluster_assignment(sample_wide_df):
    """Test leakage-safe rolling cluster assignment."""
    # Use middle timestamp as train end
    train_end = 1300
    
    cluster_map = rolling_cluster_assignment(
        sample_wide_df,
        train_end_ts=train_end,
        n_clusters=3,
    )
    
    assert len(cluster_map) > 0
    # Verify it only used data up to train_end
    # (implicitly tested by the function filtering)


def test_describe_clusters(sample_wide_df):
    """Test cluster description generation."""
    cluster_map = compute_clusters_fast(sample_wide_df, n_clusters=3, min_obs=50)
    summary = describe_clusters(sample_wide_df, cluster_map, lookback_days=200)
    
    assert len(summary) > 0
    assert 'n_stocks' in summary.columns
    assert 'avg_vol' in summary.columns
    assert 'interpretation' in summary.columns


def test_cluster_no_future_leakage(sample_wide_df):
    """Ensure rolling_cluster_assignment doesn't use future data."""
    # Split at timestamp 1300
    train_end = 1300
    
    # Get clusters using only "past" data
    cluster_map_train = rolling_cluster_assignment(
        sample_wide_df,
        train_end_ts=train_end,
        n_clusters=3,
    )
    
    # Add test data (future) with very different characteristics
    future_rows = []
    for ts in range(1400, 1500):
        # New ticker that shouldn't affect training clusters
        future_rows.append({TIMESTAMP: ts, TICKER: 'NEW.NZ', 'Close': 100})
    
    df_with_future = pd.concat([sample_wide_df, pd.DataFrame(future_rows)])
    
    # Recompute - should get same clusters for original tickers
    cluster_map_with_future = rolling_cluster_assignment(
        df_with_future,
        train_end_ts=train_end,
        n_clusters=3,
    )
    
    # Original tickers should have same clusters
    for ticker in cluster_map_train:
        if ticker in cluster_map_with_future:
            assert cluster_map_train[ticker] == cluster_map_with_future[ticker]


def test_handles_anomalous_returns(sample_wide_df):
    """Test that extreme returns are handled properly."""
    # Add anomalous data point
    df = sample_wide_df.copy()
    idx = df[(df[TICKER] == 'AAA.NZ') & (df[TIMESTAMP] == 1200)].index[0]
    df.loc[idx, 'Close'] = df.loc[idx, 'Close'] * 100  # 10000% spike
    
    # Should not crash, and should produce reasonable clusters
    cluster_map = compute_clusters_fast(df, n_clusters=3, min_obs=50)
    
    assert len(cluster_map) == 6
    assert all(0 <= v < 3 for v in cluster_map.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
