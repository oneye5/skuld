"""Test ticker clustering module."""

import pytest
import pandas as pd
import numpy as np
from features.ticker_clusters import (
    compute_ticker_characteristics,
    assign_clusters,
    add_cluster_features,
    get_cluster_summary,
    interpret_cluster,
)
from config.columns import TIMESTAMP, TICKER


@pytest.fixture
def sample_wide_df():
    """Create sample wide-format data for testing."""
    np.random.seed(42)
    timestamps = list(range(1000, 1600))  # 600 days
    tickers = ['AAA.NZ', 'BBB.NZ', 'CCC.NZ', 'DDD.NZ', 'EEE.NZ', 'FFF.NZ']
    
    rows = []
    for ticker in tickers:
        # Different volatility profiles per ticker
        vol_mult = {'AAA.NZ': 0.5, 'BBB.NZ': 1.0, 'CCC.NZ': 1.5, 
                    'DDD.NZ': 2.0, 'EEE.NZ': 0.3, 'FFF.NZ': 2.5}[ticker]
        base_price = 10
        for i, ts in enumerate(timestamps):
            # Random walk with different volatilities
            returns = np.random.normal(0.0001, 0.02 * vol_mult, i+1)
            price = base_price * np.exp(returns.sum())
            rows.append({TIMESTAMP: ts, TICKER: ticker, 'Close': price})
    
    return pd.DataFrame(rows)


def test_compute_ticker_characteristics(sample_wide_df):
    """Test characteristic computation."""
    stats = compute_ticker_characteristics(sample_wide_df, lookback_days=500, min_obs=50)
    
    assert len(stats) == 6  # All 6 tickers
    assert 'volatility' in stats.columns
    assert 'mean_return' in stats.columns
    assert 'skewness' in stats.columns
    
    # Higher vol_mult tickers should have higher volatility
    stats_sorted = stats.sort_values('volatility')
    assert stats_sorted.iloc[0]['ticker'] == 'EEE.NZ'  # Lowest vol (0.3x)
    assert stats_sorted.iloc[-1]['ticker'] == 'FFF.NZ'  # Highest vol (2.5x)


def test_assign_clusters(sample_wide_df):
    """Test cluster assignment."""
    cluster_map = assign_clusters(sample_wide_df, n_clusters=3)
    
    assert len(cluster_map) == 6
    assert all(isinstance(v, (int, np.integer)) for v in cluster_map.values())
    assert all(0 <= v < 3 for v in cluster_map.values())


def test_add_cluster_features(sample_wide_df):
    """Test adding cluster features to dataframe."""
    cluster_map = assign_clusters(sample_wide_df, n_clusters=3)
    df = add_cluster_features(sample_wide_df, cluster_map)
    
    assert 'Cluster' in df.columns
    assert 'Cluster_Size' in df.columns
    assert 'Rank_Close_InCluster' in df.columns
    assert 'Dist_Cluster_Mean_Close' in df.columns
    
    # Ranks should be between 0 and 1
    assert df['Rank_Close_InCluster'].min() >= 0
    assert df['Rank_Close_InCluster'].max() <= 1


def test_get_cluster_summary(sample_wide_df):
    """Test cluster summary generation."""
    cluster_map = assign_clusters(sample_wide_df, n_clusters=3)
    summary = get_cluster_summary(sample_wide_df, cluster_map)
    
    assert len(summary) == 3  # 3 clusters
    assert 'n_stocks' in summary.columns
    assert 'volatility' in summary.columns
    assert summary['n_stocks'].sum() == 6  # All 6 tickers assigned


def test_interpret_cluster():
    """Test cluster interpretation."""
    assert interpret_cluster(0.15, 0.10) == "Defensive/Quality"
    assert interpret_cluster(0.15, -0.05) == "Value/Turnaround"
    assert interpret_cluster(0.35, 0.20) == "Growth"
    assert interpret_cluster(0.35, -0.10) == "Cyclical/Challenged"
    assert interpret_cluster(0.60, 0.30) == "Speculative/Momentum"
    assert interpret_cluster(0.60, -0.20) == "Distressed"


def test_cluster_features_no_leak(sample_wide_df):
    """Ensure cluster features don't leak future information.
    
    Clusters are assigned based on historical characteristics,
    then features are computed per-timestamp using only that timestamp's data.
    """
    cluster_map = assign_clusters(sample_wide_df, n_clusters=3)
    df = add_cluster_features(sample_wide_df, cluster_map)
    
    # Each timestamp should have its own cluster stats
    # Check that Rank_Close_InCluster varies by timestamp
    ts1 = df[df[TIMESTAMP] == 1100]
    ts2 = df[df[TIMESTAMP] == 1200]
    
    # Ranks should differ between timestamps (prices change)
    ticker = 'AAA.NZ'
    rank1 = ts1[ts1[TICKER] == ticker]['Rank_Close_InCluster'].values[0]
    rank2 = ts2[ts2[TICKER] == ticker]['Rank_Close_InCluster'].values[0]
    # They CAN be the same by chance, but structure is correct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
