"""Tests for cluster reporting utilities."""

import pytest
import pandas as pd
import numpy as np

from features.ticker_clusters import (
    get_cluster_membership_report,
    format_cluster_report_text,
    get_cluster_performance_by_predictions,
    compute_ticker_characteristics,
)


@pytest.fixture
def sample_cluster_map():
    """Sample cluster assignments for testing."""
    return {
        'FPH.NZ': 0, 'EBO.NZ': 0, 'RYM.NZ': 0,  # Cluster 0
        'ATM.NZ': 1, 'SKL.NZ': 1,               # Cluster 1
        'SPK.NZ': 2, 'MEL.NZ': 2, 'AIR.NZ': 2, 'FBU.NZ': 2,  # Cluster 2
    }


@pytest.fixture
def sample_stats_df():
    """Sample ticker characteristics for testing."""
    return pd.DataFrame({
        'ticker': ['FPH.NZ', 'EBO.NZ', 'RYM.NZ', 'ATM.NZ', 'SKL.NZ', 'SPK.NZ', 'MEL.NZ', 'AIR.NZ', 'FBU.NZ'],
        'volatility': [0.15, 0.18, 0.20, 0.55, 0.60, 0.30, 0.32, 0.35, 0.28],
        'mean_return': [0.08, 0.10, 0.05, 0.20, -0.10, -0.05, 0.02, 0.03, 0.01],
        'skewness': [0.1, -0.2, 0.0, 0.5, -0.3, 0.1, 0.2, -0.1, 0.0],
        'max_drawdown': [-0.15, -0.20, -0.18, -0.40, -0.50, -0.25, -0.22, -0.28, -0.20],
        'pos_days_pct': [0.52, 0.54, 0.51, 0.55, 0.45, 0.48, 0.50, 0.49, 0.51],
    })


@pytest.fixture
def sample_predictions_df(sample_cluster_map):
    """Sample predictions DataFrame for testing."""
    np.random.seed(42)
    tickers = list(sample_cluster_map.keys()) * 10
    return pd.DataFrame({
        'timestamp': [i // 9 + 1 for i in range(len(tickers))],
        'ticker': tickers,
        'predicted_score': np.random.randn(len(tickers)),
        'actual_return': np.random.randn(len(tickers)) * 0.1,
    })


class TestClusterMembershipReport:
    """Tests for get_cluster_membership_report."""
    
    def test_basic_report_structure(self, sample_cluster_map):
        """Test that report has expected structure."""
        report = get_cluster_membership_report(sample_cluster_map)
        
        assert 'clusters' in report
        assert 'ticker_lookup' in report
        
    def test_cluster_contents(self, sample_cluster_map):
        """Test cluster contents are correct."""
        report = get_cluster_membership_report(sample_cluster_map)
        
        # Check cluster 0
        assert report['clusters'][0]['n_stocks'] == 3
        assert 'FPH.NZ' in report['clusters'][0]['tickers']
        
        # Check cluster 2 (largest)
        assert report['clusters'][2]['n_stocks'] == 4
        
    def test_ticker_lookup(self, sample_cluster_map):
        """Test ticker lookup works."""
        report = get_cluster_membership_report(sample_cluster_map)
        
        assert report['ticker_lookup']['FPH.NZ']['cluster'] == 0
        assert report['ticker_lookup']['ATM.NZ']['cluster'] == 1
        
    def test_with_characteristics(self, sample_cluster_map, sample_stats_df):
        """Test report with characteristics included."""
        report = get_cluster_membership_report(sample_cluster_map, sample_stats_df)
        
        # Check characteristics are populated
        chars = report['clusters'][0]['characteristics']
        assert 'volatility' in chars
        assert 'mean_return' in chars
        
        # Cluster 0 should have low volatility (FPH, EBO, RYM)
        assert chars['volatility'] < 0.25  # Low vol stocks
        
    def test_cluster_labels_generated(self, sample_cluster_map, sample_stats_df):
        """Test that cluster labels are auto-generated based on characteristics."""
        report = get_cluster_membership_report(sample_cluster_map, sample_stats_df)
        
        # Check labels exist
        for cluster_id in report['clusters']:
            assert 'label' in report['clusters'][cluster_id]
            assert report['clusters'][cluster_id]['label'] != ''


class TestFormatClusterReport:
    """Tests for format_cluster_report_text."""
    
    def test_format_basic(self, sample_cluster_map, sample_stats_df):
        """Test basic text formatting."""
        report = get_cluster_membership_report(sample_cluster_map, sample_stats_df)
        text = format_cluster_report_text(report)
        
        assert 'CLUSTER MEMBERSHIP REPORT' in text
        assert 'CLUSTER 0' in text
        assert 'FPH.NZ' in text
        
    def test_format_includes_characteristics(self, sample_cluster_map, sample_stats_df):
        """Test that characteristics are in the text output."""
        report = get_cluster_membership_report(sample_cluster_map, sample_stats_df)
        text = format_cluster_report_text(report)
        
        assert 'Volatility' in text
        assert 'Annual Return' in text


class TestClusterPerformance:
    """Tests for get_cluster_performance_by_predictions."""
    
    def test_basic_performance(self, sample_predictions_df, sample_cluster_map):
        """Test basic performance computation."""
        perf = get_cluster_performance_by_predictions(
            sample_predictions_df,
            sample_cluster_map,
            ticker_col='ticker',
            actual_col='actual_return',
            predicted_col='predicted_score',
        )
        
        assert len(perf) > 0
        assert 'cluster' in perf.columns
        assert 'pearson_ic' in perf.columns
        assert 'hit_rate' in perf.columns
        
    def test_performance_metrics_bounded(self, sample_predictions_df, sample_cluster_map):
        """Test that performance metrics are in valid ranges."""
        perf = get_cluster_performance_by_predictions(
            sample_predictions_df,
            sample_cluster_map,
        )
        
        # IC should be between -1 and 1
        assert perf['pearson_ic'].between(-1, 1).all()
        assert perf['rank_ic'].between(-1, 1).all()
        
        # Hit rate should be between 0 and 1
        assert perf['hit_rate'].between(0, 1).all()
        
    def test_performance_per_cluster(self, sample_predictions_df, sample_cluster_map):
        """Test that we get performance for each cluster."""
        perf = get_cluster_performance_by_predictions(
            sample_predictions_df,
            sample_cluster_map,
        )
        
        # Should have results for clusters with enough data
        assert len(perf) >= 2  # At least 2 clusters with enough predictions
