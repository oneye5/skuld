"""Tests for ranking_metrics module - IC, RankIC, ICIR, quintile calculations."""

import pandas as pd
import numpy as np
import pytest

from config.columns import TIMESTAMP, TICKER


class TestICCalculations:
    """Tests for Information Coefficient calculations."""
    
    def test_perfect_ranking_ic_equals_one(self):
        """If predicted scores exactly equal actual returns, IC = 1.0"""
        from evaluation.ranking_metrics import compute_ic
        
        predicted = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        actual = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        
        ic = compute_ic(predicted, actual)
        
        assert abs(ic - 1.0) < 1e-6
    
    def test_inverse_ranking_ic_equals_minus_one(self):
        """If predicted scores are inverse of actual returns, IC = -1.0"""
        from evaluation.ranking_metrics import compute_ic
        
        predicted = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        actual = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        
        ic = compute_ic(predicted, actual)
        
        assert abs(ic - (-1.0)) < 1e-6
    
    def test_random_ranking_ic_near_zero(self):
        """Random predictions should have IC ≈ 0 on average."""
        from evaluation.ranking_metrics import compute_ic
        
        np.random.seed(42)
        ics = []
        
        for _ in range(100):
            predicted = pd.Series(np.random.randn(50))
            actual = pd.Series(np.random.randn(50))
            ics.append(compute_ic(predicted, actual))
        
        mean_ic = np.mean(ics)
        # Random should be close to zero (within ±0.15 for 100 trials)
        assert abs(mean_ic) < 0.15


class TestRankICCalculations:
    """Tests for Rank Information Coefficient (Spearman) calculations."""
    
    def test_perfect_rank_ic_equals_one(self):
        """If predicted ranks exactly match actual ranks, Rank IC = 1.0"""
        from evaluation.ranking_metrics import compute_rank_ic
        
        predicted = pd.Series([1, 2, 3, 4, 5])
        actual = pd.Series([10, 20, 30, 40, 50])
        
        rank_ic = compute_rank_ic(predicted, actual)
        
        assert abs(rank_ic - 1.0) < 1e-6
    
    def test_rank_ic_robust_to_outliers(self):
        """Rank IC should be less affected by outliers than Pearson IC."""
        from evaluation.ranking_metrics import compute_ic, compute_rank_ic
        
        # Normal data
        predicted = pd.Series([1, 2, 3, 4, 5])
        actual = pd.Series([1, 2, 3, 4, 5])
        
        # Add extreme outlier
        predicted_outlier = pd.Series([1, 2, 3, 4, 100])  # extreme value
        actual_outlier = pd.Series([1, 2, 3, 4, 5])
        
        rank_ic_clean = compute_rank_ic(predicted, actual)
        rank_ic_outlier = compute_rank_ic(predicted_outlier, actual_outlier)
        
        # Rank IC should still be 1.0 because rank ordering is preserved
        assert abs(rank_ic_clean - 1.0) < 1e-6
        assert abs(rank_ic_outlier - 1.0) < 1e-6


class TestICIRCalculations:
    """Tests for Information Coefficient Information Ratio."""
    
    def test_icir_formula(self):
        """ICIR = mean(IC) / std(IC) * sqrt(annualization_factor)"""
        from evaluation.ranking_metrics import compute_icir
        
        # Known IC series
        ic_series = pd.Series([0.05, 0.04, 0.06, 0.05, 0.05])
        
        mean_ic = ic_series.mean()  # 0.05
        std_ic = ic_series.std()
        
        icir = compute_icir(ic_series, annualize=False)
        
        expected = mean_ic / std_ic
        assert abs(icir - expected) < 1e-6
    
    def test_icir_annualized(self):
        """Annualized ICIR multiplies by sqrt(252) for daily data."""
        from evaluation.ranking_metrics import compute_icir
        
        ic_series = pd.Series([0.05, 0.04, 0.06, 0.05, 0.05])
        
        icir_raw = compute_icir(ic_series, annualize=False)
        icir_annual = compute_icir(ic_series, annualize=True, periods_per_year=252)
        
        assert abs(icir_annual - icir_raw * np.sqrt(252)) < 1e-6


class TestQuintileCalculations:
    """Tests for quintile return analysis."""
    
    def test_quintile_split_correct_sizes(self):
        """Quintiles should split data into 5 roughly equal groups."""
        from evaluation.ranking_metrics import assign_quintiles
        
        # 100 items should give 20 per quintile
        scores = pd.Series(range(100))
        
        quintiles = assign_quintiles(scores)
        
        # Should have 5 groups
        assert set(quintiles.unique()) == {1, 2, 3, 4, 5}
        
        # Each quintile should have ~20 items (may vary by ±1 due to ties)
        for q in range(1, 6):
            assert 18 <= (quintiles == q).sum() <= 22
    
    def test_quintile_5_has_highest_scores(self):
        """Q5 (top quintile) should have the highest predicted scores."""
        from evaluation.ranking_metrics import assign_quintiles
        
        scores = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        quintiles = assign_quintiles(scores)
        
        # Score 10 (and 9) should be in Q5
        assert quintiles.iloc[9] == 5  # highest score
        assert quintiles.iloc[0] == 1  # lowest score
    
    def test_quintile_returns_monotonic_for_perfect_model(self):
        """Top quintile should have highest returns for good model."""
        from evaluation.ranking_metrics import compute_quintile_returns
        
        # Perfect model: predicted = actual
        df = pd.DataFrame({
            "predicted_score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "actual_return": [0.01, 0.02, 0.03, 0.04, 0.05, 
                             0.06, 0.07, 0.08, 0.09, 0.10],
        })
        
        quintile_returns = compute_quintile_returns(
            df["predicted_score"], df["actual_return"]
        )
        
        # Returns should be monotonically increasing Q1 -> Q5
        assert quintile_returns[5] > quintile_returns[4]
        assert quintile_returns[4] > quintile_returns[3]
        assert quintile_returns[3] > quintile_returns[2]
        assert quintile_returns[2] > quintile_returns[1]
    
    def test_quintile_spread_calculation(self):
        """Quintile spread = Q5_return - Q1_return."""
        from evaluation.ranking_metrics import compute_quintile_spread
        
        quintile_returns = {1: -0.02, 2: -0.01, 3: 0.00, 4: 0.01, 5: 0.03}
        
        spread = compute_quintile_spread(quintile_returns)
        
        assert abs(spread - 0.05) < 1e-6  # 0.03 - (-0.02) = 0.05


class TestHitRate:
    """Tests for hit rate calculation."""
    
    def test_hit_rate_perfect_predictions(self):
        """All top-N predictions have positive returns -> hit rate = 100%."""
        from evaluation.ranking_metrics import compute_hit_rate
        
        predicted = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        actual = pd.Series([0.10, 0.08, -0.05, -0.10, -0.15])  # Top 2 positive
        
        hit_rate = compute_hit_rate(predicted, actual, top_n=2)
        
        assert abs(hit_rate - 1.0) < 1e-6
    
    def test_hit_rate_no_hits(self):
        """No top-N predictions have positive returns -> hit rate = 0%."""
        from evaluation.ranking_metrics import compute_hit_rate
        
        predicted = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        actual = pd.Series([-0.10, -0.08, 0.05, 0.10, 0.15])  # Top 2 negative
        
        hit_rate = compute_hit_rate(predicted, actual, top_n=2)
        
        assert abs(hit_rate - 0.0) < 1e-6
    
    def test_hit_rate_partial(self):
        """50% of top-N predictions have positive returns."""
        from evaluation.ranking_metrics import compute_hit_rate
        
        predicted = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        actual = pd.Series([0.10, -0.08, 0.05, 0.10, 0.15])  # Top 2: 1 pos, 1 neg
        
        hit_rate = compute_hit_rate(predicted, actual, top_n=2)
        
        assert abs(hit_rate - 0.5) < 1e-6


class TestCrossTimestampMetrics:
    """Tests for computing metrics across multiple timestamps."""
    
    def test_compute_metrics_per_timestamp(self):
        """Compute IC for each timestamp separately, then aggregate."""
        from evaluation.ranking_metrics import compute_cross_sectional_ic_series
        
        # Two timestamps, each with 5 stocks
        df = pd.DataFrame({
            TIMESTAMP: [1] * 5 + [2] * 5,
            "predicted_score": [0.1, 0.2, 0.3, 0.4, 0.5] + [0.5, 0.4, 0.3, 0.2, 0.1],
            "actual_return": [0.1, 0.2, 0.3, 0.4, 0.5] + [0.5, 0.4, 0.3, 0.2, 0.1],
        })
        
        ic_series = compute_cross_sectional_ic_series(
            df, 
            timestamp_col=TIMESTAMP,
            predicted_col="predicted_score",
            actual_col="actual_return"
        )
        
        # Both timestamps should have IC = 1.0
        assert len(ic_series) == 2
        assert abs(ic_series[1] - 1.0) < 1e-6
        assert abs(ic_series[2] - 1.0) < 1e-6
    
    def test_skip_timestamp_with_few_stocks(self):
        """Timestamps with <min_stocks should be skipped."""
        from evaluation.ranking_metrics import compute_cross_sectional_ic_series
        
        # Timestamp 1 has 10 stocks, timestamp 2 has only 3
        df = pd.DataFrame({
            TIMESTAMP: [1] * 10 + [2] * 3,
            "predicted_score": list(range(10)) + [1, 2, 3],
            "actual_return": list(range(10)) + [1, 2, 3],
        })
        
        ic_series = compute_cross_sectional_ic_series(
            df, 
            timestamp_col=TIMESTAMP,
            predicted_col="predicted_score",
            actual_col="actual_return",
            min_stocks=5
        )
        
        # Only timestamp 1 should be included
        assert len(ic_series) == 1
        assert 1 in ic_series.index


class TestRankingMetricsDataclass:
    """Tests for the RankingMetrics dataclass."""
    
    def test_from_predictions_basic(self):
        """Create RankingMetrics from prediction DataFrame."""
        from evaluation.ranking_metrics import RankingMetrics
        
        # Create simple prediction data
        np.random.seed(42)
        n_timestamps = 20
        n_stocks = 50
        
        data = []
        for ts in range(n_timestamps):
            scores = np.random.randn(n_stocks)
            returns = scores * 0.1 + np.random.randn(n_stocks) * 0.05  # Correlated
            for i in range(n_stocks):
                data.append({
                    TIMESTAMP: ts,
                    TICKER: f"STOCK_{i}",
                    "predicted_score": scores[i],
                    "actual_return": returns[i],
                })
        
        df = pd.DataFrame(data)
        
        metrics = RankingMetrics.from_predictions(
            df,
            timestamp_col=TIMESTAMP,
            predicted_col="predicted_score",
            actual_col="actual_return"
        )
        
        # Should have positive IC (model has signal)
        assert metrics.mean_ic > 0
        assert metrics.mean_rank_ic > 0
        assert metrics.num_timestamps == n_timestamps
        assert abs(metrics.avg_stocks_per_timestamp - n_stocks) < 1
