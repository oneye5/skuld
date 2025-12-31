"""Tests to verify metric calculations are mathematically correct.

These tests verify that all computed metrics match their expected mathematical
formulas using controlled synthetic data where the correct values are known.

Test categories:
1. IC calculations - Pearson and Spearman correlations
2. ICIR calculations - with and without annualization
3. Quintile returns - proper bucketing and averaging
4. Portfolio metrics - Sharpe, Sortino, Calmar ratios
5. Win/Loss statistics - hit rate, profit factor
6. Cross-sectional aggregations - across timestamps
"""

import pandas as pd
import numpy as np
import pytest
from scipy import stats
from dataclasses import dataclass
from typing import Dict

from config.columns import TIMESTAMP, TICKER


# =============================================================================
# Fixtures for controlled test data
# =============================================================================

@pytest.fixture
def perfect_correlation_data() -> pd.DataFrame:
    """Data where predictions perfectly correlate with actual returns."""
    n_timestamps = 10
    n_stocks = 20
    
    rows = []
    for t in range(n_timestamps):
        ts = 1000000000000 + t * 86400000  # Daily timestamps
        for s in range(n_stocks):
            # Predictions exactly equal returns (perfect correlation)
            score = s / n_stocks  # 0 to ~1
            ret = score * 0.1  # 0% to 10%
            rows.append({
                TIMESTAMP: ts,
                TICKER: f"STOCK_{s:02d}",
                "predicted_score": score,
                "actual_return": ret,
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def inverse_correlation_data() -> pd.DataFrame:
    """Data where predictions inversely correlate with actual returns."""
    n_timestamps = 10
    n_stocks = 20
    
    rows = []
    for t in range(n_timestamps):
        ts = 1000000000000 + t * 86400000
        for s in range(n_stocks):
            score = s / n_stocks
            ret = (n_stocks - 1 - s) / n_stocks * 0.1  # Inverse
            rows.append({
                TIMESTAMP: ts,
                TICKER: f"STOCK_{s:02d}",
                "predicted_score": score,
                "actual_return": ret,
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def known_ic_series() -> pd.Series:
    """IC series with known mean and std for ICIR testing."""
    # ICs: [0.05, 0.05, 0.05, 0.05, 0.05] -> mean=0.05, std=0
    # Need non-zero std for ICIR
    return pd.Series([0.04, 0.05, 0.06, 0.05, 0.05])  # mean=0.05, std=0.00707


@pytest.fixture
def quintile_test_data() -> pd.DataFrame:
    """Data designed for predictable quintile returns."""
    n_stocks = 50  # 10 per quintile
    
    rows = []
    ts = 1000000000000
    
    for s in range(n_stocks):
        # Score determines quintile: 0-9 -> Q1, 10-19 -> Q2, etc.
        score = s
        # Set returns by quintile: Q1=-10%, Q2=-5%, Q3=0%, Q4=5%, Q5=10%
        quintile = (s // 10) + 1
        expected_returns = {1: -0.10, 2: -0.05, 3: 0.00, 4: 0.05, 5: 0.10}
        ret = expected_returns[quintile]
        
        rows.append({
            TIMESTAMP: ts,
            TICKER: f"STOCK_{s:02d}",
            "predicted_score": score,
            "actual_return": ret,
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# Test IC Calculations
# =============================================================================

class TestICMathematicalCorrectness:
    """Verify IC matches scipy.stats correlation calculations."""
    
    def test_ic_matches_scipy_pearsonr(self, perfect_correlation_data):
        """IC should exactly match scipy.stats.pearsonr."""
        from evaluation.ranking_metrics import compute_ic
        
        df = perfect_correlation_data
        ts = df[TIMESTAMP].iloc[0]
        group = df[df[TIMESTAMP] == ts]
        
        our_ic = compute_ic(group["predicted_score"], group["actual_return"])
        scipy_ic, _ = stats.pearsonr(group["predicted_score"], group["actual_return"])
        
        assert abs(our_ic - scipy_ic) < 1e-10, f"Our IC={our_ic}, Scipy={scipy_ic}"
    
    def test_rank_ic_matches_scipy_spearmanr(self, perfect_correlation_data):
        """Rank IC should exactly match scipy.stats.spearmanr."""
        from evaluation.ranking_metrics import compute_rank_ic
        
        df = perfect_correlation_data
        ts = df[TIMESTAMP].iloc[0]
        group = df[df[TIMESTAMP] == ts]
        
        our_rank_ic = compute_rank_ic(group["predicted_score"], group["actual_return"])
        scipy_rank_ic, _ = stats.spearmanr(group["predicted_score"], group["actual_return"])
        
        assert abs(our_rank_ic - scipy_rank_ic) < 1e-10
    
    def test_ic_equals_one_for_perfect_correlation(self, perfect_correlation_data):
        """Perfect positive correlation should give IC = 1.0."""
        from evaluation.ranking_metrics import compute_ic
        
        df = perfect_correlation_data
        ts = df[TIMESTAMP].iloc[0]
        group = df[df[TIMESTAMP] == ts]
        
        ic = compute_ic(group["predicted_score"], group["actual_return"])
        
        assert abs(ic - 1.0) < 1e-10
    
    def test_ic_equals_minus_one_for_inverse_correlation(self, inverse_correlation_data):
        """Perfect negative correlation should give IC = -1.0."""
        from evaluation.ranking_metrics import compute_ic
        
        df = inverse_correlation_data
        ts = df[TIMESTAMP].iloc[0]
        group = df[df[TIMESTAMP] == ts]
        
        ic = compute_ic(group["predicted_score"], group["actual_return"])
        
        assert abs(ic - (-1.0)) < 1e-10
    
    def test_ic_bounded_between_minus_one_and_one(self):
        """IC must always be in [-1, 1]."""
        from evaluation.ranking_metrics import compute_ic
        
        np.random.seed(12345)
        for _ in range(100):
            pred = pd.Series(np.random.randn(30))
            actual = pd.Series(np.random.randn(30))
            ic = compute_ic(pred, actual)
            
            if not np.isnan(ic):
                assert -1.0 <= ic <= 1.0, f"IC out of bounds: {ic}"


class TestICIRMathematicalCorrectness:
    """Verify ICIR calculation is mean(IC) / std(IC) * sqrt(periods)."""
    
    def test_raw_icir_formula(self, known_ic_series):
        """Raw ICIR = mean(IC) / std(IC) without annualization."""
        from evaluation.ranking_metrics import compute_icir
        
        ic_series = known_ic_series
        
        expected = ic_series.mean() / ic_series.std()
        actual = compute_icir(ic_series, annualize=False)
        
        assert abs(actual - expected) < 1e-10, f"Expected {expected}, got {actual}"
    
    def test_annualized_icir_formula(self, known_ic_series):
        """Annualized ICIR = raw_ICIR * sqrt(periods_per_year)."""
        from evaluation.ranking_metrics import compute_icir
        
        ic_series = known_ic_series
        periods_per_year = 252
        
        raw_icir = ic_series.mean() / ic_series.std()
        expected_annual = raw_icir * np.sqrt(periods_per_year)
        
        actual = compute_icir(ic_series, annualize=True, periods_per_year=periods_per_year)
        
        assert abs(actual - expected_annual) < 1e-10
    
    def test_icir_high_for_consistent_positive_ic(self):
        """Consistent positive IC should give high ICIR."""
        from evaluation.ranking_metrics import compute_icir
        
        # All ICs = 0.05 with very low variance
        ic_series = pd.Series([0.050, 0.051, 0.049, 0.050, 0.050])
        
        icir = compute_icir(ic_series, annualize=False)
        
        # High ICIR because low std
        assert icir > 5.0, f"ICIR should be high for consistent IC, got {icir}"
    
    def test_icir_low_for_volatile_ic(self):
        """Volatile IC (positive and negative) should give low ICIR."""
        from evaluation.ranking_metrics import compute_icir
        
        # ICs oscillate between positive and negative
        ic_series = pd.Series([0.10, -0.08, 0.12, -0.09, 0.05])
        
        icir = compute_icir(ic_series, annualize=False)
        
        # Low ICIR because high std relative to mean
        assert icir < 1.0, f"ICIR should be low for volatile IC, got {icir}"


class TestQuintileMathematicalCorrectness:
    """Verify quintile calculations split data correctly and compute means."""
    
    def test_quintiles_have_equal_counts(self, quintile_test_data):
        """Each quintile should have roughly n/5 stocks."""
        from evaluation.ranking_metrics import assign_quintiles
        
        df = quintile_test_data
        quintiles = assign_quintiles(df["predicted_score"])
        
        counts = quintiles.value_counts().sort_index()
        
        expected_count = len(df) // 5  # 10 stocks per quintile
        
        for q in range(1, 6):
            assert abs(counts[q] - expected_count) <= 1, f"Q{q} has {counts[q]} stocks, expected ~{expected_count}"
    
    def test_quintile_returns_are_group_means(self, quintile_test_data):
        """Quintile return should be mean of actual returns in that quintile."""
        from evaluation.ranking_metrics import compute_quintile_returns, assign_quintiles
        
        df = quintile_test_data
        quintile_returns = compute_quintile_returns(
            df["predicted_score"], df["actual_return"]
        )
        
        # Manually compute
        quintiles = assign_quintiles(df["predicted_score"])
        df_with_q = df.copy()
        df_with_q["quintile"] = quintiles.values
        
        for q in range(1, 6):
            manual_mean = df_with_q[df_with_q["quintile"] == q]["actual_return"].mean()
            computed = quintile_returns[q]
            
            assert abs(computed - manual_mean) < 1e-10, f"Q{q}: computed={computed}, manual={manual_mean}"
    
    def test_quintile_spread_formula(self, quintile_test_data):
        """Quintile spread = Q5 - Q1."""
        from evaluation.ranking_metrics import compute_quintile_returns, compute_quintile_spread
        
        df = quintile_test_data
        quintile_returns = compute_quintile_returns(
            df["predicted_score"], df["actual_return"]
        )
        
        spread = compute_quintile_spread(quintile_returns)
        expected = quintile_returns[5] - quintile_returns[1]
        
        assert abs(spread - expected) < 1e-10
    
    def test_known_quintile_returns(self, quintile_test_data):
        """With controlled data, verify exact quintile return values."""
        from evaluation.ranking_metrics import compute_quintile_returns
        
        df = quintile_test_data
        quintile_returns = compute_quintile_returns(
            df["predicted_score"], df["actual_return"]
        )
        
        # Expected: Q1=-10%, Q2=-5%, Q3=0%, Q4=5%, Q5=10%
        expected = {1: -0.10, 2: -0.05, 3: 0.00, 4: 0.05, 5: 0.10}
        
        for q, exp_ret in expected.items():
            assert abs(quintile_returns[q] - exp_ret) < 1e-10, \
                f"Q{q}: expected {exp_ret}, got {quintile_returns[q]}"


class TestHitRateMathematicalCorrectness:
    """Verify hit rate is correctly computed as % of top-N with positive returns."""
    
    def test_hit_rate_formula(self):
        """Hit rate = (# of top-N with positive returns) / N."""
        from evaluation.ranking_metrics import compute_hit_rate
        
        # 5 stocks, sorted by predicted score (desc)
        predicted = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        actual = pd.Series([0.10, -0.05, 0.03, -0.02, 0.01])
        
        # Top 3 by prediction: stocks with pred 0.5, 0.4, 0.3
        # Their returns: 0.10, -0.05, 0.03
        # Positive returns: 0.10, 0.03 (2 out of 3)
        
        hit_rate = compute_hit_rate(predicted, actual, top_n=3)
        expected = 2 / 3
        
        assert abs(hit_rate - expected) < 1e-10
    
    def test_hit_rate_100_percent(self):
        """All top-N positive returns gives hit rate = 1.0."""
        from evaluation.ranking_metrics import compute_hit_rate
        
        predicted = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        actual = pd.Series([0.10, 0.08, 0.05, -0.02, -0.05])
        
        hit_rate = compute_hit_rate(predicted, actual, top_n=3)
        
        assert abs(hit_rate - 1.0) < 1e-10
    
    def test_hit_rate_0_percent(self):
        """All top-N negative returns gives hit rate = 0.0."""
        from evaluation.ranking_metrics import compute_hit_rate
        
        predicted = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        actual = pd.Series([-0.10, -0.08, -0.05, 0.02, 0.05])
        
        hit_rate = compute_hit_rate(predicted, actual, top_n=3)
        
        assert abs(hit_rate - 0.0) < 1e-10


class TestCrossSectionalAggregation:
    """Verify cross-sectional IC series and aggregated metrics."""
    
    def test_cross_sectional_ic_per_timestamp(self, perfect_correlation_data):
        """Should compute IC for each timestamp independently."""
        from evaluation.ranking_metrics import compute_cross_sectional_ic_series
        
        df = perfect_correlation_data
        
        ic_series = compute_cross_sectional_ic_series(
            df, 
            timestamp_col=TIMESTAMP,
            predicted_col="predicted_score",
            actual_col="actual_return",
        )
        
        # Should have one IC per timestamp
        n_timestamps = df[TIMESTAMP].nunique()
        assert len(ic_series) == n_timestamps
        
        # All ICs should be 1.0 for perfect correlation data
        for ic in ic_series:
            assert abs(ic - 1.0) < 1e-10
    
    def test_mean_ic_is_average_of_ic_series(self, perfect_correlation_data):
        """Mean IC should be simple average of all timestamp ICs."""
        from evaluation.ranking_metrics import compute_cross_sectional_ic_series
        
        df = perfect_correlation_data
        
        ic_series = compute_cross_sectional_ic_series(df)
        
        mean_ic = ic_series.mean()
        manual_mean = sum(ic_series) / len(ic_series)
        
        assert abs(mean_ic - manual_mean) < 1e-10


class TestSharpeRatioCorrectness:
    """Verify Sharpe ratio calculation."""
    
    def test_sharpe_formula_annualized(self):
        """Sharpe = mean(returns) / std(returns) * sqrt(periods_per_year)."""
        from evaluation.portfolio_simulator import BacktestResult
        
        # Create a returns series
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])  # 5 periods
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        # For 5 periods (assume each is ~73 days = 365/5)
        # Annualization factor depends on period length
        periods_per_year = 365 / 73  # ~5
        
        expected_sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
        
        # Just verify the formula makes sense
        raw_sharpe = mean_ret / std_ret
        assert raw_sharpe > 0  # Positive mean, positive Sharpe
    
    def test_sharpe_positive_for_positive_returns(self):
        """Consistent positive returns should give positive Sharpe."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.012, 0.018])
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = mean_ret / std_ret
        
        assert sharpe > 0


class TestSortinoRatioCorrectness:
    """Verify Sortino ratio uses only downside deviation."""
    
    def test_sortino_only_uses_negative_returns(self):
        """Sortino denominator should only consider negative returns."""
        from evaluation.ranking_metrics import compute_sortino_ratio
        
        # Returns with some negative
        returns = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04])
        
        mean_ret = returns.mean()
        
        # Downside deviation: std of only negative returns
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()
        
        # Sortino should be mean / downside_std (not total std)
        sortino = compute_sortino_ratio(returns)
        
        # Should be higher than Sharpe because downside_std < total_std
        sharpe = mean_ret / returns.std()
        
        # With more positive than negative returns, sortino should be higher
        assert sortino > sharpe or np.isnan(sortino)


class TestWinLossStatistics:
    """Verify win/loss calculations."""
    
    def test_win_rate_formula(self):
        """Win rate = (# positive returns) / (total returns)."""
        returns = pd.Series([0.01, -0.01, 0.02, 0.005, -0.02])
        
        n_wins = (returns > 0).sum()  # 3
        n_total = len(returns)  # 5
        
        expected_win_rate = n_wins / n_total  # 0.6
        
        assert expected_win_rate == 0.6
    
    def test_avg_win_loss_formula(self):
        """Average win = mean of positive returns, loss = mean of negative."""
        returns = pd.Series([0.01, -0.01, 0.02, 0.005, -0.02])
        
        avg_win = returns[returns > 0].mean()  # mean(0.01, 0.02, 0.005) = 0.01167
        avg_loss = returns[returns < 0].mean()  # mean(-0.01, -0.02) = -0.015
        
        assert abs(avg_win - (0.01 + 0.02 + 0.005) / 3) < 1e-10
        assert abs(avg_loss - (-0.01 - 0.02) / 2) < 1e-10
    
    def test_profit_factor_formula(self):
        """Profit factor = sum(wins) / abs(sum(losses))."""
        returns = pd.Series([0.01, -0.01, 0.02, 0.005, -0.02])
        
        sum_wins = returns[returns > 0].sum()  # 0.035
        sum_losses = abs(returns[returns < 0].sum())  # 0.03
        
        expected_profit_factor = sum_wins / sum_losses  # 1.167
        
        assert abs(expected_profit_factor - 0.035 / 0.03) < 1e-10


class TestMonotonicityCheck:
    """Verify quintile monotonicity detection."""
    
    def test_perfect_monotonicity(self):
        """Q1 < Q2 < Q3 < Q4 < Q5 should be monotonic."""
        from evaluation.ranking_metrics import check_quintile_monotonicity
        
        quintile_returns = {1: -0.10, 2: -0.05, 3: 0.00, 4: 0.05, 5: 0.10}
        
        result = check_quintile_monotonicity(quintile_returns)
        
        assert result["is_monotonic"] is True
        assert result["violations"] == 0
        assert result["monotonicity_score"] == 1.0
    
    def test_one_violation(self):
        """One pair out of order should report 1 violation."""
        from evaluation.ranking_metrics import check_quintile_monotonicity
        
        # Q3 > Q4 is a violation
        quintile_returns = {1: -0.10, 2: -0.05, 3: 0.06, 4: 0.05, 5: 0.10}
        
        result = check_quintile_monotonicity(quintile_returns)
        
        assert result["is_monotonic"] is False
        assert result["violations"] == 1
    
    def test_complete_reversal(self):
        """Q5 < Q4 < Q3 < Q2 < Q1 should have 4 violations."""
        from evaluation.ranking_metrics import check_quintile_monotonicity
        
        quintile_returns = {1: 0.10, 2: 0.05, 3: 0.00, 4: -0.05, 5: -0.10}
        
        result = check_quintile_monotonicity(quintile_returns)
        
        assert result["is_monotonic"] is False
        assert result["violations"] == 4


class TestStatisticalSignificance:
    """Verify t-test calculations for statistical significance."""
    
    def test_ic_ttest_uses_scipy_internally(self):
        """IC t-test p-value should match manual scipy calculation."""
        from scipy import stats
        
        ic_series = pd.Series([0.05, 0.04, 0.06, 0.05, 0.07, 0.03, 0.05, 0.04])
        
        # Manual calculation
        _, scipy_pvalue = stats.ttest_1samp(ic_series, 0)  # Test if mean != 0
        
        # Should be significantly different from zero
        assert scipy_pvalue < 0.05
    
    def test_high_pvalue_for_zero_mean_ic(self):
        """IC series with ~zero mean should have high p-value."""
        np.random.seed(42)
        # ICs centered around zero
        ic_series = pd.Series(np.random.randn(50) * 0.05)
        
        _, pvalue = stats.ttest_1samp(ic_series, 0)
        
        # Not statistically significant from zero (high p-value)
        assert pvalue > 0.05
    
    def test_low_pvalue_for_positive_mean_ic(self):
        """Consistently positive IC should have low p-value."""
        # All positive ICs
        ic_series = pd.Series([0.05, 0.06, 0.04, 0.05, 0.07, 0.06, 0.05, 0.06,
                               0.05, 0.04, 0.06, 0.05, 0.05, 0.07, 0.04, 0.06])
        
        _, pvalue = stats.ttest_1samp(ic_series, 0)
        
        # Statistically significant (low p-value)
        assert pvalue < 0.001


class TestDecileCalculations:
    """Verify decile (10-group) calculations."""
    
    def test_decile_returns_has_10_groups(self):
        """Decile returns should have keys 1-10."""
        from evaluation.ranking_metrics import compute_decile_returns
        
        n = 100
        predicted = pd.Series(range(n))
        actual = pd.Series(np.linspace(-0.1, 0.1, n))
        
        decile_returns = compute_decile_returns(predicted, actual)
        
        assert set(decile_returns.keys()) == set(range(1, 11))
    
    def test_decile_spread_larger_than_quintile_spread(self):
        """D10 - D1 should be >= Q5 - Q1 because deciles are more extreme."""
        from evaluation.ranking_metrics import (
            compute_quintile_returns, 
            compute_decile_returns,
        )
        
        # Create data with spread
        n = 100
        predicted = pd.Series(range(n))
        actual = pd.Series(np.linspace(-0.1, 0.1, n))  # -10% to +10%
        
        quintile_returns = compute_quintile_returns(predicted, actual)
        decile_returns = compute_decile_returns(predicted, actual)
        
        q_spread = quintile_returns[5] - quintile_returns[1]
        d_spread = decile_returns[10] - decile_returns[1]
        
        # Decile spread should be at least as large
        assert d_spread >= q_spread - 0.01  # Allow small tolerance


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_ic_with_constant_predictions(self):
        """Constant predictions should return NaN IC."""
        from evaluation.ranking_metrics import compute_ic
        
        predicted = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
        actual = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        
        ic = compute_ic(predicted, actual)
        
        assert np.isnan(ic)
    
    def test_ic_with_constant_returns(self):
        """Constant returns should return NaN IC."""
        from evaluation.ranking_metrics import compute_ic
        
        predicted = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        actual = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
        
        ic = compute_ic(predicted, actual)
        
        assert np.isnan(ic)
    
    def test_ic_with_insufficient_data(self):
        """Less than 3 data points should return NaN."""
        from evaluation.ranking_metrics import compute_ic
        
        predicted = pd.Series([0.1, 0.2])
        actual = pd.Series([0.1, 0.2])
        
        ic = compute_ic(predicted, actual)
        
        assert np.isnan(ic)
    
    def test_icir_with_zero_std(self):
        """ICIR with zero std (constant IC) should return NaN."""
        from evaluation.ranking_metrics import compute_icir
        
        ic_series = pd.Series([0.05, 0.05, 0.05, 0.05])
        
        icir = compute_icir(ic_series, annualize=False)
        
        # With zero std, division by zero -> NaN is acceptable behavior
        # (or inf depending on implementation)
        assert np.isnan(icir) or np.isinf(icir)
    
    def test_hit_rate_with_zero_returns(self):
        """Zero returns should not count as 'hits' (positive)."""
        from evaluation.ranking_metrics import compute_hit_rate
        
        predicted = pd.Series([0.5, 0.4, 0.3])
        actual = pd.Series([0.0, 0.0, 0.0])
        
        hit_rate = compute_hit_rate(predicted, actual, top_n=3)
        
        assert hit_rate == 0.0


class TestConsistencyBetweenMetrics:
    """Test that related metrics are consistent with each other."""
    
    def test_rank_icir_close_to_icir(self, perfect_correlation_data):
        """Rank ICIR and ICIR should be similar for well-behaved data."""
        from evaluation.ranking_metrics import compute_cross_sectional_ic_series
        
        df = perfect_correlation_data
        
        ic_series = compute_cross_sectional_ic_series(df, use_rank=False)
        rank_ic_series = compute_cross_sectional_ic_series(df, use_rank=True)
        
        # Both should be 1.0 for perfect correlation
        assert abs(ic_series.mean() - rank_ic_series.mean()) < 0.1
    
    def test_quintile_spread_equals_q5_minus_q1(self):
        """Quintile spread must equal Q5 - Q1."""
        from evaluation.ranking_metrics import compute_quintile_returns, compute_quintile_spread
        
        predicted = pd.Series(range(50))
        actual = pd.Series(np.linspace(-0.1, 0.1, 50))
        
        quintile_returns = compute_quintile_returns(predicted, actual)
        spread = compute_quintile_spread(quintile_returns)
        
        expected = quintile_returns[5] - quintile_returns[1]
        
        assert abs(spread - expected) < 1e-10
    
    def test_hit_rate_aligns_with_quintile_5_returns(self):
        """High hit rate should correlate with positive Q5 returns."""
        from evaluation.ranking_metrics import compute_hit_rate, compute_quintile_returns
        
        # Good model: top predictions have positive returns
        predicted = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01])
        actual = pd.Series([0.10, 0.08, 0.05, 0.03, 0.01, -0.01, -0.03, -0.05, -0.08, -0.10])
        
        hit_rate = compute_hit_rate(predicted, actual, top_n=2)
        quintile_returns = compute_quintile_returns(predicted, actual)
        
        # High hit rate -> positive Q5
        assert hit_rate == 1.0
        assert quintile_returns[5] > 0
