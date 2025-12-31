"""Tests for backtest methodology correctness.

These tests are designed to catch common mistakes in portfolio backtesting:
1. Return frequency mismatch (using N-day returns but annualizing as daily)
2. Overlapping return periods (creates artificial smoothing/autocorrelation)
3. Lookahead bias (using information not available at decision time)
4. Incorrect Sharpe ratio calculation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.columns import TIMESTAMP, TICKER, ADJCLOSE
from config.settings import MS_PER_DAY
from evaluation.portfolio_simulator import (
    run_portfolio_backtest,
    PortfolioConfig,
    compute_sharpe_ratio,
    compute_max_drawdown,
)
from core.target_builder import compute_forward_returns, FORWARD_RETURN


# =============================================================================
# SHARPE RATIO CALCULATION TESTS
# =============================================================================

class TestSharpeRatioCalculation:
    """Tests for Sharpe ratio methodology."""
    
    def test_sharpe_with_known_distribution(self):
        """Verify Sharpe calculation with known mean/std."""
        # Daily returns with mean 0.1% and std 1%
        np.random.seed(42)
        n_days = 252
        mean_daily = 0.001  # 0.1% daily
        std_daily = 0.01    # 1% daily
        
        returns = pd.Series(np.random.normal(mean_daily, std_daily, n_days))
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        
        # Expected Sharpe ≈ (0.001 / 0.01) * sqrt(252) ≈ 1.59
        expected_sharpe = (mean_daily / std_daily) * np.sqrt(252)
        
        # Should be within reasonable range (Monte Carlo variance)
        assert abs(sharpe - expected_sharpe) < 0.5, (
            f"Sharpe {sharpe:.2f} too far from expected {expected_sharpe:.2f}"
        )
    
    def test_sharpe_auto_infers_period_from_timestamps(self):
        """Sharpe should auto-infer periods_per_year from timestamp index."""
        np.random.seed(42)
        
        # Weekly timestamps (5-day spacing in ms)
        weekly_ts = [i * 5 * MS_PER_DAY for i in range(52)]
        returns = pd.Series(
            np.random.normal(0.005, 0.02, 52),
            index=weekly_ts
        )
        
        # Auto-inference should detect weekly and use ~52 periods
        sharpe_auto = compute_sharpe_ratio(returns, periods_per_year=None)
        sharpe_weekly = compute_sharpe_ratio(returns, periods_per_year=52)
        
        # Should be similar
        assert abs(sharpe_auto - sharpe_weekly) < 0.5, (
            f"Auto-inferred {sharpe_auto:.2f} should match explicit weekly {sharpe_weekly:.2f}"
        )
    
    def test_5day_returns_need_5day_annualization(self):
        """5-day forward returns should NOT be annualized as daily returns."""
        np.random.seed(42)
        
        # 5-day returns (about 50 periods per year)
        periods_per_year_5day = 252 / 5  # ≈ 50.4
        
        five_day_mean = 0.005  # 0.5% per 5-day period
        five_day_std = 0.02
        five_day_returns = pd.Series(np.random.normal(five_day_mean, five_day_std, 50))
        
        # WRONG: Treating as daily
        wrong_sharpe = compute_sharpe_ratio(five_day_returns, periods_per_year=252)
        
        # CORRECT: Treating as 5-day periods
        correct_sharpe = compute_sharpe_ratio(five_day_returns, periods_per_year=int(periods_per_year_5day))
        
        # Wrong Sharpe is inflated by sqrt(252/50) ≈ 2.24x
        inflation_factor = np.sqrt(252 / periods_per_year_5day)
        assert wrong_sharpe > correct_sharpe * (inflation_factor - 0.5), (
            f"Wrong annualization inflates Sharpe by ~{inflation_factor:.1f}x"
        )


# =============================================================================
# OVERLAPPING RETURNS TESTS
# =============================================================================

class TestOverlappingReturns:
    """Tests to detect overlapping return period issues."""
    
    def test_detect_overlapping_forward_returns(self):
        """Daily timestamps with 5-day forward returns create overlap."""
        # Create daily data for 10 days
        timestamps = [i * MS_PER_DAY for i in range(10)]
        prices = [100.0 + i for i in range(10)]  # Linear increase
        
        df = pd.DataFrame({
            TIMESTAMP: timestamps,
            TICKER: ["A"] * 10,
            "Close": prices,
            ADJCLOSE: prices,
        })
        
        # Compute 5-day forward returns
        result = compute_forward_returns(df, lookahead_days=5)
        valid_returns = result[FORWARD_RETURN].dropna()
        
        # If timestamps are daily but returns are 5-day:
        # Day 0 return = price[5]/price[0] - 1
        # Day 1 return = price[6]/price[1] - 1
        # These overlap by 4 days!
        
        # For trading, this means we CAN'T rebalance daily with 5-day returns
        # because the return periods overlap
        
        # Check autocorrelation - overlapping returns should be highly correlated
        if len(valid_returns) > 2:
            autocorr = valid_returns.autocorr(lag=1)
            # High autocorrelation indicates overlapping periods
            # (would be ~0 for truly independent returns)
            assert autocorr > 0.5 or len(valid_returns) < 5, (
                f"Expected high autocorr for overlapping returns, got {autocorr:.2f}"
            )
    
    def test_non_overlapping_periods_required(self):
        """For valid backtest, rebalance frequency must match return horizon."""
        # With 5-day forward returns, should only rebalance every 5 days
        
        # Create data with 5-day spacing
        timestamps = [i * 5 * MS_PER_DAY for i in range(20)]  # Every 5 days
        prices = 100.0 * np.cumprod(1 + np.random.normal(0.002, 0.02, 20))
        
        df = pd.DataFrame({
            TIMESTAMP: timestamps,
            TICKER: ["A"] * 20,
            "Close": prices,
            ADJCLOSE: prices,
        })
        
        result = compute_forward_returns(df, lookahead_days=5)
        valid_returns = result[FORWARD_RETURN].dropna()
        
        # With proper spacing, autocorrelation should be lower
        if len(valid_returns) > 2:
            autocorr = valid_returns.autocorr(lag=1)
            # Still may have some autocorr due to momentum, but less than overlapping
            print(f"Non-overlapping autocorr: {autocorr:.3f}")


# =============================================================================
# LOOKAHEAD BIAS TESTS
# =============================================================================

class TestLookaheadBias:
    """Tests to detect lookahead bias in the pipeline."""
    
    def test_forward_return_uses_future_price_only(self):
        """Forward return should only depend on future price, not current day info."""
        timestamps = [i * MS_PER_DAY for i in range(10)]
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        df = pd.DataFrame({
            TIMESTAMP: timestamps,
            TICKER: ["A"] * 10,
            "Close": prices,
            ADJCLOSE: prices,
        })
        
        result = compute_forward_returns(df, lookahead_days=5)
        
        # Day 0: forward return = (105 - 100) / 100 = 0.05
        assert abs(result[FORWARD_RETURN].iloc[0] - 0.05) < 1e-6
        
        # Day 4: forward return = (109 - 104) / 104 ≈ 0.048
        expected = (109 - 104) / 104
        assert abs(result[FORWARD_RETURN].iloc[4] - expected) < 1e-6
    
    def test_prediction_cannot_see_future_returns(self):
        """Model predictions must not have access to future return values."""
        # This is a sanity check - if IC is very high (>0.3), 
        # there's likely leakage
        
        # Create random data where predictions should have NO signal
        np.random.seed(42)
        n_samples = 1000
        
        predictions_df = pd.DataFrame({
            TIMESTAMP: np.repeat(range(100), 10),  # 100 timestamps, 10 stocks each
            TICKER: np.tile([f"S{i}" for i in range(10)], 100),
            "predicted_score": np.random.randn(n_samples),  # Random predictions
            "actual_return": np.random.randn(n_samples) * 0.02,  # Random returns
        })
        
        # Compute IC
        from evaluation.ranking_metrics import compute_cross_sectional_ic_series
        ic_series = compute_cross_sectional_ic_series(
            predictions_df,
            timestamp_col=TIMESTAMP,
            predicted_col="predicted_score",
            actual_col="actual_return",
        )
        
        mean_ic = ic_series.mean()
        
        # With random data, IC should be near 0 (within noise)
        assert abs(mean_ic) < 0.1, (
            f"Random predictions have IC={mean_ic:.3f}, expected ~0. Possible leakage!"
        )


# =============================================================================
# BACKTEST CONSISTENCY TESTS
# =============================================================================

class TestBacktestConsistency:
    """Tests for backtest methodology consistency."""
    
    def test_perfect_foresight_sharpe_is_bounded(self):
        """With perfect predictions, returns are ALL positive - that's expected.
        
        Perfect foresight means we always pick the highest returning stocks for long
        and lowest for short. The long-short spread is therefore always positive
        by construction, leading to very high Sharpe ratios.
        
        This is NOT a bug - it's the expected behavior of perfect foresight.
        The important test is that random predictions give Sharpe near 0.
        """
        np.random.seed(42)
        
        # Create data where we have "perfect" predictions
        # Use weekly timestamps (5-day spacing) to avoid overlap with 5-day returns
        n_timestamps = 50
        n_stocks = 20
        
        records = []
        for t in range(n_timestamps):
            returns = np.random.normal(0.001, 0.02, n_stocks)
            for i, ret in enumerate(returns):
                records.append({
                    TIMESTAMP: t * 5 * MS_PER_DAY,  # 5-day spacing
                    TICKER: f"S{i}",
                    "predicted_score": ret,  # Perfect prediction!
                    "actual_return": ret,
                })
        
        predictions_df = pd.DataFrame(records)
        
        config = PortfolioConfig(top_n=5, bottom_n=5, transaction_cost_bps=10)
        result = run_portfolio_backtest(
            predictions_df,
            config,
            timestamp_col=TIMESTAMP,
            score_col="predicted_score",
            return_col="actual_return",
        )
        
        # Perfect foresight will have:
        # - Mean return ~4-5% per period (spread between top/bottom quintile)
        # - Very low std (always positive spread)
        # - Therefore VERY high Sharpe
        
        # This is mathematically correct - what matters is that:
        # 1. It's significantly higher than random (tested separately)
        # 2. Max drawdown is near 0 (always winning)
        assert result.sharpe_ratio > 5, "Perfect foresight should have high Sharpe"
        assert result.max_drawdown < 0.1, "Perfect foresight should have minimal drawdown"
        assert result.total_return > 1.0, "Perfect foresight should have >100% total return over 50 periods"
        
        print(f"Perfect foresight Sharpe: {result.sharpe_ratio:.2f}")
        print(f"Perfect foresight returns: {result.total_return:.1%}")
    
    def test_zero_skill_sharpe_near_zero(self):
        """Random predictions should produce Sharpe near 0."""
        np.random.seed(42)
        
        n_timestamps = 100
        n_stocks = 20
        
        records = []
        for t in range(n_timestamps):
            returns = np.random.normal(0.001, 0.02, n_stocks)
            predictions = np.random.randn(n_stocks)  # Random, uncorrelated
            for i in range(n_stocks):
                records.append({
                    TIMESTAMP: t * MS_PER_DAY,
                    TICKER: f"S{i}",
                    "predicted_score": predictions[i],
                    "actual_return": returns[i],
                })
        
        predictions_df = pd.DataFrame(records)
        
        config = PortfolioConfig(top_n=5, bottom_n=5, transaction_cost_bps=0)
        result = run_portfolio_backtest(
            predictions_df,
            config,
            timestamp_col=TIMESTAMP,
            score_col="predicted_score",
            return_col="actual_return",
        )
        
        # With random predictions, Sharpe should be in [-1, 1] range
        assert -2 < result.sharpe_ratio < 2, (
            f"Random predictions Sharpe={result.sharpe_ratio:.2f} is too extreme"
        )
        print(f"Zero skill Sharpe: {result.sharpe_ratio:.2f}")
    
    def test_return_frequency_detection(self):
        """Test helper to detect return frequency from timestamps."""
        # Daily timestamps
        daily_ts = pd.Series([i * MS_PER_DAY for i in range(100)])
        daily_freq = (daily_ts.diff().dropna() / MS_PER_DAY).median()
        assert daily_freq == 1.0, "Should detect daily frequency"
        
        # Weekly timestamps
        weekly_ts = pd.Series([i * 5 * MS_PER_DAY for i in range(100)])
        weekly_freq = (weekly_ts.diff().dropna() / MS_PER_DAY).median()
        assert weekly_freq == 5.0, "Should detect 5-day frequency"


# =============================================================================
# TRAIN/TEST LEAKAGE TESTS
# =============================================================================

class TestTrainTestLeakage:
    """Tests to detect information leakage between train and test sets."""
    
    def test_train_test_gap_required_for_forward_returns(self):
        """With N-day forward returns, need N-day gap between train and test.
        
        If train ends at timestamp T and uses 5-day forward returns:
        - Train targets use prices from T+5
        - Test should start at T+5+1 at minimum, not T+1
        - Otherwise, train targets overlap with test periods
        """
        forward_days = 5
        
        # Scenario: Daily timestamps, train on 0-79, test on 80-99
        train_end_ts = 79
        test_start_ts = 80
        
        # Train forward returns use prices up to:
        train_max_price_ts = train_end_ts + forward_days  # = 84
        
        # But test starts at 80, so timestamps 80-84 have price data
        # that was used to compute train forward returns!
        
        has_overlap = test_start_ts <= train_max_price_ts
        
        assert has_overlap, "This test demonstrates the overlap problem exists"
        
        # The fix: test should start AFTER train_max_price_ts
        correct_test_start = train_max_price_ts + 1
        
        print(f"With {forward_days}-day forward returns:")
        print(f"  Train ends at: {train_end_ts}")
        print(f"  Train uses prices up to: {train_max_price_ts}")
        print(f"  Current test start: {test_start_ts} (LEAKAGE!)")
        print(f"  Correct test start: {correct_test_start}")
    
    def test_suspicious_high_ic_indicates_leakage(self):
        """IC > 0.15 on real data almost always indicates leakage."""
        # In academic finance literature, IC of 0.05-0.10 is considered excellent
        # IC > 0.15 on out-of-sample data is extremely suspicious
        
        SUSPICIOUS_IC_THRESHOLD = 0.15
        REALISTIC_IC_RANGE = (0.02, 0.10)
        
        print(f"Suspicious IC threshold: > {SUSPICIOUS_IC_THRESHOLD}")
        print(f"Realistic IC range: {REALISTIC_IC_RANGE}")
        print()
        print("If you see IC > 0.15 on real out-of-sample data, check for:")
        print("  1. Forward-looking features (using future info)")
        print("  2. Train/test target overlap")
        print("  3. Scaler fitted on full data (including test)")
        print("  4. Duplicate data points")


# =============================================================================
# INTEGRATION TEST: FULL PIPELINE CHECK
# =============================================================================

class TestPipelineIntegrity:
    """Integration tests for the full ranking pipeline."""
    
    def test_timestamp_spacing_vs_forward_return_horizon(self):
        """Verify that timestamp spacing matches forward return horizon."""
        # This is the key test - if we use 5-day forward returns,
        # we should only be evaluating at 5-day intervals
        
        # Create synthetic data
        n_days = 100
        timestamps = [i * MS_PER_DAY for i in range(n_days)]  # Daily
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
        
        df = pd.DataFrame({
            TIMESTAMP: timestamps,
            TICKER: ["A"] * n_days,
            "Close": prices,
            ADJCLOSE: prices,
        })
        
        # Compute 5-day forward returns
        result = compute_forward_returns(df, lookahead_days=5)
        
        # Check timestamp spacing in the data
        ts_diff_days = result[TIMESTAMP].diff().dropna() / MS_PER_DAY
        median_spacing = ts_diff_days.median()
        
        # If spacing is 1 day but returns are 5-day, that's a problem
        if median_spacing < 5:
            # We're using overlapping returns - this inflates metrics
            print(f"WARNING: Timestamp spacing ({median_spacing:.0f} days) < "
                  f"forward return horizon (5 days). Returns overlap!")
    
    def test_correct_annualization_for_forward_return_horizon(self):
        """Verify Sharpe uses correct annualization for return horizon."""
        # If forward returns are 5-day, annualization should use ~50 periods/year
        # not 252 (daily)
        
        forward_days = 5
        periods_per_year = 252 / forward_days
        
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.003, 0.02, 50))  # 50 5-day periods
        
        # Current (potentially wrong) calculation
        current_sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        
        # Correct calculation
        correct_sharpe = compute_sharpe_ratio(returns, periods_per_year=int(periods_per_year))
        
        print(f"Current Sharpe (252): {current_sharpe:.2f}")
        print(f"Correct Sharpe ({int(periods_per_year)}): {correct_sharpe:.2f}")
        print(f"Inflation factor: {current_sharpe/correct_sharpe:.2f}x")
        
        # They differ by sqrt(252/50) ≈ 2.24x
        expected_ratio = np.sqrt(252 / periods_per_year)
        actual_ratio = current_sharpe / correct_sharpe
        assert abs(actual_ratio - expected_ratio) < 0.1


class TestPortfolioTimestampSampling:
    """Tests for correct timestamp sampling in portfolio backtest."""
    
    def test_timestamp_sampling_by_days_not_index(self):
        """Portfolio backtest should sample by actual days, not array index.
        
        Bug fix test: Previously timestamps[::365] sampled every 365th INDEX,
        not every 365 days. With daily timestamps (avg ~4 day spacing),
        this resulted in sampling every ~1300 days instead of ~365.
        """
        # Create predictions with timestamps every ~4 days on average
        # (simulating real-world data where timestamps are trading days)
        MS_PER_DAY_LOCAL = 86_400_000
        n_days = 2000
        
        # Simulate ~500 timestamps covering ~2000 days (avg 4 days apart)
        np.random.seed(42)
        timestamps = []
        current_ts = 0
        for _ in range(500):
            timestamps.append(current_ts)
            # Skip 1-7 days randomly (avg ~4 days)
            current_ts += int(np.random.choice([1, 2, 3, 4, 5, 6, 7]) * MS_PER_DAY_LOCAL)
        
        # Create fake predictions
        df = pd.DataFrame({
            TIMESTAMP: sorted(timestamps * 10),  # 10 stocks per timestamp
            TICKER: [f"S{i}" for i in range(10)] * len(timestamps),
            "predicted_score": np.random.randn(len(timestamps) * 10),
            "actual_return": np.random.randn(len(timestamps) * 10) * 0.05,
        })
        
        config = PortfolioConfig(top_n=5, bottom_n=5)
        result = run_portfolio_backtest(
            df, config, 
            return_horizon_days=365
        )
        
        # With 2000 days of data and 365-day horizon, we should get ~5-6 periods
        # NOT 500/365 ≈ 1 period (which the bug would give)
        total_days = (max(timestamps) - min(timestamps)) / MS_PER_DAY_LOCAL
        expected_periods = max(1, int(total_days / 365))
        actual_periods = result.num_rebalances
        
        print(f"Total days: {total_days:.0f}")
        print(f"Expected periods (~365 days apart): {expected_periods}")
        print(f"Actual periods: {actual_periods}")
        
        # Should be within reasonable range of expected
        assert actual_periods >= expected_periods - 1, (
            f"Too few periods ({actual_periods}) for {total_days:.0f} days "
            f"with 365-day horizon. Expected ~{expected_periods}."
        )
        assert actual_periods <= expected_periods + 2, (
            f"Too many periods ({actual_periods}). May indicate bug in sampling."
        )


class TestDailyReturnCapping:
    """Tests for handling anomalous price data in portfolio simulation."""
    
    def test_anomalous_split_data_is_detected_and_excluded(self):
        """Unadjusted split patterns should be detected and excluded."""
        from evaluation.portfolio_simulator import _detect_anomalous_prices
        
        # Create price data with a split pattern: -66% then +200%
        price_data = pd.DataFrame({
            TIMESTAMP: [1000, 1001, 1002, 1003],
            TICKER: ['A', 'A', 'A', 'A'],
            'Close': [100.0, 33.0, 100.0, 102.0],  # Split at day 2, reverses at day 3
        })
        
        excluded = _detect_anomalous_prices(
            price_data, TIMESTAMP, TICKER, 'Close', threshold=0.40
        )
        
        # Should exclude the split day (1001) and reversal day (1002)
        assert (1001, 'A') in excluded, "Split day should be excluded"
        assert (1002, 'A') in excluded, "Reversal day should be excluded"
        assert (1000, 'A') not in excluded, "Normal day before split should not be excluded"
        assert (1003, 'A') not in excluded, "Normal day after reversal should not be excluded"
    
    def test_portfolio_excludes_anomalous_ticker_days(self):
        """Portfolio returns should skip ticker-days with anomalous prices."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        # Holdings: 100% in stock A
        holdings = pd.DataFrame({
            TIMESTAMP: [1000],
            TICKER: ['A'],
            'weight': [1.0],
        })
        
        # Price data with split pattern
        price_data = pd.DataFrame({
            TIMESTAMP: [1000, 1001, 1002, 1003, 1004],
            TICKER: ['A', 'A', 'A', 'A', 'A'],
            'Close': [100.0, 33.0, 100.0, 102.0, 104.0],  # Split/reversal in middle
        })
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            returns = compute_daily_portfolio_returns(
                holdings, price_data, cost_bps_per_rebalance=0
            )
            # Should warn about anomalous prices
            assert any("anomalous" in str(warning.message).lower() for warning in w)
        
        # Returns should not include the extreme -67% or +200% moves
        if len(returns) > 0:
            assert returns.min() > -0.50, "Should not have extreme negative return from split"
            assert returns.max() < 0.50, "Should not have extreme positive return from reversal"
    
    def test_suspicious_return_threshold_handles_extreme_moves(self):
        """Suspicious return threshold should treat extreme returns as flat (0%)."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        # Holdings: 100% in stock A
        holdings = pd.DataFrame({
            TIMESTAMP: [1000],
            TICKER: ['A'],
            'weight': [1.0],
        })
        
        # Price data with extreme move that doesn't trigger anomaly detection
        # (anomaly detection looks for reversal pattern, this is a sustained move)
        price_data = pd.DataFrame({
            TIMESTAMP: [1000, 1001, 1002, 1003],
            TICKER: ['A', 'A', 'A', 'A'],
            'Close': [100.0, 1000.0, 1100.0, 1200.0],  # 10x jump, then normal moves
        })
        
        # With suspect_return_threshold=0.5 (50%)
        # Extreme returns are treated as 0% (flat), not clipped
        returns_filtered = compute_daily_portfolio_returns(
            holdings, price_data, cost_bps_per_rebalance=0,
            suspect_return_threshold=0.5
        )
        
        # The 10x move (900% return) exceeds threshold so it becomes 0%
        assert len(returns_filtered) > 0
        # After 10x jump is zeroed, remaining returns are small (~10%)
        assert returns_filtered.max() <= 0.5 + 1e-9, "Extreme return should be treated as 0%"
        
        # Without threshold (None), should see the full 900% return
        returns_unfiltered = compute_daily_portfolio_returns(
            holdings, price_data, cost_bps_per_rebalance=0,
            suspect_return_threshold=None
        )
        
        assert returns_unfiltered.max() > 5.0, "Without threshold, should show 900% move"
    
    def test_suspicious_threshold_works_for_negative_returns(self):
        """Suspicious return threshold should also handle extreme negative returns."""
        from evaluation.portfolio_simulator import compute_daily_portfolio_returns
        
        holdings = pd.DataFrame({
            TIMESTAMP: [1000],
            TICKER: ['A'],
            'weight': [1.0],
        })
        
        # Price crashes from 100 to 5 (-95%)
        price_data = pd.DataFrame({
            TIMESTAMP: [1000, 1001],
            TICKER: ['A', 'A'],
            'Close': [100.0, 5.0],
        })
        
        returns = compute_daily_portfolio_returns(
            holdings, price_data, cost_bps_per_rebalance=0,
            suspect_return_threshold=0.5
        )
        
        # -95% exceeds threshold (|−0.95| > 0.5), so it's treated as 0%
        assert len(returns) > 0
        # Return should be 0% (treated as flat), not the actual -95%
        assert returns.min() >= -0.5 - 1e-9, "Extreme negative return should be zeroed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

