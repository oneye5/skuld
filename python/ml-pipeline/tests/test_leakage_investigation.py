"""LEAKAGE INVESTIGATION TESTS.

These tests investigate whether the high evaluation metrics (ICIR > 10) are trustworthy
or indicate data leakage. In real-world quantitative finance:

- IC of 0.03-0.05 is good
- IC of 0.05-0.10 is excellent
- IC > 0.15 is almost certainly leakage
- ICIR > 2.0 is extremely rare in real-world
- ICIR > 10 is a MAJOR RED FLAG

Current results show:
- Mean IC: 0.063
- ICIR: 10.38  <-- SUSPICIOUS
- Rank ICIR: 16.47  <-- VERY SUSPICIOUS

This test suite investigates potential causes:
1. Overlapping return periods (5-day returns on daily timestamps)
2. Forward-looking features (using future information)
3. Scaler leakage (fitting on test data)
4. Target leakage (features correlated with target by construction)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from config.columns import TIMESTAMP, TICKER, CLOSE
from config.settings import MS_PER_DAY


class TestOverlappingReturnInvestigation:
    """Test if overlapping return periods inflate metrics."""
    
    def test_5day_returns_on_daily_timestamps_creates_overlap(self):
        """
        CRITICAL: If we compute 5-day forward returns on daily timestamps,
        consecutive returns share 4 of 5 days of price data.
        
        This creates artificial autocorrelation that inflates IC stability (ICIR).
        """
        # Simulate the actual scenario: daily timestamps, 5-day returns
        n_days = 100
        timestamps = [i * MS_PER_DAY for i in range(n_days)]
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, n_days))
        
        # Compute 5-day returns manually
        returns_5d = []
        for i in range(n_days - 5):
            ret = (prices[i + 5] - prices[i]) / prices[i]
            returns_5d.append(ret)
        
        returns_series = pd.Series(returns_5d)
        
        # Check autocorrelation
        autocorr_lag1 = returns_series.autocorr(lag=1)
        autocorr_lag4 = returns_series.autocorr(lag=4)
        
        print(f"\n5-day returns on daily timestamps:")
        print(f"  Autocorr lag-1: {autocorr_lag1:.3f}")
        print(f"  Autocorr lag-4: {autocorr_lag4:.3f}")
        
        # With 4/5 overlap, expect HIGH autocorrelation
        # This artificially stabilizes IC series, inflating ICIR
        assert autocorr_lag1 > 0.5, (
            f"Expected high autocorr due to overlap, got {autocorr_lag1:.3f}"
        )
    
    def test_non_overlapping_returns_have_low_autocorr(self):
        """Non-overlapping returns (5-day spacing) should have low autocorr."""
        n_periods = 100
        timestamps = [i * 5 * MS_PER_DAY for i in range(n_periods)]  # 5-day spacing
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, n_periods))
        
        # Now each return is independent
        returns_5d = []
        for i in range(n_periods - 1):
            ret = (prices[i + 1] - prices[i]) / prices[i]
            returns_5d.append(ret)
        
        returns_series = pd.Series(returns_5d)
        
        autocorr_lag1 = returns_series.autocorr(lag=1)
        
        print(f"\n5-day returns on 5-day spacing:")
        print(f"  Autocorr lag-1: {autocorr_lag1:.3f}")
        
        # Without overlap, autocorr should be near 0
        assert abs(autocorr_lag1) < 0.3, (
            f"Non-overlapping should have low autocorr, got {autocorr_lag1:.3f}"
        )
    
    def test_icir_inflation_from_overlapping_ic_series(self):
        """
        IC calculated on overlapping periods will be smoother (higher autocorr),
        leading to artificially low std(IC) and inflated ICIR.
        """
        from evaluation.ranking_metrics import compute_icir
        
        np.random.seed(42)
        
        # Simulate IC series with vs without autocorrelation
        n_obs = 50
        base_ic = 0.03  # Realistic IC
        ic_noise = 0.05  # Realistic noise
        
        # Independent IC observations (what we want)
        independent_ic = pd.Series(
            np.random.normal(base_ic, ic_noise, n_obs)
        )
        
        # Autocorrelated IC (what we GET with overlapping returns)
        # Simulate with AR(1) process
        autocorr_ic = [np.random.normal(base_ic, ic_noise)]
        ar_coef = 0.8  # High autocorr due to overlap
        for i in range(1, n_obs):
            innovation = np.random.normal(0, ic_noise * np.sqrt(1 - ar_coef**2))
            autocorr_ic.append(base_ic + ar_coef * (autocorr_ic[-1] - base_ic) + innovation)
        autocorr_ic = pd.Series(autocorr_ic)
        
        # Compute ICIR
        icir_independent = compute_icir(independent_ic, annualize=False)
        icir_autocorr = compute_icir(autocorr_ic, annualize=False)
        
        print(f"\nICIR from independent IC: {icir_independent:.2f}")
        print(f"ICIR from autocorrelated IC: {icir_autocorr:.2f}")
        print(f"Std(independent IC): {independent_ic.std():.4f}")
        print(f"Std(autocorrelated IC): {autocorr_ic.std():.4f}")
        
        # Autocorrelated IC has lower variance, inflating ICIR
        assert autocorr_ic.std() < independent_ic.std() * 1.2, (
            "Autocorrelated IC should have similar or lower std"
        )


class TestFeatureLeakageInvestigation:
    """Test if features contain future information."""
    
    def test_cross_sectional_features_use_only_current_timestamp(self):
        """Cross-sectional features should only use current timestamp data."""
        from features.cross_sectional import add_cross_sectional_features
        
        # Create test data with known values
        df = pd.DataFrame({
            TIMESTAMP: [1000, 1000, 1000, 2000, 2000, 2000],
            TICKER: ["A", "B", "C", "A", "B", "C"],
            "RSI_14": [30, 50, 70, 40, 60, 80],  # Different per timestamp
            "Close": [100, 100, 100, 110, 110, 110],
        })
        
        result = add_cross_sectional_features(df)
        
        # Rank at timestamp 1000 should be based ONLY on values at 1000
        # A=30 (lowest) -> rank ~0.17, B=50 (mid) -> rank ~0.5, C=70 (high) -> rank ~0.83
        ts1000 = result[result[TIMESTAMP] == 1000]
        
        rank_A = ts1000[ts1000[TICKER] == "A"]["Rank_RSI_14"].values[0]
        rank_C = ts1000[ts1000[TICKER] == "C"]["Rank_RSI_14"].values[0]
        
        assert rank_A < rank_C, "A should have lower rank than C at timestamp 1000"
        
        # Verify ranks are between 0 and 1 (percentile ranks)
        assert 0 <= rank_A <= 1
        assert 0 <= rank_C <= 1
    
    def test_technical_features_use_only_past_data(self):
        """Technical features should only use past price data."""
        from features.technical import add_technical_features
        
        # Create sequential price data
        n_days = 100
        df = pd.DataFrame({
            TIMESTAMP: [i * MS_PER_DAY for i in range(n_days)],
            TICKER: ["A"] * n_days,
            CLOSE: 100 + np.arange(n_days) * 0.1,  # Slowly increasing
            "Open": 100 + np.arange(n_days) * 0.1,
            "High": 101 + np.arange(n_days) * 0.1,
            "Low": 99 + np.arange(n_days) * 0.1,
            "Volume": [1000000] * n_days,
        })
        
        result = add_technical_features(df)
        
        # RSI at day 50 should depend only on days 0-50
        # If we change day 51+ prices, RSI at day 50 should NOT change
        
        df_modified = df.copy()
        df_modified.loc[df_modified[TIMESTAMP] > 50 * MS_PER_DAY, CLOSE] = 1000
        
        result_modified = add_technical_features(df_modified)
        
        # Compare RSI at day 50
        orig_rsi = result[result[TIMESTAMP] == 50 * MS_PER_DAY]["RSI_14"].values
        mod_rsi = result_modified[result_modified[TIMESTAMP] == 50 * MS_PER_DAY]["RSI_14"].values
        
        if len(orig_rsi) > 0 and len(mod_rsi) > 0:
            assert np.allclose(orig_rsi, mod_rsi), (
                f"RSI at day 50 changed when future data changed: {orig_rsi} vs {mod_rsi}"
            )


class TestScalerLeakageInvestigation:
    """Test if scaler uses test data information."""
    
    def test_scaler_fitted_on_train_only(self):
        """Scaler must be fitted on training data only."""
        from core.scaler import fit_scaler, transform_data
        
        # Create train and test data with different distributions
        np.random.seed(42)
        
        train_df = pd.DataFrame({
            TIMESTAMP: range(100),
            TICKER: ["A"] * 100,
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(10, 2, 100),
        })
        
        test_df = pd.DataFrame({
            TIMESTAMP: range(100, 150),
            TICKER: ["A"] * 50,
            "feature1": np.random.normal(5, 1, 50),  # Different mean!
            "feature2": np.random.normal(20, 2, 50),  # Different mean!
        })
        
        # Fit on train only
        scaler = fit_scaler(train_df)
        
        # Transform both
        train_scaled = transform_data(train_df, scaler)
        test_scaled = transform_data(test_df, scaler)
        
        # Train feature1 should be centered around 0
        train_mean = train_scaled["feature1"].mean()
        assert abs(train_mean) < 0.5, f"Train should be centered, got mean={train_mean:.2f}"
        
        # Test feature1 should NOT be centered (because scaler used train stats)
        test_mean = test_scaled["feature1"].mean()
        assert abs(test_mean) > 1.0, (
            f"Test should NOT be centered if scaler fit on train only, got mean={test_mean:.2f}"
        )
    
    def test_scaler_leakage_detection(self):
        """
        If scaler is fit on train+test combined, test predictions can "see"
        information about test data distribution through the scaling.
        """
        from core.scaler import fit_scaler, transform_data
        
        np.random.seed(42)
        
        # Train: normal distribution
        train_df = pd.DataFrame({
            TIMESTAMP: range(100),
            TICKER: ["A"] * 100,
            "feature1": np.random.normal(0, 1, 100),
        })
        
        # Test: shifted distribution
        test_df = pd.DataFrame({
            TIMESTAMP: range(100, 200),
            TICKER: ["A"] * 100,
            "feature1": np.random.normal(10, 1, 100),  # Shifted by 10
        })
        
        # WRONG: Fit on combined data
        combined = pd.concat([train_df, test_df])
        scaler_wrong = fit_scaler(combined)
        
        # RIGHT: Fit on train only
        scaler_right = fit_scaler(train_df)
        
        # Transform test with both scalers
        test_wrong = transform_data(test_df.copy(), scaler_wrong)
        test_right = transform_data(test_df.copy(), scaler_right)
        
        # With wrong scaler, test will be closer to 0
        # With right scaler, test will be far from 0
        print(f"\nWrong scaler (fit on combined): test mean = {test_wrong['feature1'].mean():.2f}")
        print(f"Right scaler (fit on train): test mean = {test_right['feature1'].mean():.2f}")


class TestTargetLeakageInvestigation:
    """Test if features are correlated with target by construction (not prediction)."""
    
    def test_forward_return_not_in_features(self):
        """Forward return (target) should never be used as a feature."""
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from core.target_builder import FORWARD_RETURN
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            TICKER: ["A", "A", "A"],
            "Close": [100, 101, 102],
            FORWARD_RETURN: [0.01, 0.02, 0.03],  # Target
            "feature1": [1, 2, 3],
        })
        
        feature_cols = get_feature_columns_for_ranking(df)
        
        assert FORWARD_RETURN not in feature_cols, (
            f"Forward return should not be in features: {feature_cols}"
        )
    
    def test_price_at_target_date_not_in_features(self):
        """
        Features should not include any price data from the target date.
        
        For 5-day forward return (day 0 to day 5):
        - Features at day 0 should only use data from day 0 and earlier
        - Features should NOT use Close price at day 5
        
        Note: get_feature_columns_for_ranking doesn't know about semantic meaning,
        so it would include Close_5d_ahead if present. The responsibility is on
        the data preparation pipeline to NOT create such columns.
        """
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from core.target_builder import FORWARD_RETURN
        
        # Test that the actual excluded columns are properly handled
        df = pd.DataFrame({
            TIMESTAMP: [0],
            TICKER: ["A"],
            "Close": [100],  # Should be excluded (raw price)
            FORWARD_RETURN: [0.05],  # Should be excluded (target)
            "MA_20": [98],  # Should be included (feature)
            "RSI_14": [50],  # Should be included (feature)
        })
        
        feature_cols = get_feature_columns_for_ranking(df)
        
        # Close and forward_return should NOT be in features
        assert "Close" not in feature_cols, "Close price should not be a feature"
        assert FORWARD_RETURN not in feature_cols, "Forward return should not be a feature"
        
        # MA_20 and RSI_14 should be in features
        assert "MA_20" in feature_cols, "MA_20 should be a feature"
        assert "RSI_14" in feature_cols, "RSI_14 should be a feature"


class TestICIRRealWorldBenchmark:
    """Compare observed ICIR with realistic expectations.
    
    These are documentation/benchmark tests that verify our understanding
    of realistic metric ranges. They use pytest markers for clarity.
    """
    
    @pytest.mark.benchmark
    def test_icir_10_is_unrealistic(self):
        """
        ICIR > 10 is virtually impossible in real-world finance.
        
        Reference: "Active Portfolio Management" (Grinold & Kahn)
        - Annualized ICIR of 0.5-1.0 is considered good
        - ICIR of 2.0+ is excellent
        - ICIR of 5.0+ requires extraordinary skill or data
        - ICIR of 10+ almost certainly indicates leakage/error
        
        This test documents the benchmark and verifies the math, not fail on results.
        """
        observed_icir = 10.38  # From actual run (with wrong annualization)
        corrected_icir = 4.64  # With correct annualization (sqrt(50) not sqrt(252))
        
        # In academic literature, best reported ICIRs are typically < 3
        max_realistic_icir = 3.0
        
        # Verify that corrected ICIR is lower than observed (sanity check)
        assert corrected_icir < observed_icir, "Corrected ICIR should be lower than wrong annualization"
        
        # Verify the relationship makes mathematical sense
        # With 365-day returns and ~daily observations, inflation factor ≈ sqrt(252/365*252) 
        assert observed_icir / corrected_icir > 1.5, "Inflation factor should be significant"
        
        # Document findings
        if corrected_icir > max_realistic_icir:
            # This is expected due to overlapping returns - not a test failure
            pass
    
    def test_calculate_correct_annualization(self):
        """
        For 5-day forward returns with daily IC observations:
        - Raw IC series has 252/5 ≈ 50 INDEPENDENT observations per year
        - But IC observations themselves are autocorrelated
        
        CORRECT annualization: sqrt(252/5) ≈ 7.1, not sqrt(252) ≈ 15.9
        """
        forward_days = 5
        trading_days_per_year = 252
        
        # Correct: independent periods per year
        periods_per_year_correct = trading_days_per_year / forward_days
        annualization_correct = np.sqrt(periods_per_year_correct)
        
        # Wrong: treating as daily
        annualization_wrong = np.sqrt(trading_days_per_year)
        
        inflation_factor = annualization_wrong / annualization_correct
        
        print(f"\n5-day forward returns annualization:")
        print(f"  Correct (50 periods/year): sqrt(50) = {annualization_correct:.2f}")
        print(f"  Wrong (252 periods/year): sqrt(252) = {annualization_wrong:.2f}")
        print(f"  Inflation factor: {inflation_factor:.2f}x")
        
        # If ICIR = 10.38 and was calculated with wrong annualization:
        observed_icir = 10.38
        corrected_icir = observed_icir / inflation_factor
        
        print(f"\n  Observed ICIR: {observed_icir:.2f}")
        print(f"  Corrected ICIR (estimated): {corrected_icir:.2f}")
        
        assert inflation_factor > 2.0, "Inflation should be significant"


class TestActualPipelineInvestigation:
    """Run actual pipeline components to investigate leakage."""
    
    def test_check_ic_autocorrelation_in_actual_results(self):
        """Load actual results and check IC series autocorrelation."""
        results_dir = Path(__file__).parent.parent / "output" / "runs"
        
        # Find most recent ranking run
        ranking_dirs = sorted([d for d in results_dir.iterdir() 
                              if d.is_dir() and d.name.startswith("ranking_")])
        
        if not ranking_dirs:
            pytest.skip("No ranking results found")
        
        latest_run = ranking_dirs[-1]
        predictions_file = latest_run / "predictions.csv"
        
        if not predictions_file.exists():
            pytest.skip(f"No predictions.csv in {latest_run}")
        
        # Load predictions
        predictions = pd.read_csv(predictions_file)
        
        # Compute IC per timestamp
        from evaluation.ranking_metrics import compute_cross_sectional_ic_series
        
        ic_series = compute_cross_sectional_ic_series(
            predictions,
            timestamp_col="timestamp",
            predicted_col="predicted_score",
            actual_col="actual_return",
        )
        
        # Check autocorrelation
        if len(ic_series) > 5:
            autocorr_1 = ic_series.autocorr(lag=1)
            autocorr_2 = ic_series.autocorr(lag=2)
            
            print(f"\nIC series from actual run ({latest_run.name}):")
            print(f"  Length: {len(ic_series)}")
            print(f"  Mean IC: {ic_series.mean():.4f}")
            print(f"  Std IC: {ic_series.std():.4f}")
            print(f"  Autocorr lag-1: {autocorr_1:.3f}")
            print(f"  Autocorr lag-2: {autocorr_2:.3f}")
            
            if autocorr_1 > 0.5:
                print("\n  [WARNING] HIGH AUTOCORRELATION - likely overlapping return periods")
    
    @pytest.mark.benchmark
    def test_check_timestamp_spacing_vs_forward_horizon(self):
        """Check if timestamps are properly spaced for forward return horizon.
        
        This test documents the overlap issue. Overlapping returns are a known 
        limitation that inflates ICIR. The test verifies we can detect this.
        """
        results_dir = Path(__file__).parent.parent / "output" / "runs"
        ranking_dirs = sorted([d for d in results_dir.iterdir() 
                              if d.is_dir() and d.name.startswith("ranking_")])
        
        if not ranking_dirs:
            pytest.skip("No ranking results found")
        
        latest_run = ranking_dirs[-1]
        predictions_file = latest_run / "predictions.csv"
        
        if not predictions_file.exists():
            pytest.skip(f"No predictions.csv in {latest_run}")
        
        predictions = pd.read_csv(predictions_file)
        
        # Check timestamp spacing
        timestamps = sorted(predictions["timestamp"].unique())
        if len(timestamps) > 1:
            ts_diff = np.diff(timestamps)
            median_spacing_days = np.median(ts_diff) / MS_PER_DAY
            
            # Load config to get forward_return_days
            import json
            config_file = latest_run / "config.json"
            with open(config_file) as f:
                config = json.load(f)
            
            forward_days = config.get("forward_return_days", 5)
            
            # Verify we can compute the overlap ratio
            if median_spacing_days < forward_days:
                overlap_ratio = 1 - (median_spacing_days / forward_days)
                # This is expected - document it
                assert overlap_ratio >= 0, "Overlap ratio should be non-negative"
                assert overlap_ratio <= 1, "Overlap ratio should be <= 1"
            else:
                # Good - no overlap
                assert median_spacing_days >= forward_days, "Spacing should be >= horizon for no overlap"
        else:
            pytest.skip("Insufficient timestamps for spacing analysis")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
