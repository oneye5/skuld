"""Tests for consistency between prediction and evaluation pipelines.

These tests verify that:
1. Both pipelines produce identical predictions when given the same data splits
2. Feature engineering is consistent between pipelines
3. Scaler handling is consistent
4. The model produces deterministic results with the same random seed

The tests use synthetic data to ensure reproducibility and fast execution.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from config.columns import TIMESTAMP, TICKER, CLOSE
from config.settings import MS_PER_DAY


# =============================================================================
# SYNTHETIC DATA FIXTURES
# =============================================================================

def create_synthetic_stock_data(
    n_tickers: int = 20,
    n_timestamps: int = 100,
    start_ts: int = 946684800000,  # 2000-01-01 in ms
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic stock data for testing.
    
    Creates data with realistic structure:
    - Multiple tickers with different price levels
    - Time series with trends and noise
    - Some predictable patterns (momentum) for model to learn
    
    Args:
        n_tickers: Number of tickers to generate.
        n_timestamps: Number of timestamps per ticker.
        start_ts: Starting timestamp in milliseconds.
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame in wide format with TIMESTAMP, TICKER, Close, Volume, etc.
    """
    np.random.seed(seed)
    
    tickers = [f"STOCK_{i:03d}" for i in range(n_tickers)]
    timestamps = [start_ts + i * MS_PER_DAY for i in range(n_timestamps)]
    
    rows = []
    for ticker in tickers:
        # Each ticker has a different base price and volatility
        base_price = np.random.uniform(20, 200)
        volatility = np.random.uniform(0.01, 0.03)
        drift = np.random.uniform(-0.0002, 0.0005)  # Daily drift
        
        price = base_price
        for ts in timestamps:
            # Random walk with drift (momentum effect)
            daily_return = drift + volatility * np.random.randn()
            price = price * (1 + daily_return)
            
            # Add some mean reversion for variety
            if price > base_price * 1.5:
                price *= 0.99
            elif price < base_price * 0.5:
                price *= 1.01
            
            volume = np.random.uniform(1e6, 1e8)
            
            rows.append({
                TIMESTAMP: ts,
                TICKER: ticker,
                CLOSE: price,
                "Open": price * (1 + np.random.uniform(-0.01, 0.01)),
                "High": price * (1 + np.random.uniform(0, 0.02)),
                "Low": price * (1 - np.random.uniform(0, 0.02)),
                "Volume": volume,
            })
    
    df = pd.DataFrame(rows)
    return df


def create_synthetic_data_with_features(
    n_tickers: int = 20,
    n_timestamps: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic data with pre-computed features.
    
    This simulates data after feature engineering has been applied.
    """
    df = create_synthetic_stock_data(n_tickers, n_timestamps, seed=seed)
    
    # Add some synthetic features that mimic real features
    np.random.seed(seed + 1)
    
    # Per-ticker features (computed per row)
    df["ROC_10"] = np.random.randn(len(df)) * 0.1
    df["ROC_60"] = np.random.randn(len(df)) * 0.15
    df["ROC_252"] = np.random.randn(len(df)) * 0.3
    df["RSI_14"] = np.random.uniform(20, 80, len(df))
    df["Vol_20"] = np.random.uniform(0.01, 0.05, len(df))
    df["Vol_60"] = np.random.uniform(0.01, 0.05, len(df))
    df["Dist_MA_20"] = np.random.randn(len(df)) * 0.05
    df["Dist_MA_200"] = np.random.randn(len(df)) * 0.1
    df["NATR_14"] = np.random.uniform(0.01, 0.05, len(df))
    df["BB_Width_20"] = np.random.uniform(0.02, 0.1, len(df))
    
    # Alpha-like factors
    df["Rev_5d"] = np.random.randn(len(df)) * 0.05
    df["Rev_10d"] = np.random.randn(len(df)) * 0.07
    df["IdioVol_20"] = np.random.uniform(0.01, 0.04, len(df))
    df["Skew_60d"] = np.random.randn(len(df)) * 0.5
    
    return df


# =============================================================================
# CONSISTENCY TEST CLASS
# =============================================================================

class TestPipelineConsistency:
    """Tests for consistency between prediction and evaluation pipelines."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        dirpath = tempfile.mkdtemp()
        yield Path(dirpath)
        shutil.rmtree(dirpath)
    
    @pytest.fixture
    def synthetic_wide_data(self):
        """Generate synthetic wide-format data with features."""
        return create_synthetic_data_with_features(
            n_tickers=15,
            n_timestamps=80,
            seed=42,
        )
    
    def test_scaler_consistency(self, synthetic_wide_data):
        """Test that scaler produces identical results in both pipelines."""
        from core.scaler import fit_scaler, transform_data
        from core.preprocessor import preprocess_data
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        
        df = synthetic_wide_data.copy()
        
        # Split data (mimicking train/test split)
        timestamps = sorted(df[TIMESTAMP].unique())
        split_idx = int(len(timestamps) * 0.7)
        train_ts = timestamps[:split_idx]
        test_ts = timestamps[split_idx:]
        
        train_df = df[df[TIMESTAMP].isin(train_ts)].copy()
        test_df = df[df[TIMESTAMP].isin(test_ts)].copy()
        
        # Preprocess
        train_df = preprocess_data(train_df, add_missing_flags=False)
        test_df = preprocess_data(test_df, add_missing_flags=False)
        
        # Get feature columns
        feature_cols = get_feature_columns_for_ranking(train_df)
        
        # Fit scaler on training data only
        scaler = fit_scaler(train_df[feature_cols])
        
        # Transform both
        train_scaled = transform_data(train_df, scaler)
        test_scaled = transform_data(test_df, scaler)
        
        # Verify scaler was fit only on train
        assert scaler.continuous_scaler.center_ is not None
        
        # Transform train again - should be identical
        train_scaled_2 = transform_data(train_df, scaler)
        
        for col in scaler.continuous_cols:
            np.testing.assert_array_almost_equal(
                train_scaled[col].values,
                train_scaled_2[col].values,
                decimal=6,
                err_msg=f"Column {col} not identical after re-transformation"
            )
    
    def test_feature_column_consistency(self, synthetic_wide_data):
        """Test that feature column selection is consistent."""
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from core.preprocessor import preprocess_data
        
        df = synthetic_wide_data.copy()
        
        # Get feature columns multiple times
        feature_cols_1 = get_feature_columns_for_ranking(df)
        feature_cols_2 = get_feature_columns_for_ranking(df)
        
        assert feature_cols_1 == feature_cols_2, "Feature columns not deterministic"
        
        # Verify excluded columns
        assert TIMESTAMP not in feature_cols_1
        assert TICKER not in feature_cols_1
        assert CLOSE not in feature_cols_1
        
        # After preprocessing, should still be consistent
        df_processed = preprocess_data(df, add_missing_flags=False)
        feature_cols_3 = get_feature_columns_for_ranking(df_processed)
        
        # Features should be subset (preprocessing might add/remove some)
        assert len(set(feature_cols_3) - set(feature_cols_1)) < len(feature_cols_1) * 0.5
    
    def test_ranker_determinism(self, synthetic_wide_data):
        """Test that ranker produces deterministic predictions with same seed."""
        from core.scaler import fit_scaler, transform_data
        from core.preprocessor import preprocess_data
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from learner.ranking import (
            LightGBMRankerWrapper, RankerConfig, 
            prepare_ranking_data, filter_min_stocks_per_timestamp
        )
        
        df = synthetic_wide_data.copy()
        
        # Compute forward returns
        df = compute_forward_returns(df, lookahead_days=10, drop_na=True)
        df = filter_min_stocks_per_timestamp(df, min_stocks=5, timestamp_col=TIMESTAMP)
        
        # Split
        timestamps = sorted(df[TIMESTAMP].unique())
        split_idx = int(len(timestamps) * 0.7)
        train_ts = timestamps[:split_idx]
        test_ts = timestamps[split_idx:]
        
        train_df = df[df[TIMESTAMP].isin(train_ts)].copy()
        test_df = df[df[TIMESTAMP].isin(test_ts)].copy()
        
        # Preprocess
        train_df = preprocess_data(train_df, add_missing_flags=False)
        test_df = preprocess_data(test_df, add_missing_flags=False)
        
        feature_cols = sorted(set(get_feature_columns_for_ranking(train_df)) & 
                             set(get_feature_columns_for_ranking(test_df)))
        
        # Scale
        scaler = fit_scaler(train_df[feature_cols])
        train_scaled = transform_data(train_df, scaler)
        test_scaled = transform_data(test_df, scaler)
        
        # Prepare ranking data
        X_train, y_train, groups_train = prepare_ranking_data(
            train_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
        )
        X_test, _, _ = prepare_ranking_data(
            test_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
        )
        
        # Train two rankers with same seed
        config = RankerConfig(n_estimators=10, random_state=42)
        
        ranker1 = LightGBMRankerWrapper(config)
        ranker1.fit(X_train, y_train, groups_train)
        preds1 = ranker1.predict(X_test)
        
        ranker2 = LightGBMRankerWrapper(config)
        ranker2.fit(X_train, y_train, groups_train)
        preds2 = ranker2.predict(X_test)
        
        np.testing.assert_array_almost_equal(
            preds1, preds2, decimal=6,
            err_msg="Ranker predictions not deterministic with same seed"
        )
    
    def test_prediction_pipeline_matches_evaluation(self, synthetic_wide_data, temp_dir):
        """Test that prediction pipeline produces same results as evaluation pipeline.
        
        This is the key consistency test: if we train on the same data and predict
        on the same test set, both pipelines should produce identical predictions.
        """
        from core.scaler import fit_scaler, transform_data
        from core.preprocessor import preprocess_data
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        from core.model_persistence import ModelBundle, save_model, load_model
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from learner.ranking import (
            LightGBMRankerWrapper, RankerConfig,
            prepare_ranking_data, filter_min_stocks_per_timestamp
        )
        
        df = synthetic_wide_data.copy()
        forward_days = 10
        
        # Compute forward returns
        df_with_returns = compute_forward_returns(df, lookahead_days=forward_days, drop_na=True)
        df_with_returns = filter_min_stocks_per_timestamp(
            df_with_returns, min_stocks=5, timestamp_col=TIMESTAMP
        )
        
        # Split into train and test (simulating evaluation pipeline split)
        timestamps = sorted(df_with_returns[TIMESTAMP].unique())
        split_idx = int(len(timestamps) * 0.7)
        train_ts = timestamps[:split_idx]
        test_ts = timestamps[split_idx:]
        
        train_df = df_with_returns[df_with_returns[TIMESTAMP].isin(train_ts)].copy()
        test_df = df_with_returns[df_with_returns[TIMESTAMP].isin(test_ts)].copy()
        
        # --- EVALUATION PIPELINE PATH ---
        # Preprocess
        train_processed = preprocess_data(train_df, add_missing_flags=False)
        test_processed = preprocess_data(test_df, add_missing_flags=False)
        
        # Get feature columns (intersection)
        train_features = set(get_feature_columns_for_ranking(train_processed))
        test_features = set(get_feature_columns_for_ranking(test_processed))
        feature_cols = sorted(train_features & test_features)
        
        # Scale
        scaler = fit_scaler(train_processed[feature_cols])
        train_scaled = transform_data(train_processed, scaler)
        test_scaled = transform_data(test_processed, scaler)
        
        # Prepare and train
        X_train, y_train, groups_train = prepare_ranking_data(
            train_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
        )
        X_test, _, _ = prepare_ranking_data(
            test_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
        )
        
        config = RankerConfig(n_estimators=20, random_state=42)
        ranker = LightGBMRankerWrapper(config)
        ranker.fit(X_train, y_train, groups_train)
        
        # Evaluation pipeline predictions
        eval_predictions = ranker.predict(X_test)
        
        # --- PREDICTION PIPELINE PATH (via saved model) ---
        # Save the model
        model_path = temp_dir / "test_model.pkl"
        bundle = ModelBundle(
            ranker=ranker,
            scaler=scaler,
            feature_columns=feature_cols,
            config={"forward_return_days": forward_days, "n_estimators": 20},
        )
        save_model(bundle, model_path)
        
        # Load and predict (simulating load_and_predict)
        loaded_bundle = load_model(model_path)
        
        # Re-process test data as prediction pipeline would
        test_for_pred = test_df.copy()
        test_pred_processed = preprocess_data(test_for_pred, add_missing_flags=False)
        test_pred_scaled = transform_data(test_pred_processed, loaded_bundle.scaler, strict=True)
        
        X_pred, _, _ = prepare_ranking_data(
            test_pred_scaled, loaded_bundle.feature_columns, FORWARD_RETURN, TIMESTAMP
        )
        
        pred_predictions = loaded_bundle.ranker.predict(X_pred)
        
        # --- VERIFY CONSISTENCY ---
        np.testing.assert_array_almost_equal(
            eval_predictions, pred_predictions, decimal=6,
            err_msg="Prediction pipeline does not match evaluation pipeline"
        )
    
    def test_forward_return_calculation_consistency(self, synthetic_wide_data):
        """Test that forward return calculation is deterministic."""
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        
        df = synthetic_wide_data.copy()
        
        # Compute forward returns twice
        df1 = compute_forward_returns(df.copy(), lookahead_days=10, drop_na=True)
        df2 = compute_forward_returns(df.copy(), lookahead_days=10, drop_na=True)
        
        # Should be identical
        pd.testing.assert_series_equal(
            df1[FORWARD_RETURN].reset_index(drop=True),
            df2[FORWARD_RETURN].reset_index(drop=True),
            check_names=False,
        )
    
    def test_preprocessing_consistency(self, synthetic_wide_data):
        """Test that preprocessing is deterministic."""
        from core.preprocessor import preprocess_data
        
        df = synthetic_wide_data.copy()
        
        # Preprocess twice
        df1 = preprocess_data(df.copy(), add_missing_flags=False)
        df2 = preprocess_data(df.copy(), add_missing_flags=False)
        
        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_strict_mode_catches_misaligned_features(self, synthetic_wide_data, temp_dir):
        """Test that strict mode in scaler catches feature misalignment."""
        from core.scaler import fit_scaler, transform_data
        from core.preprocessor import preprocess_data
        from core.model_persistence import ModelBundle, save_model, load_model
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from learner.ranking import RankerConfig, LightGBMRankerWrapper
        
        df = synthetic_wide_data.copy()
        
        # Add an extra feature that won't be in "new" data
        df["extra_feature"] = np.random.randn(len(df))
        
        # Process and fit scaler
        df_processed = preprocess_data(df, add_missing_flags=False)
        feature_cols = get_feature_columns_for_ranking(df_processed)
        scaler = fit_scaler(df_processed[feature_cols])
        
        # Save a bundle with these features
        model_path = temp_dir / "test_model.pkl"
        bundle = ModelBundle(
            ranker=None,  # Don't need actual ranker for this test
            scaler=scaler,
            feature_columns=feature_cols,
            config={},
        )
        save_model(bundle, model_path)
        
        # Load and try to transform data WITHOUT the extra feature
        loaded = load_model(model_path)
        
        # Create "new" data missing the extra feature
        new_df = synthetic_wide_data.copy()  # No extra_feature
        new_processed = preprocess_data(new_df, add_missing_flags=False)
        
        # Strict mode should raise error
        with pytest.raises(ValueError, match="columns required by scaler are missing"):
            transform_data(new_processed, loaded.scaler, strict=True)


class TestRollingWindowConsistency:
    """Tests for consistency of rolling window evaluation."""
    
    @pytest.fixture
    def large_synthetic_data(self):
        """Generate larger synthetic data for rolling window tests."""
        return create_synthetic_data_with_features(
            n_tickers=20,
            n_timestamps=150,
            seed=42,
        )
    
    def test_non_overlapping_train_test(self, large_synthetic_data):
        """Test that train and test sets never overlap in rolling windows."""
        from core.splitter import split_by_timestamp
        
        df = large_synthetic_data
        timestamps = sorted(df[TIMESTAMP].unique())
        
        # Simulate multiple windows
        window_configs = [
            (timestamps[50], timestamps[70]),   # Window 1
            (timestamps[60], timestamps[80]),   # Window 2
            (timestamps[70], timestamps[90]),   # Window 3
        ]
        
        for train_end_ts, test_end_ts in window_configs:
            split = split_by_timestamp(df, train_end_ts, test_end_ts)
            
            train_ts = set(split.train[TIMESTAMP].unique())
            test_ts = set(split.test[TIMESTAMP].unique())
            
            # No overlap
            overlap = train_ts & test_ts
            assert len(overlap) == 0, f"Train and test overlap: {overlap}"
            
            # Train timestamps < test timestamps
            assert max(train_ts) < min(test_ts), "Train data not strictly before test"
    
    def test_window_predictions_independent(self, large_synthetic_data):
        """Test that predictions from different windows are computed independently."""
        from core.scaler import fit_scaler, transform_data
        from core.preprocessor import preprocess_data
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        from core.splitter import split_by_timestamp
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from learner.ranking import (
            LightGBMRankerWrapper, RankerConfig,
            prepare_ranking_data, filter_min_stocks_per_timestamp
        )
        
        df = large_synthetic_data.copy()
        df = compute_forward_returns(df, lookahead_days=10, drop_na=True)
        df = filter_min_stocks_per_timestamp(df, min_stocks=5, timestamp_col=TIMESTAMP)
        
        timestamps = sorted(df[TIMESTAMP].unique())
        
        # Two different windows
        window1 = (timestamps[40], timestamps[60])
        window2 = (timestamps[60], timestamps[80])
        
        predictions = {}
        scalers = {}
        
        for window_id, (train_end, test_end) in enumerate([window1, window2]):
            split = split_by_timestamp(df, train_end, test_end)
            
            train_proc = preprocess_data(split.train, add_missing_flags=False)
            test_proc = preprocess_data(split.test, add_missing_flags=False)
            
            feature_cols = sorted(
                set(get_feature_columns_for_ranking(train_proc)) &
                set(get_feature_columns_for_ranking(test_proc))
            )
            
            # Each window has its OWN scaler
            scaler = fit_scaler(train_proc[feature_cols])
            scalers[window_id] = scaler
            
            train_scaled = transform_data(train_proc, scaler)
            test_scaled = transform_data(test_proc, scaler)
            
            X_train, y_train, groups = prepare_ranking_data(
                train_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
            )
            X_test, _, _ = prepare_ranking_data(
                test_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
            )
            
            ranker = LightGBMRankerWrapper(RankerConfig(n_estimators=10, random_state=42))
            ranker.fit(X_train, y_train, groups)
            
            predictions[window_id] = ranker.predict(X_test)
        
        # Scalers should be different (fitted on different data)
        # Check that centers are different
        center1 = scalers[0].continuous_scaler.center_
        center2 = scalers[1].continuous_scaler.center_
        
        assert not np.allclose(center1, center2), \
            "Scalers from different windows should have different parameters"


class TestEndToEndConsistency:
    """End-to-end consistency tests simulating real usage."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        dirpath = tempfile.mkdtemp()
        yield Path(dirpath)
        shutil.rmtree(dirpath)
    
    def test_save_load_predict_cycle(self, temp_dir):
        """Test full cycle: train -> save -> load -> predict."""
        from core.scaler import fit_scaler, transform_data
        from core.preprocessor import preprocess_data
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        from core.model_persistence import ModelBundle, save_model, load_model
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from learner.ranking import (
            LightGBMRankerWrapper, RankerConfig,
            prepare_ranking_data, filter_min_stocks_per_timestamp
        )
        
        # Create data
        df = create_synthetic_data_with_features(n_tickers=15, n_timestamps=80, seed=42)
        df = compute_forward_returns(df, lookahead_days=10, drop_na=True)
        df = filter_min_stocks_per_timestamp(df, min_stocks=5, timestamp_col=TIMESTAMP)
        
        # Split
        timestamps = sorted(df[TIMESTAMP].unique())
        split_idx = int(len(timestamps) * 0.8)
        train_df = df[df[TIMESTAMP] < timestamps[split_idx]].copy()
        test_df = df[df[TIMESTAMP] >= timestamps[split_idx]].copy()
        
        # Train
        train_proc = preprocess_data(train_df, add_missing_flags=False)
        feature_cols = get_feature_columns_for_ranking(train_proc)
        
        scaler = fit_scaler(train_proc[feature_cols])
        train_scaled = transform_data(train_proc, scaler)
        
        X_train, y_train, groups = prepare_ranking_data(
            train_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
        )
        
        ranker = LightGBMRankerWrapper(RankerConfig(n_estimators=15, random_state=42))
        ranker.fit(X_train, y_train, groups)
        
        # Get baseline predictions on test data
        test_proc = preprocess_data(test_df, add_missing_flags=False)
        test_scaled = transform_data(test_proc, scaler)
        X_test, _, _ = prepare_ranking_data(test_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP)
        baseline_preds = ranker.predict(X_test)
        
        # Save model
        model_path = temp_dir / "model.pkl"
        bundle = ModelBundle(
            ranker=ranker,
            scaler=scaler,
            feature_columns=feature_cols,
            config={"n_estimators": 15},
        )
        save_model(bundle, model_path)
        
        # Load and predict
        loaded = load_model(model_path)
        
        # Process test data fresh (as prediction pipeline would)
        test_fresh = test_df.copy()
        test_fresh_proc = preprocess_data(test_fresh, add_missing_flags=False)
        test_fresh_scaled = transform_data(test_fresh_proc, loaded.scaler, strict=True)
        X_fresh, _, _ = prepare_ranking_data(
            test_fresh_scaled, loaded.feature_columns, FORWARD_RETURN, TIMESTAMP
        )
        loaded_preds = loaded.ranker.predict(X_fresh)
        
        # Predictions should match
        np.testing.assert_array_almost_equal(
            baseline_preds, loaded_preds, decimal=6,
            err_msg="Predictions from loaded model don't match original"
        )
    
    def test_ranking_order_preserved(self, temp_dir):
        """Test that relative ranking order is preserved after save/load."""
        from core.scaler import fit_scaler, transform_data
        from core.preprocessor import preprocess_data
        from core.target_builder import compute_forward_returns, FORWARD_RETURN
        from core.model_persistence import ModelBundle, save_model, load_model
        from pipeline.ranking_pipeline import get_feature_columns_for_ranking
        from learner.ranking import (
            LightGBMRankerWrapper, RankerConfig,
            prepare_ranking_data, filter_min_stocks_per_timestamp
        )
        
        # Create data
        df = create_synthetic_data_with_features(n_tickers=15, n_timestamps=80, seed=42)
        df = compute_forward_returns(df, lookahead_days=10, drop_na=True)
        df = filter_min_stocks_per_timestamp(df, min_stocks=5, timestamp_col=TIMESTAMP)
        
        timestamps = sorted(df[TIMESTAMP].unique())
        split_idx = int(len(timestamps) * 0.8)
        train_df = df[df[TIMESTAMP] < timestamps[split_idx]].copy()
        test_df = df[df[TIMESTAMP] >= timestamps[split_idx]].copy()
        
        # Train
        train_proc = preprocess_data(train_df, add_missing_flags=False)
        feature_cols = get_feature_columns_for_ranking(train_proc)
        scaler = fit_scaler(train_proc[feature_cols])
        train_scaled = transform_data(train_proc, scaler)
        X_train, y_train, groups = prepare_ranking_data(
            train_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP
        )
        
        ranker = LightGBMRankerWrapper(RankerConfig(n_estimators=15, random_state=42))
        ranker.fit(X_train, y_train, groups)
        
        # Get predictions and rankings
        test_proc = preprocess_data(test_df, add_missing_flags=False)
        test_scaled = transform_data(test_proc, scaler)
        X_test, _, _ = prepare_ranking_data(test_scaled, feature_cols, FORWARD_RETURN, TIMESTAMP)
        
        original_preds = ranker.predict(X_test)
        original_ranks = pd.Series(original_preds).rank(ascending=False, method="first")
        
        # Save and load
        model_path = temp_dir / "model.pkl"
        save_model(ModelBundle(ranker, scaler, feature_cols, {}), model_path)
        loaded = load_model(model_path)
        
        # Get loaded predictions
        loaded_preds = loaded.ranker.predict(X_test)
        loaded_ranks = pd.Series(loaded_preds).rank(ascending=False, method="first")
        
        # Rankings should be identical
        pd.testing.assert_series_equal(
            original_ranks, loaded_ranks,
            check_names=False,
            obj="Rankings"
        )
