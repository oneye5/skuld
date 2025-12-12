"""
End-to-end integration tests for the complete prediction pipeline.

Tests the full workflow from data to predictions, ensuring:
1. Data flows correctly through all stages
2. No data leakage occurs
3. Model is properly trained and evaluated
4. Output is in expected format
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
from pathlib import Path

from src.preprocessing.long_to_wide_csv import long_to_wide_and_impute
from src.preprocessing.pre_split_preprocessing import pre_split_preprocess
from src.preprocessing.train_test_split import split_last_occurring_tickers
from src.preprocessing.post_split_preprocessing import (
    post_split_preprocessing_train, post_split_preprocessing_test
)
from src.learner.learner import train_model, predict
from src.config.config import LABEL_COL, PREDICTION_COL


class TestEndToEndPipeline:
    """Test complete pipeline workflow."""

    def test_simplified_pipeline_train_predict(self):
        """Complete train and predict workflow should work end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create sample train data
            train_df = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'feature_3': np.random.randn(100),
                LABEL_COL: np.random.randint(0, 2, 100),
            })
            
            # Step 2: Create sample test data
            test_df = pd.DataFrame({
                'feature_1': np.random.randn(30),
                'feature_2': np.random.randn(30),
                'feature_3': np.random.randn(30),
            })
            
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            test_scaled_path = f"{tmpdir}/test_scaled.csv"
            model_path = f"{tmpdir}/model.pkl"
            pred_path = f"{tmpdir}/predictions.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Step 3: Scale training data
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            assert Path(scaler_path).exists()
            assert Path(train_scaled_path).exists()
            
            # Step 4: Scale test data with training scaler
            post_split_preprocessing_test(test_path, test_scaled_path, scaler_path)
            assert Path(test_scaled_path).exists()
            
            # Step 5: Train model
            train_model(train_scaled_path, model_path)
            assert Path(model_path).exists()
            
            # Verify model is loadable
            model = joblib.load(model_path)
            assert model is not None
            
            # Step 6: Make predictions
            predict(model_path, test_scaled_path, pred_path)
            assert Path(pred_path).exists()
            
            predictions = pd.read_csv(pred_path)
            assert len(predictions) == len(test_df)
            assert PREDICTION_COL in predictions.columns
            
            # Verify prediction values are probabilities
            probs = predictions[PREDICTION_COL]
            assert (probs >= 0).all() and (probs <= 1).all()

    def test_pipeline_no_data_leakage(self):
        """Pipeline should not leak test data into training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create train data with features at specific scale/center
            train_df = pd.DataFrame({
                'feature_1': np.linspace(0, 10, 100),  # Ranges 0-10
                'feature_2': np.linspace(0, 100, 100),  # Ranges 0-100
                LABEL_COL: np.random.randint(0, 2, 100),
            })
            
            # Create test data with different distribution
            test_df = pd.DataFrame({
                'feature_1': np.linspace(50, 100, 30),  # Ranges 50-100 (out-of-distribution)
                'feature_2': np.linspace(500, 1000, 30),  # Ranges 500-1000
            })
            
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            test_scaled_path = f"{tmpdir}/test_scaled.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Fit scaler on train
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            
            # Load scaler and verify it was fit on train statistics
            scaler = joblib.load(scaler_path)
            
            # RobustScaler uses median
            train_median = train_df[['feature_1', 'feature_2']].median().values
            
            # Scaler should use train statistics, not test
            assert np.allclose(scaler.center_, train_median, rtol=1e-5), \
                "Scaler should be fit on training data statistics only, not test"

    def test_scaler_isolation(self):
        """Scaler fit on train, applied to test (not vice versa)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple train/test data
            train_df = pd.DataFrame({
                'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'feature_2': [10.0, 20.0, 30.0, 40.0, 50.0],
                LABEL_COL: [0, 1, 0, 1, 0],
            })
            
            test_df = pd.DataFrame({
                'feature_1': [6.0, 7.0, 8.0],
                'feature_2': [60.0, 70.0, 80.0],
            })
            
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            test_scaled_path = f"{tmpdir}/test_scaled.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Fit scaler on train
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            
            # Load and verify scaler was fit on train stats
            scaler = joblib.load(scaler_path)
            train_numeric = train_df[['feature_1', 'feature_2']]
            train_median = train_numeric.median().values
            
            assert np.allclose(scaler.center_, train_median, rtol=1e-5), \
                "Scaler should be fit on training statistics"
            
            # Apply to test (should not refit scaler)
            post_split_preprocessing_test(test_path, test_scaled_path, scaler_path)
            
            # Verify scaler statistics didn't change
            scaler_after = joblib.load(scaler_path)
            assert np.allclose(scaler.center_, scaler_after.center_), \
                "Scaler should not be refitted during test preprocessing"

    def test_model_trains_on_train_data_only(self):
        """Model should be trained on training set, not test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple train/test split
            train_df = pd.DataFrame({
                'feature_1': np.random.randn(50),
                'feature_2': np.random.randn(50),
                LABEL_COL: np.random.randint(0, 2, 50),
            })
            
            test_df = pd.DataFrame({
                'feature_1': np.random.randn(20),
                'feature_2': np.random.randn(20),
                LABEL_COL: np.random.randint(0, 2, 20),
            })
            
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            model_path = f"{tmpdir}/model.pkl"
            pred_path = f"{tmpdir}/predictions.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Train model on train data
            train_model(train_path, model_path)
            
            # Model should exist and be trainable on train data only
            assert Path(model_path).exists()
            
            # Predictions on test should work
            test_features = test_df.drop(columns=[LABEL_COL])
            test_features.to_csv(test_path, index=False)
            
            predict(model_path, test_path, pred_path)
            assert Path(pred_path).exists()


class TestPipelineDataIntegrity:
    """Test data integrity through simplified pipeline."""

    def test_scaler_output_valid(self):
        """Scaler output should be valid (no NaN, no inf)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple test data
            train_data = pd.DataFrame({
                'feature_1': np.random.randn(50),
                'feature_2': np.random.randn(50),
                LABEL_COL: np.random.randint(0, 2, 50),
            })
            
            train_path = f"{tmpdir}/train.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            
            train_data.to_csv(train_path, index=False)
            
            # Scale data
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            
            # Check output validity
            scaled_data = pd.read_csv(train_scaled_path)
            
            numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                assert not scaled_data[col].isna().any(), \
                    f"Column {col} contains NaN values"
                assert not np.isinf(scaled_data[col]).any(), \
                    f"Column {col} contains infinite values"


class TestPipelineRobustness:
    """Test pipeline robustness to edge cases."""

    def test_small_dataset(self):
        """Pipeline should work with small datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_df = pd.DataFrame({
                'feature_1': [1.0, 2.0, 3.0],
                'feature_2': [4.0, 5.0, 6.0],
                LABEL_COL: [0, 1, 0],
            })
            
            test_df = pd.DataFrame({
                'feature_1': [1.5, 2.5],
                'feature_2': [4.5, 5.5],
            })
            
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            model_path = f"{tmpdir}/model.pkl"
            pred_path = f"{tmpdir}/predictions.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Should handle small data
            train_model(train_path, model_path)
            
            predict(model_path, test_path, pred_path)
            assert Path(pred_path).exists()

    def test_high_dimensional_features(self):
        """Pipeline should handle high-dimensional data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n_samples = 100
            n_features = 200
            
            train_df = pd.DataFrame({
                f'feature_{i}': np.random.randn(n_samples)
                for i in range(n_features)
            })
            train_df[LABEL_COL] = np.random.randint(0, 2, n_samples)
            
            train_path = f"{tmpdir}/train.csv"
            model_path = f"{tmpdir}/model.pkl"
            
            train_df.to_csv(train_path, index=False)
            
            # Should handle high dimensions
            train_model(train_path, model_path)
            assert Path(model_path).exists()


class TestOutputValidation:
    """Test that outputs are properly formatted."""

    def test_predictions_have_required_columns(self):
        """Predictions should have required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_df = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                LABEL_COL: np.random.randint(0, 2, 100),
            })
            
            test_df = pd.DataFrame({
                'feature_1': np.random.randn(20),
                'feature_2': np.random.randn(20),
            })
            
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            model_path = f"{tmpdir}/model.pkl"
            pred_path = f"{tmpdir}/predictions.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            train_model(train_path, model_path)
            predict(model_path, test_path, pred_path)
            
            predictions = pd.read_csv(pred_path)
            
            # Should have prediction column
            assert PREDICTION_COL in predictions.columns or \
                   any('prediction' in col.lower() for col in predictions.columns)

    def test_predictions_all_valid_numbers(self):
        """All prediction values should be valid numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_df = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                LABEL_COL: np.random.randint(0, 2, 100),
            })
            
            test_df = pd.DataFrame({
                'feature_1': np.random.randn(20),
                'feature_2': np.random.randn(20),
            })
            
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            model_path = f"{tmpdir}/model.pkl"
            pred_path = f"{tmpdir}/predictions.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            train_model(train_path, model_path)
            predict(model_path, test_path, pred_path)
            
            predictions = pd.read_csv(pred_path)
            
            # No NaN or inf in predictions
            numeric_cols = predictions.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                assert not predictions[col].isna().any(), \
                    f"Prediction column {col} contains NaN"
                assert not np.isinf(predictions[col]).any(), \
                    f"Prediction column {col} contains infinite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
