"""
Tests to detect data leakage in the ML pipeline.

Data leakage occurs when:
1. Test data statistics influence training (e.g., scaler fit on test data)
2. Information from future time periods enters training
3. Target variable leaks into features
4. Model is evaluated on data it was trained on
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
from pathlib import Path
from sklearn.preprocessing import RobustScaler

from src.preprocessing.post_split_preprocessing import (
    post_split_preprocessing_train, post_split_preprocessing_test
)
from src.preprocessing.feature_engineering import scale_data_with_scaler
from src.learner.learner import train_model, predict
from src.config.config import LABEL_COL


class TestScalerFitting:
    """Test that scalers are properly fit on training data only."""

    def test_scaler_fit_only_on_train_not_test(self):
        """Scaler should be fit on train data and applied to test."""
        # Create training data with specific statistics
        train_data = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'timestamp': range(5),
            LABEL_COL: [0, 1, 0, 1, 0]
        })
        
        # Create test data with different statistics
        # (this simulates out-of-distribution data)
        test_data = pd.DataFrame({
            'feature_1': [100.0, 200.0, 300.0],
            'feature_2': [1000.0, 2000.0, 3000.0],
            'timestamp': range(5, 8),
            LABEL_COL: [1, 0, 1]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            test_scaled_path = f"{tmpdir}/test_scaled.csv"
            
            # Save test data
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            # Fit scaler on training data
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            
            # Apply trained scaler to test data
            post_split_preprocessing_test(test_path, test_scaled_path, scaler_path)
            
            # Load scaler and verify it was fit on train statistics
            scaler = joblib.load(scaler_path)
            
            # The scaler's center should be close to train data mean, not test
            train_mean = train_data[['feature_1', 'feature_2']].mean().values
            
            # RobustScaler uses median, not mean
            train_median = train_data[['feature_1', 'feature_2']].median().values
            
            assert np.allclose(scaler.center_, train_median, rtol=1e-5), \
                "Scaler should be fit on training data statistics, not test"

    def test_scaler_persists_continuous_columns(self):
        """Scaler should track which columns were continuous during fit."""
        train_data = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'binary_feature': [0.0, 1.0, 0.0, 1.0, 0.0],  # Binary in training
            'timestamp': range(5),
            LABEL_COL: [0, 1, 0, 1, 0]
        })
        
        test_data = pd.DataFrame({
            'numeric_feature': [6.0, 7.0, 8.0],
            'binary_feature': [0.0, 0.0, 0.0],  # Becomes constant in test
            'timestamp': range(5, 8),
            LABEL_COL: [1, 0, 1]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            test_scaled_path = f"{tmpdir}/test_scaled.csv"
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            # Fit scaler
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            
            # Verify continuous columns were saved
            continuous_cols_path = scaler_path.replace('.pkl', '_continuous_cols.pkl')
            assert Path(continuous_cols_path).exists(), \
                "Continuous columns info should be saved"
            
            continuous_cols = joblib.load(continuous_cols_path)
            
            # numeric_feature should be in continuous cols
            assert 'numeric_feature' in continuous_cols, \
                "Numeric feature should be in continuous columns"
            
            # Apply to test using same continuous columns
            post_split_preprocessing_test(test_path, test_scaled_path, scaler_path)
            
            # Verify test was scaled correctly
            test_scaled = pd.read_csv(test_scaled_path)
            assert not test_scaled.empty
            assert not test_scaled.isna().any().any(), \
                "No NaN values should be introduced during scaling"


class TestColumnAlignment:
    """Test that train and test data have aligned columns (no feature mismatch)."""

    def test_missing_columns_added_with_zeros(self):
        """Missing columns in test should be added with zeros."""
        train_data = pd.DataFrame({
            'feature_a': [1.0, 2.0, 3.0],
            'feature_b': [4.0, 5.0, 6.0],
            'feature_c': [7.0, 8.0, 9.0],
            'timestamp': range(3),
            LABEL_COL: [0, 1, 0]
        })
        
        # Test data missing feature_b
        test_data = pd.DataFrame({
            'feature_a': [10.0, 11.0],
            'feature_c': [12.0, 13.0],
            'timestamp': range(3, 5),
            LABEL_COL: [1, 0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            test_scaled_path = f"{tmpdir}/test_scaled.csv"
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            # Fit on train
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            
            # Apply to test (should add missing feature_b)
            post_split_preprocessing_test(test_path, test_scaled_path, scaler_path)
            
            test_scaled = pd.read_csv(test_scaled_path)
            
            # Verify all train columns are present in test
            train_scaled = pd.read_csv(train_scaled_path)
            assert set(test_scaled.columns) == set(train_scaled.columns), \
                "Test should have exactly the same columns as training"
            
            # Verify feature_b was added and has values
            assert 'feature_b' in test_scaled.columns
            # Should be scaled zeros (scaler center will be subtracted)
            assert test_scaled['feature_b'].notna().all()

    def test_extra_columns_removed(self):
        """Extra columns in test (not in train) should be removed."""
        train_data = pd.DataFrame({
            'feature_a': [1.0, 2.0, 3.0],
            'feature_b': [4.0, 5.0, 6.0],
            'timestamp': range(3),
            LABEL_COL: [0, 1, 0]
        })
        
        # Test data has extra column
        test_data = pd.DataFrame({
            'feature_a': [10.0, 11.0],
            'feature_b': [12.0, 13.0],
            'feature_c_extra': [14.0, 15.0],  # Extra column
            'timestamp': range(3, 5),
            LABEL_COL: [1, 0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            test_scaled_path = f"{tmpdir}/test_scaled.csv"
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            post_split_preprocessing_test(test_path, test_scaled_path, scaler_path)
            
            test_scaled = pd.read_csv(test_scaled_path)
            train_scaled = pd.read_csv(train_scaled_path)
            
            # Verify extra column was removed
            assert 'feature_c_extra' not in test_scaled.columns
            assert set(test_scaled.columns) == set(train_scaled.columns)

    def test_column_order_matches(self):
        """Test and train columns should be in same order."""
        train_data = pd.DataFrame({
            'z_feature': [1.0, 2.0, 3.0],
            'a_feature': [4.0, 5.0, 6.0],
            'm_feature': [7.0, 8.0, 9.0],
            'timestamp': range(3),
            LABEL_COL: [0, 1, 0]
        })
        
        # Test has columns in different order
        test_data = pd.DataFrame({
            'a_feature': [10.0, 11.0],
            'z_feature': [12.0, 13.0],
            'm_feature': [14.0, 15.0],
            'timestamp': range(3, 5),
            LABEL_COL: [1, 0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            test_path = f"{tmpdir}/test.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            test_scaled_path = f"{tmpdir}/test_scaled.csv"
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            post_split_preprocessing_test(test_path, test_scaled_path, scaler_path)
            
            train_scaled = pd.read_csv(train_scaled_path)
            test_scaled = pd.read_csv(test_scaled_path)
            
            # Verify columns are in exact same order
            assert list(test_scaled.columns) == list(train_scaled.columns), \
                "Train and test columns must be in same order"


class TestNoNaNAfterScaling:
    """Test that scaling doesn't introduce NaN values."""

    def test_no_nan_introduced_by_scaler(self):
        """Scaling should not introduce NaN values."""
        data = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'timestamp': range(5),
            LABEL_COL: [0, 1, 0, 1, 0]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            scaler_path = f"{tmpdir}/scaler.pkl"
            train_scaled_path = f"{tmpdir}/train_scaled.csv"
            
            data.to_csv(train_path, index=False)
            post_split_preprocessing_train(train_path, train_scaled_path, scaler_path)
            
            scaled_data = pd.read_csv(train_scaled_path)
            
            # Check for NaN values
            assert not scaled_data.isna().any().any(), \
                "Scaling should not introduce NaN values"
            
            # Check for infinite values
            numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns
            assert not np.isinf(scaled_data[numeric_cols].values).any(), \
                "Scaling should not introduce infinite values"


class TestModelTrainingValidation:
    """Test that model training validates data properly."""

    def test_training_detects_empty_data(self):
        """Training should fail gracefully on empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            model_path = f"{tmpdir}/model.pkl"
            
            # Create empty CSV with just headers
            pd.DataFrame(columns=['feature_1', 'feature_2', LABEL_COL]).to_csv(
                train_path, index=False
            )
            
            with pytest.raises(ValueError, match="empty"):
                train_model(train_path, model_path)

    def test_training_requires_label_column(self):
        """Training should fail if label column is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            model_path = f"{tmpdir}/model.pkl"
            
            # Create data without label column
            pd.DataFrame({
                'feature_1': [1.0, 2.0, 3.0],
                'feature_2': [4.0, 5.0, 6.0],
            }).to_csv(train_path, index=False)
            
            with pytest.raises(ValueError, match="Label column"):
                train_model(train_path, model_path)

    def test_model_handles_nan_values(self):
        """Model should handle NaN values gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = f"{tmpdir}/train.csv"
            model_path = f"{tmpdir}/model.pkl"
            
            # Create data with NaN
            data = pd.DataFrame({
                'feature_1': [1.0, 2.0, np.nan, 4.0, 5.0],
                'feature_2': [4.0, 5.0, 6.0, 7.0, 8.0],
                LABEL_COL: [0, 1, 0, 1, 0]
            })
            data.to_csv(train_path, index=False)
            
            # Should complete with warning (NaN handling in place)
            # Note: Depending on implementation, may need to fill NaN first
            try:
                train_model(train_path, model_path)
            except (ValueError, RuntimeError) as e:
                # If training fails, should be due to NaN handling
                assert "NaN" in str(e) or "nan" in str(e).lower()


class TestPredictionValidation:
    """Test that predictions are valid."""

    def test_predictions_are_probabilities(self):
        """Predictions should be probabilities in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train minimal model
            train_data = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                LABEL_COL: np.random.randint(0, 2, 100)
            })
            
            train_path = f"{tmpdir}/train.csv"
            model_path = f"{tmpdir}/model.pkl"
            
            train_data.to_csv(train_path, index=False)
            train_model(train_path, model_path)
            
            # Create test data
            test_data = train_data.drop(columns=[LABEL_COL]).head(10)
            test_path = f"{tmpdir}/test.csv"
            pred_path = f"{tmpdir}/predictions.csv"
            
            test_data.to_csv(test_path, index=False)
            
            # Generate predictions
            predict(model_path, test_path, pred_path)
            
            predictions = pd.read_csv(pred_path)
            
            # Check probability bounds
            from src.config.config import PREDICTION_COL
            if PREDICTION_COL in predictions.columns:
                probs = predictions[PREDICTION_COL]
                assert (probs >= 0).all() and (probs <= 1).all(), \
                    "Predictions should be probabilities in [0, 1]"

    def test_prediction_shape_matches_input(self):
        """Number of predictions should equal number of test samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train model
            train_data = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                LABEL_COL: np.random.randint(0, 2, 100)
            })
            
            train_path = f"{tmpdir}/train.csv"
            model_path = f"{tmpdir}/model.pkl"
            train_data.to_csv(train_path, index=False)
            train_model(train_path, model_path)
            
            # Create test data
            n_test = 25
            test_data = pd.DataFrame({
                'feature_1': np.random.randn(n_test),
                'feature_2': np.random.randn(n_test),
            })
            
            test_path = f"{tmpdir}/test.csv"
            pred_path = f"{tmpdir}/predictions.csv"
            test_data.to_csv(test_path, index=False)
            
            predict(model_path, test_path, pred_path)
            
            predictions = pd.read_csv(pred_path)
            
            # Should have same number of rows as input
            assert len(predictions) == n_test, \
                f"Expected {n_test} predictions, got {len(predictions)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
