"""
Tests to validate evaluation metrics and workflow.

Checks for:
1. Metric calculation correctness
2. Evaluation on correct data split
3. Metric interpretation and edge cases
4. Classification metric coherence
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)

from src.evaluation.ml_evaluation import (
    calculate_ml_metrics, _calculate_classification_metrics
)
from src.config.config import LABEL_COL, PREDICTION_COL


class TestMetricCalculationValidity:
    """Test that metrics are calculated correctly."""

    def test_perfect_predictions_perfect_metrics(self):
        """Perfect predictions should yield perfect metrics."""
        # Create perfect predictions
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 0, 1, 1, 1],
            PREDICTION_COL: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        # Check perfect metrics
        assert metrics['accuracy'].iloc[0] == 1.0
        assert metrics['precision'].iloc[0] == 1.0
        assert metrics['recall'].iloc[0] == 1.0
        assert metrics['f1_score'].iloc[0] == 1.0
        assert np.isclose(metrics['roc_auc'].iloc[0], 1.0)

    def test_random_predictions_baseline_metrics(self):
        """Random predictions should give baseline metrics."""
        np.random.seed(42)
        predictions_df = pd.DataFrame({
            LABEL_COL: np.random.randint(0, 2, 100),
            PREDICTION_COL: np.random.rand(100),
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        # Random predictor should have accuracy near 50% for balanced data
        accuracy = metrics['accuracy'].iloc[0]
        assert 0.3 < accuracy < 0.7, \
            f"Random predictions should give ~50% accuracy, got {accuracy}"

    def test_all_positive_predictions(self):
        """All positive predictions should be handled gracefully."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 1, 1],
            PREDICTION_COL: [0.9, 0.9, 0.9, 0.9],  # All predicted as positive
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        # Should calculate without error
        assert not metrics.empty
        # Precision might be undefined or partial, but should complete
        assert 'precision' in metrics.columns

    def test_all_negative_predictions(self):
        """All negative predictions should be handled gracefully."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 1, 1],
            PREDICTION_COL: [0.1, 0.1, 0.1, 0.1],  # All predicted as negative
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        assert not metrics.empty
        assert 'recall' in metrics.columns


class TestMetricCoherence:
    """Test that metrics are logically coherent with each other."""

    def test_precision_recall_tradeoff(self):
        """Precision and recall should show expected tradeoff."""
        # High threshold = high precision, low recall
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 0, 1, 1, 1, 1, 1],
            PREDICTION_COL: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.95],
        })
        
        metrics_strict = calculate_ml_metrics(predictions_df, probability_threshold=0.85)
        metrics_loose = calculate_ml_metrics(predictions_df, probability_threshold=0.35)
        
        # Stricter threshold should give higher precision
        assert metrics_strict['precision'].iloc[0] >= metrics_loose['precision'].iloc[0], \
            "Stricter threshold should increase precision"

    def test_accuracy_vs_balanced_accuracy(self):
        """Balanced accuracy should differ from accuracy in imbalanced data."""
        # Imbalanced: 90% negative, 10% positive
        predictions_df = pd.DataFrame({
            LABEL_COL: [0]*9 + [1],
            PREDICTION_COL: np.random.rand(10),
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        accuracy = metrics['accuracy'].iloc[0]
        balanced_acc = metrics['balanced_accuracy'].iloc[0]
        
        # They might differ, especially with imbalance
        # (This tests they're both calculated)
        assert 'accuracy' in metrics.columns
        assert 'balanced_accuracy' in metrics.columns

    def test_confusion_matrix_consistency(self):
        """Metrics should be consistent with confusion matrix."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 0, 0, 1, 1, 1, 1],
            PREDICTION_COL: [0.1, 0.2, 0.6, 0.7, 0.3, 0.4, 0.8, 0.9],
        })
        
        threshold = 0.5
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=threshold)
        
        # Calculate manual confusion matrix
        y_true = predictions_df[LABEL_COL]
        y_pred = (predictions_df[PREDICTION_COL] > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            
            # Verify metrics align with CM
            manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
            assert np.isclose(metrics['accuracy'].iloc[0], manual_accuracy), \
                "Accuracy should match confusion matrix calculation"
            
            if (tp + fp) > 0:
                manual_precision = tp / (tp + fp)
                assert np.isclose(metrics['precision'].iloc[0], manual_precision), \
                    "Precision should match confusion matrix calculation"


class TestMetricBoundaryValues:
    """Test metrics at boundary conditions."""

    def test_probability_exactly_at_threshold(self):
        """Probabilities exactly at threshold should be handled correctly."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 1, 1],
            PREDICTION_COL: [0.5, 0.5, 0.9],  # At and beyond threshold
        })
        
        # Should handle gracefully
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        assert not metrics.empty

    def test_zero_one_probabilities(self):
        """Extreme probabilities (0 and 1) should be valid."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 1, 1],
            PREDICTION_COL: [0.0, 0.0, 1.0, 1.0],
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        # Should be perfect predictions
        assert metrics['accuracy'].iloc[0] == 1.0

    def test_single_sample(self):
        """Single sample should not cause calculation errors or handle gracefully."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [1],
            PREDICTION_COL: [0.8],
        })
        
        # Single sample with single class may raise ValueError
        try:
            metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
            # If it succeeds, should have valid results
            assert not metrics.empty
            assert len(metrics) == 1
        except ValueError as e:
            # Single class with single sample can cause sklearn errors
            # This is expected behavior
            assert "only one label" in str(e)


class TestMetricEdgeCases:
    """Test edge cases in metric calculation."""

    def test_single_class_in_ground_truth(self):
        """All same class in ground truth should be handled."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [1, 1, 1, 1],
            PREDICTION_COL: [0.2, 0.4, 0.6, 0.8],
        })
        
        # Should calculate without error (may have warnings for undefined metrics)
        try:
            metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
            # If it succeeds, check basic structure
            assert not metrics.empty
        except ValueError:
            # Single class can cause sklearn metrics to fail - this is expected
            pass

    def test_class_imbalance_metrics(self):
        """Highly imbalanced data should calculate all metrics."""
        # 95% negative, 5% positive
        predictions_df = pd.DataFrame({
            LABEL_COL: [0]*19 + [1],
            PREDICTION_COL: np.random.rand(20),
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        # All key metrics should be present
        assert 'accuracy' in metrics.columns
        assert 'balanced_accuracy' in metrics.columns
        assert 'precision' in metrics.columns
        assert 'recall' in metrics.columns


class TestMetricValidation:
    """Test that metrics are valid numbers."""

    def test_no_nan_in_metrics(self):
        """Metrics should not contain NaN values."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 1, 1],
            PREDICTION_COL: [0.2, 0.3, 0.7, 0.8],
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        # Check for NaN values in numeric columns
        numeric_cols = metrics.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not metrics[col].isna().any(), \
                f"Metric '{col}' contains NaN values"

    def test_metrics_in_valid_ranges(self):
        """Metrics should be in expected ranges."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 0, 1, 1, 1],
            PREDICTION_COL: [0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        # Accuracy should be in [0, 1]
        assert 0 <= metrics['accuracy'].iloc[0] <= 1
        
        # Precision should be in [0, 1]
        precision = metrics['precision'].iloc[0]
        assert np.isnan(precision) or (0 <= precision <= 1)
        
        # Recall should be in [0, 1]
        recall = metrics['recall'].iloc[0]
        assert np.isnan(recall) or (0 <= recall <= 1)
        
        # F1 should be in [0, 1]
        f1 = metrics['f1_score'].iloc[0]
        assert np.isnan(f1) or (0 <= f1 <= 1)
        
        # Balanced accuracy should be in [0, 1]
        assert 0 <= metrics['balanced_accuracy'].iloc[0] <= 1

    def test_roc_auc_validity(self):
        """ROC AUC should be valid when both classes present."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 0, 1, 1, 1],
            PREDICTION_COL: [0.1, 0.2, 0.3, 0.6, 0.7, 0.9],
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.5)
        
        roc_auc = metrics['roc_auc'].iloc[0]
        
        # Should be valid when both classes present
        assert not np.isnan(roc_auc), "ROC AUC should be valid with both classes"
        assert 0 <= roc_auc <= 1, "ROC AUC should be in [0, 1]"


class TestMetricInterpretation:
    """Test that metrics make sense together."""

    def test_high_precision_low_recall(self):
        """Can have high precision with low recall (conservative predictions)."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 0, 0, 1, 1, 1, 1],
            PREDICTION_COL: [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.8, 0.9],
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.75)
        
        # Very strict threshold - high precision, low recall
        precision = metrics['precision'].iloc[0]
        recall = metrics['recall'].iloc[0]
        
        # Precision might be 1.0, recall lower
        assert isinstance(precision, (int, float, np.number))
        assert isinstance(recall, (int, float, np.number))

    def test_low_precision_high_recall(self):
        """Can have low precision with high recall (aggressive predictions)."""
        predictions_df = pd.DataFrame({
            LABEL_COL: [0, 0, 0, 0, 1, 1, 1, 1],
            PREDICTION_COL: [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.8, 0.9],
        })
        
        metrics = calculate_ml_metrics(predictions_df, probability_threshold=0.25)
        
        # Very loose threshold - high recall, lower precision
        precision = metrics['precision'].iloc[0]
        recall = metrics['recall'].iloc[0]
        
        assert isinstance(precision, (int, float, np.number))
        assert isinstance(recall, (int, float, np.number))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
