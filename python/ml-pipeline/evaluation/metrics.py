"""Classification metrics module."""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from config.columns import TARGET, PREDICTION, PREDICTION_PROB


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int
    positive_samples: int
    negative_samples: int


def compute_classification_metrics(
    actuals_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> ClassificationMetrics:
    """Compute classification metrics.
    
    Args:
        actuals_df: DataFrame with TARGET column (true labels).
        predictions_df: DataFrame with PREDICTION and PREDICTION_PROB columns.
    
    Returns:
        ClassificationMetrics with all metrics.
    """
    y_true = actuals_df[TARGET].values
    y_pred = predictions_df[PREDICTION].values
    y_proba = predictions_df[PREDICTION_PROB].values
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Handle edge cases for AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        # All samples belong to one class
        auc = 0.5
    
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc_roc=float(auc),
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        total_samples=len(y_true),
        positive_samples=int(y_true.sum()),
        negative_samples=int((y_true == 0).sum()),
    )


def metrics_to_dict(metrics: ClassificationMetrics) -> dict:
    """Convert ClassificationMetrics to dictionary.
    
    Args:
        metrics: ClassificationMetrics instance.
    
    Returns:
        Dictionary representation.
    """
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "auc_roc": metrics.auc_roc,
        "confusion_matrix": {
            "true_positives": metrics.true_positives,
            "true_negatives": metrics.true_negatives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
        },
        "samples": {
            "total": metrics.total_samples,
            "positive": metrics.positive_samples,
            "negative": metrics.negative_samples,
        },
    }
