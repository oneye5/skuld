"""Module for model evaluation metrics."""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config.column_names import TARGET, PREDICTION, PREDICTION_PROB, TIMESTAMP, TICKER


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: np.ndarray
    classification_report: str


from config.column_names import TARGET, PREDICTION, PREDICTION_PROB, TIMESTAMP, TICKER


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> ClassificationMetrics:
    """
    Evaluate model predictions against actual targets.
    
    Args:
        predictions_df: DataFrame with TIMESTAMP, TICKER, PREDICTION, PREDICTION_PROB.
        actuals_df: DataFrame with TIMESTAMP, TICKER, TARGET column.
    
    Returns:
        ClassificationMetrics containing all evaluation metrics.
    """
    # Merge predictions with actuals to ensure proper alignment
    merged = predictions_df.merge(
        actuals_df[[TIMESTAMP, TICKER, TARGET]],
        on=[TIMESTAMP, TICKER],
        how='inner'
    )
    
    if merged.empty:
        raise ValueError("No matching rows between predictions and actuals")
    
    y_true = merged[TARGET].values
    y_pred = merged[PREDICTION].values
    y_prob = merged[PREDICTION_PROB].values
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC (handle case where only one class is present)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = None
    
    # Confusion matrix and report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=cm,
        classification_report=report,
    )


def metrics_to_dict(metrics: ClassificationMetrics) -> dict:
    """Convert ClassificationMetrics to dictionary for serialization."""
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "confusion_matrix": metrics.confusion_matrix.tolist(),
    }


def aggregate_metrics(metrics_list: list[ClassificationMetrics]) -> dict:
    """Aggregate metrics across multiple windows."""
    aggregated = {
        "accuracy_mean": np.mean([m.accuracy for m in metrics_list]),
        "accuracy_std": np.std([m.accuracy for m in metrics_list]),
        "precision_mean": np.mean([m.precision for m in metrics_list]),
        "precision_std": np.std([m.precision for m in metrics_list]),
        "recall_mean": np.mean([m.recall for m in metrics_list]),
        "recall_std": np.std([m.recall for m in metrics_list]),
        "f1_mean": np.mean([m.f1 for m in metrics_list]),
        "f1_std": np.std([m.f1 for m in metrics_list]),
    }
    
    roc_aucs = [m.roc_auc for m in metrics_list if m.roc_auc is not None]
    if roc_aucs:
        aggregated["roc_auc_mean"] = np.mean(roc_aucs)
        aggregated["roc_auc_std"] = np.std(roc_aucs)
    
    return aggregated
