import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, log_loss, balanced_accuracy_score,
    precision_recall_curve, average_precision_score, auc
)

from src.config.config import *


# =======================================================
# === MACHINE LEARNING METRICS ==========================
# =======================================================

def calculate_ml_metrics(
        predictions_df: pd.DataFrame,
        probability_threshold: float
) -> pd.DataFrame:
    """Calculate comprehensive ML classification metrics."""
    df = predictions_df.copy()
    df["prediction"] = (df[PREDICTION_COL] > probability_threshold).astype(int)

    y_true = df[LABEL_COL]
    y_pred = df["prediction"]
    y_prob = df[PREDICTION_COL]

    metrics = _calculate_classification_metrics(y_true, y_pred, y_prob)
    _display_ml_metrics(metrics, probability_threshold)

    return pd.DataFrame([metrics])


def _calculate_classification_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: pd.Series
) -> Dict:
    """Calculate comprehensive classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Basic metrics
    n_samples = len(y_true)
    n_positive = y_true.sum()
    n_negative = n_samples - n_positive
    class_imbalance = n_positive / n_samples if n_samples > 0 else 0

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Probability-based metrics
    has_both_classes = len(np.unique(y_true)) > 1

    roc_auc = np.nan
    avg_precision = np.nan
    pr_auc = np.nan

    if has_both_classes:
        roc_auc = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        # Calculate PR AUC manually
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)

    logloss = log_loss(y_true, y_prob)
    brier_score = np.mean((y_prob - y_true) ** 2)

    # Cohen's Kappa
    p_observed = accuracy
    p_expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n_samples ** 2)
    cohens_kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected != 1 else 0

    return {
        "n_samples": n_samples,
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "class_imbalance": class_imbalance,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "mcc": mcc,
        "cohens_kappa": cohens_kappa,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,  # Added
        "avg_precision": avg_precision,
        "log_loss": logloss,
        "brier_score": brier_score,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def _display_ml_metrics(metrics: Dict, threshold: float):
    """Display ML metrics in formatted table."""
    print(f"\n{'=' * 60}")
    print(f"CLASSIFICATION METRICS (Threshold: {threshold})")
    print(f"{'=' * 60}\n")

    print("DATASET COMPOSITION")
    print("-" * 60)
    print(f"{'Total Samples':<30} {metrics['n_samples']:>15,}")
    print(f"{'Positive Class':<30} {metrics['n_positive']:>15,}")
    print(f"{'Negative Class':<30} {metrics['n_negative']:>15,}")
    print(f"{'Class Imbalance Ratio':<30} {metrics['class_imbalance']:>14.2%}")

    print(f"\nCONFUSION MATRIX")
    print("-" * 60)
    print(f"{'True Positives':<30} {metrics['true_positives']:>15,}")
    print(f"{'False Positives':<30} {metrics['false_positives']:>15,}")
    print(f"{'True Negatives':<30} {metrics['true_negatives']:>15,}")
    print(f"{'False Negatives':<30} {metrics['false_negatives']:>15,}")

    print(f"\nCLASSIFICATION PERFORMANCE")
    print("-" * 60)
    print(f"{'Accuracy':<30} {metrics['accuracy']:>14.4f}")
    print(f"{'Balanced Accuracy':<30} {metrics['balanced_accuracy']:>14.4f}")
    print(f"{'Precision':<30} {metrics['precision']:>14.4f}")
    print(f"{'Recall (Sensitivity)':<30} {metrics['recall']:>14.4f}")
    print(f"{'Specificity':<30} {metrics['specificity']:>14.4f}")
    print(f"{'F1 Score':<30} {metrics['f1_score']:>14.4f}")

    print(f"\nADVANCED METRICS")
    print("-" * 60)
    print(f"{'Matthews Corr Coef (MCC)':<30} {metrics['mcc']:>14.4f}")
    print(f"{'Cohens Kappa':<30} {metrics['cohens_kappa']:>14.4f}")

    if not np.isnan(metrics['roc_auc']):
        print(f"{'ROC AUC':<30} {metrics['roc_auc']:>14.4f}")
        print(f"{'PR AUC':<30} {metrics['pr_auc']:>14.4f}")
        print(f"{'Avg Precision Score':<30} {metrics['avg_precision']:>14.4f}")

    print(f"{'Log Loss':<30} {metrics['log_loss']:>14.4f}")
    print(f"{'Brier Score':<30} {metrics['brier_score']:>14.4f}")

    print(f"\n{'=' * 60}\n")
