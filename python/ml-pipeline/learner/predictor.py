"""Module for making predictions with trained models."""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from config.column_names import TIMESTAMP, TICKER, PREDICTION_PROB, PREDICTION


def predict(
    model: XGBClassifier,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Make predictions on test data.
    
    Args:
        model: Trained XGBClassifier.
        test_df: Test DataFrame with features.
        feature_cols: List of feature column names used in training.
    
    Returns:
        DataFrame with timestamp, ticker, prediction probability, and prediction.
    """
    X = test_df[feature_cols].values
    
    # Handle any remaining NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Get probability predictions
    probs = model.predict_proba(X)[:, 1]  # Probability of class 1
    predictions = model.predict(X)
    
    result = pd.DataFrame({
        TIMESTAMP: test_df[TIMESTAMP].values,
        TICKER: test_df[TICKER].values,
        PREDICTION_PROB: probs,
        PREDICTION: predictions,
    })
    
    return result
