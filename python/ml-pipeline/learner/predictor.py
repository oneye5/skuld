"""Module for making predictions with trained models."""

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, PREDICTION_PROB, PREDICTION


def predict(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Make predictions on test data.
    
    Args:
        model: Trained model with predict_proba() method.
        test_df: Test DataFrame with features.
        feature_cols: List of feature column names used in training.
    
    Returns:
        DataFrame with timestamp, ticker, prediction probability, and prediction.
    """
    X = test_df[feature_cols].values
    
    # Handle any remaining NaN
    X = np.nan_to_num(X, nan=0.0)
    
    probs = model.predict_proba(X)[:, 1]  # Probability of class 1
    predictions = (probs >= 0.5).astype(int)
    
    result = pd.DataFrame({
        TIMESTAMP: test_df[TIMESTAMP].values,
        TICKER: test_df[TICKER].values,
        PREDICTION_PROB: probs,
        PREDICTION: predictions,
    })
    
    return result
