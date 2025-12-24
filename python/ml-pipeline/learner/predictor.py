"""Model prediction module."""

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, PREDICTION_PROB, PREDICTION
from config.settings import PREDICTION_THRESHOLD


def predict(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = PREDICTION_THRESHOLD,
    ticker_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate predictions for test data.
    
    Args:
        model: Trained model with predict_proba method.
        test_df: Test DataFrame with features.
        feature_cols: List of feature column names.
        threshold: Probability threshold for positive prediction.
        ticker_info: Optional DataFrame with TIMESTAMP and TICKER columns.
            Use this when test_df has ticker one-hot encoded.
    
    Returns:
        DataFrame with timestamp, ticker, probability, and prediction columns.
    """
    X_test = test_df[feature_cols].values
    
    # Get probability of positive class
    proba = model.predict_proba(X_test)[:, 1]
    
    # Apply threshold
    predictions = (proba >= threshold).astype(int)
    
    # Get ticker/timestamp info
    if ticker_info is not None:
        timestamps = ticker_info[TIMESTAMP].values
        tickers = ticker_info[TICKER].values
    elif TIMESTAMP in test_df.columns:
        timestamps = test_df[TIMESTAMP].values
        tickers = test_df[TICKER].values if TICKER in test_df.columns else [""] * len(test_df)
    else:
        timestamps = np.arange(len(test_df))
        tickers = [""] * len(test_df)
    
    # Build result DataFrame
    result = pd.DataFrame({
        TIMESTAMP: timestamps,
        TICKER: tickers,
        PREDICTION_PROB: proba,
        PREDICTION: predictions,
    })
    
    return result


def get_prediction_summary(predictions_df: pd.DataFrame) -> dict:
    """Get summary statistics for predictions.
    
    Args:
        predictions_df: DataFrame with prediction columns.
    
    Returns:
        Dictionary with summary statistics.
    """
    proba = predictions_df[PREDICTION_PROB]
    preds = predictions_df[PREDICTION]
    
    return {
        "total_predictions": len(predictions_df),
        "positive_predictions": int(preds.sum()),
        "negative_predictions": int((preds == 0).sum()),
        "mean_probability": float(proba.mean()),
        "std_probability": float(proba.std()),
        "min_probability": float(proba.min()),
        "max_probability": float(proba.max()),
    }
