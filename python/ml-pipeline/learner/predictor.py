"""Model prediction module."""

import pandas as pd
import numpy as np

from config.columns import TIMESTAMP, TICKER, PREDICTION_PROB, PREDICTION
from config.settings import PREDICTION_THRESHOLD, TOP_N_PREDICTIONS


def predict(
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = PREDICTION_THRESHOLD,
    top_n: int | None = TOP_N_PREDICTIONS,
    ticker_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate predictions for test data.
    
    Args:
        model: Trained model with predict_proba method.
        test_df: Test DataFrame with features.
        feature_cols: List of feature column names.
        threshold: Probability threshold for positive prediction (legacy).
        top_n: If set, select top N predictions per timestamp instead of threshold.
               This is more realistic for trading - pick best N opportunities.
        ticker_info: Optional DataFrame with TIMESTAMP and TICKER columns.
            Use this when test_df has ticker one-hot encoded.
    
    Returns:
        DataFrame with timestamp, ticker, probability, and prediction columns.
    """
    X_test = test_df[feature_cols]
    
    # Get probability of positive class
    proba = model.predict_proba(X_test)[:, 1]
    
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
    })
    
    # Apply selection strategy
    if top_n is not None and top_n > 0:
        # Select top N predictions per timestamp
        result = _select_top_n_per_timestamp(result, top_n)
    else:
        # Legacy: apply threshold
        result[PREDICTION] = (result[PREDICTION_PROB] >= threshold).astype(int)
    
    return result


def _select_top_n_per_timestamp(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select top N predictions per unique timestamp.
    
    This is more realistic for trading - on each decision date, 
    pick the top N most confident predictions.
    
    Args:
        df: DataFrame with TIMESTAMP, TICKER, PREDICTION_PROB columns.
        top_n: Number of top predictions to select per timestamp.
    
    Returns:
        DataFrame with PREDICTION column added (1 for top N, 0 otherwise).
    """
    df = df.copy()
    df[PREDICTION] = 0
    
    # Group by timestamp and select top N
    for ts in df[TIMESTAMP].unique():
        mask = df[TIMESTAMP] == ts
        ts_df = df.loc[mask].copy()
        
        # Get indices of top N by probability
        if len(ts_df) <= top_n:
            # If fewer than N samples, select all
            top_indices = ts_df.index
        else:
            # Select top N by probability
            top_indices = ts_df.nlargest(top_n, PREDICTION_PROB).index
        
        df.loc[top_indices, PREDICTION] = 1
    
    return df


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
