"""Model training module."""

import pandas as pd

from config.columns import TARGET
from config.settings import create_model


def train_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
):
    """Train a model on the training data.
    
    Args:
        train_df: Training DataFrame with features and target.
        feature_cols: List of feature column names to use.
    
    Returns:
        Trained model instance.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    
    model = create_model()
    model.fit(X_train, y_train)
    
    return model


def get_feature_columns(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> list[str]:
    """Get list of feature columns from DataFrame.
    
    Args:
        df: DataFrame to extract feature columns from.
        exclude_cols: Additional columns to exclude beyond defaults.
    
    Returns:
        List of feature column names.
    """
    from config.columns import TIMESTAMP, TICKER, TARGET
    
    default_exclude = {TIMESTAMP, TICKER, TARGET}
    
    if exclude_cols:
        default_exclude.update(exclude_cols)
    
    feature_cols = [
        col for col in df.columns 
        if col not in default_exclude
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    return feature_cols
