"""Module for training models."""

from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from config.column_names import TIMESTAMP, TICKER, TARGET
from config.model_config import create_model


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns for training (exclude metadata and target).
    
    Note: TIMESTAMP is included as a feature (should be min-max scaled first).
    """
    exclude = {TICKER, TARGET}  # TIMESTAMP now included as feature
    return [
        col for col in df.columns
        if col not in exclude
        and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]


def train_model(train_df: pd.DataFrame) -> tuple[object, list[str]]:
    """
    Train model on the provided data.

    Args:
        train_df: Training DataFrame with features and target column.

    Returns:
        Tuple of (trained model, list of feature column names).
    """
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols].to_numpy(copy=False)
    y = train_df[TARGET].to_numpy(copy=False)

    # Handle any remaining NaN
    np.nan_to_num(X, copy=False, nan=0.0)
    
    model = create_model()
    model.fit(X, y)

    return model, feature_cols


def save_model(model, output_path: Path) -> None:
    """Save trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path: Path):
    """Load trained model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)
