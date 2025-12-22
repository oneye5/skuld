"""Module for training models."""

from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from config.column_names import TIMESTAMP, TICKER, TARGET
from config.model_config import (
    initialize_model,
    calculate_class_weight,
    USE_ENSEMBLE,
)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns for training (exclude metadata and target)."""
    exclude = {TIMESTAMP, TICKER, TARGET}
    return [
        col for col in df.columns
        if col not in exclude
        and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]


def train_model(
    train_df: pd.DataFrame,
    params: dict | None = None,
) -> tuple[object, list[str]]:
    """
    Train model with class weight balancing.
    
    Uses ensemble (XGBoost + LightGBM + CatBoost) if USE_ENSEMBLE is True,
    otherwise uses XGBoost only.

    Args:
        train_df: Training DataFrame with features and target column.
        params: Parameters for the model.

    Returns:
        Tuple of (trained model, list of feature column names).
    """
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols].to_numpy(copy=False)
    y = train_df[TARGET].to_numpy(copy=False)

    # Handle any remaining NaN (replace with 0)
    np.nan_to_num(X, copy=False, nan=0.0)
    
    if USE_ENSEMBLE:
        # Use ensemble of models
        from .ensemble import EnsembleModel, EnsembleConfig
        config = EnsembleConfig(
            use_xgboost=True,
            use_lightgbm=True,
            use_catboost=True,
            calibrate_probabilities=False,  # Avoid data leakage
        )
        model = EnsembleModel(config)
        model.fit(X, y)
    else:
        # Use single XGBoost model
        class_weight = calculate_class_weight(y)
        model = initialize_model(params, class_weight=class_weight)
        model.fit(X, y)

    return model, feature_cols


def save_model(model: XGBClassifier, output_path: Path) -> None:
    """Save trained model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path: Path) -> XGBClassifier:
    """Load trained model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)
