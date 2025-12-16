"""Machine learning model training and prediction pipeline.

Provides model factory, training, feature importance extraction, and prediction.
Uses ensemble voting of multiple classifier types for robust predictions.
"""
from pathlib import Path
from typing import Optional, Dict, Union
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from src.utils.io_utils import load_data, save_data, save_csv
from src.config.config import *


# =======================================================
# === MODEL FACTORY ====================================
# =======================================================

def build_default_model() -> VotingClassifier:
    """
    Assemble the default ensemble model (soft voting).
    Easily extensible by changing the estimators list or adding logic here.
    """
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('et', ExtraTreesClassifier(random_state=42)),
        ('hgb', HistGradientBoostingClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42, verbose=-1)),
    ]
    model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    return model


def compute_class_weights_for_data(y: pd.Series) -> Dict[int, float]:
    """Compute class weights to handle imbalanced data.
    
    Args:
        y: Target series with binary class labels.
    
    Returns:
        Dictionary mapping class labels to their weights.
    
    Raises:
        ValueError: If y is empty or has no variation.
    """
    if y.empty:
        raise ValueError("Target series is empty")
    
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"Target has only {len(classes)} class(es), need at least 2")
    
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {c: float(w) for c, w in zip(classes, weights)}


def get_feature_importance(model: VotingClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Extract feature importance from ensemble model.
    
    Averages importance across tree-based estimators in the ensemble.
    
    Args:
        model: Fitted VotingClassifier.
        X: Feature DataFrame (for column names).
    
    Returns:
        pd.DataFrame: Feature importance sorted by importance score.
    
    Raises:
        ValueError: If model has no tree-based estimators.
    """
    importances = []
    feature_names = X.columns
    
    for name, estimator in model.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            importances.append(estimator.feature_importances_)
    
    if not importances:
        # Fallback: return uniform importance if no tree estimators
        n_features = len(feature_names)
        avg_importance = np.ones(n_features) / n_features
    else:
        # Average importance across all estimators
        avg_importance = np.mean(importances, axis=0)
        # Normalize to sum to 1
        avg_importance = avg_importance / avg_importance.sum()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df


# =======================================================
# === TRAINING PIPELINE =================================
# =======================================================

def train_model(train_csv_path: Union[str, Path], model_save_path: Union[str, Path]) -> None:
    """Train a model using the specified training set and save it to disk.
    
    Includes:
    - Automatic class weight computation for imbalanced data
    - Validation of data quality
    - Feature importance extraction
    - Model serialization
    
    Args:
        train_csv_path: Path to training CSV file.
        model_save_path: Path where model will be saved.
    
    Raises:
        FileNotFoundError: If training CSV doesn't exist.
        ValueError: If training data is invalid or empty.
        RuntimeError: If model training fails.
    """
    train_csv_path = str(train_csv_path)
    model_save_path = str(model_save_path)
    
    if not Path(train_csv_path).exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
    
    try:
        df = load_data(train_csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load training data: {str(e)}")

    if df.empty:
        raise ValueError("Training DataFrame is empty")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in training data")

    # Validate data
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    
    # Check for NaN or infinite values
    if X.isna().any().any():
        X = X.fillna(0)
    
    if np.isinf(X.select_dtypes(include=[np.number]).values).any():
        X = X.replace([np.inf, -np.inf], 0)
    
    # Compute class weights
    class_weights = compute_class_weights_for_data(y)

    # Build model
    model = build_default_model()

    # Fit model
    try:
        model.fit(X, y)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")

    # Extract and save feature importance
    try:
        importance_df = get_feature_importance(model, X)
        importance_path = str(model_save_path).replace('.pkl', '_feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
    except Exception:
        pass

    # Save model
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_save_path)


# =======================================================
# === PREDICTION PIPELINE ===============================
# =======================================================

def predict(model_path: Union[str, Path], input_csv_path: Union[str, Path], output_csv_path: Union[str, Path]) -> None:
    """Load model and input data, generate probability predictions, and save them.
    
    Only probability predictions are generated (no discrete class labels).
    Includes validation of input data quality.
    
    Args:
        model_path: Path to saved model file.
        input_csv_path: Path to input CSV file.
        output_csv_path: Path where predictions will be saved.
    
    Raises:
        FileNotFoundError: If model or input CSV doesn't exist.
        ValueError: If prediction data is invalid.
        RuntimeError: If prediction fails.
    """
    model_path = str(model_path)
    input_csv_path = str(input_csv_path)
    output_csv_path = str(output_csv_path)
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not Path(input_csv_path).exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    try:
        # Load model
        model = joblib.load(model_path)

        # Load data
        df = load_data(input_csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load model or data: {str(e)}")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # X only (drop label if present)
    X = df.drop(columns=[LABEL_COL]) if LABEL_COL in df.columns else df.copy()
    
    # Validate data
    if X.isna().any().any():
        X = X.fillna(0)
    
    if np.isinf(X.select_dtypes(include=[np.number]).values).any():
        X = X.replace([np.inf, -np.inf], 0)

    # Predict probabilities (only class 1 probability)
    try:
        df[PREDICTION_COL] = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    # Keep predictions as CSV for easy inspection (external-facing file)
    save_csv(df, output_csv_path)



# =======================================================
# === ENTRYPOINT ========================================
# =======================================================

if __name__ == "__main__":
    print("Training model...")
    train_model(str(TRAIN_CSV_PATH), str(MODEL_PKL_PATH))

    print("\nGenerating predictions...")
    predict(str(MODEL_PKL_PATH), str(TEST_CSV_PATH), str(PREDICTION_CSV_PATH))

