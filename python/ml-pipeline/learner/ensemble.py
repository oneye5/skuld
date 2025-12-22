"""Ensemble model combining multiple classifiers for robust predictions.

Based on research showing ensemble methods outperform single models for financial prediction:
- Different algorithms capture different patterns
- Soft voting reduces variance and improves generalization
- Calibrated probabilities improve trading decisions

Per nzx-predictor success: VotingClassifier with RF, ExtraTrees, HistGB, XGB, LGBM
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config.column_names import TIMESTAMP, TICKER, TARGET
from config.model_config import (
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    CATBOOST_PARAMS,
)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    use_random_forest: bool = True
    use_extra_trees: bool = True
    use_hist_gradient_boosting: bool = True
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = False  # Disabled - RF+ET+HGB+XGB+LGBM matches nzx-predictor
    calibrate_probabilities: bool = False  # Disabled - isotonic calibration hurts performance
    rf_weight: float = 1.0
    et_weight: float = 1.0
    hgb_weight: float = 1.0
    xgb_weight: float = 1.0
    lgb_weight: float = 1.0
    cat_weight: float = 1.0


def calculate_class_weight(y: np.ndarray) -> float:
    """Calculate class weight ratio for imbalanced data."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


class EnsembleModel:
    """Ensemble of gradient boosting models with soft voting and calibration."""
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models: dict[str, object] = {}
        self.calibrated_models: dict[str, CalibratedClassifierCV] = {}
        self.weights: dict[str, float] = {}
        self.is_fitted = False
        self.is_calibrated = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """Fit all enabled models with optional probability calibration."""
        class_weight = calculate_class_weight(y)
        
        # RandomForest - bagging ensemble, good for diverse feature interactions
        if self.config.use_random_forest:
            self.models["random_forest"] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self.models["random_forest"].fit(X, y)
            self.weights["random_forest"] = self.config.rf_weight
        
        # ExtraTrees - more randomized than RF, reduces variance further
        if self.config.use_extra_trees:
            self.models["extra_trees"] = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self.models["extra_trees"].fit(X, y)
            self.weights["extra_trees"] = self.config.et_weight
        
        # HistGradientBoosting - fast native sklearn boosting
        if self.config.use_hist_gradient_boosting:
            self.models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=8,
                min_samples_leaf=50,
                random_state=42,
            )
            self.models["hist_gradient_boosting"].fit(X, y)
            self.weights["hist_gradient_boosting"] = self.config.hgb_weight
        
        # XGBoost - optimized gradient boosting
        if self.config.use_xgboost:
            xgb_params = XGBOOST_PARAMS.copy()
            xgb_params["scale_pos_weight"] = class_weight
            self.models["xgboost"] = XGBClassifier(random_state=42, **xgb_params)
            self.models["xgboost"].fit(X, y)
            self.weights["xgboost"] = self.config.xgb_weight
        
        # LightGBM - fast gradient boosting
        if self.config.use_lightgbm:
            lgb_params = LIGHTGBM_PARAMS.copy()
            self.models["lightgbm"] = LGBMClassifier(random_state=42, **lgb_params)
            self.models["lightgbm"].fit(X, y)
            self.weights["lightgbm"] = self.config.lgb_weight
        
        # CatBoost - handles categorical features well
        if self.config.use_catboost:
            cat_params = CATBOOST_PARAMS.copy()
            self.models["catboost"] = CatBoostClassifier(random_seed=42, **cat_params)
            self.models["catboost"].fit(X, y)
            self.weights["catboost"] = self.config.cat_weight
        
        # Calibrate probabilities using isotonic regression for better decisions
        # Isotonic is more flexible than sigmoid/Platt scaling
        if self.config.calibrate_probabilities and len(self.models) > 0:
            try:
                # Use 3-fold CV for calibration to avoid data leakage
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                for name, model in self.models.items():
                    calibrated = CalibratedClassifierCV(
                        model, method='isotonic', cv=cv, n_jobs=-1
                    )
                    calibrated.fit(X, y)
                    self.calibrated_models[name] = calibrated
                self.is_calibrated = True
            except Exception:
                # Fall back to uncalibrated if calibration fails
                self.is_calibrated = False
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted average of probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        total_weight = sum(self.weights.values())
        weighted_probs = None
        
        # Use calibrated models if available
        models_to_use = self.calibrated_models if self.is_calibrated else self.models
        
        for name, model in models_to_use.items():
            probs = model.predict_proba(X)[:, 1]  # Probability of positive class
            weight = self.weights[name] / total_weight
            
            if weighted_probs is None:
                weighted_probs = probs * weight
            else:
                weighted_probs += probs * weight
        
        # Return as 2D array [prob_0, prob_1]
        prob_1 = weighted_probs
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get class predictions."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self) -> dict[str, np.ndarray]:
        """Get feature importance from base (uncalibrated) models."""
        importances = {}
        
        # Always use base models for feature importance (calibrated don't have it)
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances[name] = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                importances[name] = model.get_feature_importance()
        
        return importances
    
    def get_averaged_importance(self) -> Optional[np.ndarray]:
        """Get averaged feature importance across models."""
        importances = self.get_feature_importance()
        if not importances:
            return None
        
        # Stack and average
        all_imp = np.stack(list(importances.values()))
        return all_imp.mean(axis=0)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns for training (exclude metadata and target)."""
    exclude = {TIMESTAMP, TICKER, TARGET}
    return [
        col for col in df.columns
        if col not in exclude
        and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]


def train_ensemble(
    train_df: pd.DataFrame,
    config: EnsembleConfig = None,
) -> tuple[EnsembleModel, list[str]]:
    """
    Train ensemble model.

    Args:
        train_df: Training DataFrame with features and target column.
        config: Ensemble configuration.

    Returns:
        Tuple of (trained ensemble model, list of feature column names).
    """
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols].to_numpy(copy=False)
    y = train_df[TARGET].to_numpy(copy=False)

    # Handle any remaining NaN (replace with 0)
    np.nan_to_num(X, copy=False, nan=0.0)
    
    model = EnsembleModel(config)
    model.fit(X, y)

    return model, feature_cols


def predict_ensemble(
    model: EnsembleModel,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Make predictions using ensemble model.

    Args:
        model: Trained ensemble model.
        test_df: Test DataFrame.
        feature_cols: Feature column names (from training).

    Returns:
        DataFrame with predictions and probabilities.
    """
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X = test_df[feature_cols].to_numpy(copy=False)
    np.nan_to_num(X, copy=False, nan=0.0)

    probs = model.predict_proba(X)

    result = test_df[[TIMESTAMP, TICKER]].copy()
    result["prediction"] = (probs[:, 1] >= 0.5).astype(int)
    result["probability"] = probs[:, 1]
    
    # Add actual if available
    if TARGET in test_df.columns:
        result["actual"] = test_df[TARGET].values

    return result
