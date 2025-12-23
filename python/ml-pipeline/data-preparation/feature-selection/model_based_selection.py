"""Model-based feature selection methods.

These methods use a model to evaluate feature importance and select
the most predictive features. They are more computationally expensive
than filter methods but can capture feature interactions.

Methods included:
- Tree-based importance (using LightGBM/XGBoost feature importances)
- Recursive feature elimination (RFE) placeholder
"""

import pandas as pd
import numpy as np
from typing import Tuple

from config.column_names import TIMESTAMP, TICKER, TARGET

from base import BaseFeatureSelector, FeatureSelectionResult


class TreeImportanceSelector(BaseFeatureSelector):
    """Select features based on tree model feature importance.
    
    Uses a gradient boosting model to compute feature importances,
    then keeps the top N features or features above a threshold.
    
    Args:
        n_features: Number of top features to keep. If None, use threshold.
        importance_threshold: Minimum importance to keep (if n_features is None).
        model_type: Type of tree model ('lightgbm' or 'xgboost').
    """
    
    def __init__(
        self,
        n_features: int | None = None,
        importance_threshold: float = 0.001,
        model_type: str = "lightgbm",
    ):
        self.n_features = n_features
        self.importance_threshold = importance_threshold
        self.model_type = model_type
        
        self._selected_features: list[str] = []
        self._dropped_features: list[str] = []
        self._importances: dict[str, float] = {}
        self._is_fitted = False
    
    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns (excluding metadata and target)."""
        exclude_cols = {TIMESTAMP, TICKER, TARGET, "index"}
        return [col for col in df.columns if col not in exclude_cols]
    
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "TreeImportanceSelector":
        """Fit the selector using a tree model.
        
        Args:
            df: Training DataFrame with features.
            y: Target variable (required for this method).
        
        Returns:
            self for method chaining.
        """
        if y is None:
            if TARGET in df.columns:
                y = df[TARGET]
            else:
                raise ValueError("Target variable required for TreeImportanceSelector")
        
        feature_cols = self._get_feature_columns(df)
        X = df[feature_cols].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Train a quick model to get importances
        if self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                verbosity=-1,
                random_state=42,
            )
        elif self.model_type == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                verbosity=0,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        model.fit(X, y)
        
        # Extract importances
        importances = model.feature_importances_
        self._importances = dict(zip(feature_cols, importances))
        
        # Sort by importance
        sorted_features = sorted(
            self._importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select features
        if self.n_features is not None:
            self._selected_features = [f for f, _ in sorted_features[:self.n_features]]
        else:
            self._selected_features = [
                f for f, imp in sorted_features if imp >= self.importance_threshold
            ]
        
        self._dropped_features = [
            f for f in feature_cols if f not in self._selected_features
        ]
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame by selecting important features.
        
        Args:
            df: DataFrame to transform.
        
        Returns:
            DataFrame with only selected features (plus metadata/target).
        """
        if not self._is_fitted:
            raise RuntimeError("Selector must be fitted before transform")
        
        # Keep metadata and target columns
        metadata_cols = [col for col in [TIMESTAMP, TICKER] if col in df.columns]
        target_cols = [TARGET] if TARGET in df.columns else []
        
        final_cols = metadata_cols + self._selected_features + target_cols
        
        return df[final_cols].copy()
    
    def get_result(self) -> FeatureSelectionResult:
        """Get detailed result with importance scores."""
        if not self._is_fitted:
            raise RuntimeError("Selector must be fitted first")
        
        return FeatureSelectionResult(
            selected_features=self._selected_features,
            dropped_features=self._dropped_features,
            metadata={
                "importances": self._importances,
                "model_type": self.model_type,
                "n_features": self.n_features,
                "importance_threshold": self.importance_threshold,
            }
        )


def select_by_tree_importance(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_features: int | None = None,
    importance_threshold: float = 0.001,
    model_type: str = "lightgbm",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Convenience function for tree-based feature selection.
    
    Args:
        train_df: Training DataFrame with features and target.
        test_df: Test DataFrame.
        n_features: Number of top features to keep.
        importance_threshold: Minimum importance if n_features is None.
        model_type: 'lightgbm' or 'xgboost'.
    
    Returns:
        Tuple of (train_selected, test_selected, importances_dict).
    """
    selector = TreeImportanceSelector(
        n_features=n_features,
        importance_threshold=importance_threshold,
        model_type=model_type,
    )
    
    selector.fit(train_df)
    train_selected = selector.transform(train_df)
    test_selected = selector.transform(test_df)
    result = selector.get_result()
    
    return train_selected, test_selected, result.metadata["importances"]
