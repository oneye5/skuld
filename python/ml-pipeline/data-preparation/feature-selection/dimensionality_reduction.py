"""Dimensionality reduction methods for feature transformation.

These methods transform features into a lower-dimensional space,
capturing the most important variance in the data.

Methods included:
- PCA (Principal Component Analysis)
- PCA augmentation (add components as features, keep originals)

Note: These transform features rather than select them.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from config.column_names import TIMESTAMP, TICKER, TARGET


@dataclass
class PCAResult:
    """Result of PCA transformation.
    
    Attributes:
        n_components: Number of components used.
        explained_variance_ratio: Variance explained by each component.
        total_variance_explained: Sum of variance ratios.
        component_names: Names of the PCA feature columns.
    """
    n_components: int
    explained_variance_ratio: list[float]
    total_variance_explained: float
    component_names: list[str]


class PCATransformer:
    """PCA-based dimensionality reduction.
    
    Can either replace features with principal components or augment
    existing features by adding components alongside.
    
    Args:
        n_components: Number of components to keep. Can be:
            - int: exact number of components
            - float (0-1): variance to explain (selects components automatically)
            - None: keep all components
        augment: If True, add PCA features alongside originals. 
                 If False, replace features with PCA components.
        prefix: Prefix for PCA feature names.
    """
    
    def __init__(
        self,
        n_components: int | float | None = 10,
        augment: bool = True,
        prefix: str = "PCA_",
    ):
        self.n_components = n_components
        self.augment = augment
        self.prefix = prefix
        
        self._pca = None
        self._feature_cols: list[str] = []
        self._is_fitted = False
        self._result: PCAResult | None = None
    
    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns (excluding metadata and target)."""
        exclude_cols = {TIMESTAMP, TICKER, TARGET, "index"}
        return [col for col in df.columns if col not in exclude_cols]
    
    def fit(self, df: pd.DataFrame) -> "PCATransformer":
        """Fit PCA on the feature data.
        
        Args:
            df: Training DataFrame with features.
        
        Returns:
            self for method chaining.
        """
        from sklearn.decomposition import PCA
        
        self._feature_cols = self._get_feature_columns(df)
        X = df[self._feature_cols].copy()
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Fit PCA
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(X)
        
        # Store result
        n_actual = self._pca.n_components_
        self._result = PCAResult(
            n_components=n_actual,
            explained_variance_ratio=self._pca.explained_variance_ratio_.tolist(),
            total_variance_explained=sum(self._pca.explained_variance_ratio_),
            component_names=[f"{self.prefix}{i+1}" for i in range(n_actual)],
        )
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using fitted PCA.
        
        Args:
            df: DataFrame to transform.
        
        Returns:
            DataFrame with PCA features (augmented or replaced).
        """
        if not self._is_fitted:
            raise RuntimeError("PCATransformer must be fitted before transform")
        
        # Get feature matrix
        X = df[self._feature_cols].copy()
        X = X.fillna(0)
        
        # Transform
        pca_values = self._pca.transform(X)
        
        # Create PCA DataFrame
        pca_df = pd.DataFrame(
            pca_values,
            columns=self._result.component_names,
            index=df.index,
        )
        
        if self.augment:
            # Add PCA features alongside originals
            result = pd.concat([df, pca_df], axis=1)
        else:
            # Replace features with PCA components
            metadata_cols = [col for col in [TIMESTAMP, TICKER] if col in df.columns]
            target_cols = [TARGET] if TARGET in df.columns else []
            
            result = pd.concat([
                df[metadata_cols],
                pca_df,
                df[target_cols] if target_cols else pd.DataFrame(),
            ], axis=1)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def get_result(self) -> PCAResult:
        """Get PCA result with variance explained."""
        if not self._is_fitted:
            raise RuntimeError("PCATransformer must be fitted first")
        return self._result


def add_pca_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_components: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, PCAResult]:
    """Add PCA features to train and test data (augmentation mode).
    
    Fits PCA on training data only, then transforms both.
    Original features are preserved.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        n_components: Number of PCA components to add.
    
    Returns:
        Tuple of (train_with_pca, test_with_pca, pca_result).
    """
    transformer = PCATransformer(
        n_components=n_components,
        augment=True,
    )
    
    transformer.fit(train_df)
    train_out = transformer.transform(train_df)
    test_out = transformer.transform(test_df)
    result = transformer.get_result()
    
    return train_out, test_out, result


def reduce_with_pca(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variance_to_explain: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, PCAResult]:
    """Replace features with PCA components explaining specified variance.
    
    Fits PCA on training data only, then transforms both.
    Original features are replaced with components.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        variance_to_explain: Target variance to explain (0-1).
    
    Returns:
        Tuple of (train_pca, test_pca, pca_result).
    """
    transformer = PCATransformer(
        n_components=variance_to_explain,
        augment=False,
    )
    
    transformer.fit(train_df)
    train_out = transformer.transform(train_df)
    test_out = transformer.transform(test_df)
    result = transformer.get_result()
    
    return train_out, test_out, result
