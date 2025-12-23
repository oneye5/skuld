"""Base classes and interfaces for feature selection methods.

This module defines the interface that all feature selection methods
should implement, enabling consistent usage across different approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass
class FeatureSelectionResult:
    """Result of a feature selection operation.
    
    Attributes:
        selected_features: List of feature names that were kept.
        dropped_features: List of feature names that were removed.
        metadata: Dict with method-specific information (e.g., importance scores).
    """
    selected_features: list[str]
    dropped_features: list[str]
    metadata: dict


class BaseFeatureSelector(ABC):
    """Abstract base class for feature selection methods.
    
    All feature selection implementations should inherit from this class
    and implement the `fit` and `transform` methods.
    
    Example usage:
        selector = SomeFeatureSelector(threshold=0.01)
        selector.fit(train_df)
        train_selected = selector.transform(train_df)
        test_selected = selector.transform(test_df)
    """
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "BaseFeatureSelector":
        """Fit the selector on training data.
        
        Args:
            df: Training DataFrame with features.
            y: Optional target variable (required for some methods).
        
        Returns:
            self for method chaining.
        """
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame by selecting features.
        
        Args:
            df: DataFrame to transform.
        
        Returns:
            DataFrame with only selected features.
        """
        pass
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            df: Training DataFrame with features.
            y: Optional target variable.
        
        Returns:
            Transformed DataFrame with selected features.
        """
        return self.fit(df, y).transform(df)
    
    @abstractmethod
    def get_result(self) -> FeatureSelectionResult:
        """Get detailed result of feature selection.
        
        Returns:
            FeatureSelectionResult with selected/dropped features and metadata.
        """
        pass


def select_features_with_selector(
    selector: BaseFeatureSelector,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureSelectionResult]:
    """Convenience function to apply a selector to train and test data.
    
    Args:
        selector: Feature selector instance.
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        y_train: Optional target for training data.
    
    Returns:
        Tuple of (train_selected, test_selected, result).
    """
    selector.fit(train_df, y_train)
    train_selected = selector.transform(train_df)
    test_selected = selector.transform(test_df)
    result = selector.get_result()
    
    return train_selected, test_selected, result
