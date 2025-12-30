"""Ranking model wrappers for Learning-to-Rank stock prediction.

This module provides wrappers for ranking models (LGBMRanker, XGBRanker) with
a sklearn-like interface. The primary model is LightGBM's LGBMRanker which
uses LambdaRank/LambdaMART for NDCG optimization.

Critical Implementation Notes:
1. Data MUST be sorted by timestamp (group) before training
2. The `group` parameter is a list of group sizes: [n_stocks_ts1, n_stocks_ts2, ...]
3. sum(group) must equal len(X)
4. LGBMRanker requires INTEGER labels (relevance grades) - continuous returns
   are converted to quintile-based relevance grades (0-4) for training
5. Higher relevance = better (higher predicted return)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from config.columns import TIMESTAMP, TICKER


# =============================================================================
# LABEL CONVERSION FOR RANKING
# =============================================================================

def returns_to_relevance_grades(
    y: pd.Series | np.ndarray,
    n_grades: int = 5,
) -> np.ndarray:
    """Convert continuous returns to integer relevance grades.
    
    LGBMRanker requires integer labels representing relevance grades.
    We convert continuous returns to grades based on percentile ranking.
    
    Args:
        y: Continuous target values (forward returns).
        n_grades: Number of relevance grades (default 5 for quintiles).
    
    Returns:
        Integer array with values 0 to (n_grades - 1).
        Higher values = higher returns = better.
    
    Raises:
        ValueError: If input contains NaN values.
    """
    y_arr = np.asarray(y)
    
    # Check for NaN values - LGBMRanker requires non-negative labels
    if np.isnan(y_arr).any():
        raise ValueError(
            f"Input contains {np.isnan(y_arr).sum()} NaN values. "
            "Forward returns must not contain NaN for ranking."
        )
    
    # Use percentile-based assignment to handle ties and ensure balance
    try:
        grades = pd.qcut(y_arr, q=n_grades, labels=False, duplicates='drop')
    except ValueError:
        # Too few unique values, fall back to rank-based assignment
        ranks = pd.Series(y_arr).rank(method='first')
        grades = pd.cut(ranks, bins=n_grades, labels=False)
    
    # Final safety check - grades should not have NaN
    grades_arr = np.asarray(grades)
    if np.isnan(grades_arr).any():
        raise ValueError("Grade conversion produced NaN values unexpectedly.")
    
    return grades_arr.astype(np.int32)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RankerConfig:
    """Configuration for ranking models.
    
    Attributes:
        n_estimators: Number of boosting iterations.
        learning_rate: Learning rate (shrinkage).
        num_leaves: Maximum number of leaves per tree.
        max_depth: Maximum tree depth (-1 = no limit).
        min_child_samples: Minimum samples in a leaf.
        subsample: Fraction of samples for each iteration.
        colsample_bytree: Fraction of features for each tree.
        objective: Ranking objective ('lambdarank', 'rank_xendcg', 'pairwise').
        metric: Evaluation metric ('ndcg', 'map').
        eval_at: Positions to evaluate NDCG at.
        random_state: Random seed for reproducibility.
        device: Device for training ('cpu' or 'gpu').
        early_stopping_rounds: Early stopping rounds. None to disable.
    """
    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: Tuple[int, ...] = (5, 10, 20)
    random_state: int = 42
    verbose: int = -1
    device: str = "cpu"
    early_stopping_rounds: Optional[int] = None


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class BaseRanker(ABC):
    """Abstract base class for ranking models."""
    
    @abstractmethod
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        group: List[int],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        group_val: Optional[List[int]] = None,
    ) -> "BaseRanker":
        """Fit the ranking model.
        
        Args:
            X: Feature DataFrame, MUST be sorted by timestamp.
            y: Target Series (forward returns), same order as X.
            group: List of group sizes. sum(group) == len(X).
            X_val: Optional validation features.
            y_val: Optional validation targets.
            group_val: Optional validation group sizes.
        
        Returns:
            Self for method chaining.
        
        Raises:
            ValueError: If sum(group) != len(X).
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Array of ranking scores (higher = better predicted rank).
        
        Raises:
            RuntimeError: If model not fitted.
        """
        pass
    
    @abstractmethod
    def feature_importances(self) -> np.ndarray:
        """Get feature importances.
        
        Returns:
            Array of feature importance values.
        
        Raises:
            RuntimeError: If model not fitted.
        """
        pass


# =============================================================================
# LIGHTGBM RANKER WRAPPER
# =============================================================================

class LightGBMRankerWrapper(BaseRanker):
    """Wrapper for LightGBM LGBMRanker with sklearn-like interface.
    
    LGBMRanker uses LambdaRank/LambdaMART algorithm to optimize NDCG
    (Normalized Discounted Cumulative Gain) directly via gradient approximation.
    
    Example:
        >>> from learner.ranking import LightGBMRankerWrapper, RankerConfig
        >>> config = RankerConfig(n_estimators=100)
        >>> ranker = LightGBMRankerWrapper(config)
        >>> ranker.fit(X_train, y_train, groups_train)
        >>> predictions = ranker.predict(X_test)
    """
    
    def __init__(self, config: Optional[RankerConfig] = None):
        """Initialize the ranker.
        
        Args:
            config: RankerConfig with model parameters. Uses defaults if None.
        """
        self.config = config or RankerConfig()
        self.model = None
        self._feature_names: Optional[List[str]] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group: List[int],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        group_val: Optional[List[int]] = None,
        n_relevance_grades: int = 5,
    ) -> "LightGBMRankerWrapper":
        """Fit the LGBMRanker model.
        
        Note: LGBMRanker requires integer relevance grades, not continuous values.
        Continuous returns are automatically converted to relevance grades (0 to n-1)
        based on quintile ranking within each group.
        
        Args:
            X: Feature DataFrame, MUST be sorted by timestamp.
            y: Target Series (forward returns - will be converted to grades).
            group: List of group sizes per timestamp.
            X_val: Optional validation features.
            y_val: Optional validation targets.
            group_val: Optional validation group sizes.
            n_relevance_grades: Number of relevance grades (default 5).
        
        Returns:
            Self for method chaining.
        
        Raises:
            ValueError: If sum(group) != len(X).
        """
        from lightgbm import LGBMRanker
        
        # Validate group sizes
        if sum(group) != len(X):
            raise ValueError(
                f"sum(group)={sum(group)} != len(X)={len(X)}. "
                "Ensure data is sorted by timestamp and group sizes are correct."
            )
        
        # Store feature names
        self._feature_names = list(X.columns)
        
        # Convert continuous returns to relevance grades per group
        y_grades = self._convert_to_relevance_grades(y, group, n_relevance_grades)
        
        # Create model
        model_params = {
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "min_child_samples": self.config.min_child_samples,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "objective": self.config.objective,
            "metric": self.config.metric,
            "random_state": self.config.random_state,
            "n_jobs": -1,
            "verbose": self.config.verbose,
        }
        
        # Add device (GPU support)
        if self.config.device == "gpu":
            model_params["device"] = "gpu"
        
        self.model = LGBMRanker(**model_params)
        
        # Prepare validation data if provided
        eval_set = None
        eval_group = None
        callbacks = None
        
        if X_val is not None and y_val is not None and group_val is not None:
            if sum(group_val) != len(X_val):
                raise ValueError(
                    f"sum(group_val)={sum(group_val)} != len(X_val)={len(X_val)}"
                )
            y_val_grades = self._convert_to_relevance_grades(y_val, group_val, n_relevance_grades)
            eval_set = [(X_val, y_val_grades)]
            eval_group = [group_val]
            
            # Add early stopping callback if configured
            if self.config.early_stopping_rounds is not None:
                from lightgbm import early_stopping
                callbacks = [early_stopping(self.config.early_stopping_rounds, verbose=False)]
        
        # Fit model
        fit_params = {
            "X": X,
            "y": y_grades,
            "group": group,
            "eval_set": eval_set,
            "eval_group": eval_group,
            "eval_at": self.config.eval_at,
        }
        if callbacks:
            fit_params["callbacks"] = callbacks
        
        self.model.fit(**fit_params)
        
        return self
    
    def _convert_to_relevance_grades(
        self,
        y: pd.Series | np.ndarray,
        group: List[int],
        n_grades: int,
    ) -> np.ndarray:
        """Convert continuous returns to relevance grades within each group.
        
        Args:
            y: Continuous target values.
            group: Group sizes.
            n_grades: Number of relevance grades.
        
        Returns:
            Integer array with relevance grades (0 to n_grades-1).
        """
        y_arr = np.asarray(y)
        grades = np.zeros(len(y_arr), dtype=np.int32)
        
        start_idx = 0
        for group_size in group:
            end_idx = start_idx + group_size
            group_y = y_arr[start_idx:end_idx]
            
            # Convert this group's returns to grades
            group_grades = returns_to_relevance_grades(group_y, n_grades)
            grades[start_idx:end_idx] = group_grades
            
            start_idx = end_idx
        
        return grades
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Array of ranking scores (higher = better predicted rank).
        
        Raises:
            RuntimeError: If model not fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def feature_importances(self) -> np.ndarray:
        """Get feature importances.
        
        Returns:
            Array of feature importance values (gain-based).
        
        Raises:
            RuntimeError: If model not fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.model.feature_importances_
    
    def feature_importance_df(self) -> pd.DataFrame:
        """Get feature importances as a sorted DataFrame.
        
        Returns:
            DataFrame with columns ['feature', 'importance'], sorted descending.
        
        Raises:
            RuntimeError: If model not fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        importances = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self._feature_names,
            'importance': importances,
        })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_group_from_timestamps(
    df: pd.DataFrame, 
    timestamp_col: str = TIMESTAMP,
) -> List[int]:
    """Build group sizes from a DataFrame sorted by timestamp.
    
    The `group` parameter for LGBMRanker is a list where each element
    represents the number of samples (stocks) in that group (timestamp).
    
    Args:
        df: DataFrame that MUST be sorted by timestamp_col.
        timestamp_col: Name of timestamp column.
    
    Returns:
        List of group sizes: [n_stocks_ts1, n_stocks_ts2, ...].
        sum(result) == len(df).
    
    Raises:
        ValueError: If DataFrame is not sorted by timestamp.
    
    Example:
        >>> df = pd.DataFrame({'timestamp': [1, 1, 1, 2, 2], 'ticker': list('ABCAB')})
        >>> build_group_from_timestamps(df)
        [3, 2]
    """
    timestamps = df[timestamp_col].values
    
    # Verify data is sorted
    if len(timestamps) > 1:
        if not np.all(timestamps[:-1] <= timestamps[1:]):
            raise ValueError(
                f"DataFrame must be sorted by {timestamp_col} for LGBMRanker. "
                "Use df.sort_values(timestamp_col) before building groups."
            )
    
    # Count samples per timestamp
    return df.groupby(timestamp_col).size().tolist()


def prepare_ranking_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    timestamp_col: str = TIMESTAMP,
) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
    """Prepare DataFrame for ranking model training.
    
    Sorts data by timestamp and builds the group parameter required by LGBMRanker.
    Drops rows with NaN in target column to ensure valid labels.
    
    Args:
        df: DataFrame with features, target, and timestamp.
        feature_cols: List of feature column names.
        target_col: Name of target column (forward returns).
        timestamp_col: Name of timestamp column.
    
    Returns:
        Tuple of (X, y, groups) ready for ranker.fit().
        - X: Feature DataFrame, sorted by timestamp
        - y: Target Series, same order as X
        - groups: List of group sizes
    
    Example:
        >>> X, y, groups = prepare_ranking_data(df, ['feat1', 'feat2'], 'return')
        >>> ranker.fit(X, y, groups)
    """
    # Drop rows with NaN in target - LGBMRanker requires valid labels
    df_clean = df.dropna(subset=[target_col]).copy()
    
    if len(df_clean) < len(df):
        dropped = len(df) - len(df_clean)
        import warnings
        warnings.warn(f"Dropped {dropped} rows with NaN in target column '{target_col}'")
    
    # Sort by timestamp
    df_sorted = df_clean.sort_values(timestamp_col).reset_index(drop=True)
    
    # Extract features and target
    X = df_sorted[feature_cols].copy()
    y = df_sorted[target_col].copy()
    
    # Build groups
    groups = build_group_from_timestamps(df_sorted, timestamp_col)
    
    return X, y, groups


def filter_min_stocks_per_timestamp(
    df: pd.DataFrame,
    min_stocks: int,
    timestamp_col: str = TIMESTAMP,
) -> pd.DataFrame:
    """Filter out timestamps with fewer than min_stocks.
    
    LGBMRanker performs better with sufficient items per group.
    This function removes timestamps that don't meet the minimum requirement.
    
    Args:
        df: DataFrame with timestamp column.
        min_stocks: Minimum stocks required per timestamp.
        timestamp_col: Name of timestamp column.
    
    Returns:
        Filtered DataFrame with only timestamps having >= min_stocks.
    """
    counts = df.groupby(timestamp_col).size()
    valid_timestamps = counts[counts >= min_stocks].index
    
    return df[df[timestamp_col].isin(valid_timestamps)].copy()
