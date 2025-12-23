"""Data preparation module exports."""

import sys
from pathlib import Path

# Handle hyphenated directory names by importing directly
_base = Path(__file__).parent

# Import from long-to-wide
sys.path.insert(0, str(_base / "long-to-wide"))
from converter import long_to_wide

# Import from data-splitting
sys.path.insert(0, str(_base / "data-splitting" / "train-test"))
from splitter import split_by_timestamp, TrainTestSplit

# Import from labeling
sys.path.insert(0, str(_base / "labeling"))
from labeler import create_labels

# Import from transformations
sys.path.insert(0, str(_base / "transformations"))
from macro_prefix import add_macro_prefix
from imputation import impute_data, compute_imputation_stats, ImputationStats
from feature_engineering import add_cyclical_time_features
from scaling import (
    fit_scalers,
    transform_data,
    save_scalers,
    load_scalers,
    ScalerSet,
    get_macro_columns,
    get_ticker_columns,
)

# Import from feature-selection
sys.path.insert(0, str(_base / "feature-selection"))
from filter_selection import select_features, get_feature_columns, compute_feature_stats
from model_based_selection import TreeImportanceSelector, select_by_tree_importance
from dimensionality_reduction import (
    PCATransformer,
    add_pca_features,
    reduce_with_pca,
)
from sparse_handling import (
    SparseConfig,
    SparseHandlingResult,
    analyze_sparsity,
    forward_fill_sparse,
    aggregate_correlated_sparse,
    identify_sparse_groups,
    handle_sparse_features,
    select_representative_features,
)

__all__ = [
    "long_to_wide",
    "split_by_timestamp",
    "TrainTestSplit",
    "create_labels",
    "add_macro_prefix",
    "impute_data",
    "compute_imputation_stats",
    "ImputationStats",
    "add_cyclical_time_features",
    "fit_scalers",
    "transform_data",
    "save_scalers",
    "load_scalers",
    "ScalerSet",
    "get_macro_columns",
    "get_ticker_columns",
    # Feature selection
    "select_features",
    "get_feature_columns",
    "compute_feature_stats",
    "TreeImportanceSelector",
    "select_by_tree_importance",
    # Dimensionality reduction
    "PCATransformer",
    "add_pca_features",
    "reduce_with_pca",
    # Sparse handling
    "SparseConfig",
    "SparseHandlingResult",
    "analyze_sparsity",
    "forward_fill_sparse",
    "aggregate_correlated_sparse",
    "identify_sparse_groups",
    "handle_sparse_features",
    "select_representative_features",
]
