"""Feature selection module exports."""

from filter_selection import (
    select_features,
    get_feature_columns,
    compute_feature_stats,
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
    # Filter selection
    "select_features",
    "get_feature_columns",
    "compute_feature_stats",
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
