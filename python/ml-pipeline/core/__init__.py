"""Core package - data loading, transformation, preprocessing, and utilities."""

from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix
from core.target_builder import compute_forward_returns, FORWARD_RETURN
from core.splitter import split_by_timestamp, TrainTestSplit, calculate_window_timestamps
from core.scaler import fit_scaler, transform_data
from core.preprocessor import (
    preprocess_data,
    detect_price_anomalies,
    filter_anomalous_data,
    get_anomaly_summary,
)
from core.validation import (
    ValidationError,
    validate_dataframe,
    validate_wide_data,
    validate_no_lookahead,
    validate_groups_match_data,
    check_data_quality_report,
)
from core.experiment_tracking import (
    ExperimentManifest,
    create_experiment_manifest,
    compare_experiments,
    find_best_experiment,
)
from core.logging_config import (
    setup_logging,
    get_logger,
    log_timing,
    timed,
    log_dataframe_info,
    log_metrics,
    ProgressLogger,
)

__all__ = [
    # Data loading
    "load_long_data",
    "long_to_wide",
    "add_macro_prefix",
    # Target building
    "compute_forward_returns",
    "FORWARD_RETURN",
    # Splitting
    "split_by_timestamp",
    "TrainTestSplit",
    "calculate_window_timestamps",
    # Scaling
    "fit_scaler",
    "transform_data",
    # Preprocessing
    "preprocess_data",
    # Validation
    "ValidationError",
    "validate_dataframe",
    "validate_wide_data",
    "validate_no_lookahead",
    "validate_groups_match_data",
    "check_data_quality_report",
    # Experiment tracking
    "ExperimentManifest",
    "create_experiment_manifest",
    "compare_experiments",
    "find_best_experiment",
    # Logging
    "setup_logging",
    "get_logger",
    "log_timing",
    "timed",
    "log_dataframe_info",
    "log_metrics",
    "ProgressLogger",
]
