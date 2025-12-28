"""Core package - data loading, transformation, and preprocessing."""

from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix
from core.target_builder import compute_forward_returns, FORWARD_RETURN
from core.splitter import split_by_timestamp, TrainTestSplit, calculate_window_timestamps
from core.scaler import fit_scaler, transform_data
from core.preprocessor import preprocess_data

__all__ = [
    "load_long_data",
    "long_to_wide",
    "add_macro_prefix",
    "compute_forward_returns",
    "FORWARD_RETURN",
    "split_by_timestamp",
    "TrainTestSplit",
    "calculate_window_timestamps",
    "fit_scaler",
    "transform_data",
    "preprocess_data",
]
