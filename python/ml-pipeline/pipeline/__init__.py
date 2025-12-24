"""Pipeline package - orchestration for single and rolling window runs."""

from pipeline.single_window import run_single_window, SingleWindowResult
from pipeline.rolling_window import run_rolling_windows, RollingWindowResult

__all__ = [
    "run_single_window",
    "SingleWindowResult",
    "run_rolling_windows",
    "RollingWindowResult",
]
