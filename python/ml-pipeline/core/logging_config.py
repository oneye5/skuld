"""Logging configuration for the ML pipeline.

This module provides a consistent logging setup across the pipeline with:
- Structured log formatting
- File and console handlers
- Performance timing utilities
- Log level configuration
"""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional, Callable, Any


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Custom log format with timestamp, level, and module
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log level (can be overridden)
DEFAULT_LOG_LEVEL = logging.INFO


def setup_logging(
    level: int = DEFAULT_LOG_LEVEL,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Set up logging for the pipeline.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
        console: Whether to log to console.
    
    Returns:
        Configured root logger.
    """
    # Get root logger for the pipeline
    logger = logging.getLogger("skuld")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Module name (typically __name__).
    
    Returns:
        Logger instance.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Processing data...")
    """
    return logging.getLogger(f"skuld.{name}")


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@contextmanager
def log_timing(operation: str, logger: Optional[logging.Logger] = None):
    """Context manager to log operation timing.
    
    Args:
        operation: Name of the operation being timed.
        logger: Logger to use (defaults to skuld logger).
    
    Usage:
        with log_timing("feature engineering"):
            df = add_features(df)
    """
    if logger is None:
        logger = get_logger("timing")
    
    start_time = time.perf_counter()
    logger.info(f"Starting: {operation}")
    
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        
        # Format time appropriately
        if elapsed < 1:
            time_str = f"{elapsed * 1000:.1f}ms"
        elif elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m {seconds:.1f}s"
        
        logger.info(f"Completed: {operation} ({time_str})")


def timed(func: Callable) -> Callable:
    """Decorator to log function execution time.
    
    Usage:
        @timed
        def slow_function():
            ...
    """
    logger = get_logger("timing")
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    return wrapper


# =============================================================================
# STRUCTURED LOGGING HELPERS
# =============================================================================

def log_dataframe_info(
    df,
    name: str,
    logger: Optional[logging.Logger] = None,
):
    """Log DataFrame shape and memory usage.
    
    Args:
        df: DataFrame to log info about.
        name: Name/description of the DataFrame.
        logger: Logger to use.
    """
    if logger is None:
        logger = get_logger("data")
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols ({memory_mb:.1f} MB)")


def log_metrics(
    metrics: dict,
    prefix: str = "",
    logger: Optional[logging.Logger] = None,
):
    """Log a dictionary of metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric name -> value.
        prefix: Optional prefix for the log message.
        logger: Logger to use.
    """
    if logger is None:
        logger = get_logger("metrics")
    
    header = f"{prefix} Metrics" if prefix else "Metrics"
    logger.info(f"=== {header} ===")
    
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")


def log_config(
    config: dict,
    logger: Optional[logging.Logger] = None,
):
    """Log configuration parameters.
    
    Args:
        config: Configuration dictionary.
        logger: Logger to use.
    """
    if logger is None:
        logger = get_logger("config")
    
    logger.info("=== Configuration ===")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


# =============================================================================
# PROGRESS LOGGING
# =============================================================================

class ProgressLogger:
    """Simple progress logger for iterative operations.
    
    Usage:
        progress = ProgressLogger(total=100, desc="Processing")
        for item in items:
            process(item)
            progress.update()
        progress.finish()
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        logger: Optional[logging.Logger] = None,
        log_every: int = 10,
    ):
        """Initialize progress logger.
        
        Args:
            total: Total number of items.
            desc: Description of the operation.
            logger: Logger to use.
            log_every: Log progress every N percent.
        """
        self.total = total
        self.desc = desc
        self.logger = logger or get_logger("progress")
        self.log_every = log_every
        self.current = 0
        self.start_time = time.perf_counter()
        self.last_logged_pct = -log_every
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        
        if self.total > 0:
            pct = (self.current / self.total) * 100
            if pct - self.last_logged_pct >= self.log_every:
                elapsed = time.perf_counter() - self.start_time
                rate = self.current / elapsed if elapsed > 0 else 0
                eta = (self.total - self.current) / rate if rate > 0 else 0
                
                self.logger.info(
                    f"{self.desc}: {pct:.0f}% ({self.current}/{self.total}) "
                    f"[{elapsed:.1f}s elapsed, ETA {eta:.1f}s]"
                )
                self.last_logged_pct = pct
    
    def finish(self):
        """Log completion message."""
        elapsed = time.perf_counter() - self.start_time
        self.logger.info(
            f"{self.desc}: Complete! {self.total} items in {elapsed:.1f}s"
        )


# =============================================================================
# PIPELINE LOGGING SHORTCUTS
# =============================================================================

def log_window_start(window_id: int, total_windows: int):
    """Log the start of a rolling window."""
    logger = get_logger("pipeline")
    logger.info(f"=" * 50)
    logger.info(f"Window {window_id + 1}/{total_windows}")
    logger.info(f"=" * 50)


def log_window_result(
    window_id: int,
    train_size: int,
    test_size: int,
    ic: float,
    sharpe: Optional[float] = None,
):
    """Log results from a single window."""
    logger = get_logger("pipeline")
    msg = f"Window {window_id + 1}: train={train_size:,}, test={test_size:,}, IC={ic:.4f}"
    if sharpe is not None:
        msg += f", Sharpe={sharpe:.2f}"
    logger.info(msg)


def log_pipeline_summary(
    mean_ic: float,
    icir: float,
    sharpe: float,
    n_windows: int,
):
    """Log final pipeline summary."""
    logger = get_logger("pipeline")
    logger.info("=" * 50)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Windows:     {n_windows}")
    logger.info(f"  Mean IC:     {mean_ic:.4f}")
    logger.info(f"  ICIR:        {icir:.4f}")
    logger.info(f"  Sharpe:      {sharpe:.2f}")
    logger.info("=" * 50)
