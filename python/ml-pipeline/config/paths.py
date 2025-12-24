"""File path constants for the ML pipeline.

All paths should be defined here. Paths are resolved relative to project root.
"""

from pathlib import Path
from datetime import datetime


# =============================================================================
# PROJECT STRUCTURE
# =============================================================================
def _get_project_root() -> Path:
    """Get the project root (skuld/) directory."""
    # This file is at: skuld/python/ml-pipeline/config/paths.py
    # Project root is: skuld/
    return Path(__file__).parent.parent.parent.parent


def _get_pipeline_root() -> Path:
    """Get the ml-pipeline directory."""
    return Path(__file__).parent.parent


PROJECT_ROOT: Path = _get_project_root()
PIPELINE_ROOT: Path = _get_pipeline_root()


# =============================================================================
# DATA PATHS
# =============================================================================
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_LONG_CSV: Path = DATA_DIR / "data_long.csv"


# =============================================================================
# OUTPUT PATHS
# =============================================================================
OUTPUT_DIR: Path = PIPELINE_ROOT / "output"
RUNS_DIR: Path = OUTPUT_DIR / "runs"


# =============================================================================
# RUN DIRECTORY MANAGEMENT
# =============================================================================
_current_run_dir: Path | None = None


def get_run_dir() -> Path:
    """Get current run directory, creating timestamped one if needed.
    
    Returns:
        Path to the current run directory.
    """
    global _current_run_dir
    if _current_run_dir is None:
        _current_run_dir = _create_run_dir()
    return _current_run_dir


def _create_run_dir() -> Path:
    """Create a new timestamped run directory.
    
    Returns:
        Path to the newly created run directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def reset_run_dir() -> None:
    """Reset the run directory (for testing or new runs)."""
    global _current_run_dir
    _current_run_dir = None


def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    run_dir = get_run_dir()
    
    subdirs = ["figures"]
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
