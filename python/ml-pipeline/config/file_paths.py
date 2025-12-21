"""File path constants used throughout the pipeline."""

from pathlib import Path
from datetime import datetime
from utils.file_utils import get_skuld_root

_root = get_skuld_root()

# Data paths
DATA_DIR = _root / "data"
DATA_LONG_CSV = DATA_DIR / "data_long.csv"
LEGACY_DIR = DATA_DIR / "legacy"

# ML pipeline paths
ML_PIPELINE_DIR = _root / "python" / "ml-pipeline"

# Output paths
OUTPUT_DIR = ML_PIPELINE_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
SCALERS_DIR = OUTPUT_DIR / "scalers"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

# Current run directory (set dynamically)
_current_run_dir: Path | None = None


def get_run_dir() -> Path:
    """Get current run directory, creating timestamped one if needed."""
    global _current_run_dir
    if _current_run_dir is None:
        _current_run_dir = create_run_dir()
    return _current_run_dir


def create_run_dir() -> Path:
    """Create a timestamped directory for this run."""
    global _current_run_dir
    timestamp = datetime.now().strftime("%d-%m-%Y-%H%M")
    run_dir = OUTPUT_DIR / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    _current_run_dir = run_dir
    return run_dir


def get_run_evaluation_dir() -> Path:
    """Get evaluation directory for current run."""
    return get_run_dir() / "evaluation"


def get_run_predictions_dir() -> Path:
    """Get predictions directory for current run."""
    return get_run_dir() / "predictions"


def get_run_models_dir() -> Path:
    """Get models directory for current run."""
    return get_run_dir() / "models"


def get_run_scalers_dir() -> Path:
    """Get scalers directory for current run."""
    return get_run_dir() / "scalers"


# Ensure output directories exist
def ensure_output_dirs():
    """Create output directories if they don't exist."""
    # Legacy directories
    for dir_path in [OUTPUT_DIR, MODELS_DIR, PREDICTIONS_DIR, SCALERS_DIR, EVALUATION_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Run-specific directories
    run_dir = get_run_dir()
    for subdir in ["evaluation", "predictions", "models", "scalers", "evaluation/figures"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)