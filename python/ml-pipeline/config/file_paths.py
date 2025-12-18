"""File path constants used throughout the pipeline."""

from pathlib import Path
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

# Ensure output directories exist
def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, MODELS_DIR, PREDICTIONS_DIR, SCALERS_DIR, EVALUATION_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)