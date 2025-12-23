"""Centralized path setup for modules with hyphenated directory names."""

import sys
from pathlib import Path


def setup_paths() -> Path:
    """
    Add hyphenated directories to sys.path for imports.
    
    Returns:
        Path to ml-pipeline root directory.
    """
    # Determine ml-pipeline root (works from any submodule)
    current = Path(__file__).resolve()
    ml_pipeline = current.parent.parent  # utils -> ml-pipeline
    
    paths_to_add = [
        ml_pipeline,
        ml_pipeline / "data-preparation",
        ml_pipeline / "data-preparation" / "transformations",
        ml_pipeline / "data-preparation" / "long-to-wide",
        ml_pipeline / "data-preparation" / "data-splitting" / "train-test",
        ml_pipeline / "data-preparation" / "labeling",
        ml_pipeline / "data-preparation" / "feature-selection",
        ml_pipeline / "evaluation",
        ml_pipeline / "evaluation" / "model-evaluation",
        ml_pipeline / "evaluation" / "trade-simulation",
    ]
    
    for path in paths_to_add:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    return ml_pipeline


# Auto-setup when imported
ML_PIPELINE_ROOT = setup_paths()
