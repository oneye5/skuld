"""Pytest configuration for ml-pipeline tests."""

import sys
from pathlib import Path

# Add the ml-pipeline root to the path
ml_pipeline_root = Path(__file__).parent
sys.path.insert(0, str(ml_pipeline_root))

# Add subdirectories with hyphenated names
sys.path.insert(0, str(ml_pipeline_root / "data-preparation"))
sys.path.insert(0, str(ml_pipeline_root / "data-preparation" / "transformations"))
sys.path.insert(0, str(ml_pipeline_root / "data-preparation" / "long-to-wide"))
sys.path.insert(0, str(ml_pipeline_root / "data-preparation" / "data-splitting" / "train-test"))
sys.path.insert(0, str(ml_pipeline_root / "data-preparation" / "labeling"))
sys.path.insert(0, str(ml_pipeline_root / "data-preparation" / "feature-selection"))
sys.path.insert(0, str(ml_pipeline_root / "evaluation"))
sys.path.insert(0, str(ml_pipeline_root / "evaluation" / "model-evaluation"))
sys.path.insert(0, str(ml_pipeline_root / "evaluation" / "trade-simulation"))
