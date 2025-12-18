from pathlib import Path


def get_skuld_root() -> Path:
    """Return the absolute path to the /skuld project root."""
    return Path(__file__).resolve().parents[3]


def get_ml_pipeline_root() -> Path:
    """Return the absolute path to the ml-pipeline directory."""
    return get_skuld_root() / "python" / "ml-pipeline"
