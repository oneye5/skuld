"""Utility functions for path management."""
from pathlib import Path


def get_skuld_root() -> Path:
    """
    Return the absolute path to the /skuld project root.
    
    Works regardless of the script location by traversing up 3 levels from the config module.
    
    Returns:
        Path: Absolute path to the project root directory.
    """
    return Path(__file__).resolve().parents[3]
