from pathlib import Path

def get_skuld_root() -> Path:
    """
    Return the absolute path to the /skuld project root.
    Works regardless of the script location.
    """
    return Path(__file__).resolve().parents[3]
