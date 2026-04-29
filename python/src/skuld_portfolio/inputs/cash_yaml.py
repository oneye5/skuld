"""Read cash balance from YAML file."""
from __future__ import annotations

from pathlib import Path

import yaml


class CashYAMLError(ValueError):
    """Raised when cash YAML is malformed or missing required keys."""
    pass


def read_cash_yaml(path: Path) -> float:
    """Read cash balance from a YAML file.
    
    Args:
        path: Path to YAML file containing 'cash_nzd: <number>'.
    
    Returns:
        Cash balance in NZD.
    
    Raises:
        CashYAMLError: if YAML is malformed or missing 'cash_nzd' key.
        FileNotFoundError: if path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cash YAML not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise CashYAMLError(f"Failed to parse YAML from {path}: {e}") from e
    
    if not isinstance(data, dict):
        raise CashYAMLError(
            f"Cash YAML must be a YAML mapping (dict), got {type(data).__name__}"
        )
    
    if "cash_nzd" not in data:
        raise CashYAMLError(
            "Cash YAML missing required key 'cash_nzd'.\n"
            "Expected format:\n"
            "  cash_nzd: <number>\n"
            "  asof: <date>  # optional"
        )
    
    cash = data["cash_nzd"]
    try:
        return float(cash)
    except (TypeError, ValueError) as e:
        raise CashYAMLError(
            f"cash_nzd must be a number, got {type(cash).__name__}: {cash}"
        ) from e
