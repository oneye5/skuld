"""Load BacktestSpec from YAML files."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from skuld_research.config.spec import BacktestSpec


class SpecValidationError(ValueError):
    """Raised when a spec YAML fails validation."""
    pass


def find_python_root() -> Path:
    """Find the python/ project root by walking up from this file.
    
    Returns:
        Path to directory containing pyproject.toml.
    
    Raises:
        FileNotFoundError: if no pyproject.toml found in parent chain.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find pyproject.toml in parent chain")


def load_spec(path: Path | str) -> BacktestSpec:
    """Load and validate a BacktestSpec from a YAML file.
    
    Args:
        path: Path to YAML file.
    
    Returns:
        Validated BacktestSpec.
    
    Raises:
        SpecValidationError: if YAML is invalid or fails validation.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return BacktestSpec.model_validate(data)
    except (yaml.YAMLError, ValidationError) as e:
        raise SpecValidationError(f"Failed to load spec from {path}: {e}") from e
    except FileNotFoundError as e:
        raise SpecValidationError(f"Spec file not found: {path}") from e


def iter_preregistered_specs(root: Path | None = None) -> list[Path]:
    """Find all pre-registered spec YAML files.
    
    Args:
        root: Python project root (defaults to auto-detected root).
    
    Returns:
        Sorted list of paths to *.yaml under configs/preregistered/.
    """
    if root is None:
        root = find_python_root()
    
    preregistered_dir = root / "configs" / "preregistered"
    if not preregistered_dir.exists():
        return []
    
    return sorted(preregistered_dir.glob("*.yaml"))
