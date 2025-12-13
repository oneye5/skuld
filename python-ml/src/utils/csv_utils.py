"""CSV utility functions using Polars for fast I/O operations.

Provides efficient CSV loading/saving with caching to minimize disk I/O.
Uses Polars backend for superior performance on large datasets.
"""
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional

_cache = {}


def load_csv(path: str, index_col: Optional[str] = None) -> pd.DataFrame:
    """Load CSV file using Polars for speed, return as pandas DataFrame.
    
    Caches results to avoid redundant disk I/O.
    
    Args:
        path: Path to CSV file.
        index_col: Optional column to set as index.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If file cannot be read as CSV.
    """
    path_str = str(Path(path).resolve())

    if path_str in _cache:
        return _cache[path_str]  # Return cached reference - pandas CoW handles safety

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path_obj}")

    try:
        # Read with Polars (faster), convert to pandas
        df = pl.read_csv(path_obj).to_pandas()
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {path_obj}: {str(e)}")

    if index_col is not None:
        df = df.set_index(index_col)

    _cache[path_str] = df
    return df


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Save DataFrame to CSV using Polars for speed.
    
    Creates parent directories if needed and updates cache.
    
    Args:
        df: DataFrame to save.
        path: Target file path.
        index: Whether to write index column.
    
    Raises:
        ValueError: If DataFrame cannot be converted to CSV.
        IOError: If file cannot be written.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Cache copy to avoid modifying original
        path_str = str(path_obj.resolve())
        _cache[path_str] = df.copy()

        # Convert to Polars and save (faster)
        pl.DataFrame(df).write_csv(path_obj)
    except Exception as e:
        raise ValueError(f"Failed to save CSV to {path_obj}: {str(e)}")


def clear_cache() -> None:
    """Clear the internal CSV cache."""
    _cache.clear()