"""Efficient I/O utilities for CSV and Parquet with caching.

Provides fast data loading/saving with automatic format detection.
Parquet is used for intermediate files (10-100x faster, better compression).
CSV is used for external-facing files for easy inspection.
"""
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional, Literal

_cache = {}

FileFormat = Literal["csv", "parquet", "auto"]


def load_data(path: str, format: FileFormat = "auto", index_col: Optional[str] = None) -> pd.DataFrame:
    """Load data from CSV or Parquet with caching.
    
    Automatically detects format from file extension if format='auto'.
    Caches results to avoid redundant disk I/O.
    
    Args:
        path: Path to data file.
        format: File format ('csv', 'parquet', or 'auto').
        index_col: Optional column to set as index.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file cannot be read or format is invalid.
    """
    path_str = str(Path(path).resolve())

    if path_str in _cache:
        return _cache[path_str]

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")

    # Auto-detect format from extension
    if format == "auto":
        ext = path_obj.suffix.lower()
        if ext == ".csv":
            format = "csv"
        elif ext in [".parquet", ".pq"]:
            format = "parquet"
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")

    try:
        if format == "csv":
            df = pl.read_csv(path_obj).to_pandas()
        elif format == "parquet":
            df = pl.read_parquet(path_obj).to_pandas()
        else:
            raise ValueError(f"Invalid format: {format}. Use 'csv', 'parquet', or 'auto'.")
    except Exception as e:
        raise ValueError(f"Failed to read {format} file {path_obj}: {str(e)}")

    if index_col is not None:
        df = df.set_index(index_col)

    _cache[path_str] = df
    return df


def save_data(df: pd.DataFrame, path: str, format: FileFormat = "auto", index: bool = False) -> None:
    """Save DataFrame to CSV or Parquet with caching.
    
    Automatically detects format from file extension if format='auto'.
    Creates parent directories if needed.
    
    Args:
        df: DataFrame to save.
        path: Target file path.
        format: File format ('csv', 'parquet', or 'auto').
        index: Whether to write index column.
    
    Raises:
        ValueError: If DataFrame cannot be saved or format is invalid.
        IOError: If file cannot be written.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect format from extension
    if format == "auto":
        ext = path_obj.suffix.lower()
        if ext == ".csv":
            format = "csv"
        elif ext in [".parquet", ".pq"]:
            format = "parquet"
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")

    try:
        # Cache copy to avoid modifying original
        path_str = str(path_obj.resolve())
        _cache[path_str] = df.copy()

        # Save in requested format
        pl_df = pl.DataFrame(df)
        if format == "csv":
            pl_df.write_csv(path_obj)
        elif format == "parquet":
            pl_df.write_parquet(path_obj, compression="zstd", compression_level=3)
        else:
            raise ValueError(f"Invalid format: {format}. Use 'csv', 'parquet', or 'auto'.")
    except Exception as e:
        raise ValueError(f"Failed to save {format} to {path_obj}: {str(e)}")


def clear_cache() -> None:
    """Clear the internal data cache."""
    _cache.clear()


# Legacy CSV compatibility functions
def load_csv(path: str, index_col: Optional[str] = None) -> pd.DataFrame:
    """Legacy CSV loading function for backward compatibility."""
    return load_data(path, format="csv", index_col=index_col)


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Legacy CSV saving function for backward compatibility."""
    save_data(df, path, format="csv", index=index)
