import polars as pl
import pandas as pd
from pathlib import Path

_cache = {}


def load_csv(path: str, index_col=None) -> pd.DataFrame:
    """Load CSV using Polars, return as pandas DataFrame."""
    path_str = str(Path(path).resolve())

    if path_str in _cache:
        return _cache[path_str].copy()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Read with Polars (faster), convert to pandas
    df = pl.read_csv(path).to_pandas()

    if index_col is not None:
        df = df.set_index(index_col)

    _cache[path_str] = df
    return df


def save_csv(df: pd.DataFrame, path: str, index=False):
    """Save using Polars (faster writing)."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Cache before saving
    _cache[str(path_obj.resolve())] = df

    # Convert to Polars and save (faster)
    pl.DataFrame(df).write_csv(path_obj)


def clear_cache():
    _cache.clear()