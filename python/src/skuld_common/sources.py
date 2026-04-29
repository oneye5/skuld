"""Source-legend loader for the long-format dataset.

Numeric source IDs in `data_long.csv` are loaded from the sibling
`source_legend.csv`. This module is intentionally agnostic about which
sources mean "prices", "fundamentals", etc. — that policy lives with the
consumer (e.g. `skuld_research.data.config`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LEGEND_FILENAME = "source_legend.csv"


def load_source_legend(path: Path) -> dict[str, int]:
    """Load `name -> id` mapping from a source legend CSV.

    Args:
        path: Path to source_legend.csv (columns: id, name).

    Returns:
        Mapping from source name to numeric id.

    Raises:
        FileNotFoundError: if the legend file does not exist.
        ValueError: if the legend has duplicate names or missing columns.
    """
    df = pd.read_csv(path)
    missing = {"id", "name"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if df["name"].duplicated().any():
        dups = df.loc[df["name"].duplicated(), "name"].tolist()
        raise ValueError(f"{path} has duplicate source names: {dups}")
    return dict(zip(df["name"], df["id"].astype(int), strict=True))


def default_legend_path(data_csv_path: Path) -> Path:
    """Return the conventional legend path beside a data CSV."""
    return Path(data_csv_path).with_name(LEGEND_FILENAME)
