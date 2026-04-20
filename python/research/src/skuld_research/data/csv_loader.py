"""Load the long-format CSV into typed DataFrames, split by source/feature.

No PIT filtering here — that's pit_loader's job. This module only parses,
pivots, and categorises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Source IDs (from data/source_legend.csv)
SRC_PRICES = 6
SRC_FUNDAMENTALS = 12

# Price features that become per-ticker columns
PRICE_FEATURE = "adj_close"
VOLUME_FEATURE = "volume"
CORPORATE_ACTION_FEATURES = {"dividend", "split"}


@dataclass
class RawData:
    """All data from the CSV, categorised but unfiltered."""

    prices: pd.DataFrame  # index=date, columns=ticker, values=adj_close
    volumes: pd.DataFrame  # index=date, columns=ticker, values=volume
    fundamentals: pd.DataFrame  # MultiIndex (ticker, publication_date), columns=feature
    macro: pd.DataFrame  # index=date, columns=feature
    corporate_actions: pd.DataFrame  # columns: ticker, ex_date, type, factor


def load_raw_csv(path: Path) -> RawData:
    """Load long-format CSV and split into categorised DataFrames.

    Args:
        path: Path to data_long.csv

    Returns:
        RawData with all observations categorised.
    """
    df = pd.read_csv(
        path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str, "src": "int8"},
    )
    # Fill NaN tickers (macro rows) with empty string
    df["ticker"] = df["ticker"].fillna("")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)

    prices = _pivot_price_feature(df, PRICE_FEATURE)
    volumes = _pivot_price_feature(df, VOLUME_FEATURE)
    fundamentals = _build_fundamentals(df)
    macro = _build_macro(df)
    corporate_actions = _build_corporate_actions(df)

    return RawData(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        macro=macro,
        corporate_actions=corporate_actions,
    )


def _pivot_price_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Pivot a single price-source feature into date x ticker."""
    mask = (df["src"] == SRC_PRICES) & (df["feature"] == feature) & (df["ticker"] != "")
    subset = df.loc[mask, ["date", "ticker", "value"]]
    if subset.empty:
        return pd.DataFrame()
    pivoted = subset.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    pivoted.index.name = "date"
    pivoted = pivoted.sort_index()
    return pivoted


def _build_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Build fundamentals with MultiIndex (ticker, publication_date)."""
    mask = (df["src"] == SRC_FUNDAMENTALS) & (df["ticker"] != "")
    subset = df.loc[mask, ["ticker", "date", "feature", "value"]]
    if subset.empty:
        return pd.DataFrame(
            columns=pd.Index([], dtype=str),
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"]),
        )
    pivoted = subset.pivot_table(
        index=["ticker", "date"], columns="feature", values="value", aggfunc="last"
    )
    pivoted.index = pivoted.index.set_names(["ticker", "publication_date"])
    return pivoted


def _build_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Build macro DataFrame: date x feature for rows with empty ticker."""
    mask = df["ticker"] == ""
    subset = df.loc[mask, ["date", "feature", "value"]]
    if subset.empty:
        return pd.DataFrame()
    pivoted = subset.pivot_table(index="date", columns="feature", values="value", aggfunc="last")
    pivoted.index.name = "date"
    pivoted = pivoted.sort_index()
    return pivoted


def _build_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Extract dividend and split rows into a flat DataFrame."""
    mask = (df["src"] == SRC_PRICES) & (df["feature"].isin(CORPORATE_ACTION_FEATURES))
    subset = df.loc[mask, ["ticker", "date", "feature", "value"]].copy()
    subset = subset.rename(columns={"date": "ex_date", "feature": "type", "value": "factor"})
    subset = subset.reset_index(drop=True)
    return subset
