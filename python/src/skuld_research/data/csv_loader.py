"""Load the long-format CSV into typed DataFrames, split by feature/shape.

Categorisation is driven entirely by what the data *is*, not by where it
came from:

  * ticker present, feature in PRICE_FEATURES         → prices / volumes
  * ticker present, feature in CORPORATE_ACTIONS      → corporate_actions
  * ticker present, any other feature                 → fundamentals
  * ticker absent                                     → macro

The `src` column in the long CSV is intentionally ignored here — it is
provenance metadata for staleness/audit reporting, not a routing key.
No PIT filtering happens in this module; that is pit_loader's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from skuld_common.features import (
    ADJ_CLOSE,
    CLOSE,
    CORPORATE_ACTIONS,
    HIGH,
    LOW,
    PRICE_FEATURES,
    VOLUME,
)


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
        path: Path to data_long.csv.

    Returns:
        RawData with all observations categorised by feature/shape.
    """
    df = pd.read_csv(
        path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str},
        usecols=["timestamp", "ticker", "feature", "value"],
    )
    df["ticker"] = df["ticker"].fillna("")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)

    has_ticker = df["ticker"] != ""
    is_price_feature = df["feature"].isin(PRICE_FEATURES)
    is_corp_action = df["feature"].isin(CORPORATE_ACTIONS)

    return RawData(
        prices=_pivot_ticker_feature(df, has_ticker & (df["feature"] == ADJ_CLOSE)),
        volumes=_pivot_ticker_feature(df, has_ticker & (df["feature"] == VOLUME)),
        fundamentals=_build_fundamentals(df, has_ticker & ~is_price_feature & ~is_corp_action),
        macro=_build_macro(df, ~has_ticker),
        corporate_actions=_build_corporate_actions(df, has_ticker & is_corp_action),
    )


def _pivot_ticker_feature(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Pivot date×ticker for a single feature mask."""
    subset = df.loc[mask, ["date", "ticker", "value"]]
    if subset.empty:
        return pd.DataFrame()
    pivoted = subset.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    pivoted.index.name = "date"
    return pivoted.sort_index()


def _build_fundamentals(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Build fundamentals with MultiIndex (ticker, publication_date)."""
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


def _build_macro(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Build macro DataFrame: date × feature for ticker-less rows."""
    subset = df.loc[mask, ["date", "feature", "value"]]
    if subset.empty:
        return pd.DataFrame()
    pivoted = subset.pivot_table(index="date", columns="feature", values="value", aggfunc="last")
    pivoted.index.name = "date"
    return pivoted.sort_index()


def _build_corporate_actions(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Extract dividend and split rows into a flat event-shaped DataFrame."""
    subset = df.loc[mask, ["ticker", "date", "feature", "value"]].copy()
    subset = subset.rename(columns={"date": "ex_date", "feature": "type", "value": "factor"})
    return subset.reset_index(drop=True)


def load_raw_ohlc(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw (unadjusted) high/low/close per ticker per date.

    Returned separately from `RawData` because OHLC is only required by the
    cost-modelling layer (Abdi-Ranaldo spread estimator). Keeping this off
    the core PIT contract avoids forcing every snapshot consumer to carry
    OHLC frames it doesn't need.

    Args:
        path: Path to data_long.csv.

    Returns:
        (high, low, close) — each is a date x ticker DataFrame of raw
        (unadjusted) prices. Use these inputs ONLY for spread estimation
        and similar microstructure-derived metrics; use `RawData.prices`
        (adjusted close) for return calculations.
    """
    df = pd.read_csv(
        path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str},
        usecols=["timestamp", "ticker", "feature", "value"],
    )
    df["ticker"] = df["ticker"].fillna("")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    has_ticker = df["ticker"] != ""
    return (
        _pivot_ticker_feature(df, has_ticker & (df["feature"] == HIGH)),
        _pivot_ticker_feature(df, has_ticker & (df["feature"] == LOW)),
        _pivot_ticker_feature(df, has_ticker & (df["feature"] == CLOSE)),
    )
