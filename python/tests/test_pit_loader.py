"""Tests for the point-in-time loader.

The critical invariant: as_of(t) never returns data with timestamps >= t.
"""

from pathlib import Path

import pandas as pd

from skuld_research.data.csv_loader import load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_common.contracts import PITSnapshot


def test_pit_loader_returns_snapshot(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """as_of returns a PITSnapshot."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    assert isinstance(snap, PITSnapshot)
    assert snap.asof == asof_timestamp


def test_future_data_excluded(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Row at t=asof (2025-01-15) must NOT appear in the snapshot.

    The synthetic CSV has an ANZ.NZ adj_close=52.0 at 2025-01-15T00:00Z
    which equals asof. The PIT invariant is strictly-before, so this row
    must be absent.
    """
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    # Prices should have only day13 and day14 for ANZ.NZ, not day15
    max_price_date = snap.prices.index.max()
    assert max_price_date < asof_timestamp.tz_localize(None), (
        f"PIT violation: max price date {max_price_date} >= asof {asof_timestamp}"
    )


def test_negative_prices_excluded(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Negative price rows are filtered out of the snapshot."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    # After cleaning, no cell should be explicitly negative (NaN is acceptable)
    non_null_prices = snap.prices.stack().dropna()
    assert (non_null_prices >= 0).all(), "Negative prices found in PIT snapshot"


def test_fundamentals_filtered(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Fundamentals with publication_date >= asof are excluded."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    if not snap.fundamentals.empty:
        pub_dates = snap.fundamentals.index.get_level_values("publication_date")
        asof_naive = asof_timestamp.tz_localize(None)
        assert (pub_dates < asof_naive).all(), "Fundamental data at or after asof found"


def test_macro_filtered(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Macro data with date >= asof is excluded."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    if not snap.macro.empty:
        asof_naive = asof_timestamp.tz_localize(None)
        assert (snap.macro.index < asof_naive).all(), "Macro data at or after asof found"


def test_corporate_actions_filtered(synthetic_csv_path: Path, asof_timestamp: pd.Timestamp):
    """Corporate actions with ex_date >= asof are excluded."""
    loader = PITLoader(load_raw_csv(synthetic_csv_path))
    snap = loader.as_of(asof_timestamp)
    if not snap.corporate_actions.empty:
        asof_naive = asof_timestamp.tz_localize(None)
        assert (snap.corporate_actions["ex_date"] < asof_naive).all()
