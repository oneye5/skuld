"""Tests for PITSnapshot contract type."""

import pandas as pd

from skuld_common.contracts import PITSnapshot


def test_pit_snapshot_construction():
    """PITSnapshot can be constructed with valid DataFrames."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    snap = PITSnapshot(
        prices=pd.DataFrame(
            {"ANZ.NZ": [50.0, 51.0]},
            index=pd.to_datetime(["2025-01-13", "2025-01-14"]),
        ),
        volumes=pd.DataFrame(
            {"ANZ.NZ": [100_000.0, 120_000.0]},
            index=pd.to_datetime(["2025-01-13", "2025-01-14"]),
        ),
        fundamentals=pd.DataFrame(
            {"annual_net_income_common_stockholders": [1_000_000.0]},
            index=pd.MultiIndex.from_tuples(
                [("ANZ.NZ", pd.Timestamp("2024-06-30"))],
                names=["ticker", "publication_date"],
            ),
        ),
        macro=pd.DataFrame(
            {"oecd_bcicp": [100.5]},
            index=pd.to_datetime(["2025-01-10"]),
        ),
        corporate_actions=pd.DataFrame(
            {
                "ticker": ["ANZ.NZ"],
                "ex_date": [pd.Timestamp("2024-12-01")],
                "type": ["dividend"],
                "factor": [0.50],
            }
        ),
        asof=ts,
    )
    assert snap.asof == ts
    assert list(snap.prices.columns) == ["ANZ.NZ"]
    assert snap.prices.shape == (2, 1)


def test_pit_snapshot_rejects_future_prices():
    """PITSnapshot raises if prices contain dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    future_prices = pd.DataFrame(
        {"ANZ.NZ": [50.0, 51.0, 52.0]},
        index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
    )
    try:
        PITSnapshot(
            prices=future_prices,
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "future" in str(e).lower() or "asof" in str(e).lower()


def test_pit_snapshot_rejects_future_volumes():
    """PITSnapshot raises if volumes contain dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(
                {"ANZ.NZ": [100_000.0, 120_000.0, 130_000.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "volumes" in str(e).lower()


def test_pit_snapshot_rejects_future_macro():
    """PITSnapshot raises if macro contains dates >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(
                {"oecd_bcicp": [100.5, 101.0]},
                index=pd.to_datetime(["2025-01-14", "2025-01-15"]),
            ),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "macro" in str(e).lower()


def test_pit_snapshot_rejects_future_fundamentals():
    """PITSnapshot raises if fundamentals have publication_date >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(
                {"annual_net_income_common_stockholders": [1_000_000.0]},
                index=pd.MultiIndex.from_tuples(
                    [("ANZ.NZ", pd.Timestamp("2025-01-15"))],
                    names=["ticker", "publication_date"],
                ),
            ),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "fundamentals" in str(e).lower()


def test_pit_snapshot_rejects_future_corporate_actions():
    """PITSnapshot raises if corporate_actions have ex_date >= asof."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(),
            volumes=pd.DataFrame(),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(
                {
                    "ticker": ["ANZ.NZ"],
                    "ex_date": [pd.Timestamp("2025-01-15")],
                    "type": ["dividend"],
                    "factor": [0.50],
                }
            ),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "corporate_actions" in str(e).lower()


def test_pit_snapshot_reports_all_violations():
    """PITSnapshot reports all violations in a single error, not just the first."""
    ts = pd.Timestamp("2025-01-15", tz="UTC")
    try:
        PITSnapshot(
            prices=pd.DataFrame(
                {"ANZ.NZ": [50.0, 51.0, 52.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            volumes=pd.DataFrame(
                {"ANZ.NZ": [100_000.0, 120_000.0, 130_000.0]},
                index=pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]),
            ),
            fundamentals=pd.DataFrame(),
            macro=pd.DataFrame(),
            corporate_actions=pd.DataFrame(),
            asof=ts,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        msg = str(e).lower()
        assert "prices" in msg and "volumes" in msg, (
            f"Expected both 'prices' and 'volumes' in error, got: {e}"
        )
