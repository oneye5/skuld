"""Tests for the book-to-market factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


def _make_panel_with_book_values() -> PreparedPanel:
    dates = pd.bdate_range("2024-01-02", periods=40)
    month_ends = pd.date_range("2024-01-31", periods=2, freq="BME")
    tickers = ["VALUE.NZ", "GROWTH.NZ", "MISSING.NZ"]

    fundamentals = pd.DataFrame(
        {
            "annual_stockholders_equity": [300_000_000.0, 100_000_000.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("VALUE.NZ", pd.Timestamp("2024-03-31")),
                ("GROWTH.NZ", pd.Timestamp("2024-03-31")),
            ],
            names=["ticker", "publication_date"],
        ),
    )

    return PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=month_ends, columns=tickers),
        market_cap=pd.DataFrame(
            {
                "VALUE.NZ": np.full(len(dates), 100_000_000.0),
                "GROWTH.NZ": np.full(len(dates), 200_000_000.0),
                "MISSING.NZ": np.full(len(dates), 150_000_000.0),
            },
            index=dates,
        ),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=month_ends, columns=tickers),
        macro=pd.DataFrame(index=dates),
        fundamentals=fundamentals,
        asof=pd.Timestamp("2024-06-01"),
    )


def test_book_to_market_conforms_to_signal_generator_protocol() -> None:
    from skuld_research.factors.book_to_market import BookToMarketFactor
    from skuld_research.factors.protocols import SignalGenerator

    factor = BookToMarketFactor()

    assert isinstance(factor, SignalGenerator)
    assert factor.name == "book_to_market"


def test_book_to_market_scores_higher_for_higher_book_relative_to_market_cap() -> None:
    from skuld_research.factors.book_to_market import BookToMarketFactor

    panel = _make_panel_with_book_values()
    factor = BookToMarketFactor()

    scores = factor.score(panel, pd.Timestamp("2024-05-01"), ["VALUE.NZ", "GROWTH.NZ"])

    assert scores["VALUE.NZ"] > scores["GROWTH.NZ"]
    assert scores["VALUE.NZ"] == 3.0
    assert scores["GROWTH.NZ"] == 0.5


def test_book_to_market_returns_nan_when_book_value_missing() -> None:
    from skuld_research.factors.book_to_market import BookToMarketFactor

    panel = _make_panel_with_book_values()
    factor = BookToMarketFactor()

    scores = factor.score(panel, pd.Timestamp("2024-05-01"), ["MISSING.NZ"])

    assert pd.isna(scores["MISSING.NZ"])


def test_book_to_market_uses_only_fundamentals_strictly_before_rebalance_date() -> None:
    from skuld_research.factors.book_to_market import BookToMarketFactor

    panel = _make_panel_with_book_values()
    future_row = pd.DataFrame(
        {"annual_stockholders_equity": [900_000_000.0]},
        index=pd.MultiIndex.from_tuples(
            [("VALUE.NZ", pd.Timestamp("2024-05-01"))],
            names=["ticker", "publication_date"],
        ),
    )
    panel = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=panel.universe_mask,
        macro=panel.macro,
        fundamentals=pd.concat([panel.fundamentals, future_row]),
        asof=panel.asof,
    )

    factor = BookToMarketFactor()
    scores = factor.score(panel, pd.Timestamp("2024-05-01"), ["VALUE.NZ"])

    assert scores["VALUE.NZ"] == 3.0
