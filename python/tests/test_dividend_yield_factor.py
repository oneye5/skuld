"""Tests for the dividend-yield factor."""

from __future__ import annotations

import pandas as pd

from skuld_common.contracts import PreparedPanel


def _panel_with_dividends() -> PreparedPanel:
    dates = pd.bdate_range("2022-01-03", "2024-06-28")
    tickers = ["HIGH.NZ", "LOW.NZ", "NONE.NZ"]
    returns_daily = pd.DataFrame(0.0, index=dates, columns=tickers)
    prices = pd.DataFrame(100.0, index=dates, columns=tickers)
    returns_monthly = pd.DataFrame(
        0.0, index=pd.date_range("2022-01-31", "2024-06-30", freq="BME"), columns=tickers
    )
    market_cap = pd.DataFrame(1_000_000.0, index=dates, columns=tickers)
    universe_mask = pd.DataFrame(True, index=returns_monthly.index, columns=tickers)
    corporate_actions = pd.DataFrame(
        [
            {
                "ticker": "HIGH.NZ",
                "ex_date": pd.Timestamp("2023-07-03"),
                "type": "dividend",
                "factor": 4.0,
            },
            {
                "ticker": "LOW.NZ",
                "ex_date": pd.Timestamp("2023-07-03"),
                "type": "dividend",
                "factor": 1.0,
            },
            {
                "ticker": "HIGH.NZ",
                "ex_date": pd.Timestamp("2024-07-03"),
                "type": "dividend",
                "factor": 100.0,
            },
        ]
    )

    return PreparedPanel(
        returns_daily=returns_daily,
        returns_monthly=returns_monthly,
        market_cap=market_cap,
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=universe_mask,
        macro=pd.DataFrame(index=dates),
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        asof=pd.Timestamp("2024-07-01"),
        prices=prices,
        corporate_actions=corporate_actions,
    )


def test_dividend_yield_scores_trailing_dividends_over_price() -> None:
    from skuld_research.factors.dividend_yield import DividendYieldFactor

    panel = _panel_with_dividends()
    factor = DividendYieldFactor(lookback_months=12, min_dividends=1)

    scores = factor.score(panel, pd.Timestamp("2024-07-01"), ["HIGH.NZ", "LOW.NZ", "NONE.NZ"])

    assert scores["HIGH.NZ"] == 0.04
    assert scores["LOW.NZ"] == 0.01
    assert pd.isna(scores["NONE.NZ"])


def test_dividend_yield_ignores_future_dividends() -> None:
    from skuld_research.factors.dividend_yield import DividendYieldFactor

    panel = _panel_with_dividends()
    factor = DividendYieldFactor(lookback_months=12, min_dividends=1)

    scores = factor.score(panel, pd.Timestamp("2024-07-01"), ["HIGH.NZ"])

    assert scores["HIGH.NZ"] == 0.04


def test_dividend_yield_conforms_to_signal_generator_protocol() -> None:
    from skuld_research.factors.dividend_yield import DividendYieldFactor
    from skuld_research.factors.protocols import SignalGenerator

    factor = DividendYieldFactor()

    assert isinstance(factor, SignalGenerator)
    assert factor.name == "dividend_yield"


def test_walk_forward_fold_restriction_preserves_dividend_inputs() -> None:
    from skuld_research.backtest.walk_forward import FoldSpec, _restrict_panel_to_fold

    panel = _panel_with_dividends()
    fold = _restrict_panel_to_fold(
        panel,
        FoldSpec(
            fold_id=0,
            test_start=panel.universe_mask.index[0],
            test_end=panel.universe_mask.index[-1],
        ),
    )

    assert not fold.prices.empty
    assert not fold.corporate_actions.empty
