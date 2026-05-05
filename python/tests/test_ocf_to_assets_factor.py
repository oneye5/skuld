"""Tests for the OCF-to-assets factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PreparedPanel


def _make_panel_with_ocf_and_assets() -> PreparedPanel:
    dates = pd.bdate_range("2024-01-02", periods=40)
    month_ends = pd.date_range("2024-01-31", periods=2, freq="BME")
    tickers = ["STRONG.NZ", "WEAK.NZ", "MISSING.NZ"]

    fundamentals = pd.DataFrame(
        {
            "annual_cash_flowsfromusedin_operating_activities_direct": [120_000_000.0, 30_000_000.0],
            "annual_total_assets": [600_000_000.0, 600_000_000.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("STRONG.NZ", pd.Timestamp("2024-03-31")),
                ("WEAK.NZ", pd.Timestamp("2024-03-31")),
            ],
            names=["ticker", "publication_date"],
        ),
    )

    return PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=month_ends, columns=tickers),
        market_cap=pd.DataFrame(
            {ticker: np.full(len(dates), 100_000_000.0) for ticker in tickers},
            index=dates,
        ),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=month_ends, columns=tickers),
        macro=pd.DataFrame(index=dates),
        fundamentals=fundamentals,
        asof=pd.Timestamp("2024-06-01"),
    )


def test_ocf_to_assets_conforms_to_signal_generator_protocol() -> None:
    from skuld_research.factors.ocf_to_assets import OcfToAssetsFactor
    from skuld_research.factors.protocols import SignalGenerator

    factor = OcfToAssetsFactor()

    assert isinstance(factor, SignalGenerator)
    assert factor.name == "ocf_to_assets"


def test_ocf_to_assets_scores_higher_for_stronger_cash_generation() -> None:
    from skuld_research.factors.ocf_to_assets import OcfToAssetsFactor

    panel = _make_panel_with_ocf_and_assets()
    factor = OcfToAssetsFactor()

    scores = factor.score(panel, pd.Timestamp("2024-05-01"), ["STRONG.NZ", "WEAK.NZ"])

    assert scores["STRONG.NZ"] > scores["WEAK.NZ"]
    assert scores["STRONG.NZ"] == 0.2
    assert scores["WEAK.NZ"] == 0.05


def test_ocf_to_assets_returns_nan_when_inputs_missing() -> None:
    from skuld_research.factors.ocf_to_assets import OcfToAssetsFactor

    panel = _make_panel_with_ocf_and_assets()
    factor = OcfToAssetsFactor()

    scores = factor.score(panel, pd.Timestamp("2024-05-01"), ["MISSING.NZ"])

    assert pd.isna(scores["MISSING.NZ"])


def test_ocf_to_assets_uses_only_fundamentals_strictly_before_rebalance_date() -> None:
    from skuld_research.factors.ocf_to_assets import OcfToAssetsFactor

    panel = _make_panel_with_ocf_and_assets()
    future_row = pd.DataFrame(
        {
            "annual_cash_flowsfromusedin_operating_activities_direct": [300_000_000.0],
            "annual_total_assets": [300_000_000.0],
        },
        index=pd.MultiIndex.from_tuples(
            [("STRONG.NZ", pd.Timestamp("2024-05-01"))],
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

    factor = OcfToAssetsFactor()
    scores = factor.score(panel, pd.Timestamp("2024-05-01"), ["STRONG.NZ"])

    assert scores["STRONG.NZ"] == 0.2


def test_ocf_to_assets_uses_latest_non_null_inputs_even_when_dates_differ() -> None:
    from skuld_research.factors.ocf_to_assets import OcfToAssetsFactor

    panel = _make_panel_with_ocf_and_assets()
    panel = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=panel.universe_mask,
        macro=panel.macro,
        fundamentals=pd.concat(
            [
                panel.fundamentals,
                pd.DataFrame(
                    {
                        "annual_cash_flowsfromusedin_operating_activities_direct": [150_000_000.0, np.nan],
                        "annual_total_assets": [np.nan, 750_000_000.0],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            ("STRONG.NZ", pd.Timestamp("2024-04-15")),
                            ("STRONG.NZ", pd.Timestamp("2024-04-20")),
                        ],
                        names=["ticker", "publication_date"],
                    ),
                ),
                pd.DataFrame(
                    {
                        "ticker_specific_other_feature": [1.0],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [("STRONG.NZ", pd.Timestamp("2024-04-25"))],
                        names=["ticker", "publication_date"],
                    ),
                ),
            ]
        ).sort_index(),
        asof=panel.asof,
    )

    factor = OcfToAssetsFactor()
    scores = factor.score(panel, pd.Timestamp("2024-05-01"), ["STRONG.NZ"])

    assert scores["STRONG.NZ"] == 0.2
