"""Tests for the return-on-risk factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PITSnapshot, PreparedPanel


def _make_prepared_panel(
    n_days: int = 800,
    tickers: list[str] | None = None,
    seed: int = 42,
    asof: str = "2025-01-01",
    custom_returns: dict[str, pd.Series] | None = None,
) -> PreparedPanel:
    """Build a synthetic PreparedPanel with controllable return/vol profiles."""
    if tickers is None:
        tickers = ["AAA.NZ", "BBB.NZ", "CCC.NZ"]

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    if custom_returns is not None:
        prices_data = {}
        for ticker, rets in custom_returns.items():
            prices_data[ticker] = 10.0 * (1 + rets).cumprod()
        prices = pd.DataFrame(prices_data, index=dates)
    else:
        prices_data = {}
        for t in tickers:
            px = 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
            prices_data[t] = px
        prices = pd.DataFrame(prices_data, index=dates)

    prices.index.name = "date"

    volumes = pd.DataFrame(
        {t: np.full(n_days, 500_000.0) for t in (custom_returns.keys() if custom_returns else tickers)},
        index=dates,
    )
    volumes.index.name = "date"

    asof_ts = pd.Timestamp(asof)

    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [], names=["ticker", "publication_date"]
            )
        ),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(
            columns=["ticker", "ex_date", "type", "factor"]
        ),
        asof=asof_ts,
    )

    from skuld_research.data.prepared_panel import build_prepared_panel

    return build_prepared_panel(snap, nzx_only=False, rebalance_start="2021-01-01")


# ---------------------------------------------------------------------------
# Test: NaN when insufficient history
# ---------------------------------------------------------------------------


def test_return_on_risk_insufficient_history_returns_nan():
    """Tickers with fewer than min_months of daily returns get NaN."""
    from skuld_research.factors.return_on_risk import ReturnOnRiskFactor

    n_days = 600
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(456)

    # LONG has full history, SHORT only has last ~60 days (3 months)
    long_returns = pd.Series(0.001 * rng.standard_normal(n_days), index=dates)
    short_returns = pd.Series(np.nan, index=dates)
    short_returns.iloc[-60:] = 0.001 * rng.standard_normal(60)

    panel = _make_prepared_panel(
        n_days=n_days,
        custom_returns={"LONG.NZ": long_returns, "SHORT.NZ": short_returns},
        asof="2025-01-01",
    )

    factor = ReturnOnRiskFactor(lookback_months=12, min_months=6)
    t = pd.Timestamp("2024-06-01")
    scores = factor.score(panel, t, ["LONG.NZ", "SHORT.NZ"])

    assert not pd.isna(scores["LONG.NZ"]), "LONG should have a valid score"
    assert pd.isna(scores["SHORT.NZ"]), "SHORT should be NaN (insufficient history)"


# ---------------------------------------------------------------------------
# Test: zero-vol tickers return NaN not inf
# ---------------------------------------------------------------------------


def test_return_on_risk_zero_vol_returns_nan():
    """A ticker with constant daily returns (zero volatility) scores NaN, not inf."""
    from skuld_research.factors.return_on_risk import ReturnOnRiskFactor

    n_days = 600
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Constant positive daily return → zero std → division by zero risk
    const_returns = pd.Series(0.001, index=dates)

    panel = _make_prepared_panel(
        n_days=n_days,
        custom_returns={"CONST.NZ": const_returns},
        asof="2025-01-01",
    )

    factor = ReturnOnRiskFactor(lookback_months=12, min_months=6)
    t = pd.Timestamp("2024-06-01")
    scores = factor.score(panel, t, ["CONST.NZ"])

    assert pd.isna(scores["CONST.NZ"]), "Zero-vol ticker should score NaN, not inf"
    assert not np.isinf(scores["CONST.NZ"]), "Score must not be inf"


# ---------------------------------------------------------------------------
# Test: higher return / same vol scores higher
# ---------------------------------------------------------------------------


def test_return_on_risk_higher_return_same_vol_scores_higher():
    """Higher-return ticker with same vol scores higher than lower-return same-vol ticker."""
    from skuld_research.factors.return_on_risk import ReturnOnRiskFactor

    n_days = 600
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(42)
    vol = 0.01

    # Both have same vol, HIGH has higher mean daily return
    noise = vol * rng.standard_normal(n_days)
    high_returns = pd.Series(0.002 + noise, index=dates)
    low_returns = pd.Series(-0.001 + noise, index=dates)  # same noise, lower mean

    panel = _make_prepared_panel(
        n_days=n_days,
        custom_returns={"HIGH.NZ": high_returns, "LOW.NZ": low_returns},
        asof="2025-01-01",
    )

    factor = ReturnOnRiskFactor(lookback_months=12, min_months=6)
    t = pd.Timestamp("2024-06-01")
    scores = factor.score(panel, t, ["HIGH.NZ", "LOW.NZ"])

    assert not scores.isna().any(), "Both tickers should have valid scores"
    assert scores["HIGH.NZ"] > scores["LOW.NZ"], "Higher-return ticker should score higher"


# ---------------------------------------------------------------------------
# Test: output indexed by universe
# ---------------------------------------------------------------------------


def test_return_on_risk_output_indexed_by_universe():
    """Output Series is indexed by the input universe, including unknown tickers as NaN."""
    from skuld_research.factors.return_on_risk import ReturnOnRiskFactor

    n_days = 600
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(99)
    returns = pd.Series(0.001 * rng.standard_normal(n_days), index=dates)

    panel = _make_prepared_panel(
        n_days=n_days,
        custom_returns={"A.NZ": returns},
        asof="2025-01-01",
    )

    factor = ReturnOnRiskFactor(lookback_months=12, min_months=6)
    t = pd.Timestamp("2024-06-01")
    universe = ["A.NZ", "MISSING.NZ"]
    scores = factor.score(panel, t, universe)

    assert list(scores.index) == universe, "Index must match input universe"
    assert pd.isna(scores["MISSING.NZ"]), "Unknown ticker should be NaN"
