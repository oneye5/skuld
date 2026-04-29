"""Tests for the Stage 5 portfolio constructor.

These tests are written before the implementation (TDD red phase).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import CombinedScores, PreparedPanel, TargetPortfolio


def _make_combined_scores(
    universe: list[str],
    score_values: list[float],
    t: pd.Timestamp,
) -> CombinedScores:
    """Build a minimal CombinedScores for testing."""
    scores = pd.Series(score_values, index=universe, name="combined")
    component = pd.DataFrame({"momentum": score_values}, index=universe)
    return CombinedScores(scores=scores, component_scores=component, asof=t)


def _make_panel_for_portfolio(
    tickers: list[str],
    n_days: int = 300,
    seed: int = 0,
    asof: str | None = None,
) -> PreparedPanel:
    """Build a PreparedPanel with synthetic daily returns for use in portfolio tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-06-01", periods=n_days)

    prices_data = {}
    for t in tickers:
        px = 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
        prices_data[t] = px

    prices = pd.DataFrame(prices_data, index=dates)
    prices.index.name = "date"
    volumes = pd.DataFrame({t: 500_000.0 for t in tickers}, index=dates)
    volumes.index.name = "date"

    # Default asof: last date in the price series + 3 months (safely after all prices)
    if asof is None:
        last_date = dates[-1]
        asof_ts = last_date + pd.DateOffset(months=3)
    else:
        asof_ts = pd.Timestamp(asof)

    from skuld_common.contracts import PITSnapshot
    from skuld_research.data.prepared_panel import build_prepared_panel

    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(
            columns=["ticker", "ex_date", "type", "factor"]
        ),
        asof=asof_ts,
    )
    return build_prepared_panel(snap, nzx_only=False, rebalance_start="2023-06-01")


# ---------------------------------------------------------------------------
# Test: output is TargetPortfolio with valid contracts
# ---------------------------------------------------------------------------


def test_portfolio_returns_target_portfolio():
    """build_target_portfolio returns a TargetPortfolio with valid contracts."""
    from skuld_research.portfolio.optimizer import build_target_portfolio

    n = 20
    tickers = [f"T{i:02d}.NZ" for i in range(n)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers)

    # Half positive, half negative combined scores
    scores = _make_combined_scores(
        tickers,
        [(1.0 - i * 0.1) for i in range(n)],
        t,
    )

    result = build_target_portfolio(scores, panel, t, cash_floor=0.05)

    assert isinstance(result, TargetPortfolio)
    # Contract invariants are enforced by TargetPortfolio.__post_init__


# ---------------------------------------------------------------------------
# Test: weights sum to (1 - cash_weight)
# ---------------------------------------------------------------------------


def test_portfolio_weights_sum_to_equity():
    """Equity weights sum to exactly (1 - cash_weight)."""
    from skuld_research.portfolio.optimizer import build_target_portfolio

    n = 20
    tickers = [f"T{i:02d}.NZ" for i in range(n)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers)
    scores = _make_combined_scores(tickers, list(range(n, 0, -1)), t)

    result = build_target_portfolio(scores, panel, t, cash_floor=0.05)

    assert abs(result.weights.sum() - (1.0 - result.cash_weight)) < 1e-4


# ---------------------------------------------------------------------------
# Test: cash_weight >= cash_floor
# ---------------------------------------------------------------------------


def test_portfolio_cash_weight_respects_floor():
    """cash_weight is never below the configured cash_floor."""
    from skuld_research.portfolio.optimizer import build_target_portfolio

    n = 20
    tickers = [f"T{i:02d}.NZ" for i in range(n)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers)
    scores = _make_combined_scores(tickers, list(range(n, 0, -1)), t)

    for floor in [0.0, 0.05, 0.10]:
        result = build_target_portfolio(scores, panel, t, cash_floor=floor)
        assert result.cash_weight >= floor - 1e-9, (
            f"cash_weight {result.cash_weight:.4f} < floor {floor}"
        )


# ---------------------------------------------------------------------------
# Test: all weights non-negative
# ---------------------------------------------------------------------------


def test_portfolio_no_negative_weights():
    """No ticker weight is negative."""
    from skuld_research.portfolio.optimizer import build_target_portfolio

    n = 20
    tickers = [f"T{i:02d}.NZ" for i in range(n)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers)
    scores = _make_combined_scores(tickers, list(range(n, 0, -1)), t)

    result = build_target_portfolio(scores, panel, t)

    assert (result.weights >= -1e-9).all(), (
        f"Negative weights found: {result.weights[result.weights < 0].to_dict()}"
    )


# ---------------------------------------------------------------------------
# Test: per-name cap (default 5%) is respected
# ---------------------------------------------------------------------------


def test_portfolio_respects_per_name_cap():
    """No single ticker weight exceeds max_position."""
    from skuld_research.portfolio.optimizer import build_target_portfolio

    n = 20
    tickers = [f"T{i:02d}.NZ" for i in range(n)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers)
    scores = _make_combined_scores(tickers, list(range(n, 0, -1)), t)

    result = build_target_portfolio(scores, panel, t, max_position=0.05)

    assert (result.weights <= 0.05 + 1e-6).all(), (
        f"Weight exceeds 5% cap: {result.weights[result.weights > 0.05 + 1e-6].to_dict()}"
    )


# ---------------------------------------------------------------------------
# Test: only positive-score tickers get weight
# ---------------------------------------------------------------------------


def test_portfolio_excludes_negative_score_tickers():
    """Tickers with non-positive combined scores receive zero weight."""
    from skuld_research.portfolio.optimizer import build_target_portfolio

    tickers = [f"T{i:02d}.NZ" for i in range(10)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers)

    # First 5 positive, last 5 negative
    score_vals = [2.0, 1.5, 1.0, 0.5, 0.1, -0.1, -0.5, -1.0, -1.5, -2.0]
    scores = _make_combined_scores(tickers, score_vals, t)

    result = build_target_portfolio(scores, panel, t, cash_floor=0.0)

    negative_score_tickers = [tickers[i] for i in range(5, 10)]
    for ticker in negative_score_tickers:
        w = result.weights.get(ticker, 0.0)
        assert w < 1e-9, f"Ticker {ticker} with negative score got weight {w:.6f}"


# ---------------------------------------------------------------------------
# Test: top quintile selection
# ---------------------------------------------------------------------------


def test_portfolio_selects_top_quintile():
    """Only the top quintile of the universe (by combined score) is used."""
    from skuld_research.portfolio.optimizer import build_target_portfolio

    n = 20
    tickers = [f"T{i:02d}.NZ" for i in range(n)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers)

    # Clear ranking: T19 highest, T00 lowest
    scores = _make_combined_scores(tickers, list(range(n)), t)

    # Top quintile = top 4 tickers (20% of 20)
    result = build_target_portfolio(scores, panel, t, cash_floor=0.0)

    # Non-zero weights should only be for top quintile tickers
    nonzero = result.weights[result.weights > 1e-6].index.tolist()
    top_quintile = sorted(tickers)[-4:]  # T16, T17, T18, T19
    for ticker in nonzero:
        assert ticker in top_quintile, (
            f"Ticker {ticker} is outside the top quintile but has weight {result.weights[ticker]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: method is 'RiskParity' or 'EqualWeight'
# ---------------------------------------------------------------------------


def test_portfolio_method_is_recorded():
    """TargetPortfolio.method is set to a recognised optimisation method."""
    from skuld_research.portfolio.optimizer import build_target_portfolio

    n = 20
    tickers = [f"T{i:02d}.NZ" for i in range(n)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers)
    scores = _make_combined_scores(tickers, list(range(n, 0, -1)), t)

    result = build_target_portfolio(scores, panel, t)

    assert result.method in ("RiskParity", "EqualWeight", "RiskParity(fallback)")


# ---------------------------------------------------------------------------
# Test: smoke — full pipeline with real-ish synthetic data
# ---------------------------------------------------------------------------


def test_portfolio_smoke_end_to_end():
    """Full pipeline: momentum factor → combiner → portfolio constructor."""
    from skuld_research.factors.combiner import combine_signals
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_research.portfolio.optimizer import build_target_portfolio

    n = 30
    tickers = [f"T{i:02d}.NZ" for i in range(n)]
    t = pd.Timestamp("2024-09-30")
    panel = _make_panel_for_portfolio(tickers, n_days=600)

    factor = MomentumFactor()
    raw_scores = factor.score(panel, t, tickers)

    sector = pd.Series("Unknown", index=tickers, name="sector")
    combined = combine_signals({"momentum": raw_scores}, tickers, sector, t)

    result = build_target_portfolio(combined, panel, t, cash_floor=0.05)

    assert isinstance(result, TargetPortfolio)
    assert result.cash_weight >= 0.05
    assert result.weights.sum() == pytest.approx(1.0 - result.cash_weight, abs=1e-4)
    assert (result.weights >= -1e-9).all()
    assert (result.weights <= 0.05 + 1e-6).all()
