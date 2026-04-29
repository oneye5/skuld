"""Tests for alpha decay analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import DecayReport


def test_alpha_decay_basic_structure():
    """Decay report has correct structure."""
    from skuld_research.diagnostics.decay import alpha_decay

    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"])
    tickers = ["A", "B", "C", "D", "E"]

    factor_panel = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]] * 3,
        index=dates,
        columns=tickers,
    )

    returns_panel = pd.DataFrame(
        np.random.randn(15, 5) * 0.03,
        index=pd.date_range("2024-01-31", periods=15, freq="BME"),
        columns=tickers,
    )

    result = alpha_decay(
        factor_panel,
        returns_panel,
        horizons=(1, 2, 3),
        factor_name="test_factor",
        min_cross_section=3,
    )

    assert isinstance(result, DecayReport)
    assert result.factor_name == "test_factor"
    assert result.horizons == (1, 2, 3)
    assert len(result.ic_by_horizon) == 3
    assert result.peak_horizon in (1, 2, 3)


def test_alpha_decay_peak_horizon():
    """Peak horizon is argmax(ic_mean)."""
    from skuld_research.diagnostics.decay import alpha_decay

    # Create a factor with some predictive power
    np.random.seed(123)
    n_dates = 20
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="BME")
    tickers = [f"T{i}" for i in range(20)]

    # Scores with moderate autocorrelation
    scores = np.random.randn(n_dates, 20).cumsum(axis=0)
    factor_panel = pd.DataFrame(scores, index=dates, columns=tickers)

    # Returns crafted to correlate with factor scores
    returns_data = np.zeros((n_dates + 12, 20))
    for i in range(n_dates + 12):
        # Add some correlation with past scores at various lags
        base_signal = np.zeros(20)
        for lag in range(min(i + 1, n_dates)):
            if i - lag >= 0 and i - lag < n_dates:
                base_signal += scores[i - lag] * (0.01 / (1 + lag))
        returns_data[i] = base_signal + np.random.randn(20) * 0.02

    returns_panel = pd.DataFrame(
        returns_data,
        index=pd.date_range("2020-01-31", periods=n_dates + 12, freq="BME"),
        columns=tickers,
    )

    result = alpha_decay(
        factor_panel,
        returns_panel,
        horizons=(1, 2, 3, 6),
        min_cross_section=10,
    )

    # Verify IC at peak_horizon is indeed highest
    ic_means = {h: result.ic_by_horizon[h].ic_mean for h in result.horizons}
    assert ic_means[result.peak_horizon] == max(ic_means.values())
    # Also verify all horizons were computed
    assert all(result.ic_by_horizon[h].n_obs > 0 for h in (1, 2, 3, 6))


def test_alpha_decay_handles_insufficient_data():
    """Decay handles horizons with insufficient forward data gracefully."""
    from skuld_research.diagnostics.decay import alpha_decay

    # Very short panel
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    tickers = ["A", "B", "C"]

    factor_panel = pd.DataFrame(
        [[1.0, 2.0, 3.0]] * 2,
        index=dates,
        columns=tickers,
    )

    # Only 4 months of returns
    returns_panel = pd.DataFrame(
        [[0.01, 0.02, 0.03]] * 4,
        index=pd.date_range("2024-01-31", periods=4, freq="BME"),
        columns=tickers,
    )

    result = alpha_decay(
        factor_panel,
        returns_panel,
        horizons=(1, 3, 6, 12),
        min_cross_section=2,
    )

    # horizon=1 and horizon=3 might have some observations
    # horizon=6 and horizon=12 will have zero observations
    assert result.ic_by_horizon[1].n_obs >= 0
    assert result.ic_by_horizon[12].n_obs == 0
