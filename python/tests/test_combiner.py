"""Tests for the signal combiner (Stage 4).

These tests are written before the implementation (TDD red phase).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import CombinedScores


def _make_signals(
    universe: list[str],
    values: dict[str, list[float]],
) -> dict[str, pd.Series]:
    """Build a signals dict from lists of values aligned to universe."""
    return {
        name: pd.Series(vals, index=universe, dtype=float)
        for name, vals in values.items()
    }


def _make_sector(universe: list[str], sector: str = "Unknown") -> pd.Series:
    return pd.Series(sector, index=universe, name="sector")


# ---------------------------------------------------------------------------
# Test: output type and NaN-free contract
# ---------------------------------------------------------------------------


def test_combine_returns_combined_scores():
    """combine_signals returns a CombinedScores with NaN-free scores."""
    from skuld_research.factors.combiner import combine_signals

    universe = ["A.NZ", "B.NZ", "C.NZ", "D.NZ", "E.NZ"]
    signals = _make_signals(universe, {"momentum": [1.0, 0.5, 0.0, -0.5, -1.0]})
    sector = _make_sector(universe)

    result = combine_signals(signals, universe, sector, pd.Timestamp("2024-01-01"))

    assert isinstance(result, CombinedScores)
    assert result.scores.index.tolist() == universe
    assert not result.scores.isna().any(), "scores must be NaN-free"


# ---------------------------------------------------------------------------
# Test: z-scoring — output has mean ≈ 0, std ≈ 1
# ---------------------------------------------------------------------------


def test_combine_output_is_unit_normal():
    """Combined scores are re-z'd to approximately mean=0, std=1."""
    from skuld_research.factors.combiner import combine_signals

    universe = [f"T{i}.NZ" for i in range(20)]
    rng = np.random.default_rng(42)
    signals = _make_signals(universe, {"momentum": rng.standard_normal(20).tolist()})
    sector = _make_sector(universe)

    result = combine_signals(signals, universe, sector, pd.Timestamp("2024-01-01"))

    assert abs(result.scores.mean()) < 0.1, "Expected mean ≈ 0"
    assert abs(result.scores.std() - 1.0) < 0.15, "Expected std ≈ 1"


# ---------------------------------------------------------------------------
# Test: winsorisation clips extreme values
# ---------------------------------------------------------------------------


def test_combine_winsorises_extremes():
    """Extreme raw scores are clipped to ±winsor_limit after z-scoring."""
    from skuld_research.factors.combiner import combine_signals

    universe = [f"T{i}.NZ" for i in range(20)]
    # Create one extreme outlier
    vals = [0.0] * 20
    vals[0] = 1000.0  # massive outlier
    signals = _make_signals(universe, {"momentum": vals})
    sector = _make_sector(universe)

    result = combine_signals(
        signals, universe, sector, pd.Timestamp("2024-01-01"), winsor_limit=3.0
    )

    # The component score for the outlier must be at most the winsor limit
    assert result.component_scores.loc["T0.NZ", "momentum"] <= 3.0 + 1e-9


# ---------------------------------------------------------------------------
# Test: NaN inputs become 0 in component_scores
# ---------------------------------------------------------------------------


def test_combine_nan_imputed_to_zero():
    """NaN signal values are imputed to 0 in the component_scores."""
    from skuld_research.factors.combiner import combine_signals

    universe = ["A.NZ", "B.NZ", "C.NZ"]
    # C has NaN momentum — can't be scored this period
    signals = _make_signals(universe, {"momentum": [1.0, -1.0, np.nan]})
    sector = _make_sector(universe)

    result = combine_signals(signals, universe, sector, pd.Timestamp("2024-01-01"))

    assert result.component_scores.loc["C.NZ", "momentum"] == pytest.approx(0.0)
    assert not result.scores.isna().any()


# ---------------------------------------------------------------------------
# Test: component_scores preserved per factor
# ---------------------------------------------------------------------------


def test_combine_component_scores_has_one_column_per_factor():
    """component_scores has exactly one column per factor in signals."""
    from skuld_research.factors.combiner import combine_signals

    universe = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    signals = _make_signals(
        universe,
        {
            "momentum": [1.0, 0.5, -0.5, -1.0],
            "value": [0.2, 0.8, 0.1, 0.9],
        },
    )
    sector = _make_sector(universe)

    result = combine_signals(signals, universe, sector, pd.Timestamp("2024-01-01"))

    assert set(result.component_scores.columns) == {"momentum", "value"}
    assert result.component_scores.index.tolist() == universe


# ---------------------------------------------------------------------------
# Test: shrinkage toward sector mean reduces spread
# ---------------------------------------------------------------------------


def test_combine_shrinkage_reduces_score_spread():
    """Higher shrinkage produces less spread in scores."""
    from skuld_research.factors.combiner import combine_signals

    universe = [f"T{i}.NZ" for i in range(10)]
    rng = np.random.default_rng(7)
    vals = rng.standard_normal(10).tolist()
    signals = _make_signals(universe, {"momentum": vals})
    sector = _make_sector(universe)
    t = pd.Timestamp("2024-01-01")

    result_low = combine_signals(
        signals, universe, sector, t, shrinkage=0.0, winsor_limit=10.0
    )
    result_high = combine_signals(
        signals, universe, sector, t, shrinkage=0.8, winsor_limit=10.0
    )

    # High shrinkage should produce lower std in component scores before re-z
    std_low = result_low.component_scores["momentum"].std()
    std_high = result_high.component_scores["momentum"].std()
    assert std_high < std_low, (
        f"Higher shrinkage should reduce component score spread: low={std_low:.4f}, high={std_high:.4f}"
    )


# ---------------------------------------------------------------------------
# Test: ordering preserved — higher raw signal → higher combined score
# ---------------------------------------------------------------------------


def test_combine_preserves_rank_order_single_factor():
    """With a single factor and no NaN, rank order must be preserved."""
    from skuld_research.factors.combiner import combine_signals

    universe = [f"T{i}.NZ" for i in range(10)]
    # Monotonically increasing raw scores
    vals = list(range(10))
    signals = _make_signals(universe, {"momentum": vals})
    sector = _make_sector(universe)

    result = combine_signals(signals, universe, sector, pd.Timestamp("2024-01-01"))

    combined_sorted = result.scores.sort_values(ascending=False).index.tolist()
    # Highest raw score (T9) should rank first
    assert combined_sorted[0] == "T9.NZ"
    assert combined_sorted[-1] == "T0.NZ"


# ---------------------------------------------------------------------------
# Test: asof propagated correctly
# ---------------------------------------------------------------------------


def test_combine_asof_is_preserved():
    """CombinedScores.asof matches the input timestamp."""
    from skuld_research.factors.combiner import combine_signals

    universe = ["A.NZ", "B.NZ"]
    signals = _make_signals(universe, {"momentum": [1.0, -1.0]})
    sector = _make_sector(universe)
    t = pd.Timestamp("2025-03-31")

    result = combine_signals(signals, universe, sector, t)
    assert result.asof == t
