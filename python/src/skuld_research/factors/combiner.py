"""Signal combiner: cross-sectional normalisation and equal-weight averaging.

Pipeline per factor at each rebalance date (cross-sectional, within-sector):
  1. Z-score within sector group (NaN excluded from mean/std).
  2. Winsorise at ±winsor_limit.
  3. Shrink toward sector mean (Jorion 1986-style James-Stein shrinkage).
  4. Impute NaN → 0 (missing factor score = average cross-sectional position).

Then: equal-weight average of all component z-scores → re-z the combined
score so the output is unit-variance for downstream consumers.

The "Unknown" sector is handled naturally: when all tickers share the same
sector label (including the default "Unknown"), the within-sector z-score
is identical to the universe-wide z-score, and shrinkage is toward the
grand mean (≈ 0 after z-scoring).
"""

from __future__ import annotations

import pandas as pd

from skuld_common.contracts import CombinedScores


def combine_signals(
    signals: dict[str, pd.Series],
    universe: list[str],
    sector: pd.Series,
    asof: pd.Timestamp,
    *,
    winsor_limit: float = 3.0,
    shrinkage: float = 0.2,
) -> CombinedScores:
    """Combine per-factor raw scores into a single normalised combined score.

    Args:
        signals: ``{factor_name: Series[float]}``, index = universe.
            NaN where a factor cannot score a ticker.
        universe: Ordered list of tickers to include in the output.
        sector: Series[str] indexed by tickers; ``"Unknown"`` treated as a
            single group covering all tickers without real sector data.
        asof: Rebalance date for the resulting ``CombinedScores``.
        winsor_limit: Z-scores are clipped to ``[-winsor_limit, +winsor_limit]``
            after within-sector standardisation.
        shrinkage: Jorion-style shrinkage intensity (0 = none, 1 = full pull
            toward sector mean). Default 0.2.

    Returns:
        ``CombinedScores`` with NaN-free ``scores`` over ``universe``.
    """
    if not signals:
        raise ValueError("signals must contain at least one factor")

    # Align all signals to the universe
    df = pd.DataFrame(
        {name: sig.reindex(universe) for name, sig in signals.items()},
        index=universe,
        dtype=float,
    )
    sector_aligned = sector.reindex(universe).fillna("Unknown")

    # Process each factor column through the normalisation pipeline
    processed = pd.DataFrame(index=universe, dtype=float)
    for col in df.columns:
        col_scores = _sector_zscore(df[col], sector_aligned)
        col_scores = col_scores.clip(-winsor_limit, winsor_limit)
        col_scores = _sector_shrink(col_scores, sector_aligned, shrinkage)
        # Impute NaN → 0 (unknown factor = average score)
        col_scores = col_scores.fillna(0.0)
        processed[col] = col_scores

    # Equal-weight average of component scores
    combined = processed.mean(axis=1)

    # Re-z the combined score to unit variance (or de-mean if flat)
    std = combined.std(ddof=1)
    if std > 1e-12:
        combined = (combined - combined.mean()) / std
    else:
        combined = combined - combined.mean()

    return CombinedScores(
        scores=combined,
        component_scores=processed,
        asof=asof,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sector_zscore(scores: pd.Series, sector: pd.Series) -> pd.Series:
    """Z-score ``scores`` within each sector group.

    NaN values are excluded from the group mean/std calculation.  If a group
    has fewer than 2 non-NaN observations or zero standard deviation, the raw
    scores are left unchanged (no z-scoring for degenerate groups).
    """
    result = scores.copy().astype(float)
    for grp_tickers in sector.groupby(sector).groups.values():
        grp_valid = scores.reindex(grp_tickers).dropna()
        if len(grp_valid) < 2:
            continue
        std = grp_valid.std(ddof=1)
        if std < 1e-12:
            continue
        mean = grp_valid.mean()
        for ticker in grp_valid.index:
            result[ticker] = (scores[ticker] - mean) / std
    return result


def _sector_shrink(
    z_scores: pd.Series, sector: pd.Series, shrinkage: float
) -> pd.Series:
    """Shrink z-scores toward the group (sector) mean.

    ``shrinkage = 0`` → identity; ``shrinkage = 1`` → all scores collapse to
    the sector mean.  NaN values are excluded from the mean calculation and
    left as NaN in the output.
    """
    if shrinkage == 0.0:
        return z_scores

    result = z_scores.copy()
    for grp_tickers in sector.groupby(sector).groups.values():
        grp_valid = z_scores.reindex(grp_tickers).dropna()
        if grp_valid.empty:
            continue
        grp_mean = grp_valid.mean()
        for ticker in grp_valid.index:
            result[ticker] = (1.0 - shrinkage) * z_scores[ticker] + shrinkage * grp_mean
    return result
