"""Stage 5 — portfolio constructor.

Accepts a ``CombinedScores`` object and a ``PreparedPanel``, and returns a
``TargetPortfolio`` whose weights are risk-parity-optimised over the top-quintile
positive-score names.

Constraints enforced here (per architecture §3.6):
  - Per-name cap:    ``max_position`` (default 5 %).
  - Per-sector cap:  ``max_sector`` (default 25 %) — skipped for the
                     degenerate case where all tickers share one sector (e.g.
                     "Unknown"), which would otherwise cap the entire equity
                     allocation and is meaningless without real sector data.
  - Cash floor:      ``cash_floor`` (default 5 %).
  - Liquidity cap:   optional; requires ``adv`` Series in NZD and
                     ``portfolio_nav`` in NZD (not available in PreparedPanel
                     at this stage — deferred to Milestone 4).

Optimiser: riskfolio-lib equal-risk-contribution (risk parity).
Fallback: equal weight when the optimiser fails or fewer than 2 candidates
remain after filtering.
"""

from __future__ import annotations

import warnings

import pandas as pd

from skuld_common.contracts import CombinedScores, PreparedPanel, TargetPortfolio


def build_target_portfolio(
    scores: CombinedScores,
    panel: PreparedPanel,
    t: pd.Timestamp,
    *,
    top_quintile_n: int | None = None,
    cash_floor: float = 0.05,
    max_position: float = 0.05,
    max_sector: float = 0.25,
    score_lambda: float = 0.0,
    adv: pd.Series | None = None,
    portfolio_nav: float | None = None,
    return_window_days: int = 252,
    min_return_obs: int = 63,
) -> TargetPortfolio:
    """Construct a risk-parity target portfolio from combined factor scores.

    Args:
        scores: ``CombinedScores`` from Stage 4.
        panel: ``PreparedPanel`` providing daily returns for the covariance estimate.
        t: Rebalance date. Only data strictly before ``t`` is used.
        top_quintile_n: Number of candidates to admit to the optimizer.  If
            ``None``, defaults to the top 20 % (quintile) of ``scores.scores``.
        cash_floor: Minimum cash fraction. Default 5 %.
        max_position: Per-ticker weight cap. Default 5 %.
        max_sector: Per-sector weight cap (as a fraction of the full portfolio,
            not just the equity sleeve). Skipped for single-sector universes.
            Default 25 %.
        score_lambda: Score-tilt intensity. Final weights are proportional to
            ``w_rp * max(0, 1 + λ * score)``, re-normalised.  ``λ = 0``
            (default) gives pure risk parity.
        adv: Optional Series[float] indexed by ticker; ADV in NZD for the
            liquidity cap (1 % × ADV per name). Not enforced if ``None``.
        portfolio_nav: Portfolio NAV in NZD for the liquidity cap. Required
            together with ``adv``; ignored if ``adv`` is ``None``.
        return_window_days: Lookback (trading days) for covariance estimation.
        min_return_obs: Minimum non-NaN return observations required for a
            ticker to enter the optimizer. Default 63 (≈ 3 months).

    Returns:
        ``TargetPortfolio`` with all contract invariants satisfied.
    """
    t_naive = t.tz_localize(None) if t.tzinfo else t

    # -----------------------------------------------------------------------
    # 1. Filter candidates: positive combined score, top quintile
    # -----------------------------------------------------------------------
    all_scores = scores.scores
    positive = all_scores[all_scores > 0].sort_values(ascending=False)

    if positive.empty:
        if _is_flat_positive_component_signal(scores):
            return _equal_weight_fallback(list(all_scores.index), cash_floor, t, max_position)
        return _cash_portfolio(t)

    n_top = top_quintile_n if top_quintile_n is not None else max(1, len(all_scores) // 5)
    candidates = positive.head(n_top)

    if len(candidates) < 2:
        return _equal_weight_fallback(list(candidates.index), cash_floor, t, max_position)

    # -----------------------------------------------------------------------
    # 2. Build return window for covariance estimation
    # -----------------------------------------------------------------------
    daily = panel.returns_daily
    avail_dates = daily.index[daily.index < t_naive]
    if len(avail_dates) < min_return_obs:
        return _equal_weight_fallback(list(candidates.index), cash_floor, t, max_position)

    window_dates = avail_dates[-return_window_days:]
    candidate_tickers = [tk for tk in candidates.index if tk in daily.columns]
    returns_window = daily.loc[window_dates, candidate_tickers].dropna(how="all")

    # Drop tickers with insufficient return history in the window
    valid_mask = returns_window.notna().sum() >= min_return_obs
    returns_window = returns_window.loc[:, valid_mask]

    # -----------------------------------------------------------------------
    # 3. Build initial proportions (risk parity or equal-weight fallback)
    # Proportions sum to 1; they will be scaled to the equity sleeve below.
    # -----------------------------------------------------------------------
    if len(returns_window.columns) >= 2:
        rp_weights = _risk_parity_weights(returns_window)
        if score_lambda != 0.0:
            tilt_scores = candidates.reindex(rp_weights.index).fillna(0.0)
            rp_weights = rp_weights * (1.0 + score_lambda * tilt_scores).clip(lower=0.0)
        proportions = _renorm(rp_weights)
        method = "RiskParity"
    else:
        n_eq = len(candidates)
        proportions = pd.Series(1.0 / n_eq, index=candidates.index)
        method = "EqualWeight"

    # -----------------------------------------------------------------------
    # 4–7. Apply constraints in final-weight space (% of NAV).
    #
    # Scale proportions to the equity sleeve first, then apply caps.
    # Excess weight that cannot be redistributed (all names at cap) stays
    # as cash — cash_weight = max(cash_floor, 1 − Σfinal_weights).
    # -----------------------------------------------------------------------
    equity_weights = proportions * (1.0 - cash_floor)

    # Per-name cap: redistribute excess to non-capped names; if none, cash absorbs
    equity_weights = _apply_per_name_cap(equity_weights, max_position)

    # Per-sector cap (skip degenerate single-sector case)
    unique_sectors = panel.sector.reindex(equity_weights.index).unique()
    if len(unique_sectors) > 1:
        equity_weights = _apply_sector_cap(equity_weights, panel.sector, max_sector)

    # Optional liquidity cap
    if adv is not None and portfolio_nav is not None and portfolio_nav > 0:
        for ticker in list(equity_weights.index):
            if ticker in adv and adv[ticker] > 0:
                liq_cap = 0.01 * adv[ticker] / portfolio_nav
                if equity_weights[ticker] > liq_cap:
                    equity_weights[ticker] = liq_cap

    # Non-negativity guard (floating-point artefacts)
    final_weights = equity_weights.clip(lower=0.0)
    cash_weight = max(cash_floor, 1.0 - final_weights.sum())

    return TargetPortfolio(
        weights=final_weights,
        cash_weight=cash_weight,
        method=method,
        asof=t,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
    """Equal risk contribution weights via riskfolio-lib.

    Returns a Series[float] indexed by ticker, summing to 1.  Falls back to
    equal weight if the optimizer fails (e.g. singular covariance matrix).
    """
    try:
        import riskfolio as rp

        # riskfolio-lib Portfolio expects a DatetimeIndex
        port = rp.Portfolio(returns=returns)
        # ledoit_wolf shrinkage ensures the covariance is positive definite
        # even for small samples or low-variance synthetic data.
        port.assets_stats(method_mu="hist", method_cov="ledoit_wolf")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_df = port.rp_optimization(
                model="Classic", rm="MV", hist=True, rf=0, b=None
            )

        if w_df is not None and not w_df.empty:
            weights = w_df["weights"]  # Series indexed by ticker name
            weights = weights.clip(lower=0.0)
            total = weights.sum()
            if total > 1e-12:
                return weights / total

    except Exception:  # noqa: BLE001
        pass

    # Fallback: equal weight
    n = len(returns.columns)
    return pd.Series(1.0 / n, index=returns.columns)


def _renorm(weights: pd.Series) -> pd.Series:
    """Renormalise weights to sum to 1; return equal weight if sum is ~0."""
    total = weights.sum()
    if total < 1e-12:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / total


def _apply_per_name_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Cap each ticker at ``cap`` by redistributing excess to uncapped names.

    Works in final-weight space (% of NAV), not proportions.  If all names
    are at or above the cap, no redistribution is possible and the function
    returns weights summing to ``n * cap`` — the remainder goes to cash.
    """
    weights = weights.copy()
    for _ in range(len(weights) + 1):
        above = weights > cap + 1e-9
        if not above.any():
            break
        below = ~above
        excess = (weights[above] - cap).sum()
        weights[above] = cap
        if below.any() and weights[below].sum() > 1e-12:
            # Redistribute proportionally to uncapped names
            weights[below] += excess * weights[below] / weights[below].sum()
        else:
            # All at cap — no redistribution possible; cash absorbs excess
            break
    return weights


def _apply_sector_cap(
    weights: pd.Series, sector: pd.Series, cap: float
) -> pd.Series:
    """Ensure no sector exceeds ``cap`` of total portfolio weight.

    Excess within the capped sector is scaled down and the freed weight stays
    as cash (consistent with ``_apply_per_name_cap`` behaviour).
    """
    weights = weights.copy()
    for _ in range(len(weights) + 1):
        capped_any = False
        for grp_tickers in sector.groupby(sector).groups.values():
            grp = weights.reindex(grp_tickers).dropna()
            sector_total = grp.sum()
            if sector_total > cap + 1e-9:
                scale = cap / sector_total
                for tk in grp.index:
                    weights[tk] = weights[tk] * scale
                capped_any = True
        if not capped_any:
            break
    return weights


def _equal_weight_fallback(
    tickers: list[str], cash_floor: float, asof: pd.Timestamp, max_position: float = 1.0
) -> TargetPortfolio:
    """Return an equal-weight portfolio over ``tickers``, respecting max_position."""
    n = len(tickers)
    if n == 0:
        return TargetPortfolio(
            weights=pd.Series(dtype=float),
            cash_weight=1.0,
            method="EqualWeight",
            asof=asof,
        )
    eq_w = (1.0 - cash_floor) / n
    weights = pd.Series(eq_w, index=tickers)
    # Apply per-name cap
    if max_position < 1.0:
        weights = _apply_per_name_cap(weights, max_position)
    cash_weight = max(cash_floor, 1.0 - weights.sum())
    return TargetPortfolio(
        weights=weights, cash_weight=cash_weight, method="EqualWeight", asof=asof
    )


def _cash_portfolio(asof: pd.Timestamp) -> TargetPortfolio:
    """Return a fully-cash target when no positive signal exists."""
    return TargetPortfolio(
        weights=pd.Series(dtype=float),
        cash_weight=1.0,
        method="Cash",
        asof=asof,
    )


def _is_flat_positive_component_signal(scores: CombinedScores) -> bool:
    """Detect the explicit equal-weight benchmark signal before demeaning."""
    components = scores.component_scores
    if list(components.columns) != ["constant_one"]:
        return False
    row_means = components.mean(axis=1)
    return bool((row_means > 0.0).all())
