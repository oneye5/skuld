"""Return attribution: decomposes strategy returns into signal, construction, and cost contributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from skuld_common.contracts import PreparedPanel
from skuld_research.diagnostics.panels import market_proxy_returns


@dataclass(frozen=True)
class AttributionReport:
    """Decomposes strategy returns into signal, construction+cost contributions.

    Attributes:
        market_proxy_monthly: Equal-weighted universe monthly returns (baseline).
        signal_ew_monthly: Equal-weighted signal-selected monthly returns (zero cost).
        production_monthly: Actual strategy monthly returns (with construction + costs).

        signal_contribution_ann: Annualized CAGR of signal_ew minus CAGR of market_proxy.
        construction_cost_drag_ann: Annualized CAGR of production minus CAGR of signal_ew.
        total_alpha_ann: Annualized CAGR of production minus CAGR of market_proxy.

        market_proxy_cumulative: Cumulative return index (base 1.0) for market proxy.
        signal_ew_cumulative: Cumulative return index (base 1.0) for signal EW.
        production_cumulative: Cumulative return index (base 1.0) for production.
    """

    market_proxy_monthly: pd.Series
    signal_ew_monthly: pd.Series
    production_monthly: pd.Series

    signal_contribution_ann: float
    construction_cost_drag_ann: float
    total_alpha_ann: float

    market_proxy_cumulative: pd.Series
    signal_ew_cumulative: pd.Series
    production_cumulative: pd.Series

    __hash__ = None  # pd.Series fields are unhashable; disable hash explicitly


def _cagr(returns: pd.Series) -> float:
    """Compute annualized CAGR from monthly return series (excluding NaN)."""
    clean = returns.dropna()
    if len(clean) == 0:
        return float("nan")
    n_years = len(clean) / 12.0
    if n_years <= 0:
        return float("nan")
    total = (1.0 + clean).prod()
    if total <= 0:
        return float("nan")
    return float(total ** (1.0 / n_years) - 1.0)


def _cumulative(returns: pd.Series) -> pd.Series:
    """Compute cumulative return index (base 1.0) from monthly return series.

    NaN values in the input are treated as zero-return (filled before cumproduct).
    """
    return (1.0 + returns.fillna(0.0)).cumprod()


def attribute_returns(
    scores_panel: pd.DataFrame,
    panel: PreparedPanel,
    production_returns: pd.Series,
    *,
    top_frac: float = 0.5,
) -> AttributionReport:
    """Attribute production returns into signal and construction/cost components."""
    # Market proxy
    market_proxy = market_proxy_returns(panel)

    # Signal EW: for each rebalance date, select top tickers by score
    signal_entries: dict = {}
    for t in scores_panel.index:
        universe_mask = panel.universe_mask.loc[t]
        universe = universe_mask[universe_mask].index.tolist()
        if not universe:
            continue

        scores = scores_panel.loc[t].reindex(universe).dropna()
        if len(scores) < 2:
            continue

        n_select = max(1, math.ceil(len(scores) * top_frac))
        top_tickers = scores.nlargest(n_select).index.tolist()

        future_dates = panel.returns_monthly.index[panel.returns_monthly.index > t]
        if len(future_dates) == 0:
            continue

        next_month = future_dates[0]
        rets = panel.returns_monthly.loc[next_month, top_tickers].dropna()
        if len(rets) == 0:
            continue

        signal_entries[next_month] = rets.mean()

    signal_ew = pd.Series(signal_entries, dtype=float).sort_index()

    # Align all three on common index for attribution metrics
    common_idx = market_proxy.index.intersection(signal_ew.index).intersection(
        production_returns.index
    )

    if len(common_idx) == 0:
        nan = float("nan")
        return AttributionReport(
            market_proxy_monthly=market_proxy,
            signal_ew_monthly=signal_ew,
            production_monthly=production_returns,
            signal_contribution_ann=nan,
            construction_cost_drag_ann=nan,
            total_alpha_ann=nan,
            market_proxy_cumulative=_cumulative(market_proxy) if len(market_proxy) > 0 else pd.Series(dtype=float),
            signal_ew_cumulative=_cumulative(signal_ew) if len(signal_ew) > 0 else pd.Series(dtype=float),
            production_cumulative=_cumulative(production_returns) if len(production_returns) > 0 else pd.Series(dtype=float),
        )

    market_aligned = market_proxy.reindex(common_idx)
    signal_aligned = signal_ew.reindex(common_idx)
    prod_aligned = production_returns.reindex(common_idx)

    signal_contribution_ann = _cagr(signal_aligned) - _cagr(market_aligned)
    construction_cost_drag_ann = _cagr(prod_aligned) - _cagr(signal_aligned)
    total_alpha_ann = _cagr(prod_aligned) - _cagr(market_aligned)

    return AttributionReport(
        market_proxy_monthly=market_proxy,
        signal_ew_monthly=signal_ew,
        production_monthly=production_returns,
        signal_contribution_ann=signal_contribution_ann,
        construction_cost_drag_ann=construction_cost_drag_ann,
        total_alpha_ann=total_alpha_ann,
        market_proxy_cumulative=_cumulative(market_proxy),
        signal_ew_cumulative=_cumulative(signal_ew),
        production_cumulative=_cumulative(production_returns),
    )
