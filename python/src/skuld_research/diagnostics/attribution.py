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

        ticker_contributions: date × ticker DataFrame of each ticker's weighted return
            contribution to the signal-EW portfolio each period.  NaN where the ticker
            was not selected.

        breadth_series: Rebalance-date series counting universe members (tickers passing
            all filters) at each rebalance.  Drawn from panel.universe_mask.

        factor_leg_alpha_ann: Per-factor annualized CAGR alpha vs the market proxy
            for an EW long portfolio of the top ``top_frac`` tickers by that factor's
            standalone score.  Requires ``component_score_panels`` to be supplied.
            Empty dict when component scores are not provided.
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

    ticker_contributions: pd.DataFrame
    breadth_series: pd.Series
    factor_leg_alpha_ann: dict[str, float]

    __hash__ = None  # pd.Series / pd.DataFrame fields are unhashable; disable hash explicitly


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
    component_score_panels: dict[str, pd.DataFrame] | None = None,
) -> AttributionReport:
    """Attribute production returns into signal and construction/cost components.

    Args:
        scores_panel: date × ticker DataFrame of combined scores at each rebalance.
        panel: PreparedPanel providing universe_mask and monthly returns.
        production_returns: Actual strategy monthly net returns.
        top_frac: Fraction of universe to select for the signal-EW portfolio.
        component_score_panels: Optional dict mapping factor name → date × ticker
            DataFrame of that factor's standalone scores.  When provided, a per-factor
            EW long portfolio is built and its CAGR alpha vs the market proxy is
            computed and stored in ``AttributionReport.factor_leg_alpha_ann``.

    Returns:
        AttributionReport with all attribution fields populated.
    """
    # Market proxy
    market_proxy = market_proxy_returns(panel)

    # Universe breadth: count of tickers in universe at each rebalance date.
    breadth_series = panel.universe_mask.sum(axis=1).rename("universe_breadth")

    # Signal EW: for each rebalance date, select top tickers by combined score.
    signal_entries: dict = {}
    ticker_contrib_entries: dict = {}  # date → {ticker: contribution}

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

        weight = 1.0 / len(rets)
        signal_entries[next_month] = float((rets * weight).sum())

        # Per-ticker contribution: weight × return for each selected ticker
        ticker_contrib_entries[next_month] = (rets * weight).to_dict()

    signal_ew = pd.Series(signal_entries, dtype=float).sort_index()

    # Build ticker contributions DataFrame (date × ticker, NaN for unselected)
    if ticker_contrib_entries:
        all_tickers = sorted({tk for d in ticker_contrib_entries.values() for tk in d})
        ticker_contributions = pd.DataFrame(ticker_contrib_entries, dtype=float).T
        ticker_contributions = ticker_contributions.reindex(columns=all_tickers)
        ticker_contributions.index = pd.DatetimeIndex(ticker_contributions.index)
        ticker_contributions = ticker_contributions.sort_index()
    else:
        ticker_contributions = pd.DataFrame(dtype=float)

    # Per-factor EW long portfolio returns (for factor-leg attribution)
    factor_leg_alpha_ann: dict[str, float] = {}
    if component_score_panels:
        for factor_name, comp_panel in component_score_panels.items():
            leg_entries: dict = {}
            for t in comp_panel.index:
                if t not in panel.universe_mask.index:
                    continue
                universe_mask = panel.universe_mask.loc[t]
                universe = universe_mask[universe_mask].index.tolist()
                if not universe:
                    continue
                comp_scores = comp_panel.loc[t].reindex(universe).dropna()
                if len(comp_scores) < 2:
                    continue
                n_sel = max(1, math.ceil(len(comp_scores) * top_frac))
                top_tickers_leg = comp_scores.nlargest(n_sel).index.tolist()
                future_dates = panel.returns_monthly.index[panel.returns_monthly.index > t]
                if len(future_dates) == 0:
                    continue
                next_m = future_dates[0]
                leg_rets = panel.returns_monthly.loc[next_m, top_tickers_leg].dropna()
                if len(leg_rets) == 0:
                    continue
                leg_entries[next_m] = float(leg_rets.mean())

            if leg_entries:
                leg_series = pd.Series(leg_entries, dtype=float).sort_index()
                common_leg = market_proxy.index.intersection(leg_series.index)
                if len(common_leg) > 0:
                    leg_alpha = _cagr(leg_series.reindex(common_leg)) - _cagr(market_proxy.reindex(common_leg))
                    factor_leg_alpha_ann[factor_name] = leg_alpha
                else:
                    factor_leg_alpha_ann[factor_name] = float("nan")
            else:
                factor_leg_alpha_ann[factor_name] = float("nan")

    # Align all three on common index for attribution metrics
    common_idx = market_proxy.index.intersection(signal_ew.index).intersection(
        production_returns.index
    )

    nan = float("nan")
    if len(common_idx) == 0:
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
            ticker_contributions=ticker_contributions,
            breadth_series=breadth_series,
            factor_leg_alpha_ann=factor_leg_alpha_ann,
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
        ticker_contributions=ticker_contributions,
        breadth_series=breadth_series,
        factor_leg_alpha_ann=factor_leg_alpha_ann,
    )
