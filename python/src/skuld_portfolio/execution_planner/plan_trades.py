"""Execution planner: builds a TradeList from target and current portfolios.

Implements the no-trade region, size floor, and Sharesies fee-cliff logic
identically to BacktestEngine so planner and engine agree on what a trade is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import CurrentPortfolio, TargetPortfolio, TradeList
from skuld_research.costs.model import CostModel
from skuld_research.execution.policy import ExecutionPolicyConfig, apply_execution_policy


def plan_trades(
    target: TargetPortfolio,
    current: CurrentPortfolio,
    cost_model: CostModel,
    no_trade_threshold: float,
    size_floor_nzd: float,
    size_floor_cost_multiple: float,
    sharesies_coverage_nzd: float,
    sharesies_excess_bps: float,
    config_hash: str,
    expected_alpha: pd.Series | None = None,
    execution_policy: ExecutionPolicyConfig | None = None,
) -> TradeList:
    """Plan trades from current holdings to target portfolio.

    Replicates BacktestEngine's no-trade region and size floor logic exactly.
    Additionally implements Sharesies fee-cliff deferral: trades that push
    cumulative monthly volume past sharesies_coverage_nzd are deferred unless
    their expected alpha exceeds the marginal sharesies_excess_bps cost.

    Args:
        target: Target portfolio from optimizer.
        current: Current holdings.
        cost_model: CostModel for spread cost estimation.
        no_trade_threshold: no-trade region threshold (fraction of NAV).
        size_floor_nzd: minimum trade value in NZD.
        size_floor_cost_multiple: skip if trade < N × round-trip cost.
        sharesies_coverage_nzd: volume covered by flat fee (default $5,000).
        sharesies_excess_bps: marginal fee above coverage (190 bps).
        config_hash: SHA-256 hash of spec that produced target.
        expected_alpha: Optional per-ticker expected alpha in bps. If None,
            falls back to combined_score_z magnitude from target (if available)
            or uniform ranking.

    Returns:
        TradeList with one row per ticker, action classifications, and totals.
    """
    # 1. Compute total NAV
    current_value = (current.holdings * current.prices).sum()
    total_nav = current_value + current.cash_nzd

    # 2. Build target dollar positions
    # target.weights already sums to (1 - cash_weight), representing fractions of total NAV
    target_values = target.weights * total_nav

    # 3. Get current prices for target tickers
    all_tickers = sorted(set(list(target.weights.index)) | set(list(current.holdings.index)))

    # Align current holdings and prices to all_tickers
    current_holdings_full = current.holdings.reindex(all_tickers, fill_value=0)
    current_prices_full = current.prices.reindex(all_tickers, fill_value=0.0)

    # For target tickers not in current prices, use a placeholder (can't trade them)
    # In practice, target tickers should all be in the panel; this is defensive
    target_prices = current_prices_full.copy()
    # If a ticker in target has price=0, it can't be traded

    # Target shares (round down to whole shares)
    target_values_full = target_values.reindex(all_tickers, fill_value=0.0)
    target_shares = (target_values_full / target_prices).fillna(0).apply(
        lambda x: int(x) if x > 0 else 0
    )

    # Current values
    current_values_full = current_holdings_full * current_prices_full

    # Delta
    delta_shares = target_shares - current_holdings_full
    delta_values = target_values_full - current_values_full

    # 4. Compute weights
    if total_nav > 1e-6:
        current_weights = current_values_full / total_nav
        target_weights = target_values_full / total_nav
    else:
        current_weights = pd.Series(0.0, index=all_tickers)
        target_weights = pd.Series(0.0, index=all_tickers)
    delta_weights = target_weights - current_weights
    forced = (current_weights > 1e-9) & (target_weights <= 1e-9)

    # 5. No-trade region (matches engine logic exactly)
    in_ntr = delta_weights.abs() < no_trade_threshold

    # 6. Size floor (matches engine logic exactly)
    delta_values_abs = delta_values.abs()
    spread_cost_per_trade = delta_values_abs * cost_model.config.spread_bps / 10_000
    size_floor_per_trade = np.maximum(
        size_floor_nzd,
        size_floor_cost_multiple * spread_cost_per_trade,
    )
    below_floor = delta_values_abs < size_floor_per_trade

    # 7. Initial action assignment (before fee-cliff deferral)
    action = pd.Series("HOLD", index=all_tickers)
    action[delta_shares > 0] = "BUY"
    action[delta_shares < 0] = "SELL"
    action[in_ntr | below_floor] = "HOLD"
    action[forced & (delta_shares < 0)] = "SELL"

    if expected_alpha is None:
        # Fallback: use combined_score_z magnitude if available
        # (target doesn't carry component_scores directly; planner gets it from recommender)
        # For now, use a uniform ranking as placeholder — recommender will pass expected_alpha
        alpha_proxy = delta_values_abs.copy()
    else:
        alpha_proxy = expected_alpha.reindex(all_tickers, fill_value=0.0)

    # 8. Sharesies fee-cliff deferral via the shared research execution policy.
    executable_delta_weights = delta_weights.copy()
    executable_delta_weights[~action.isin(["BUY", "SELL"])] = 0.0
    policy_config = execution_policy or ExecutionPolicyConfig(
        volume_budget_nzd=None,
        min_trade_benefit_bps=0.0,
        excess_trade_benefit_bps=sharesies_excess_bps,
    )
    policy = apply_execution_policy(
        executable_delta_weights,
        nav_nzd=total_nav,
        expected_alpha_bps=alpha_proxy,
        config=policy_config,
        forced=forced,
    )
    deferred = pd.Series(False, index=all_tickers)
    for ticker in all_tickers:
        no_executable_delta = abs(policy.executable_delta_weights[ticker]) <= 1e-12
        if action[ticker] in ["BUY", "SELL"] and no_executable_delta:
            action[ticker] = "DEFER"
            deferred[ticker] = True

    benefit_proxy = alpha_proxy * delta_weights.apply(lambda value: 1.0 if value >= 0 else -1.0)
    ranked = sorted(all_tickers, key=lambda t: benefit_proxy[t], reverse=True)

    # 8b. Assign Sharesies fee bands based on cumulative executed volume.
    # Trades executed within the sharesies_coverage_nzd cap → "flat_15".
    # Trades that push cumulative volume past the cap → "percent_19bps".
    # Non-traded rows → "subscription_only".
    fee_bands: dict[str, str] = {t: "subscription_only" for t in all_tickers}
    cum_vol_for_bands = 0.0
    for ticker in ranked:
        if action[ticker] not in ["BUY", "SELL"]:
            continue
        trade_vol = delta_values_abs[ticker]
        if cum_vol_for_bands + trade_vol <= sharesies_coverage_nzd:
            fee_bands[ticker] = "flat_15"
        else:
            fee_bands[ticker] = "percent_19bps"
        cum_vol_for_bands += trade_vol

    # 9. Build trades DataFrame
    records = []
    for ticker in all_tickers:
        trade_vol = delta_values_abs[ticker] if action[ticker] in ["BUY", "SELL"] else 0.0
        band = fee_bands[ticker]

        records.append({
            "ticker": ticker,
            "action": action[ticker],
            "current_shares": int(current_holdings_full[ticker]),
            "target_shares": int(target_shares[ticker]),
            "delta_shares": int(delta_shares[ticker]),
            "current_value_nzd": float(current_values_full[ticker]),
            "target_value_nzd": float(target_values_full[ticker]),
            "delta_value_nzd": float(delta_values[ticker]),
            "est_round_trip_cost_nzd": float(spread_cost_per_trade[ticker]),
            "in_no_trade_region": bool(in_ntr[ticker]),
            "below_size_floor": bool(below_floor[ticker]),
            "deferred_to_next_month": bool(deferred[ticker]),
            "sharesies_fee_band": band,
        })

    trades_df = pd.DataFrame(records)

    # 10. Compute totals
    executed = trades_df[trades_df["action"].isin(["BUY", "SELL"])]
    total_volume = executed["delta_value_nzd"].abs().sum() if not executed.empty else 0.0

    # Sharesies fee (using cost_model logic)
    exec_values = (
        executed["delta_value_nzd"].abs()
        if not executed.empty
        else pd.Series(dtype=float)
    )
    cost_bd = cost_model.compute_period_costs(exec_values)
    total_cost = cost_bd.total_cost_nzd

    return TradeList(
        trades=trades_df,
        total_volume_nzd=float(total_volume),
        total_estimated_cost_nzd=float(total_cost),
        asof=target.asof,
        config_hash=config_hash,
    )
