"""Portfolio recommendation pipeline.

    Orchestrates: load spec -> build panel -> compute factors -> construct target -> plan trades.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from skuld_common.contracts import (
    CombinedScores,
    CurrentPortfolio,
    PreparedPanel,
    TargetPortfolio,
    TradeList,
)
from skuld_portfolio.execution_planner.plan_trades import plan_trades
from skuld_portfolio.inputs.cash_yaml import read_cash_yaml
from skuld_portfolio.inputs.sharesies_holdings import parse_sharesies_csv
from skuld_research.config.factors import build_factors_from_specs
from skuld_research.config.hashing import spec_hash
from skuld_research.config.loader import load_spec
from skuld_research.config.spec import BacktestSpec
from skuld_research.costs.model import CostConfig, CostModel
from skuld_research.data.csv_loader import load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.execution.policy import ExecutionPolicyConfig
from skuld_research.factors.combiner import combine_signals
from skuld_research.portfolio.optimizer import build_target_portfolio


def recommend(
    spec_path: Path,
    holdings_path: Path,
    cash_yaml_path: Path,
    asof: pd.Timestamp,
    raw_csv_path: Path,
) -> tuple[TradeList, CombinedScores, dict]:
    """Generate trade recommendations for a single date.

    Args:
        spec_path: Path to strategy spec YAML.
        holdings_path: Path to Sharesies export CSV.
        cash_yaml_path: Path to cash YAML.
        asof: Rebalance date (must exist in panel).
        raw_csv_path: Path to data_long.csv.

    Returns:
        (TradeList, CombinedScores, meta_dict) where meta_dict contains
        spec_hash, spec_path, asof, low_confidence, total_volume_nzd,
        total_estimated_cost_nzd.
    """
    # 1. Load spec and compute hash
    spec: BacktestSpec = load_spec(spec_path)
    shash = spec_hash(spec)

    # 2. Load panel up to asof
    raw = load_raw_csv(raw_csv_path)
    snap = PITLoader(raw).as_of(asof)

    panel: PreparedPanel = build_prepared_panel(
        snap,
        min_adv_dollars=spec.universe.min_adv_dollars,
        min_market_cap_nzd=spec.universe.min_market_cap_nzd,
        min_history_days=spec.universe.min_history_days,
        adv_window=spec.universe.adv_window,
        mc_ffill_days=spec.universe.mc_ffill_days,
        nzx_only=spec.universe.nzx_only,
    )

    # 3. Snap asof to the latest available rebalance date in the panel
    rebalance_dates = panel.universe_mask.index
    asof_for_compare = asof.tz_localize(None) if asof.tzinfo is not None else asof
    if rebalance_dates.tz is not None:
        asof_for_compare = asof_for_compare.tz_localize(rebalance_dates.tz)
    valid = rebalance_dates[rebalance_dates <= asof_for_compare]
    if len(valid) == 0:
        raise ValueError(
            f"No rebalance date in panel on or before asof {asof}. "
            f"Panel range: {rebalance_dates[0]} to {rebalance_dates[-1]}"
        )
    rebalance_date = valid[-1]
    if rebalance_date != asof:
        print(
            f"Note: asof {asof.date()} snapped to latest available "
            f"rebalance date {rebalance_date.date()}"
        )
    asof = rebalance_date

    # 4. Get universe at asof
    universe = [
        tk for tk in panel.universe_mask.columns
        if bool(panel.universe_mask.loc[asof, tk])
    ]

    if not universe:
        raise ValueError(f"Universe is empty at {asof}")

    # 5. Compute factor scores
    factors = build_factors_from_specs(spec.factors)

    signals = {}
    for factor in factors:
        signals[factor.name] = factor.score(panel, asof, universe)

    # 6. Combine signals
    combined = combine_signals(signals, universe, panel.sector, asof)

    # 7. Build target portfolio (single date, no walk-forward)
    target: TargetPortfolio = build_target_portfolio(
        combined,
        panel,
        asof,
        cash_floor=spec.backtest.cash_floor,
        max_position=spec.backtest.max_position,
        max_sector=spec.backtest.max_sector,
        score_lambda=spec.backtest.score_lambda,
        return_window_days=spec.backtest.return_window_days,
        min_return_obs=spec.backtest.min_return_obs,
    )

    # 8. Load current holdings
    cash_nzd = read_cash_yaml(cash_yaml_path)
    current: CurrentPortfolio = parse_sharesies_csv(holdings_path, cash_nzd=cash_nzd)

    # 8b. Enrich current.prices with PIT-latest close for any target ticker
    # the user does not yet hold (planner needs prices to size BUY trades).
    target_tickers = list(target.weights.index)
    missing = [t for t in target_tickers if t not in current.prices.index]
    if missing and not snap.prices.empty:
        last_close = snap.prices.ffill().iloc[-1]
        extra_prices = last_close.reindex(missing).dropna()
        if len(extra_prices) > 0:
            merged_prices = pd.concat([current.prices, extra_prices])
            merged_holdings = current.holdings.reindex(
                merged_prices.index, fill_value=0
            ).astype(int)
            current = CurrentPortfolio(
                holdings=merged_holdings,
                prices=merged_prices,
                cash_nzd=current.cash_nzd,
            )

    # 9. Build cost model
    cost_config = CostConfig(
        spread_bps=spec.cost.spread_bps,
        sharesies_monthly_fee_nzd=spec.cost.sharesies_monthly_fee_nzd,
        sharesies_coverage_nzd=spec.cost.sharesies_coverage_nzd,
        sharesies_excess_bps=spec.cost.sharesies_excess_bps,
    )
    cost_model = CostModel(cost_config)

    # 10. Plan trades
    # expected_alpha: use combined_score_z as proxy (in bps scale)
    expected_alpha = combined.scores * 100.0  # z-score -> rough bps proxy
    execution_policy = ExecutionPolicyConfig(
        volume_budget_nzd=(
            spec.execution_policy.monthly_volume_budget_nzd
            if spec.execution_policy.kind == "volume_budget"
            else None
        ),
        min_trade_benefit_bps=(
            spec.execution_policy.min_trade_benefit_bps
            if spec.execution_policy.kind == "volume_budget"
            else 0.0
        ),
        excess_trade_benefit_bps=(
            spec.execution_policy.excess_trade_benefit_bps
            if spec.execution_policy.kind == "volume_budget"
            else 190.0
        ),
    )

    trades: TradeList = plan_trades(
        target=target,
        current=current,
        cost_model=cost_model,
        no_trade_threshold=spec.backtest.no_trade_threshold_frac,
        size_floor_nzd=spec.backtest.size_floor_nzd,
        size_floor_cost_multiple=spec.backtest.size_floor_cost_multiple,
        sharesies_coverage_nzd=spec.cost.sharesies_coverage_nzd,
        sharesies_excess_bps=spec.cost.sharesies_excess_bps,
        config_hash=shash,
        expected_alpha=expected_alpha,
        execution_policy=execution_policy,
    )

    # 11. Build metadata dict
    meta = {
        "spec_hash": shash,
        "spec_path": str(spec_path),
        "asof": asof.isoformat(),
        "low_confidence": not spec.passed_gating,
        "total_volume_nzd": trades.total_volume_nzd,
        "total_estimated_cost_nzd": trades.total_estimated_cost_nzd,
        "panel_coverage": {
            "start": panel.returns_daily.index[0].isoformat(),
            "end": panel.returns_daily.index[-1].isoformat(),
            "n_tickers": len(panel.returns_daily.columns),
        },
    }

    return trades, combined, meta
