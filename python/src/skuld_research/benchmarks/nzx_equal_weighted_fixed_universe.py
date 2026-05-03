"""NZX equal-weighted benchmark with fixed universe criteria."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from skuld_common.contracts import BacktestResult, PreparedPanel
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.costs.model import CostConfig
from skuld_research.execution.policy import ExecutionPolicyConfig
from skuld_research.factors.constant import ConstantOneSignal


def nzx_equal_weighted_fixed_universe(
    panel: PreparedPanel,
    mcap_floor_nzd: float = 20e6,
    adv_floor_shares: int = 10_000,
    share_adv: pd.DataFrame | None = None,
    backtest_config: BacktestConfig | None = None,
) -> BacktestResult:
    """NZX equal-weighted benchmark with independent universe definition.
    
    Builds a universe mask INDEPENDENTLY of panel.universe_mask to avoid gaming.
    Uses market cap floor (resampled month-end) and an optional independent
    share ADV panel when a non-default ADV floor is requested.
    
    Constructs a new PreparedPanel with the alternative universe_mask, runs
    BacktestEngine([ConstantOneSignal()], modified_panel, config) and returns
    the BacktestResult.
    
    Args:
        panel: PreparedPanel with market_cap and returns data.
        mcap_floor_nzd: minimum market cap in NZD (default 20M).
        adv_floor_shares: minimum average daily volume in shares (default 10k).
        share_adv: optional date x ticker average daily share-volume panel.
        backtest_config: BacktestConfig for the backtest engine.
    
    Returns:
        BacktestResult from the equal-weighted benchmark.
    """
    # Resample to business month-end so weekend calendar month labels align
    # with strategy rebalance dates rather than being treated as future data.
    mcap_monthly = panel.market_cap.resample("BME").last()
    mcap_monthly = mcap_monthly.ffill(limit=3)

    # Build universe mask: mcap >= floor
    universe_mask = mcap_monthly >= mcap_floor_nzd

    if adv_floor_shares > 0 and share_adv is None:
        raise ValueError("share_adv is required when adv_floor_shares is enabled")
    if share_adv is not None and adv_floor_shares > 0:
        share_adv = share_adv.copy()
        share_adv.index = pd.DatetimeIndex(share_adv.index).map(
            lambda date: pd.offsets.BMonthEnd().rollback(pd.Timestamp(date))
        )
        adv_monthly = share_adv.resample("BME").last().ffill(limit=3)
        universe_mask &= adv_monthly.reindex_like(universe_mask) >= adv_floor_shares

    # Align with panel.universe_mask index (rebalance dates)
    universe_mask = universe_mask.reindex(
        panel.universe_mask.index, method="ffill"
    ).fillna(False)

    # Ensure columns match panel.universe_mask
    universe_mask = universe_mask.reindex(
        columns=panel.universe_mask.columns, fill_value=False
    )

    # Build modified panel with new universe_mask
    modified_panel = replace(panel, universe_mask=universe_mask)

    # Run backtest with ConstantOneSignal (produces uniform +1.0 scores)
    config = replace(
        backtest_config or BacktestConfig(),
        cost_config=CostConfig(spread_bps=40.0),
        flat_haircut_bps=0.0,
        min_names=None,
        adv_participation_cap=None,
        turnover_budget_frac=None,
        execution_policy=ExecutionPolicyConfig(),
        adv_panel=None,
    )
    engine = BacktestEngine(
        factors=[ConstantOneSignal()],
        panel=modified_panel,
        config=config,
    )

    return engine.run()
