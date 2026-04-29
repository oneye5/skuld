"""NZX equal-weighted benchmark with fixed universe criteria."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from skuld_common.contracts import BacktestResult, PreparedPanel
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.factors.constant import ConstantOneSignal


def nzx_equal_weighted_fixed_universe(
    panel: PreparedPanel,
    mcap_floor_nzd: float = 20e6,
    adv_floor_shares: int = 10_000,
    backtest_config: BacktestConfig | None = None,
) -> BacktestResult:
    """NZX equal-weighted benchmark with independent universe definition.
    
    Builds a universe mask INDEPENDENTLY of panel.universe_mask to avoid gaming.
    Uses market cap floor (resampled month-end). ADV floor is documented but
    not implemented (requires daily volume series not exposed by PreparedPanel).
    
    Constructs a new PreparedPanel with the alternative universe_mask, runs
    BacktestEngine([ConstantOneSignal()], modified_panel, config) and returns
    the BacktestResult.
    
    Args:
        panel: PreparedPanel with market_cap and returns data.
        mcap_floor_nzd: minimum market cap in NZD (default 20M).
        adv_floor_shares: minimum average daily volume in shares (default 10k).
            NOTE: ADV filter not implemented pending daily volume exposure in panel.
        backtest_config: BacktestConfig for the backtest engine.
    
    Returns:
        BacktestResult from the equal-weighted benchmark.
    """
    if adv_floor_shares != 10_000:
        raise NotImplementedError(
            "ADV filter is not implemented for the NZX equal-weighted benchmark"
        )

    # Resample market_cap to month-end, forward-fill up to 3 months
    mcap_monthly = panel.market_cap.resample("ME").last()
    mcap_monthly = mcap_monthly.ffill(limit=3)
    
    # Build universe mask: mcap >= floor
    universe_mask = mcap_monthly >= mcap_floor_nzd
    
    # Align with panel.universe_mask index (rebalance dates)
    universe_mask = universe_mask.reindex(
        panel.universe_mask.index, method="ffill"
    ).fillna(False)
    
    # Ensure columns match panel.universe_mask
    universe_mask = universe_mask.reindex(
        columns=panel.universe_mask.columns, fill_value=False
    )
    
    # Note: ADV floor is only accepted at its default placeholder value until
    # PreparedPanel exposes the daily volume series needed to implement it.
    
    # Build modified panel with new universe_mask
    modified_panel = replace(panel, universe_mask=universe_mask)
    
    # Run backtest with ConstantOneSignal (produces uniform +1.0 scores)
    config = backtest_config or BacktestConfig()
    engine = BacktestEngine(
        factors=[ConstantOneSignal()],
        panel=modified_panel,
        config=config,
    )
    
    return engine.run()
