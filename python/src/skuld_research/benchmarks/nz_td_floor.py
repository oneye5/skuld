"""NZ TD floor benchmark."""
from __future__ import annotations

import pandas as pd

from skuld_common.contracts import BacktestResult, PreparedPanel
from skuld_research.backtest.metrics import compute_drawdown_series


def nz_td_floor(
    panel: PreparedPanel,
    asof: pd.Timestamp,
    default_floor: float = 0.04,
) -> BacktestResult:
    """NZ TD floor benchmark (notional term-deposit returns).
    
    Reads panel.macro["short_term_interest_rates"] (annualised %, decimal form).
    Resamples to month-end, forward-fills gaps up to 3 months, falls back to
    default_floor (4%/yr) for any remaining missing month.
    
    Monthly return = (1 + r_annual) ** (1/12) - 1.
    
    Args:
        panel: PreparedPanel with macro data.
        asof: PIT cutoff (unused, for signature consistency).
        default_floor: fallback annual rate (decimal, e.g., 0.04 for 4%).
    
    Returns:
        BacktestResult with synthesised returns, zero costs/turnover.
    """
    # Check if macro field exists
    if "short_term_interest_rates" in panel.macro.columns and not panel.macro.empty:
        rates_daily = panel.macro["short_term_interest_rates"]
    else:
        # Field missing → use default for all dates
        rates_daily = pd.Series(default_floor, index=panel.returns_daily.index)
    
    # Resample to month-end, take last available rate in each month
    rates_monthly = rates_daily.resample("ME").last()
    
    # Forward-fill gaps up to 3 months
    rates_monthly = rates_monthly.ffill(limit=3)
    
    # Fill any remaining NaNs with default_floor
    rates_monthly = rates_monthly.fillna(default_floor)
    
    # Align with panel.returns_monthly index
    aligned_index = panel.returns_monthly.index
    rates_aligned = rates_monthly.reindex(aligned_index, method="ffill").fillna(default_floor)
    
    # Monthly return: (1 + r_annual)^(1/12) - 1
    monthly_returns = (1.0 + rates_aligned) ** (1.0 / 12.0) - 1.0
    
    # Build BacktestResult
    n = len(monthly_returns)
    if n == 0:
        # Empty panel
        return BacktestResult(
            returns=pd.Series([], dtype=float),
            costs_nzd=pd.Series([], dtype=float),
            turnover=pd.Series([], dtype=float),
            drawdown=pd.Series([], dtype=float),
            sharpe_raw=0.0,
            sharpe_flat_haircut=0.0,
            start=pd.Timestamp("1970-01-01"),
            end=pd.Timestamp("1970-01-01"),
            n_periods=0,
            avg_positions=0.0,
            hit_rate=0.0,
            skewness=0.0,
            calmar_ratio=0.0,
            period_n_positions=pd.Series([], dtype=int),
        )
    
    drawdown = compute_drawdown_series(monthly_returns)
    
    # Compute Sharpe (annualised)
    mu = float(monthly_returns.mean()) * 12.0
    vol = float(monthly_returns.std(ddof=1)) * (12.0 ** 0.5) if n > 1 else 0.0
    sharpe_raw = mu / vol if vol > 1e-12 else 0.0
    
    return BacktestResult(
        returns=monthly_returns,
        costs_nzd=pd.Series(0.0, index=monthly_returns.index),
        turnover=pd.Series(0.0, index=monthly_returns.index),
        drawdown=drawdown,
        sharpe_raw=sharpe_raw,
        sharpe_flat_haircut=sharpe_raw,  # No costs for TD floor
        start=monthly_returns.index[0],
        end=monthly_returns.index[-1],
        n_periods=n,
        avg_positions=0.0,
        hit_rate=float((monthly_returns > 0).mean()),
        skewness=0.0,
        calmar_ratio=0.0,
        period_n_positions=pd.Series(0, index=monthly_returns.index, dtype=int),
    )
