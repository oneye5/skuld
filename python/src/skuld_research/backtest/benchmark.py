"""Benchmark return series computation for strategy comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _annualised_sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    rf_monthly = rf_annual / 12.0
    excess = returns - rf_monthly
    if len(excess) < 2 or excess.std(ddof=1) < 1e-12:
        return float("nan")
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(12))


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return float(dd.min())


def _mean_annual_return(returns: pd.Series) -> float:
    return float(returns.mean() * 12)


def compute_sixty_forty_returns(
    raw_df: pd.DataFrame,
    spec,
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """Compute monthly 60/40 FNZ+bond portfolio returns aligned to date_index."""
    # --- FNZ equity monthly returns ---
    fnz = raw_df[(raw_df["ticker"] == spec.benchmarks.sixty_forty_equity_proxy)]
    fnz_ac = fnz[fnz["feature"] == "adj_close"]
    if fnz_ac.empty:
        fnz_ac = fnz[fnz["feature"] == "close"]

    fnz_ac = fnz_ac.copy()
    fnz_ac["date"] = pd.to_datetime(fnz_ac["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    fnz_ac = fnz_ac.set_index("date")["value"].sort_index()
    # resample to month-end
    fnz_monthly = fnz_ac.resample("ME").last().pct_change().dropna()

    # --- Synthetic bond monthly returns ---
    ltr_field = spec.benchmarks.sixty_forty_bond_macro_field
    ltr = raw_df[raw_df["ticker"].isna() & (raw_df["feature"] == ltr_field)].copy()
    ltr["date"] = pd.to_datetime(ltr["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    ltr = ltr.set_index("date")["value"].sort_index()
    # LTR values are in percent (e.g. 5.5 = 5.5%), convert to decimal
    ltr_pct = ltr / 100.0
    ltr_monthly = ltr_pct.resample("ME").last().ffill()
    ltr_delta = ltr_monthly.diff()

    duration = spec.benchmarks.sixty_forty_bond_duration_years
    # bond total return ≈ -duration * delta_yield + yield/12
    bond_return = -duration * ltr_delta + ltr_monthly / 12.0

    # --- Align both to date_index ---
    # Normalise date_index to month-end without tz
    if hasattr(date_index, "tz") and date_index.tz is not None:
        di = date_index.tz_localize(None)
    else:
        di = date_index
    di_me = di.to_period("M").to_timestamp("M")

    fnz_aligned = fnz_monthly.reindex(di_me).fillna(0.0)
    bond_aligned = bond_return.reindex(di_me).fillna(0.0)

    haircut = spec.benchmarks.sixty_forty_flat_haircut_bps / 10_000 / 12
    blended = 0.6 * fnz_aligned + 0.4 * bond_aligned - haircut
    blended.index = date_index
    blended.name = "sixty_forty"
    return blended


def compute_td_floor_returns(spec, date_index: pd.DatetimeIndex) -> pd.Series:
    """Return flat monthly TD floor returns aligned to date_index."""
    monthly_return = spec.benchmarks.td_floor_default / 12.0
    s = pd.Series(monthly_return, index=date_index, name="td_floor")
    return s


def compute_benchmark_stats(name: str, returns: pd.Series, rf_annual: float = 0.0) -> dict:
    """Compute summary stats for a benchmark return series."""
    return {
        "name": name,
        "sharpe": _annualised_sharpe(returns, rf_annual),
        "mean_annual": _mean_annual_return(returns),
        "max_dd": _max_drawdown(returns),
    }
