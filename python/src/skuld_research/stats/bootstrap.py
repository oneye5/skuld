"""Stationary bootstrap for Sharpe ratio confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import BootstrapResult


def stationary_bootstrap_sharpe(
    returns: pd.Series,
    mean_block_len: float | None = None,
    n_resamples: int = 2000,
    rng_seed: int = 42,
    periods_per_year: int = 12,
) -> BootstrapResult:
    """Stationary bootstrap CI for annualised Sharpe ratio.
    
    Implements Politis & Romano (1994) stationary bootstrap: geometric
    block resampling that preserves temporal structure.
    
    Args:
        returns: monthly returns series (NaN-free).
        mean_block_len: geometric mean block length. If None, uses len(returns)**(1/3).
        n_resamples: number of bootstrap resamples.
        rng_seed: RNG seed for reproducibility.
        periods_per_year: annualisation factor (default 12 for monthly).
    
    Returns:
        BootstrapResult with point estimate and 95% CI percentiles.
    
    Raises:
        ValueError: if returns contains NaN.
    """
    if returns.isna().any():
        raise ValueError("returns must be NaN-free")
    
    n = len(returns)
    if n < 2:
        raise ValueError("Need at least 2 observations for bootstrap")
    
    if mean_block_len is None:
        mean_block_len = max(2.0, float(n ** (1.0 / 3.0)))
    
    # Point estimate from original series
    mu = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std < 1e-12:
        # Constant series → Sharpe undefined, return NaN
        return BootstrapResult(
            point_estimate=float("nan"),
            mean=float("nan"),
            ci_low_95=float("nan"),
            ci_median=float("nan"),
            ci_high_95=float("nan"),
            n_resamples=n_resamples,
            mean_block_len=mean_block_len,
        )
    
    point_estimate = (mu / std) * (periods_per_year ** 0.5)
    
    # Bootstrap resamples
    rng = np.random.default_rng(rng_seed)
    ret_values = returns.values
    sharpes = []
    
    p_continue = 1.0 - 1.0 / mean_block_len  # probability of continuing block
    
    for _ in range(n_resamples):
        # Stationary bootstrap: start at random index, then with prob p_continue
        # advance by 1, else jump to random index. Wrap modulo n.
        resample = []
        idx = rng.integers(0, n)
        for _ in range(n):
            resample.append(ret_values[idx])
            if rng.random() < p_continue:
                idx = (idx + 1) % n
            else:
                idx = rng.integers(0, n)
        
        resample_arr = np.array(resample)
        mu_b = resample_arr.mean()
        std_b = resample_arr.std(ddof=1)
        if std_b > 1e-12:
            sharpe_b = (mu_b / std_b) * (periods_per_year ** 0.5)
            sharpes.append(sharpe_b)
        else:
            sharpes.append(float("nan"))
    
    sharpes_arr = np.array(sharpes)
    # Filter out NaN if any resamples had zero std
    sharpes_finite = sharpes_arr[np.isfinite(sharpes_arr)]
    
    if len(sharpes_finite) == 0:
        # All resamples had zero std → return NaN
        return BootstrapResult(
            point_estimate=point_estimate,
            mean=float("nan"),
            ci_low_95=float("nan"),
            ci_median=float("nan"),
            ci_high_95=float("nan"),
            n_resamples=n_resamples,
            mean_block_len=mean_block_len,
        )
    
    return BootstrapResult(
        point_estimate=point_estimate,
        mean=float(sharpes_finite.mean()),
        ci_low_95=float(np.percentile(sharpes_finite, 2.5)),
        ci_median=float(np.percentile(sharpes_finite, 50.0)),
        ci_high_95=float(np.percentile(sharpes_finite, 97.5)),
        n_resamples=n_resamples,
        mean_block_len=mean_block_len,
    )
