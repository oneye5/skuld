"""Deflated Sharpe Ratio (Bailey & López de Prado 2014)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from skuld_common.contracts import DeflatedSharpeResult


def deflated_sharpe(
    sharpe_hat: float,
    n_obs: int,
    n_trials: int,
    *,
    sharpe_variance: float | None = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    alpha: float = 0.05,
    periods_per_year: int = 12,
) -> DeflatedSharpeResult:
    """Bailey & López de Prado deflated Sharpe ratio.
    
    Adjusts observed Sharpe for selection bias across n_trials strategies.
    
    Args:
        sharpe_hat: observed annualised Sharpe.
        n_obs: number of monthly observations (OOS).
        n_trials: total strategies tried (production + prior).
        sharpe_variance: variance of Sharpes across trials. If None, defaults to
            1.0 / n_obs (Lo 2002 asymptotic under H₀).
        skew: sample skewness of returns (default 0).
        kurtosis: sample Pearson kurtosis of returns (default 3).
        alpha: significance level (default 0.05).
        periods_per_year: annualisation factor (default 12).
    
    Returns:
        DeflatedSharpeResult with deflated Sharpe and pass/fail.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")

    if sharpe_variance is None:
        sharpe_variance = 1.0 / n_obs

    # De-annualise
    sr_periodic = sharpe_hat / (periods_per_year ** 0.5)

    # Expected maximum SR under selection bias
    # Formula from Bailey & López de Prado (2014)
    gamma = np.euler_gamma
    N = float(n_trials)

    inv_cdf_1 = norm.ppf(1.0 - 1.0 / N)
    inv_cdf_2 = norm.ppf(1.0 - 1.0 / (N * np.e))

    expected_max_sr = (sharpe_variance ** 0.5) * (
        (1.0 - gamma) * inv_cdf_1 + gamma * inv_cdf_2
    )

    # Deflated SR (periodic)
    numerator = sr_periodic - expected_max_sr
    denominator_factor = (
        1.0 - skew * sr_periodic + (kurtosis - 1.0) / 4.0 * sr_periodic ** 2
    )

    if denominator_factor <= 0:
        # Degenerate case → return zero deflated
        sharpe_deflated_annualised = 0.0
        p_value = 1.0
        passes_gate = False
    else:
        deflated_periodic = numerator * ((n_obs - 1.0) / denominator_factor) ** 0.5

        # Probabilistic Sharpe Ratio (one-sided)
        psr = norm.cdf(deflated_periodic)
        # One-sided p-value: H₀: SR ≤ SR*, H₁: SR > SR* (Bailey & López de Prado 2014)
        p_value = 1.0 - psr

        # Re-annualise deflated SR
        sharpe_deflated_annualised = numerator * (periods_per_year ** 0.5)

        # Pass only if p_value <= alpha AND deflated Sharpe is positive
        # This ensures negative original Sharpe never passes
        passes_gate = (p_value <= alpha) and (sharpe_deflated_annualised > 0) and (sharpe_hat > 0)

    return DeflatedSharpeResult(
        sharpe_hat=sharpe_hat,
        sharpe_deflated=sharpe_deflated_annualised,
        p_value=p_value,
        n_obs=n_obs,
        n_trials=n_trials,
        passes=passes_gate,
        alpha=alpha,
    )
