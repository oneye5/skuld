"""Factor decomposition of strategy returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import DecompositionReport


def factor_decomposition(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    factor_returns_dict: dict[str, pd.Series],
) -> DecompositionReport:
    """OLS decomposition of strategy returns onto market and factor returns.

    Args:
        strategy_returns: monthly strategy returns.
        market_returns: monthly market proxy returns.
        factor_returns_dict: dict of factor_name -> monthly factor returns (long-short).

    Returns:
        DecompositionReport with coefficients, t-stats, alpha, R^2.
    """
    # Build design matrix: market + factors
    regressors_list = ["market"]
    design_data = {"market": market_returns}

    for factor_name, factor_series in factor_returns_dict.items():
        regressors_list.append(factor_name)
        design_data[factor_name] = factor_series

    # Align all series on common index
    design_df = pd.DataFrame(design_data)
    y = strategy_returns

    # Inner join on index, drop NaNs
    common_idx = y.index.intersection(design_df.index)
    y_aligned = y.loc[common_idx]
    X_aligned = design_df.loc[common_idx]

    # Drop any rows with NaN
    valid = y_aligned.notna() & X_aligned.notna().all(axis=1)
    y_clean = y_aligned[valid].values
    X_clean = X_aligned[valid].values

    n_obs = len(y_clean)

    if n_obs == 0:
        # No valid observations
        return DecompositionReport(
            regressors=tuple(regressors_list),
            coefficients={name: float("nan") for name in regressors_list},
            t_stats={name: float("nan") for name in regressors_list},
            residual_alpha_annualised=float("nan"),
            residual_alpha_t_stat=float("nan"),
            r_squared=float("nan"),
            n_obs=0,
        )

    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(n_obs), X_clean])

    # OLS via numpy.linalg.lstsq
    coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y_clean, rcond=None)

    intercept = coeffs[0]
    betas = coeffs[1:]

    # Residuals
    y_pred = X_with_intercept @ coeffs
    resid = y_clean - y_pred

    # R-squared
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Newey-West HAC standard errors with Andrews automatic bandwidth
    scores_matrix = X_with_intercept * resid[:, np.newaxis]
    lag = _andrews_bandwidth(scores_matrix)
    se_nw = _newey_west_se(X_with_intercept, resid, lag)

    # T-statistics
    t_stats_values = coeffs / se_nw if not np.any(se_nw == 0) else np.full_like(coeffs, np.nan)

    # Build result
    coefficients = {name: float(beta) for name, beta in zip(regressors_list, betas)}
    t_stats = {name: float(t) for name, t in zip(regressors_list, t_stats_values[1:])}

    residual_alpha_annualised = float(intercept * 12.0)
    residual_alpha_t_stat = float(t_stats_values[0])

    return DecompositionReport(
        regressors=tuple(regressors_list),
        coefficients=coefficients,
        t_stats=t_stats,
        residual_alpha_annualised=residual_alpha_annualised,
        residual_alpha_t_stat=residual_alpha_t_stat,
        r_squared=r_squared,
        n_obs=n_obs,
    )


def _andrews_bandwidth(scores: np.ndarray) -> int:
    """Andrews (1991) data-driven bandwidth for Bartlett/Newey-West kernel.

    Fits an AR(1) to each column of the score matrix, uses the resulting
    autocorrelation to compute the optimal bandwidth via:
        m* = floor(1.1447 * (ᾱ₁ * T)^(1/3)) + 1
    where ᾱ₁ is the cross-column average of 4ρ²/(1-ρ)⁴ (Bartlett kernel
    second-derivative spectral formula from Andrews 1991, eq. 6.4).

    Args:
        scores: (n, k) score matrix (X * u per observation).

    Returns:
        Integer bandwidth ≥ 1.
    """
    n, k = scores.shape
    alphas = []
    for j in range(k):
        s = scores[:, j]
        if len(s) < 3:
            continue
        # AR(1) via OLS on demeaned series
        s_dm = s - s.mean()
        rho = np.dot(s_dm[:-1], s_dm[1:]) / max(np.dot(s_dm[:-1], s_dm[:-1]), 1e-12)
        rho = float(np.clip(rho, -0.99, 0.99))
        alphas.append(4.0 * rho ** 2 / (1.0 - rho ** 2) ** 2)

    if not alphas:
        return 3  # fallback

    alpha_bar = float(np.mean(alphas))
    if alpha_bar <= 0:
        return 1
    m = max(1, int(1.1447 * (alpha_bar * n) ** (1.0 / 3.0)))
    return m


def _newey_west_se(X: np.ndarray, resid: np.ndarray, lag: int) -> np.ndarray:
    """Compute Newey-West HAC standard errors for OLS coefficients.

    Args:
        X: design matrix (n × k) including intercept.
        resid: residuals (n,).
        lag: maximum lag for HAC estimator.

    Returns:
        Standard errors for each coefficient (k,).
    """
    n, k = X.shape

    # Meat of the sandwich: Ω = Σ_ℓ w_ℓ * E[X'uu'X]
    # where u = residual vector
    omega = np.zeros((k, k))

    # Lag 0: no weight adjustment
    for i in range(n):
        xi = X[i:i + 1].T  # (k, 1)
        ui = resid[i]
        omega += xi @ xi.T * ui**2

    # Lags 1 to L: Bartlett kernel w_ℓ = 1 - ℓ/(L+1)
    for ell in range(1, min(lag + 1, n)):
        weight = 1.0 - ell / (lag + 1)
        for i in range(n - ell):
            xi = X[i:i + 1].T
            xj = X[i + ell:i + ell + 1].T
            ui = resid[i]
            uj = resid[i + ell]
            omega += weight * (xi @ xj.T * ui * uj + xj @ xi.T * uj * ui)

    # Bread: (X'X)^{-1}
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Singular matrix
        return np.full(k, np.nan)

    # Sandwich: V = (X'X)^{-1} * Ω * (X'X)^{-1}
    V = XtX_inv @ omega @ XtX_inv

    # Standard errors
    se = np.sqrt(np.diag(V) / n)

    return se
