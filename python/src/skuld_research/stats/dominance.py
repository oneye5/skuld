"""Romano-Wolf stepwise multiple testing for strategy dominance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import DominanceResult


def romano_wolf_stepwise(
    strategy_returns: pd.Series,
    benchmark_returns_dict: dict[str, pd.Series],
    alpha: float = 0.05,
    n_resamples: int = 2000,
    rng_seed: int = 43,
) -> DominanceResult:
    """Romano-Wolf stepwise multiple testing for strategy vs benchmarks.
    
    Tests whether strategy beats each benchmark using stepwise procedure
    to control family-wise error rate.
    
    Args:
        strategy_returns: monthly returns of the strategy.
        benchmark_returns_dict: dict of benchmark_name -> returns Series.
        alpha: family-wise significance level (default 0.05).
        n_resamples: bootstrap resamples for the step-down distribution.
        rng_seed: RNG seed for reproducibility.
    
    Returns:
        DominanceResult with adjusted p-values and dominance flags.
    """
    if not benchmark_returns_dict:
        # No benchmarks → return empty result
        return DominanceResult(
            benchmark_names=(),
            adjusted_p_values={},
            dominates={},
            alpha=alpha,
            n_resamples=n_resamples,
        )

    benchmark_names = tuple(benchmark_returns_dict.keys())

    # Align all series on inner join
    all_series = {"strategy": strategy_returns}
    all_series.update(benchmark_returns_dict)
    df = pd.DataFrame(all_series).dropna()

    if len(df) < 10:
        # Too few observations → return inconclusive
        return DominanceResult(
            benchmark_names=benchmark_names,
            adjusted_p_values={name: 1.0 for name in benchmark_names},
            dominates={name: False for name in benchmark_names},
            alpha=alpha,
            n_resamples=n_resamples,
        )

    strategy_aligned = df["strategy"].values

    # Compute observed studentised t-stats per benchmark
    observed_t = {}
    benchmark_data = {}

    for name in benchmark_names:
        bench_aligned = df[name].values
        diff = strategy_aligned - bench_aligned

        # Annualised Sharpe of difference
        mu_diff = diff.mean()
        std_diff = diff.std(ddof=1)

        if std_diff < 1e-12:
            # Degenerate: difference series has zero variance.
            # If mean > 0, strategy strictly dominates (infinite t-stat).
            # If mean <= 0, no evidence of dominance.
            if mu_diff > 1e-12:
                observed_t[name] = float("inf")
            elif mu_diff < -1e-12:
                observed_t[name] = float("-inf")
            else:
                observed_t[name] = 0.0
        else:
            t_stat = mu_diff / (std_diff / (len(diff) ** 0.5))
            observed_t[name] = t_stat

        benchmark_data[name] = bench_aligned

    # Bootstrap distribution of max t-stat under the null (centered).
    # Romano–Wolf step-down requires a null distribution; we recenter each
    # bootstrap difference by subtracting the observed mean of that benchmark's
    # difference series before computing its t-stat. This makes the bootstrap
    # distribution approximate the sampling distribution of the t-stat under
    # H0: mu_diff = 0.
    rng = np.random.default_rng(rng_seed)
    n = len(strategy_aligned)

    # Pre-compute observed means (used to recenter)
    diff_means = {
        name: (strategy_aligned - benchmark_data[name]).mean()
        for name in benchmark_names
    }

    # Use stationary bootstrap for paired series
    mean_block_len = max(2.0, float(n ** (1.0 / 3.0)))
    p_continue = 1.0 - 1.0 / mean_block_len

    bootstrap_max_t = []

    for _ in range(n_resamples):
        # Resample indices using stationary bootstrap (paired across benchmarks)
        indices = []
        idx = rng.integers(0, n)
        for _ in range(n):
            indices.append(idx)
            if rng.random() < p_continue:
                idx = (idx + 1) % n
            else:
                idx = rng.integers(0, n)

        # Compute centered t-stats for all benchmarks in this resample
        resample_t = []
        for name in benchmark_names:
            strat_b = strategy_aligned[indices]
            bench_b = benchmark_data[name][indices]
            diff_b = strat_b - bench_b
            # Recenter under the null: subtract the original observed mean
            mu_b = diff_b.mean() - diff_means[name]
            std_b = diff_b.std(ddof=1)
            if std_b > 1e-12:
                t_b = mu_b / (std_b / (n ** 0.5))
                resample_t.append(t_b)
            else:
                resample_t.append(0.0)

        if resample_t:
            bootstrap_max_t.append(max(resample_t))
        else:
            bootstrap_max_t.append(0.0)

    bootstrap_max_t_arr = np.array(bootstrap_max_t)

    # Compute adjusted p-values
    adjusted_p = {}
    for name in benchmark_names:
        t_obs = observed_t[name]
        if t_obs == float("inf"):
            # Degenerate strict-dominance case: p=0.
            adjusted_p[name] = 0.0
        elif t_obs == float("-inf"):
            adjusted_p[name] = 1.0
        else:
            # Fraction of bootstrap maxes >= observed t
            p_adj = float((bootstrap_max_t_arr >= t_obs).mean())
            adjusted_p[name] = p_adj

    # Dominance: adjusted p-value <= alpha
    dominates = {name: (adjusted_p[name] <= alpha) for name in benchmark_names}

    return DominanceResult(
        benchmark_names=benchmark_names,
        adjusted_p_values=adjusted_p,
        dominates=dominates,
        alpha=alpha,
        n_resamples=n_resamples,
    )
