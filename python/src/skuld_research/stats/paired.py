"""Paired return comparison helpers for candidate-vs-baseline analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedDeltaResult:
    """Stationary-bootstrap confidence interval for paired return deltas."""

    mean_delta_monthly: float
    mean_delta_annual: float
    ci_low_95_monthly: float
    ci_median_monthly: float
    ci_high_95_monthly: float
    n_obs: int
    n_resamples: int
    mean_block_len: float


def stationary_bootstrap_paired_delta(
    candidate_returns: pd.Series,
    baseline_returns: pd.Series,
    *,
    mean_block_len: float | None = None,
    n_resamples: int = 2000,
    rng_seed: int = 42,
) -> PairedDeltaResult:
    """Bootstrap the mean paired monthly return delta preserving date alignment."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if mean_block_len is not None and mean_block_len <= 1.0:
        raise ValueError("mean_block_len must be > 1")

    aligned = pd.concat([candidate_returns, baseline_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        raise ValueError("Need at least 2 paired observations for bootstrap")

    deltas = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    n = len(deltas)
    if mean_block_len is None:
        mean_block_len = max(2.0, float(n ** (1.0 / 3.0)))

    rng = np.random.default_rng(rng_seed)
    values = deltas.to_numpy()
    p_continue = 1.0 - 1.0 / mean_block_len
    boot_means = []
    for _ in range(n_resamples):
        sample = []
        idx = rng.integers(0, n)
        for _ in range(n):
            sample.append(values[idx])
            idx = (idx + 1) % n if rng.random() < p_continue else rng.integers(0, n)
        boot_means.append(float(np.mean(sample)))

    boot = np.array(boot_means)
    return PairedDeltaResult(
        mean_delta_monthly=float(deltas.mean()),
        mean_delta_annual=float(deltas.mean() * 12.0),
        ci_low_95_monthly=float(np.percentile(boot, 2.5)),
        ci_median_monthly=float(np.percentile(boot, 50.0)),
        ci_high_95_monthly=float(np.percentile(boot, 97.5)),
        n_obs=n,
        n_resamples=n_resamples,
        mean_block_len=mean_block_len,
    )
