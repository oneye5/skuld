"""Gating decision for walk-forward results."""
from __future__ import annotations

import math

import pandas as pd
from scipy import stats as scipy_stats

from skuld_common.contracts import (
    BootstrapResult,
    DeflatedSharpeResult,
    GatingDecision,
    WalkForwardResult,
)
from skuld_research.stats.bootstrap import stationary_bootstrap_sharpe
from skuld_research.stats.deflated import deflated_sharpe
from skuld_research.stats.dominance import romano_wolf_stepwise
from skuld_research.stats.excess_return import one_sided_hac_excess_return
from skuld_research.stats.ledger import TrialLedger, n_trials_prior


def evaluate(
    walk_forward_result: WalkForwardResult,
    ledger: TrialLedger,
    *,
    benchmarks: dict[str, pd.Series] | None = None,
    td_benchmark_name: str | None = None,
    sanity_floor: float = 0.0,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    dominance_n_resamples: int | None = None,
    rng_seed: int = 44,
    n_trials_prior_override: int | None = None,
    rf_annual: float = 0.0,
) -> GatingDecision:
    """Evaluate walk-forward result against statistical gates.

    Args:
        walk_forward_result: WalkForwardResult to evaluate.
        ledger: TrialLedger for trial count.
        benchmarks: optional dict of benchmark_name -> returns Series.
        td_benchmark_name: optional benchmark key to evaluate via one-sided HAC
            excess return instead of Romano-Wolf dominance.
        sanity_floor: minimum Sharpe threshold (default 0.0).
        alpha: significance level (default 0.05).
        n_resamples: bootstrap resamples (default 2000).
        dominance_n_resamples: optional dominance-test resamples. Defaults to
            n_resamples for backward compatibility.
        rng_seed: RNG seed for reproducibility.
        rf_annual: annual risk-free rate used for bootstrap Sharpe significance.

    Returns:
        GatingDecision with pass/fail and detailed bars.
    """
    oos_returns = walk_forward_result.oos_returns

    if oos_returns.empty:
        # No OOS returns → fail all gates
        return GatingDecision(
            passes=False,
            bars={
                "sanity_floor": (False, "No OOS returns"),
                "bootstrap_ci": (False, "No OOS returns"),
                "deflated_sharpe": (False, "No OOS returns"),
            },
            deflated=DeflatedSharpeResult(
                sharpe_hat=0.0,
                sharpe_deflated=float("nan"),
                p_value=float("nan"),
                n_obs=0,
                n_trials=1,
                passes=False,
                alpha=alpha,
            ),
            bootstrap=BootstrapResult(
                point_estimate=float("nan"),
                mean=float("nan"),
                ci_low_95=float("nan"),
                ci_median=float("nan"),
                ci_high_95=float("nan"),
                n_resamples=0,
                mean_block_len=float("nan"),
            ),
            dominance=None,
            n_kept_folds=walk_forward_result.n_kept_folds,
            n_rejected_folds=walk_forward_result.n_rejected_folds,
            rejection_reasons=walk_forward_result.rejection_reasons,
            notes="No OOS returns available.",
        )

    # Bootstrap
    bootstrap = stationary_bootstrap_sharpe(
        oos_returns,
        n_resamples=n_resamples,
        rng_seed=rng_seed,
        rf_annual=rf_annual,
    )

    # Deflated Sharpe
    # Using oos_sharpe_delisting_adjusted instead of flat haircut for DSR
    n_obs = len(oos_returns)
    effective_n_trials_prior = (
        n_trials_prior_override if n_trials_prior_override is not None else n_trials_prior
    )
    n_trials = effective_n_trials_prior + ledger.n_unique_trials()

    # Compute moments
    skew = float(scipy_stats.skew(oos_returns.values, bias=False)) if n_obs >= 3 else 0.0
    kurt = (
        float(scipy_stats.kurtosis(oos_returns.values, fisher=False, bias=False))
        if n_obs >= 3
        else 3.0
    )

    deflated = deflated_sharpe(
        sharpe_hat=walk_forward_result.oos_sharpe_delisting_adjusted,
        n_obs=n_obs,
        n_trials=n_trials,
        skew=skew,
        kurtosis=kurt,
        alpha=alpha,
    )

    # Dominance
    td_excess = None
    dominance_benchmarks = benchmarks or {}
    if td_benchmark_name is not None and td_benchmark_name not in dominance_benchmarks:
        raise ValueError(f"td_benchmark_name {td_benchmark_name!r} not found in benchmarks")
    if benchmarks and td_benchmark_name in benchmarks:
        td_excess = one_sided_hac_excess_return(
            oos_returns,
            benchmarks[td_benchmark_name],
            alpha=alpha,
        )
        dominance_benchmarks = {
            name: returns for name, returns in benchmarks.items() if name != td_benchmark_name
        }

    if dominance_benchmarks:
        dominance = romano_wolf_stepwise(
            oos_returns,
            dominance_benchmarks,
            alpha=alpha,
            n_resamples=dominance_n_resamples or n_resamples,
            rng_seed=rng_seed + 1,  # different seed for independence
        )
    else:
        dominance = None

    # Construct bars
    bars = {}

    # Sanity floor
    sanity_passed = walk_forward_result.oos_sharpe_flat_haircut > sanity_floor
    bars["sanity_floor"] = (
        sanity_passed,
        f"Sharpe {walk_forward_result.oos_sharpe_flat_haircut:.2f} > {sanity_floor:.2f}"
        if sanity_passed
        else f"Sharpe {walk_forward_result.oos_sharpe_flat_haircut:.2f} ≤ {sanity_floor:.2f}",
    )

    # Bootstrap CI: the documented gate requires the 95% OOS Sharpe interval
    # to clear zero, not merely have a positive point estimate.
    bootstrap_low = bootstrap.ci_low_95
    bootstrap_passed = math.isfinite(bootstrap_low) and bootstrap_low > 0.0
    bars["bootstrap_ci"] = (
        bootstrap_passed,
        f"95% CI low {bootstrap_low:.2f} > 0"
        if bootstrap_passed
        else f"95% CI low {bootstrap_low:.2f} ≤ 0",
    )

    # Deflated Sharpe
    bars["deflated_sharpe"] = (
        deflated.passes,
        f"p={deflated.p_value:.4f} ≤ {alpha}"
        if deflated.passes
        else f"p={deflated.p_value:.4f} > {alpha}",
    )

    if td_excess is not None:
        td_mean_pct = td_excess.mean_excess_annual * 100.0
        bars["td_excess_return"] = (
            td_excess.passes,
            f"Mean excess {td_mean_pct:.2f}% > 0, p={td_excess.p_value:.4f} ≤ {alpha}"
            if td_excess.passes
            else f"Mean excess {td_mean_pct:.2f}% ≤ 0 or p={td_excess.p_value:.4f} > {alpha}",
        )

    # Dominance bars
    if dominance:
        for bench_name in dominance.benchmark_names:
            dom = dominance.dominates[bench_name]
            p_adj = dominance.adjusted_p_values[bench_name]
            bars[f"dominance_{bench_name}"] = (
                dom,
                f"p_adj={p_adj:.4f} ≤ {alpha}"
                if dom
                else f"p_adj={p_adj:.4f} > {alpha}",
            )

    # Overall pass
    passes = all(passed for passed, _ in bars.values())

    # Notes
    notes_parts = [
        f"Kept folds: {walk_forward_result.n_kept_folds}",
        f"Rejected folds: {walk_forward_result.n_rejected_folds}",
        "n_trials: "
        f"{n_trials} (prior: {effective_n_trials_prior}, "
        f"ledger: {ledger.n_unique_trials()})",
    ]
    if walk_forward_result.oos_sharpe_by_regime:
        regime_str = ", ".join(
            f"{k}={v:.2f}" for k, v in walk_forward_result.oos_sharpe_by_regime.items()
        )
        notes_parts.append(f"Per-regime Sharpe: {regime_str}")
    notes = "; ".join(notes_parts)

    return GatingDecision(
        passes=passes,
        bars=bars,
        deflated=deflated,
        bootstrap=bootstrap,
        dominance=dominance,
        n_kept_folds=walk_forward_result.n_kept_folds,
        n_rejected_folds=walk_forward_result.n_rejected_folds,
        rejection_reasons=walk_forward_result.rejection_reasons,
        notes=notes,
    )
