"""Gating decision for walk-forward results."""
from __future__ import annotations

import pandas as pd
from scipy import stats as scipy_stats

from skuld_common.contracts import GatingDecision, WalkForwardResult
from skuld_research.stats.bootstrap import stationary_bootstrap_sharpe
from skuld_research.stats.deflated import deflated_sharpe
from skuld_research.stats.dominance import romano_wolf_stepwise
from skuld_research.stats.ledger import TrialLedger, n_trials_prior


def evaluate(
    walk_forward_result: WalkForwardResult,
    ledger: TrialLedger,
    *,
    benchmarks: dict[str, pd.Series] | None = None,
    sanity_floor: float = 0.0,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    rng_seed: int = 44,
) -> GatingDecision:
    """Evaluate walk-forward result against statistical gates.
    
    Args:
        walk_forward_result: WalkForwardResult to evaluate.
        ledger: TrialLedger for trial count.
        benchmarks: optional dict of benchmark_name -> returns Series.
        sanity_floor: minimum Sharpe threshold (default 0.0).
        alpha: significance level (default 0.05).
        n_resamples: bootstrap resamples (default 2000).
        rng_seed: RNG seed for reproducibility.
    
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
                "deflated_sharpe": (False, "No OOS returns"),
            },
            deflated=deflated_sharpe(0.0, 0, 1, alpha=alpha),
            bootstrap=stationary_bootstrap_sharpe(pd.Series([0.0]), n_resamples=10, rng_seed=rng_seed),
            dominance=None,
            n_kept_folds=walk_forward_result.n_kept_folds,
            n_rejected_folds=walk_forward_result.n_rejected_folds,
            rejection_reasons=walk_forward_result.rejection_reasons,
            notes="No OOS returns available.",
        )
    
    # Bootstrap
    bootstrap = stationary_bootstrap_sharpe(oos_returns, n_resamples=n_resamples, rng_seed=rng_seed)
    
    # Deflated Sharpe
    n_obs = len(oos_returns)
    n_trials = n_trials_prior + ledger.n_unique_trials()
    
    # Compute moments
    skew = float(scipy_stats.skew(oos_returns.values, bias=False)) if n_obs >= 3 else 0.0
    kurt = float(scipy_stats.kurtosis(oos_returns.values, fisher=False, bias=False)) if n_obs >= 3 else 3.0
    
    deflated = deflated_sharpe(
        sharpe_hat=walk_forward_result.oos_sharpe_flat_haircut,
        n_obs=n_obs,
        n_trials=n_trials,
        skew=skew,
        kurtosis=kurt,
        alpha=alpha,
    )
    
    # Dominance
    if benchmarks:
        dominance = romano_wolf_stepwise(
            oos_returns,
            benchmarks,
            alpha=alpha,
            n_resamples=n_resamples,
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
    
    # Deflated Sharpe
    bars["deflated_sharpe"] = (
        deflated.passes,
        f"p={deflated.p_value:.4f} ≤ {alpha}" if deflated.passes else f"p={deflated.p_value:.4f} > {alpha}",
    )
    
    # Dominance bars
    if dominance:
        for bench_name in dominance.benchmark_names:
            dom = dominance.dominates[bench_name]
            p_adj = dominance.adjusted_p_values[bench_name]
            bars[f"dominance_{bench_name}"] = (
                dom,
                f"p_adj={p_adj:.4f} ≤ {alpha}" if dom else f"p_adj={p_adj:.4f} > {alpha}",
            )
    
    # Overall pass
    passes = all(passed for passed, _ in bars.values())
    
    # Notes
    notes_parts = [
        f"Kept folds: {walk_forward_result.n_kept_folds}",
        f"Rejected folds: {walk_forward_result.n_rejected_folds}",
        f"n_trials: {n_trials} (prior: {n_trials_prior}, ledger: {ledger.n_unique_trials()})",
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
