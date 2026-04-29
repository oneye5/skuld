"""Build MethodologyReport from walk-forward results and statistics."""
from __future__ import annotations

import pandas as pd

from skuld_common.contracts import (
    BenchmarkResult,
    DominanceResult,
    GatingDecision,
    MethodologyReport,
    WalkForwardResult,
)
from skuld_research.reporting._seeds import SPAWN_ORDER_DOC


def build_methodology_report(
    strategy_name: str,
    strategy_two_fold: WalkForwardResult,
    strategy_rolling: WalkForwardResult,
    benchmarks: tuple[BenchmarkResult, ...],
    gating: GatingDecision,
    dominance: DominanceResult,
    config_hash: str,
    git_sha: str,
    asof: pd.Timestamp,
    panel_coverage_start: pd.Timestamp,
    panel_coverage_end: pd.Timestamp,
    master_seed: int,
    n_trials_prior: int,
) -> MethodologyReport:
    """Build a MethodologyReport from walk-forward results and statistics.
    
    Pure function; no I/O, no time-of-day reads.
    
    Args:
        strategy_name: e.g., "momentum_only".
        strategy_two_fold: WalkForwardResult from 2-fold driver.
        strategy_rolling: WalkForwardResult from rolling driver (gating reference).
        benchmarks: tuple of BenchmarkResult.
        gating: GatingDecision from M5.
        dominance: DominanceResult from Romano-Wolf stepwise.
        config_hash: SHA-256 placeholder ("pre-M7" until M7 lands).
        git_sha: short SHA from git rev-parse.
        asof: PIT cutoff used.
        panel_coverage_start: panel.returns_daily.index[0].
        panel_coverage_end: panel.returns_daily.index[-1].
        master_seed: master RNG seed.
        n_trials_prior: prior trial count.
    
    Returns:
        MethodologyReport frozen dataclass.
    """
    # Compute pass/fail bars
    pass_fail_bars = []
    
    # 1. Sanity floor (TD floor)
    td_floor_bench = next((b for b in benchmarks if b.name == "NZ TD floor"), None)
    if td_floor_bench:
        strategy_sharpe = strategy_rolling.oos_sharpe_delisting_adjusted
        td_sharpe = td_floor_bench.wf_rolling.oos_sharpe_delisting_adjusted
        passed = strategy_sharpe > td_sharpe
        reason = (
            f"Strategy Sharpe {strategy_sharpe:.3f} > TD floor Sharpe {td_sharpe:.3f}"
            if passed
            else f"Strategy Sharpe {strategy_sharpe:.3f} ≤ TD floor Sharpe {td_sharpe:.3f}"
        )
        pass_fail_bars.append(("Sanity floor (TD floor)", passed, reason))
    
    # 2. Primary benchmark (NZX equal-weighted) via Romano-Wolf
    nzx_bench_name = "NZX equal-weighted"
    if nzx_bench_name in dominance.dominates:
        dominates_nzx = dominance.dominates[nzx_bench_name]
        p_adj = dominance.adjusted_p_values[nzx_bench_name]
        passed = dominates_nzx and p_adj <= 0.05
        reason = (
            f"Dominates={dominates_nzx}, p_adj={p_adj:.4f} ≤ 0.05"
            if passed
            else f"Dominates={dominates_nzx}, p_adj={p_adj:.4f} > 0.05"
        )
        pass_fail_bars.append(("Primary benchmark (NZX equal-weighted) via Romano-Wolf", passed, reason))
    
    # 3. Deflated Sharpe
    deflated_passed = gating.deflated.passes
    reason = (
        f"Deflated Sharpe p={gating.deflated.p_value:.4f} ≤ {gating.deflated.alpha}"
        if deflated_passed
        else f"Deflated Sharpe p={gating.deflated.p_value:.4f} > {gating.deflated.alpha}"
    )
    pass_fail_bars.append(("Deflated Sharpe", deflated_passed, reason))
    
    return MethodologyReport(
        config_hash=config_hash,
        git_sha=git_sha,
        asof=asof,
        panel_coverage_start=panel_coverage_start,
        panel_coverage_end=panel_coverage_end,
        master_seed=master_seed,
        n_trials_prior=n_trials_prior,
        rng_master_seed_note=SPAWN_ORDER_DOC,
        strategy_name=strategy_name,
        strategy_two_fold=strategy_two_fold,
        strategy_rolling=strategy_rolling,
        benchmarks=benchmarks,
        gating=gating,
        dominance=dominance,
        pass_fail=tuple(pass_fail_bars),
    )
