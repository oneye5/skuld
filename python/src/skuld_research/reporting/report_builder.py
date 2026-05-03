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


def _pass_fail_label(bar_name: str) -> str:
    """Map gating bar keys onto stable human-readable report labels."""
    if bar_name == "sanity_floor":
        return "Sanity floor"
    if bar_name == "bootstrap_ci":
        return "Bootstrap CI"
    if bar_name == "deflated_sharpe":
        return "Deflated Sharpe"
    if bar_name == "td_excess_return":
        return "TD excess return"
    if bar_name.startswith("dominance_"):
        benchmark_name = bar_name.removeprefix("dominance_")
        return f"Benchmark ({benchmark_name}) via Romano-Wolf"
    return bar_name


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
        strategy_name: e.g., "mom-ar-spread".
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
    if gating.dominance is not None and dominance != gating.dominance:
        raise ValueError("dominance must match gating.dominance")

    pass_fail_bars = [
        (_pass_fail_label(bar_name), passed, reason)
        for bar_name, (passed, reason) in gating.bars.items()
    ]

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
        dominance=gating.dominance or dominance,
        pass_fail=tuple(pass_fail_bars),
    )
