"""Tests for Romano-Wolf stepwise multiple testing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_research.stats.dominance import romano_wolf_stepwise


def test_strategy_beats_benchmark_with_constant_alpha():
    """Strategy = benchmark + constant alpha → dominates at alpha=0.05."""
    rng = np.random.default_rng(400)
    n = 120
    benchmark_ret = pd.Series(0.005 + 0.02 * rng.standard_normal(n))
    strategy_ret = benchmark_ret + 0.05 / 12  # +5% annualised alpha

    result = romano_wolf_stepwise(
        strategy_ret,
        {"bench": benchmark_ret},
        alpha=0.05,
        n_resamples=500,
        rng_seed=42,
    )

    assert result.benchmark_names == ("bench",)
    assert result.dominates["bench"] == True
    assert result.adjusted_p_values["bench"] <= 0.05


def test_strategy_equals_benchmark_no_dominance():
    """Strategy = benchmark + noise → does not dominate."""
    rng = np.random.default_rng(500)
    n = 100
    benchmark_ret = pd.Series(0.005 + 0.02 * rng.standard_normal(n))
    noise = 0.01 * rng.standard_normal(n)
    strategy_ret = benchmark_ret + noise

    result = romano_wolf_stepwise(
        strategy_ret,
        {"bench": benchmark_ret},
        alpha=0.05,
        n_resamples=500,
        rng_seed=99,
    )

    assert result.dominates["bench"] == False


def test_multiple_benchmarks():
    """Multiple benchmarks: dominates some but not others."""
    rng = np.random.default_rng(600)
    n = 100
    bench1 = pd.Series(0.003 + 0.015 * rng.standard_normal(n))
    bench2 = pd.Series(0.008 + 0.025 * rng.standard_normal(n))
    # Strategy beats bench1 clearly, but is close to bench2
    strategy = bench1 + 0.04 / 12

    result = romano_wolf_stepwise(
        strategy,
        {"weak": bench1, "strong": bench2},
        alpha=0.05,
        n_resamples=500,
        rng_seed=77,
    )

    assert "weak" in result.benchmark_names
    assert "strong" in result.benchmark_names
    # Expect to dominate the weak benchmark
    assert result.dominates["weak"] == True


def test_determinism():
    """Two calls with same seed produce identical adjusted p-values."""
    rng = np.random.default_rng(700)
    n = 80
    bench = pd.Series(0.004 + 0.018 * rng.standard_normal(n))
    strat = bench + 0.03 / 12

    r1 = romano_wolf_stepwise(strat, {"b": bench}, alpha=0.05, n_resamples=300, rng_seed=111)
    r2 = romano_wolf_stepwise(strat, {"b": bench}, alpha=0.05, n_resamples=300, rng_seed=111)

    # Use abs for float comparison
    assert abs(r1.adjusted_p_values["b"] - r2.adjusted_p_values["b"]) < 1e-9
    assert r1.dominates["b"] == r2.dominates["b"]
