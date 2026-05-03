"""Tests for report_builder."""
from __future__ import annotations

import pandas as pd
import pytest

from skuld_common.contracts import (
    BenchmarkResult,
    BootstrapResult,
    DeflatedSharpeResult,
    DominanceResult,
    GatingDecision,
    WalkForwardResult,
)
from skuld_research.reporting.report_builder import build_methodology_report


def _make_minimal_wf_result(sharpe: float = 0.5) -> WalkForwardResult:
    """Build minimal WalkForwardResult for testing."""
    return WalkForwardResult(
        folds=(),
        oos_returns=pd.Series([0.01, 0.02, -0.01]),
        oos_sharpe_raw=sharpe,
        oos_sharpe_flat_haircut=sharpe - 0.1,
        oos_sharpe_delisting_adjusted=sharpe - 0.05,
        oos_drawdown_observed=pd.Series([0.0, -0.01, -0.02]),
        oos_max_drawdown_observed=-0.05,
        oos_max_drawdown_augmented_median=-0.08,
        oos_max_drawdown_augmented_p90=-0.12,
        oos_avg_turnover=0.15,
        oos_total_cost_nzd=500.0,
        oos_hit_rate=0.6,
        oos_skewness=-0.3,
        oos_calmar_ratio=1.2,
        n_kept_folds=2,
        n_rejected_folds=0,
        rejection_reasons=(),
        oos_sharpe_by_regime={},
    )


def _make_minimal_gating() -> GatingDecision:
    """Build minimal GatingDecision for testing."""
    return GatingDecision(
        passes=True,
        bars={
            "sanity_floor": (True, "Sharpe 0.40 > 0.00"),
            "td_excess_return": (True, "Mean excess 2.00% > 0, p=0.0100 ≤ 0.05"),
            "bootstrap_ci": (True, "95% CI low 0.30 > 0"),
            "dominance_NZX equal-weighted": (True, "p_adj=0.0300 ≤ 0.05"),
            "deflated_sharpe": (True, "p=0.01 ≤ 0.05"),
        },
        deflated=DeflatedSharpeResult(
            sharpe_hat=0.5,
            sharpe_deflated=0.4,
            p_value=0.01,
            n_obs=50,
            n_trials=100,
            passes=True,
            alpha=0.05,
        ),
        bootstrap=BootstrapResult(
            point_estimate=0.5,
            mean=0.48,
            ci_low_95=0.3,
            ci_median=0.5,
            ci_high_95=0.7,
            n_resamples=1000,
            mean_block_len=5.0,
        ),
        dominance=None,
        n_kept_folds=2,
        n_rejected_folds=0,
        rejection_reasons=(),
        notes="Test gating",
    )


def _make_minimal_dominance() -> DominanceResult:
    """Build minimal DominanceResult for testing."""
    return DominanceResult(
        benchmark_names=("NZX equal-weighted", "60/40"),
        adjusted_p_values={
            "NZX equal-weighted": 0.03,
            "60/40": 0.08,
        },
        dominates={
            "NZX equal-weighted": True,
            "60/40": False,
        },
        alpha=0.05,
        n_resamples=2000,
    )


def test_build_methodology_report_pass_fail_bars():
    """Report builder computes expected pass/fail bars."""
    strategy_two_fold = _make_minimal_wf_result(sharpe=0.6)
    strategy_rolling = _make_minimal_wf_result(sharpe=0.5)

    # TD floor benchmark with lower Sharpe
    td_floor_wf = _make_minimal_wf_result(sharpe=0.3)
    td_bench = BenchmarkResult(
        name="NZ TD floor",
        wf_two_fold=td_floor_wf,
        wf_rolling=td_floor_wf,
        coverage_start=pd.Timestamp("2020-01-01"),
        coverage_end=pd.Timestamp("2025-01-01"),
        notes=(),
    )

    # NZX equal-weighted benchmark
    nzx_wf = _make_minimal_wf_result(sharpe=0.4)
    nzx_bench = BenchmarkResult(
        name="NZX equal-weighted",
        wf_two_fold=nzx_wf,
        wf_rolling=nzx_wf,
        coverage_start=pd.Timestamp("2020-01-01"),
        coverage_end=pd.Timestamp("2025-01-01"),
        notes=(),
    )

    benchmarks = (td_bench, nzx_bench)

    gating = _make_minimal_gating()
    dominance = _make_minimal_dominance()

    report = build_methodology_report(
        strategy_name="momentum",
        strategy_two_fold=strategy_two_fold,
        strategy_rolling=strategy_rolling,
        benchmarks=benchmarks,
        gating=gating,
        dominance=dominance,
        config_hash="pre-M7",
        git_sha="abc123",
        asof=pd.Timestamp("2026-01-01"),
        panel_coverage_start=pd.Timestamp("2015-01-01"),
        panel_coverage_end=pd.Timestamp("2025-12-31"),
        master_seed=42,
        n_trials_prior=30,
    )

    # Check pass_fail bars
    assert len(report.pass_fail) >= 5

    # Check specific bars
    bar_names = [bar[0] for bar in report.pass_fail]
    assert "Sanity floor" in bar_names
    assert "TD excess return" in bar_names
    assert "Bootstrap CI" in bar_names
    assert "Benchmark (NZX equal-weighted) via Romano-Wolf" in bar_names
    assert "Deflated Sharpe" in bar_names

    sanity_floor_bar = next(b for b in report.pass_fail if b[0] == "Sanity floor")
    assert sanity_floor_bar == ("Sanity floor", True, "Sharpe 0.40 > 0.00")

    sanity_bar = next(b for b in report.pass_fail if b[0] == "TD excess return")
    assert sanity_bar[1] is True  # passed
    assert "Mean excess 2.00% > 0" in sanity_bar[2]

    bootstrap_bar = next(b for b in report.pass_fail if b[0] == "Bootstrap CI")
    assert bootstrap_bar == ("Bootstrap CI", True, "95% CI low 0.30 > 0")

    # Check deflated passed
    deflated_bar = next(b for b in report.pass_fail if b[0] == "Deflated Sharpe")
    assert deflated_bar[1] is True  # passed


def test_build_methodology_report_uses_gating_bars_for_renamed_benchmark_and_non_default_alpha():
    """Reporting should map labels but preserve pass/fail state and reasons from gating."""
    strategy_wf = _make_minimal_wf_result()
    gating = GatingDecision(
        passes=False,
        bars={
            "dominance_Aggressive 60/40": (True, "p_adj=0.0700 ≤ 0.10"),
            "deflated_sharpe": (False, "p=0.1100 > 0.10"),
        },
        deflated=DeflatedSharpeResult(
            sharpe_hat=0.5,
            sharpe_deflated=0.4,
            p_value=0.11,
            n_obs=50,
            n_trials=100,
            passes=False,
            alpha=0.10,
        ),
        bootstrap=BootstrapResult(
            point_estimate=0.5,
            mean=0.48,
            ci_low_95=0.3,
            ci_median=0.5,
            ci_high_95=0.7,
            n_resamples=1000,
            mean_block_len=5.0,
        ),
        dominance=None,
        n_kept_folds=2,
        n_rejected_folds=0,
        rejection_reasons=(),
        notes="Test gating",
    )
    dominance = DominanceResult(
        benchmark_names=("Different benchmark",),
        adjusted_p_values={"Different benchmark": 0.99},
        dominates={"Different benchmark": False},
        alpha=0.05,
        n_resamples=2000,
    )

    report = build_methodology_report(
        strategy_name="test",
        strategy_two_fold=strategy_wf,
        strategy_rolling=strategy_wf,
        benchmarks=(),
        gating=gating,
        dominance=dominance,
        config_hash="test",
        git_sha="test",
        asof=pd.Timestamp("2026-01-01"),
        panel_coverage_start=pd.Timestamp("2015-01-01"),
        panel_coverage_end=pd.Timestamp("2025-12-31"),
        master_seed=42,
        n_trials_prior=30,
    )

    assert report.pass_fail == (
        ("Benchmark (Aggressive 60/40) via Romano-Wolf", True, "p_adj=0.0700 ≤ 0.10"),
        ("Deflated Sharpe", False, "p=0.1100 > 0.10"),
    )


def test_build_methodology_report_rejects_contradictory_dominance_input():
    """Reporting should fail fast if the separate dominance arg disagrees with gating."""
    strategy_wf = _make_minimal_wf_result()
    gating = _make_minimal_gating()
    gating = GatingDecision(
        passes=gating.passes,
        bars=gating.bars,
        deflated=gating.deflated,
        bootstrap=gating.bootstrap,
        dominance=_make_minimal_dominance(),
        n_kept_folds=gating.n_kept_folds,
        n_rejected_folds=gating.n_rejected_folds,
        rejection_reasons=gating.rejection_reasons,
        notes=gating.notes,
    )
    contradictory = DominanceResult(
        benchmark_names=("Other benchmark",),
        adjusted_p_values={"Other benchmark": 0.99},
        dominates={"Other benchmark": False},
        alpha=0.05,
        n_resamples=2000,
    )

    with pytest.raises(ValueError, match=r"gating\.dominance"):
        build_methodology_report(
            strategy_name="test",
            strategy_two_fold=strategy_wf,
            strategy_rolling=strategy_wf,
            benchmarks=(),
            gating=gating,
            dominance=contradictory,
            config_hash="test",
            git_sha="test",
            asof=pd.Timestamp("2026-01-01"),
            panel_coverage_start=pd.Timestamp("2015-01-01"),
            panel_coverage_end=pd.Timestamp("2025-12-31"),
            master_seed=42,
            n_trials_prior=30,
        )


def test_report_builder_uses_td_excess_return_bar():
    """Reporting should surface the TD excess-return gate directly."""
    strategy_wf = _make_minimal_wf_result(sharpe=0.1)
    td_floor_wf = _make_minimal_wf_result(sharpe=10.0)
    td_bench = BenchmarkResult(
        name="NZ TD floor",
        wf_two_fold=td_floor_wf,
        wf_rolling=td_floor_wf,
        coverage_start=pd.Timestamp("2020-01-01"),
        coverage_end=pd.Timestamp("2025-01-01"),
        notes=(),
    )

    gating = _make_minimal_gating()
    dominance = _make_minimal_dominance()

    report = build_methodology_report(
        strategy_name="momentum",
        strategy_two_fold=strategy_wf,
        strategy_rolling=strategy_wf,
        benchmarks=(td_bench,),
        gating=gating,
        dominance=dominance,
        config_hash="pre-M7",
        git_sha="abc123",
        asof=pd.Timestamp("2026-01-01"),
        panel_coverage_start=pd.Timestamp("2015-01-01"),
        panel_coverage_end=pd.Timestamp("2025-12-31"),
        master_seed=42,
        n_trials_prior=30,
    )

    td_bar = next(b for b in report.pass_fail if b[0] == "TD excess return")
    assert td_bar[1] is True
    assert td_bar[2] == "Mean excess 2.00% > 0, p=0.0100 ≤ 0.05"


def test_build_methodology_report_includes_seed_note():
    """Report includes RNG master seed derivation note."""
    strategy_wf = _make_minimal_wf_result()
    gating = _make_minimal_gating()
    dominance = _make_minimal_dominance()

    report = build_methodology_report(
        strategy_name="test",
        strategy_two_fold=strategy_wf,
        strategy_rolling=strategy_wf,
        benchmarks=(),
        gating=gating,
        dominance=dominance,
        config_hash="test",
        git_sha="test",
        asof=pd.Timestamp("2026-01-01"),
        panel_coverage_start=pd.Timestamp("2015-01-01"),
        panel_coverage_end=pd.Timestamp("2025-12-31"),
        master_seed=42,
        n_trials_prior=30,
    )

    # Check seed note is present
    assert "bootstrap" in report.rng_master_seed_note.lower()
    assert "spawn" in report.rng_master_seed_note.lower()
