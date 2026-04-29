"""Tests for markdown_writer determinism."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd

from skuld_common.contracts import (
    BenchmarkResult,
    BootstrapResult,
    DeflatedSharpeResult,
    DominanceResult,
    GatingDecision,
    MethodologyReport,
    WalkForwardResult,
)


def _make_test_report() -> MethodologyReport:
    """Build a minimal MethodologyReport for testing."""
    wf = WalkForwardResult(
        folds=(),
        oos_returns=pd.Series([0.01, 0.02]),
        oos_sharpe_raw=0.5,
        oos_sharpe_flat_haircut=0.4,
        oos_sharpe_delisting_adjusted=0.45,
        oos_drawdown_observed=pd.Series([0.0, -0.01]),
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
        oos_sharpe_by_regime={"bull": 0.6, "bear": 0.3},
    )
    
    bench = BenchmarkResult(
        name="NZ TD floor",
        wf_two_fold=wf,
        wf_rolling=wf,
        coverage_start=pd.Timestamp("2020-01-01"),
        coverage_end=pd.Timestamp("2025-01-01"),
        notes=("yield-only bond proxy",),
    )
    
    gating = GatingDecision(
        passes=True,
        bars={"sanity_floor": (True, "Sharpe > 0")},
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
        notes="Test",
    )
    
    dominance = DominanceResult(
        benchmark_names=("NZ TD floor",),
        adjusted_p_values={"NZ TD floor": 0.01},
        dominates={"NZ TD floor": True},
        alpha=0.05,
        n_resamples=2000,
    )
    
    return MethodologyReport(
        config_hash="pre-M7",
        git_sha="abc123",
        asof=pd.Timestamp("2026-01-01"),
        panel_coverage_start=pd.Timestamp("2015-01-01"),
        panel_coverage_end=pd.Timestamp("2025-12-31"),
        master_seed=42,
        n_trials_prior=30,
        rng_master_seed_note="Test spawn order",
        strategy_name="momentum",
        strategy_two_fold=wf,
        strategy_rolling=wf,
        benchmarks=(bench,),
        gating=gating,
        dominance=dominance,
        pass_fail=(("Sanity floor", True, "Passed"),),
    )


def test_markdown_writer_determinism():
    """Write report twice, assert SHA-256 equal."""
    from skuld_research.reporting.markdown_writer import write_methodology_report
    
    report = _make_test_report()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "report1.md"
        path2 = Path(tmpdir) / "report2.md"
        
        write_methodology_report(report, path1)
        write_methodology_report(report, path2)
        
        hash1 = hashlib.sha256(path1.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(path2.read_bytes()).hexdigest()
        
        assert hash1 == hash2


def test_markdown_writer_contains_expected_sections():
    """Written markdown contains expected section headers."""
    from skuld_research.reporting.markdown_writer import write_methodology_report
    
    report = _make_test_report()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.md"
        write_methodology_report(report, path)
        
        content = path.read_text(encoding="utf-8")
        
        # Check for expected sections
        assert "## Strategy" in content
        assert "## Benchmarks" in content
        assert "## Dominance" in content
        assert "## Gating Decision" in content
        assert "## Pass / Fail" in content
        assert "NZ TD floor" in content
