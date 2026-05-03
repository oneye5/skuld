"""Tests for markdown report writer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from skuld_common.contracts import DecayReport, DecompositionReport, ICReport


def test_write_diagnostics_report_basic_structure(tmp_path: Path):
    """Report contains expected sections."""
    from skuld_research.diagnostics.report import write_diagnostics_report

    # Create minimal reports
    ic_series = pd.Series([0.1, 0.2, 0.15], index=pd.date_range("2024-01-31", periods=3, freq="BME"))

    ic = ICReport(
        factor_name="test_factor",
        horizon_months=1,
        ic_series=ic_series,
        ic_mean=0.15,
        ic_std=0.05,
        ic_ir=2.0,
        t_stat_newey_west=3.5,
        n_obs=3,
        min_universe_per_date=10,
    )

    decay = DecayReport(
        factor_name="test_factor",
        horizons=(1, 3, 6),
        ic_by_horizon={
            1: ic,
            3: ICReport("test_factor", 3, pd.Series([0.12]), 0.12, 0.01, 1.5, 2.0, 1, 10),
            6: ICReport("test_factor", 6, pd.Series([0.08]), 0.08, 0.02, 1.0, 1.2, 1, 10),
        },
        peak_horizon=1,
    )

    decomp = DecompositionReport(
        regressors=("market", "momentum"),
        coefficients={"market": 0.5, "momentum": 0.8},
        t_stats={"market": 2.5, "momentum": 4.2},
        residual_alpha_annualised=0.025,
        residual_alpha_t_stat=1.8,
        r_squared=0.65,
        n_obs=50,
    )

    out_path = tmp_path / "test_report.md"
    write_diagnostics_report(ic, decay, decomp, out_path)

    assert out_path.exists()
    content = out_path.read_text()

    # Check for key sections
    assert "# Signal Diagnostics" in content or "# Diagnostics" in content
    assert "test_factor" in content
    assert "IC" in content or "Information Coefficient" in content
    assert "Decay" in content or "decay" in content
    assert "Decomposition" in content or "decomposition" in content


def test_write_diagnostics_report_reproducibility(tmp_path: Path):
    """Two consecutive runs produce byte-identical output."""
    from skuld_research.diagnostics.report import write_diagnostics_report

    ic_series = pd.Series([0.15, 0.18, 0.12], index=pd.date_range("2024-01-31", periods=3, freq="BME"))

    ic = ICReport(
        factor_name="momentum",
        horizon_months=1,
        ic_series=ic_series,
        ic_mean=0.15,
        ic_std=0.03,
        ic_ir=2.5,
        t_stat_newey_west=4.0,
        n_obs=3,
        min_universe_per_date=15,
    )

    decay = DecayReport(
        factor_name="momentum",
        horizons=(1, 2, 3, 6, 12),
        ic_by_horizon={
            1: ic,
            2: ICReport("momentum", 2, pd.Series([0.14]), 0.14, 0.02, 2.3, 3.8, 1, 15),
            3: ICReport("momentum", 3, pd.Series([0.13]), 0.13, 0.02, 2.1, 3.5, 1, 15),
            6: ICReport("momentum", 6, pd.Series([0.10]), 0.10, 0.03, 1.5, 2.5, 1, 15),
            12: ICReport("momentum", 12, pd.Series([0.07]), 0.07, 0.04, 1.0, 1.8, 1, 15),
        },
        peak_horizon=1,
    )

    decomp = DecompositionReport(
        regressors=("market", "momentum"),
        coefficients={"market": 0.45, "momentum": 0.75},
        t_stats={"market": 2.8, "momentum": 4.5},
        residual_alpha_annualised=0.032,
        residual_alpha_t_stat=2.1,
        r_squared=0.72,
        n_obs=60,
    )

    # Write twice
    out1 = tmp_path / "report1.md"
    out2 = tmp_path / "report2.md"

    write_diagnostics_report(ic, decay, decomp, out1)
    write_diagnostics_report(ic, decay, decomp, out2)

    # Byte-identical check
    content1 = out1.read_bytes()
    content2 = out2.read_bytes()

    assert content1 == content2

    # Also verify with hash
    hash1 = hashlib.sha256(content1).hexdigest()
    hash2 = hashlib.sha256(content2).hexdigest()

    assert hash1 == hash2


def test_write_diagnostics_report_float_formatting(tmp_path: Path):
    """Floats are formatted with consistent precision."""
    from skuld_research.diagnostics.report import write_diagnostics_report

    ic = ICReport(
        factor_name="test",
        horizon_months=1,
        ic_series=pd.Series([0.123456789]),
        ic_mean=0.123456789,
        ic_std=0.045678901,
        ic_ir=2.345678901,
        t_stat_newey_west=3.456789012,
        n_obs=1,
        min_universe_per_date=10,
    )

    decay = DecayReport(
        factor_name="test",
        horizons=(1,),
        ic_by_horizon={1: ic},
        peak_horizon=1,
    )

    decomp = DecompositionReport(
        regressors=("market",),
        coefficients={"market": 0.567890123},
        t_stats={"market": 2.678901234},
        residual_alpha_annualised=0.034567890,
        residual_alpha_t_stat=1.789012345,
        r_squared=0.678901234,
        n_obs=50,
    )

    out_path = tmp_path / "formatted.md"
    write_diagnostics_report(ic, decay, decomp, out_path)

    content = out_path.read_text()

    # Check that numbers don't have excessive precision (max 4 decimal places)
    # Should NOT find numbers like 0.123456789
    assert "0.123456789" not in content
    assert "0.045678901" not in content
