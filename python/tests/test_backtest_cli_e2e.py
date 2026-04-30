"""End-to-end test for backtest.py CLI."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from skuld_research.config import load_spec, spec_hash


@pytest.mark.slow
def test_backtest_cli_e2e():
    """Real-data backtest.py run produces report with correct spec hash (done-when d)."""
    # Find python root
    python_root = Path(__file__).parent.parent

    # Path to the strategy spec
    spec_path = python_root / "configs" / "strategy-specs" / "archive" / "m8-mom.yaml"

    # Load spec and compute hash
    spec = load_spec(spec_path)
    expected_hash = spec_hash(spec)

    # Run backtest.py
    result = subprocess.run(
        [
            sys.executable,
            "scripts/backtest.py",
            "--spec",
            str(spec_path),
        ],
        cwd=python_root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    # Check exit code
    assert result.returncode == 0, f"backtest.py failed:\n{result.stderr}"

    # Check report file exists
    report_path = python_root / "reports" / "2026-01-01_methodology.md"
    assert report_path.exists(), f"Report not found at {report_path}"

    # Read report and check for config hash
    report_text = report_path.read_text(encoding="utf-8")

    # Should contain the actual hash, not "pre-M7"
    assert "pre-M7" not in report_text, "Report still contains placeholder 'pre-M7'"
    assert expected_hash in report_text, f"Report does not contain expected hash {expected_hash}"

    # Check that the hash appears in the Config hash section
    assert f"**Config hash:** `{expected_hash}`" in report_text or \
           f"Config hash: {expected_hash}" in report_text or \
           f"Config hash:** {expected_hash}" in report_text, \
           "Config hash not found in expected format in report header"
