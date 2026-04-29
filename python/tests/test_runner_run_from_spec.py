"""Tests for run_from_spec."""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skuld_research.config import BacktestSpec, MomentumFactorSpec, run_from_spec


@pytest.fixture
def tiny_synthetic_panel_csv(tmp_path: Path) -> Path:
    """Create a tiny synthetic panel CSV for fast testing."""
    # 5 tickers, ~400 business days
    np.random.seed(12345)
    
    tickers = ["TK1.NZ", "TK2.NZ", "TK3.NZ", "TK4.NZ", "TK5.NZ"]
    dates = pd.bdate_range("2020-01-01", "2021-12-31", freq="B", tz="UTC")
    
    rows = []
    
    # Add price data
    for ticker in tickers:
        for date in dates:
            price = 100 + np.random.randn() * 10
            volume = int(10_000 + np.random.randn() * 1000)
            timestamp_ms = int(date.value // 1_000_000)  # nanoseconds to milliseconds
            
            rows.append({
                "timestamp": timestamp_ms,
                "ticker": ticker,
                "feature": "adj_close",
                "value": max(price, 1.0),
            })
            
            rows.append({
                "timestamp": timestamp_ms,
                "ticker": ticker,
                "feature": "volume",
                "value": max(volume, 1),
            })
    
    # Add fundamentals for market cap calculation
    for ticker in tickers:
        for date in pd.date_range("2020-01-01", "2021-12-31", freq="QE", tz="UTC"):
            shares = 1_000_000
            timestamp_ms = int(date.value // 1_000_000)
            
            rows.append({
                "timestamp": timestamp_ms,
                "ticker": ticker,
                "feature": "trailing_basic_average_shares",
                "value": shares,
            })
    
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


def test_run_from_spec_byte_identical_on_reruns(tiny_synthetic_panel_csv: Path, tmp_path: Path):
    """Same spec produces byte-identical numeric tables across two runs (done-when a)."""
    from skuld_research.config import SurvivorshipSpec, GatingSpec, RollingDriverSpec, WalkForwardSpec
    
    spec = BacktestSpec(
        name="test_determinism",
        asof=datetime.date(2022, 1, 1),
        master_seed=42,
        factors=[MomentumFactorSpec(min_months=11)],
        survivorship=SurvivorshipSpec(monte_carlo_seeds=10),
        gating=GatingSpec(
            bootstrap_n_resamples=50,
            dominance_n_resamples=50,
        ),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,  # not enough data for 2-fold
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )
    
    # Run twice
    result1 = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )
    
    result2 = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )
    
    # Check byte-equality of OOS returns
    assert result1.strategy_rolling.oos_returns.values.tobytes() == \
           result2.strategy_rolling.oos_returns.values.tobytes()
    
    # Check float fields
    assert result1.strategy_rolling.oos_sharpe_raw == result2.strategy_rolling.oos_sharpe_raw
    assert result1.strategy_rolling.oos_sharpe_delisting_adjusted == \
           result2.strategy_rolling.oos_sharpe_delisting_adjusted


def test_run_from_spec_ledger_deduplication(tiny_synthetic_panel_csv: Path, tmp_path: Path):
    """Ledger entry is deduplicated correctly on second run (done-when c)."""
    from skuld_research.config import SurvivorshipSpec, GatingSpec, RollingDriverSpec, WalkForwardSpec
    
    ledger_root = tmp_path / "ledger"
    
    spec = BacktestSpec(
        name="test_ledger_dedup",
        asof=datetime.date(2022, 1, 1),
        master_seed=42,
        factors=[MomentumFactorSpec(min_months=11)],
        survivorship=SurvivorshipSpec(monte_carlo_seeds=10),
        gating=GatingSpec(
            bootstrap_n_resamples=50,
            dominance_n_resamples=50,
        ),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,  # not enough data for 2-fold
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )
    
    # Run twice with write_ledger=True
    result1 = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=True,
        ledger_root=ledger_root,
    )
    
    result2 = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=True,
        ledger_root=ledger_root,
    )
    
    # Check that both runs have the same spec_hash
    assert result1.spec_hash == result2.spec_hash
    
    # Load ledger and check for unique spec_hash
    from skuld_research.stats.ledger import TrialLedger
    
    ledger = TrialLedger(ledger_root, spec.output.ledger_scope)
    entries = ledger.all_entries()
    
    spec_hashes = [e["spec_hash"] for e in entries]
    unique_hashes = set(spec_hashes)
    
    # Should have exactly one unique hash (deduplication worked)
    assert len(unique_hashes) == 1
    assert result1.spec_hash in unique_hashes


def test_run_from_spec_rejects_unsupported_nzx_adv_floor(
    tiny_synthetic_panel_csv: Path,
) -> None:
    """Non-default NZX ADV floors must not be silently ignored."""
    from skuld_research.config import BenchmarksSpec, RollingDriverSpec, WalkForwardSpec

    spec = BacktestSpec(
        name="test_unsupported_nzx_adv_floor",
        asof=datetime.date(2022, 1, 1),
        factors=[MomentumFactorSpec(min_months=11)],
        benchmarks=BenchmarksSpec(nzx_eq_adv_floor_shares=25_000),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    with pytest.raises(NotImplementedError, match="ADV filter"):
        run_from_spec(
            spec,
            raw_csv_path=tiny_synthetic_panel_csv,
            write_ledger=False,
        )
