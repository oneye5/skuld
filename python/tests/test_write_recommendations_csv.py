"""Tests for skuld_portfolio.output.write_recommendations_csv."""
from pathlib import Path

import pandas as pd
import pytest

from skuld_common.contracts import TradeList
from skuld_portfolio.output.write_recommendations_csv import write_recommendations_csv
from skuld_research.config.loader import load_spec


@pytest.fixture
def minimal_trade_list() -> TradeList:
    """Minimal valid TradeList for testing."""
    trades_df = pd.DataFrame({
        "ticker": ["AIR", "FBU"],
        "action": ["BUY", "HOLD"],
        "current_shares": [0, 100],
        "target_shares": [50, 100],
        "delta_shares": [50, 0],
        "current_value_nzd": [0.0, 400.0],
        "target_value_nzd": [125.0, 400.0],
        "delta_value_nzd": [125.0, 0.0],
        "est_round_trip_cost_nzd": [2.50, 0.0],
        "in_no_trade_region": [False, True],
        "below_size_floor": [False, False],
        "deferred_to_next_month": [False, False],
        "sharesies_fee_band": ["flat_15", "subscription_only"],
    })
    return TradeList(
        trades=trades_df,
        total_volume_nzd=125.0,
        total_estimated_cost_nzd=17.50,
        asof=pd.Timestamp("2026-01-01", tz="UTC"),
        config_hash="test_hash",
    )


@pytest.fixture
def test_spec():
    """Load the phase1_baseline spec for testing."""
    python_root = Path(__file__).parent.parent
    spec_path = python_root / "configs" / "preregistered" / "2026-04-26_phase1_baseline.yaml"
    return load_spec(spec_path)


def test_write_recommendations_csv_creates_files(tmp_path: Path, minimal_trade_list: TradeList, test_spec):
    """Write CSV, meta JSON, and overrides log."""
    output_path = tmp_path / "recommendations_2026-01-01.csv"
    meta = {
        "spec_hash": "test_hash",
        "spec_path": "test.yaml",
        "asof": "2026-01-01",
        "low_confidence": False,
        "total_volume_nzd": 125.0,
        "total_estimated_cost_nzd": 17.50,
    }
    
    write_recommendations_csv(minimal_trade_list, test_spec, meta, output_path)
    
    # Check main CSV exists
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 2
    assert "ticker" in df.columns
    assert "action" in df.columns
    assert "rebalance_date" in df.columns
    
    # Check meta JSON exists
    meta_path = output_path.with_suffix(".meta.json")
    assert meta_path.exists()
    import json
    with open(meta_path, "r") as f:
        loaded_meta = json.load(f)
    assert loaded_meta["spec_hash"] == "test_hash"
    
    # Check overrides log exists
    overrides_path = output_path.parent / "overrides_log_2026-01-01.csv"
    assert overrides_path.exists()
    overrides_df = pd.read_csv(overrides_path)
    assert "ticker" in overrides_df.columns
    assert "override_action" in overrides_df.columns


def test_write_recommendations_csv_required_columns(tmp_path: Path, minimal_trade_list: TradeList, test_spec):
    """Output CSV has all required columns from §3.9."""
    output_path = tmp_path / "test.csv"
    meta = {"spec_hash": "x", "spec_path": "x", "asof": "2026-01-01", "low_confidence": False}
    
    write_recommendations_csv(minimal_trade_list, test_spec, meta, output_path)
    
    df = pd.read_csv(output_path)
    
    # Required columns from §3.9
    required = [
        "rebalance_date", "ticker", "action",
        "current_shares", "target_shares", "delta_shares",
        "current_value_nzd", "target_value_nzd", "delta_value_nzd",
        "current_weight", "target_weight",
        "combined_score_z",
        "est_round_trip_cost_nzd",
        "sharesies_fee_band",
        "in_no_trade_region", "below_size_floor", "deferred_to_next_month",
        "rationale",
    ]
    
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Factor columns: spec has momentum only
    assert "factor_momentum_z" in df.columns


def test_write_recommendations_csv_factor_columns_match_spec(tmp_path: Path, minimal_trade_list: TradeList, test_spec):
    """Factor columns are derived from spec.factors order."""
    output_path = tmp_path / "test.csv"
    meta = {"spec_hash": "x", "spec_path": "x", "asof": "2026-01-01", "low_confidence": False}
    
    write_recommendations_csv(minimal_trade_list, test_spec, meta, output_path)
    
    df = pd.read_csv(output_path)
    
    # Phase1_baseline has only momentum
    assert "factor_momentum_z" in df.columns
    # Should NOT have other factors
    assert "factor_value_z" not in df.columns
    assert "factor_quality_z" not in df.columns
