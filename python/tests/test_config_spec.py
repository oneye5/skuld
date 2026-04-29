"""Tests for BacktestSpec schema."""
from __future__ import annotations

import datetime

import pytest
from pydantic import ValidationError

from skuld_research.config import (
    BacktestSpec,
    LowVolatilityFactorSpec,
    MomentumFactorSpec,
    SizeFactorSpec,
    spec_hash,
)


def test_spec_round_trip():
    """Spec round-trips through model_dump / model_validate."""
    spec = BacktestSpec(
        name="test_spec",
        description="Test description",
        asof=datetime.date(2026, 1, 1),
        master_seed=42,
        factors=[MomentumFactorSpec(min_months=11)],
    )
    
    dumped = spec.model_dump()
    reloaded = BacktestSpec.model_validate(dumped)
    
    assert reloaded.name == "test_spec"
    assert reloaded.description == "Test description"
    assert reloaded.asof == datetime.date(2026, 1, 1)
    assert reloaded.master_seed == 42
    assert len(reloaded.factors) == 1
    assert reloaded.factors[0].kind == "momentum"
    assert reloaded.factors[0].min_months == 11


def test_spec_extra_forbid():
    """extra='forbid' rejects unknown keys."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BacktestSpec.model_validate({
            "name": "test",
            "asof": "2026-01-01",
            "unknown_field": "should_fail",
        })


def test_spec_hash_stable():
    """Default-built spec produces a stable 64-char hex hash."""
    spec = BacktestSpec(
        name="test",
        asof=datetime.date(2026, 1, 1),
    )
    
    h = spec_hash(spec)
    
    # Check it's a valid hex string
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
    
    # Optional: pin exact hash for determinism check
    # (Comment out if spec defaults change; re-pin after stabilization)
    # expected_hash = "REPLACE_WITH_ACTUAL_HASH_AFTER_FIRST_RUN"
    # assert h == expected_hash


def test_spec_hash_changes_on_field_change():
    """spec_hash changes if any leaf field changes."""
    spec1 = BacktestSpec(
        name="test",
        asof=datetime.date(2026, 1, 1),
        master_seed=42,
    )
    
    spec2 = BacktestSpec(
        name="test",
        asof=datetime.date(2026, 1, 1),
        master_seed=43,  # different seed
    )
    
    h1 = spec_hash(spec1)
    h2 = spec_hash(spec2)
    
    assert h1 != h2


def test_spec_hash_deterministic():
    """Two equivalent specs yield identical hashes."""
    spec1 = BacktestSpec(
        name="test",
        asof=datetime.date(2026, 1, 1),
        master_seed=42,
    )
    
    spec2 = BacktestSpec(
        name="test",
        asof=datetime.date(2026, 1, 1),
        master_seed=42,
    )
    
    h1 = spec_hash(spec1)
    h2 = spec_hash(spec2)
    
    assert h1 == h2


def test_nested_spec_extra_forbid():
    """Nested specs also reject extra keys."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BacktestSpec.model_validate({
            "name": "test",
            "asof": "2026-01-01",
            "universe": {
                "min_adv_dollars": 10000.0,
                "unknown_field": "should_fail",
            },
        })


def test_low_volatility_factor_spec_round_trip():
    """LowVolatilityFactorSpec round-trips correctly."""
    spec = BacktestSpec(
        name="test_low_vol",
        asof=datetime.date(2026, 1, 1),
        factors=[
            LowVolatilityFactorSpec(lookback_months=12, min_months=6),
        ],
    )
    
    dumped = spec.model_dump()
    reloaded = BacktestSpec.model_validate(dumped)
    
    assert len(reloaded.factors) == 1
    assert reloaded.factors[0].kind == "low_vol"
    assert reloaded.factors[0].lookback_months == 12
    assert reloaded.factors[0].min_months == 6


def test_size_factor_spec_round_trip():
    """SizeFactorSpec round-trips correctly."""
    spec = BacktestSpec(
        name="test_size",
        asof=datetime.date(2026, 1, 1),
        factors=[
            SizeFactorSpec(),
        ],
    )
    
    dumped = spec.model_dump()
    reloaded = BacktestSpec.model_validate(dumped)
    
    assert len(reloaded.factors) == 1
    assert reloaded.factors[0].kind == "size"


def test_multi_factor_spec_round_trip():
    """Multiple factors round-trip correctly."""
    spec = BacktestSpec(
        name="test_multi",
        asof=datetime.date(2026, 1, 1),
        factors=[
            MomentumFactorSpec(min_months=11),
            LowVolatilityFactorSpec(lookback_months=12, min_months=6),
            SizeFactorSpec(),
        ],
    )
    
    dumped = spec.model_dump()
    reloaded = BacktestSpec.model_validate(dumped)
    
    assert len(reloaded.factors) == 3
    assert reloaded.factors[0].kind == "momentum"
    assert reloaded.factors[1].kind == "low_vol"
    assert reloaded.factors[2].kind == "size"
