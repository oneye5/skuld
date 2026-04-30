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


def test_execution_policy_spec_round_trip():
    """Cost-aware execution policy settings are part of the frozen spec."""
    from skuld_research.config.spec import ExecutionPolicySpec

    spec = BacktestSpec(
        name="test_execution_policy",
        asof=datetime.date(2026, 1, 1),
        execution_policy=ExecutionPolicySpec(
            kind="volume_budget",
            monthly_volume_budget_nzd=5_000.0,
            min_trade_benefit_bps=50.0,
            excess_trade_benefit_bps=190.0,
        ),
    )

    dumped = spec.model_dump()
    reloaded = BacktestSpec.model_validate(dumped)

    assert reloaded.execution_policy.kind == "volume_budget"
    assert reloaded.execution_policy.monthly_volume_budget_nzd == 5_000.0
    assert reloaded.execution_policy.min_trade_benefit_bps == 50.0
    assert reloaded.execution_policy.excess_trade_benefit_bps == 190.0


def test_default_execution_policy_hash_neutral():
    """Default-disabled execution policy does not change existing spec hashes."""
    from skuld_research.config.spec import ExecutionPolicySpec

    spec = BacktestSpec(
        name="test_execution_policy_hash",
        asof=datetime.date(2026, 1, 1),
    )
    explicit_default = spec.model_copy(
        update={"execution_policy": ExecutionPolicySpec(kind="none")}
    )
    enabled = spec.model_copy(
        update={
            "execution_policy": ExecutionPolicySpec(
                kind="volume_budget",
                monthly_volume_budget_nzd=5_000.0,
            )
        }
    )

    assert spec_hash(spec) == spec_hash(explicit_default)
    assert spec_hash(spec) != spec_hash(enabled)


def test_disabled_execution_policy_hash_neutral_even_with_unused_fields():
    """kind='none' disables and omits execution policy knobs from the hash."""
    from skuld_research.config.spec import ExecutionPolicySpec

    spec = BacktestSpec(
        name="test_execution_policy_disabled_hash",
        asof=datetime.date(2026, 1, 1),
    )
    disabled_with_unused_fields = spec.model_copy(
        update={
            "execution_policy": ExecutionPolicySpec(
                kind="none",
                monthly_volume_budget_nzd=5_000.0,
                min_trade_benefit_bps=50.0,
                excess_trade_benefit_bps=250.0,
            )
        }
    )

    assert spec_hash(spec) == spec_hash(disabled_with_unused_fields)


def test_volume_budget_execution_policy_requires_budget():
    """A volume-budget policy must provide the budget it claims to enforce."""
    from skuld_research.config.spec import ExecutionPolicySpec

    with pytest.raises(ValidationError, match="monthly_volume_budget_nzd"):
        ExecutionPolicySpec(kind="volume_budget")


def test_scrubbing_spec_round_trip():
    """Daily-price scrubbing settings are part of the frozen spec."""
    from skuld_research.config.spec import ScrubbingSpec

    spec = BacktestSpec(
        name="test_scrubbing",
        asof=datetime.date(2026, 1, 1),
        scrubbing=ScrubbingSpec(
            kind="round_trip",
            threshold=0.30,
            reversal_tolerance=0.10,
        ),
    )

    dumped = spec.model_dump()
    reloaded = BacktestSpec.model_validate(dumped)

    assert reloaded.scrubbing is not None
    assert reloaded.scrubbing.kind == "round_trip"
    assert reloaded.scrubbing.threshold == 0.30
    assert reloaded.scrubbing.reversal_tolerance == 0.10


def test_default_scrubbing_hash_neutral():
    """Default-disabled scrubbing does not change existing spec hashes."""
    from skuld_research.config.spec import ScrubbingSpec

    spec = BacktestSpec(
        name="test_scrubbing_hash",
        asof=datetime.date(2026, 1, 1),
    )
    explicit_none = spec.model_copy(
        update={"scrubbing": ScrubbingSpec(kind="none")}
    )
    enabled = spec.model_copy(
        update={
            "scrubbing": ScrubbingSpec(
                kind="round_trip",
                threshold=0.30,
                reversal_tolerance=0.10,
            )
        }
    )

    assert spec_hash(spec) == spec_hash(explicit_none)
    assert spec_hash(spec) != spec_hash(enabled)


def test_disabled_scrubbing_hash_neutral_even_with_unused_fields():
    """kind='none' disables and omits scrubbing knobs from the hash."""
    from skuld_research.config.spec import ScrubbingSpec

    spec = BacktestSpec(
        name="test_scrubbing_disabled_hash",
        asof=datetime.date(2026, 1, 1),
    )
    disabled_with_unused_fields = spec.model_copy(
        update={
            "scrubbing": ScrubbingSpec(
                kind="none",
                threshold=0.20,
                reversal_tolerance=0.05,
            )
        }
    )

    assert spec_hash(spec) == spec_hash(disabled_with_unused_fields)


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
