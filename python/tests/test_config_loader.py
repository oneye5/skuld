"""Tests for spec loader."""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from skuld_research.config import (
    BacktestSpec,
    SpecValidationError,
    iter_strategy_specs,
    load_spec,
    spec_hash,
)


def test_load_spec_round_trip(tmp_path: Path):
    """load_spec round-trips a YAML file."""
    spec = BacktestSpec(
        name="test_spec",
        description="Test round-trip",
        asof=datetime.date(2026, 1, 1),
        master_seed=99,
    )

    yaml_path = tmp_path / "test.yaml"

    # Write YAML manually
    yaml_content = f"""
name: {spec.name}
description: "{spec.description}"
asof: {spec.asof}
master_seed: {spec.master_seed}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # Load and verify
    loaded = load_spec(yaml_path)

    assert loaded.name == "test_spec"
    assert loaded.description == "Test round-trip"
    assert loaded.asof == datetime.date(2026, 1, 1)
    assert loaded.master_seed == 99


def test_load_spec_missing_required_field(tmp_path: Path):
    """load_spec raises SpecValidationError on missing required fields."""
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text("description: Missing name and asof\n", encoding="utf-8")

    with pytest.raises(SpecValidationError, match="Failed to load spec"):
        load_spec(yaml_path)


def test_load_spec_file_not_found(tmp_path: Path):
    """load_spec raises SpecValidationError when file doesn't exist."""
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(SpecValidationError, match="Spec file not found"):
        load_spec(missing_path)


def test_iter_strategy_specs():
    """iter_strategy_specs finds committed strategy spec YAML files."""
    specs = iter_strategy_specs()

    # Should find at least the archived momentum spec
    assert len(specs) > 0

    # All should be yaml files
    assert all(p.suffix == ".yaml" for p in specs)

    # Should find specs recursively under lifecycle folders
    names = [p.stem for p in specs]
    assert "m8-mom" in names
    assert any("strategy-specs" in p.as_posix() and "archive" in p.as_posix() for p in specs)
    assert any("strategy-specs" in p.as_posix() and "candidates" in p.as_posix() for p in specs)


def test_strategy_specs_parse_cleanly():
    """Each committed YAML parses cleanly and yields a 64-char hash."""
    specs = iter_strategy_specs()

    for spec_path in specs:
        spec = load_spec(spec_path)
        h = spec_hash(spec)

        # Verify hash format
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

        # Verify required fields present
        assert spec.name
        assert spec.asof
