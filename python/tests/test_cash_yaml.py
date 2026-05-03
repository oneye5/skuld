"""Tests for skuld_portfolio.inputs.cash_yaml."""
from pathlib import Path

import pytest

from skuld_portfolio.inputs.cash_yaml import CashYAMLError, read_cash_yaml


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_cash_yaml(fixture_dir: Path) -> Path:
    return fixture_dir / "cash_2026-04-26.yaml"


def test_read_cash_yaml_happy_path(valid_cash_yaml: Path):
    """Read cash balance from valid YAML."""
    cash = read_cash_yaml(valid_cash_yaml)
    assert cash == pytest.approx(2500.0)


def test_read_cash_yaml_missing_key(tmp_path: Path):
    """Raise CashYAMLError if 'cash_nzd' key is missing."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("asof: 2026-04-26\n")

    with pytest.raises(CashYAMLError, match="missing required key"):
        read_cash_yaml(bad_yaml)


def test_read_cash_yaml_non_numeric_value(tmp_path: Path):
    """Raise CashYAMLError if cash_nzd is not a number."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("cash_nzd: not_a_number\n")

    with pytest.raises(CashYAMLError, match="must be a number"):
        read_cash_yaml(bad_yaml)


def test_read_cash_yaml_malformed(tmp_path: Path):
    """Raise CashYAMLError if YAML is malformed."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("cash_nzd: [invalid: yaml\n")

    with pytest.raises(CashYAMLError, match="Failed to parse"):
        read_cash_yaml(bad_yaml)


def test_read_cash_yaml_not_dict(tmp_path: Path):
    """Raise CashYAMLError if YAML is not a dict."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- item1\n- item2\n")

    with pytest.raises(CashYAMLError, match="must be a YAML mapping"):
        read_cash_yaml(bad_yaml)


def test_read_cash_yaml_file_not_found():
    """Raise FileNotFoundError if path does not exist."""
    with pytest.raises(FileNotFoundError):
        read_cash_yaml(Path("/nonexistent/file.yaml"))
