"""Tests for trial ledger."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from skuld_research.stats.ledger import (
    ExplorationTrialLedger,
    ProductionTrialLedger,
    TrialLedger,
)


def test_append_and_read_single_entry(tmp_path: Path):
    """Append one entry, then read it back."""
    ledger = TrialLedger(root=tmp_path, scope="test")
    
    entry = {
        "spec_hash": "abc123",
        "spec_summary": "momentum_12m",
        "wf_sharpe": 1.5,
        "wf_n_obs": 120,
        "kept_folds": 5,
        "rejected_folds": 0,
        "git_sha": "deadbeef",
        "entered_at": "2026-01-15T10:30:00+00:00",
    }
    
    ledger.append(entry)
    
    entries = ledger.all_entries()
    assert len(entries) == 1
    assert entries[0]["spec_hash"] == "abc123"


def test_dedup_by_spec_hash(tmp_path: Path):
    """Multiple appends with same spec_hash → n_unique_trials counts once."""
    ledger = TrialLedger(root=tmp_path, scope="test")
    
    entry1 = {
        "spec_hash": "xyz",
        "spec_summary": "strategy_a",
        "wf_sharpe": 2.0,
        "wf_n_obs": 100,
        "kept_folds": 4,
        "rejected_folds": 1,
        "git_sha": None,
        "entered_at": "2026-01-10T08:00:00+00:00",
    }
    entry2 = {
        "spec_hash": "xyz",  # same hash
        "spec_summary": "strategy_a_rerun",
        "wf_sharpe": 2.1,
        "wf_n_obs": 100,
        "kept_folds": 4,
        "rejected_folds": 1,
        "git_sha": None,
        "entered_at": "2026-01-11T08:00:00+00:00",
    }
    
    ledger.append(entry1)
    ledger.append(entry2)
    
    assert ledger.n_unique_trials() == 1
    assert len(ledger.all_entries()) == 2  # both stored, but counted once


def test_cross_year_partition(tmp_path: Path):
    """Entries from different years create separate files."""
    ledger = TrialLedger(root=tmp_path, scope="test")
    
    entry_2025 = {
        "spec_hash": "hash_2025",
        "spec_summary": "old",
        "wf_sharpe": 1.0,
        "wf_n_obs": 80,
        "kept_folds": 3,
        "rejected_folds": 0,
        "git_sha": None,
        "entered_at": "2025-12-31T23:59:00+00:00",
    }
    entry_2026 = {
        "spec_hash": "hash_2026",
        "spec_summary": "new",
        "wf_sharpe": 1.5,
        "wf_n_obs": 90,
        "kept_folds": 4,
        "rejected_folds": 0,
        "git_sha": None,
        "entered_at": "2026-01-01T00:01:00+00:00",
    }
    
    ledger.append(entry_2025)
    ledger.append(entry_2026)
    
    # Check files exist
    file_2025 = ledger.root / "2025.jsonl"
    file_2026 = ledger.root / "2026.jsonl"
    
    assert file_2025.exists()
    assert file_2026.exists()
    
    # All entries readable
    entries = ledger.all_entries()
    assert len(entries) == 2
    assert ledger.n_unique_trials() == 2


def test_production_and_exploration_separate(tmp_path: Path):
    """ProductionTrialLedger and ExplorationTrialLedger do not see each other."""
    prod = ProductionTrialLedger(root=tmp_path / "production")
    expl = ExplorationTrialLedger(root=tmp_path / "exploration")
    
    entry = {
        "spec_hash": "shared_hash",
        "spec_summary": "test",
        "wf_sharpe": 1.0,
        "wf_n_obs": 100,
        "kept_folds": 5,
        "rejected_folds": 0,
        "git_sha": None,
        "entered_at": "2026-04-25T12:00:00+00:00",
    }
    
    prod.append(entry)
    
    assert prod.n_unique_trials() == 1
    assert expl.n_unique_trials() == 0


def test_contains(tmp_path: Path):
    """contains() checks if a spec_hash exists in the ledger."""
    ledger = TrialLedger(root=tmp_path, scope="test")
    
    entry = {
        "spec_hash": "findme",
        "spec_summary": "test",
        "wf_sharpe": 1.0,
        "wf_n_obs": 100,
        "kept_folds": 5,
        "rejected_folds": 0,
        "git_sha": None,
        "entered_at": "2026-04-25T12:00:00+00:00",
    }
    
    ledger.append(entry)
    
    assert ledger.contains("findme") is True
    assert ledger.contains("notfound") is False


def test_trial_ledger_does_not_double_append_scope(tmp_path: Path):
    """Passing an already-scoped root should not nest the scope twice."""
    scoped_root = tmp_path / "test"
    ledger = TrialLedger(root=scoped_root, scope="test")

    entry = {
        "spec_hash": "abc123",
        "spec_summary": "momentum_12m",
        "wf_sharpe": 1.5,
        "wf_n_obs": 120,
        "kept_folds": 5,
        "rejected_folds": 0,
        "git_sha": "deadbeef",
        "entered_at": "2026-01-15T10:30:00+00:00",
    }

    ledger.append(entry)

    assert ledger.root == scoped_root
    assert (scoped_root / "2026.jsonl").exists()
    assert not (scoped_root / "test" / "2026.jsonl").exists()
