"""Trial ledgers for tracking hypothesis tests."""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

# Implicit hypothesis tests baked into M0.5–M4
n_trials_prior: int = 30


class TrialLedger:
    """Append-only JSONL trial ledger partitioned by year.
    
    Schema per entry:
    {
        "spec_hash": str,
        "spec_summary": str,
        "wf_sharpe": float,
        "wf_n_obs": int,
        "kept_folds": int,
        "rejected_folds": int,
        "git_sha": str | None,
        "entered_at": ISO8601 str (UTC)
    }
    
    Args:
        root: base directory for ledger files.
        scope: subdirectory name (e.g., "production", "exploration").
    """

    def __init__(self, root: Path, scope: str):
        self.root = self._resolve_root(root, scope)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_root(root: Path, scope: str) -> Path:
        root_path = Path(root)
        if root_path.name == scope:
            return root_path
        return root_path / scope

    @staticmethod
    def _year_for_entry(entry: dict[str, Any]) -> int:
        entered_at = str(entry["entered_at"])
        return datetime.fromisoformat(entered_at.replace("Z", "+00:00")).year

    def _entry_file_path(self, year: int) -> Path:
        return self.root / f"{year}.jsonl"

    def _iter_entries(self) -> Iterator[dict[str, Any]]:
        for jsonl_file in sorted(self.root.glob("*.jsonl")):
            with jsonl_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        yield json.loads(stripped)

    def append(self, entry: dict) -> None:
        """Append an entry to the ledger.
        
        Partitions by year extracted from entry["entered_at"].
        """
        file_path = self._entry_file_path(self._year_for_entry(entry))

        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def all_entries(self) -> list[dict]:
        """Read all entries across all years."""
        return list(self._iter_entries())

    def n_unique_trials(self) -> int:
        """Count unique spec_hash values (dedup at read time)."""
        return len({entry["spec_hash"] for entry in self._iter_entries()})

    def contains(self, spec_hash: str) -> bool:
        """Check if a spec_hash exists in the ledger."""
        return any(entry["spec_hash"] == spec_hash for entry in self._iter_entries())


class ProductionTrialLedger(TrialLedger):
    """Production trial ledger."""

    def __init__(self, root: Path = Path("trial_ledger") / "production"):
        super().__init__(root=root, scope="production")


class ExplorationTrialLedger(TrialLedger):
    """Exploration trial ledger."""

    def __init__(self, root: Path = Path("trial_ledger") / "exploration"):
        super().__init__(root=root, scope="exploration")
