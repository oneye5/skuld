"""Test pre-registered spec immutability."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from skuld_research.config import iter_preregistered_specs, load_spec, spec_hash


def test_preregistered_specs_parse_cleanly():
    """Each committed pre-registered YAML parses and has a valid hash."""
    specs = iter_preregistered_specs()
    
    assert len(specs) > 0, "No pre-registered specs found"
    
    for spec_path in specs:
        spec = load_spec(spec_path)
        h = spec_hash(spec)
        
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


def test_preregistered_specs_not_modified():
    """git diff shows no modifications to committed pre-registered specs."""
    try:
        # Check if we're in a git repo
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            check=True,
            capture_output=True,
            cwd=Path(__file__).parent.parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Not in a git repository")
    
    # Get list of modified files in configs/preregistered/
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "python/configs/preregistered/"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    
    modified_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    
    assert len(modified_files) == 0, (
        f"Pre-registered specs have been modified: {modified_files}\n"
        f"Pre-registered specs are read-only. Create a new spec file instead."
    )
