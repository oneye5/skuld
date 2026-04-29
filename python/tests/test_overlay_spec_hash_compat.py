"""Test that overlay field does not break existing spec hashes."""
import pytest

from skuld_research.config import load_spec, spec_hash


def test_momentum_only_spec_hash_unchanged():
    """Verify that momentum_only spec hash is unchanged (M7 lock-in test).
    
    The addition of the overlay field must not change the hash of existing
    specs that have no overlay block or overlay.kind == "none".
    """
    spec = load_spec("configs/preregistered/2026-04-26_momentum_only.yaml")
    actual_hash = spec_hash(spec)
    
    # Expected hash from M7 (before overlay was added)
    expected_hash = "b19d42fb7236097b16974b3bf0e4a109ea758209d5fca8b66a094f7edb291e52"
    
    assert actual_hash == expected_hash, (
        f"momentum_only spec hash changed! Expected {expected_hash}, got {actual_hash}. "
        f"The overlay field must not participate in the hash when absent or kind='none'."
    )


def test_hash_compat_rebalance_freq_default():
    """Setting rebalance_freq explicitly to 'BME' (default) does not change the hash."""
    spec = load_spec("configs/preregistered/2026-04-26_momentum_only.yaml")
    h_default = spec_hash(spec)

    universe_explicit = spec.universe.model_copy(update={"rebalance_freq": "BME"})
    spec_explicit = spec.model_copy(update={"universe": universe_explicit})
    h_explicit = spec_hash(spec_explicit)

    assert h_default == h_explicit, (
        "rebalance_freq: 'BME' (default) must be omitted from the hash for backward compat."
    )

    # And changing it to BQE *must* change the hash.
    universe_quarterly = spec.universe.model_copy(update={"rebalance_freq": "BQE"})
    spec_quarterly = spec.model_copy(update={"universe": universe_quarterly})
    assert spec_hash(spec_quarterly) != h_default, (
        "rebalance_freq: 'BQE' must influence the hash."
    )
