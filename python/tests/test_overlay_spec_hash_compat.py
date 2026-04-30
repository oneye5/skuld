"""Test that overlay field does not break existing spec hashes."""

from skuld_research.config import load_spec, spec_hash


def test_m8_mom_spec_hash_matches_expected():
    """Verify that the archived M8 momentum spec hash is intentional."""
    spec = load_spec("configs/strategy-specs/archive/m8-mom.yaml")
    actual_hash = spec_hash(spec)

    expected_hash = "4dd5177fe965a6a703fe04c9ad0ec24e30b2eab6f00876657004a930b74ef489"

    assert actual_hash == expected_hash, (
        f"m8-mom spec hash changed! Expected {expected_hash}, got {actual_hash}."
    )


def test_hash_compat_rebalance_freq_default():
    """Setting rebalance_freq explicitly to 'BME' (default) does not change the hash."""
    spec = load_spec("configs/strategy-specs/archive/m8-mom.yaml")
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
