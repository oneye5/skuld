"""Content-addressable hashing for BacktestSpec."""
from __future__ import annotations

import hashlib
import json

from skuld_research.config.spec import BacktestSpec


def spec_hash(spec: BacktestSpec) -> str:
    """Compute SHA-256 hash of spec in canonical JSON form.

    The `overlay` and `execution_policy` fields are omitted from the hash when
    disabled. This ensures backward compatibility: specs without those blocks
    produce the same hash as before the fields were added.

    Returns:
        64-character hex digest.
    """
    dump = spec.model_dump(mode="json")

    # Omit overlay if absent or kind == "none"
    if "overlay" in dump and (
        dump["overlay"] is None or dump["overlay"].get("kind") == "none"
    ):
        del dump["overlay"]

    # Omit execution policy if disabled. Disabled policy knobs are inert and
    # must not change hashes for existing pre-registered specs.
    execution_policy = dump.get("execution_policy")
    if isinstance(execution_policy, dict) and execution_policy.get("kind") == "none":
        del dump["execution_policy"]

    # Omit scrubbing if absent or kind == "none". Disabled scrubbing must
    # not change hashes for specs registered before the field was added.
    if "scrubbing" in dump and (
        dump["scrubbing"] is None or dump["scrubbing"].get("kind") == "none"
    ):
        del dump["scrubbing"]

    # Omit adjustments if absent or kind == "off". Disabled corp-action
    # audit/repair must not change hashes for specs registered before the
    # field was added.
    if "adjustments" in dump and (
        dump["adjustments"] is None or dump["adjustments"].get("kind") == "off"
    ):
        del dump["adjustments"]

    # Omit passed_gating — it is a deployment metadata flag, not output-influencing
    dump.pop("passed_gating", None)

    # Omit momentum smoothing_months when at default (1) — preserves backward
    # compatibility for specs registered before the smoothing feature was added.
    for factor in dump.get("factors", []):
        if factor.get("kind") == "momentum" and factor.get("smoothing_months") == 1:
            del factor["smoothing_months"]

    # Omit universe.rebalance_freq when at default ("BME") — preserves backward
    # compatibility for specs registered before quarterly rebalance was added.
    universe = dump.get("universe")
    if isinstance(universe, dict) and universe.get("rebalance_freq") == "BME":
        del universe["rebalance_freq"]

    # Omit cost spread-model fields when at defaults — preserves backward
    # compatibility for specs registered before the AR estimator was added.
    # When spread_model == "flat", the estimator parameters are unused and
    # must not affect the hash.
    cost = dump.get("cost")
    if isinstance(cost, dict) and cost.get("spread_model") == "flat":
        for key in (
            "spread_model",
            "spread_estimator_window",
            "spread_estimator_min_obs",
            "spread_estimator_scale",
            "spread_estimator_min_bps_per_side",
        ):
            cost.pop(key, None)

    canonical = json.dumps(
        dump,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def short_hash(h: str, n: int = 12) -> str:
    """Return first n characters of a hash.

    Args:
        h: full hash string.
        n: number of characters to return (default 12).

    Returns:
        First n characters of h.
    """
    return h[:n]
