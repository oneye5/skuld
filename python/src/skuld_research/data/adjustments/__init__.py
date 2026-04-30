"""Corporate-action adjustment audit and repair layer.

Cross-validates Yahoo's ``adj_close`` panel against a corporate-action ledger
(``dividend`` and ``split`` rows) and detects six categories of discrepancy:

* ``missed_split`` — adj_close ratio matches a known split-style factor at a
  date with no corresponding ``split`` row in the ledger.
* ``unit_jump`` — ratio matches a 100x or 0.01x unit-confusion factor.
* ``split_mismatch`` — a ``split`` row exists but the observed ratio does
  not match the expected ``1/factor``.
* ``bad_div_adjust`` — a ``dividend`` row exists but the observed adj_close
  ratio is inconsistent with the price-drop model
  ``(raw_close[ex] - dividend) / raw_close[prev]`` beyond tolerance. Requires
  ``raw_close``; otherwise the event is recorded with severity
  ``skipped_no_raw``.
* ``orphan_action`` — corp-action ``ex_date`` falls outside the price index.
* ``duplicate_action`` — multiple rows of the same ``type`` on the same
  ``(ticker, ex_date)``.

Two repair policies (CONSERVATIVE, AGGRESSIVE) and a no-op (OFF) are
provided. CONSERVATIVE back-scales for missed splits and unit jumps;
AGGRESSIVE additionally re-derives the entire adjusted-close series for
tickers with ``bad_div_adjust`` events using the standard CRSP backward
total-return chain.

This package is intentionally decoupled from the rest of the Skuld pipeline:
inputs are bare DataFrames, outputs are bare DataFrames + frozen dataclass
wrappers, and only ``pandas``, ``numpy``, and the standard library are
imported. See spec ``docs/specs/2026-04-30-corporate-action-adjustments.md``
§4.5 for rationale.

See also ``scrubber.py`` for the complementary single-day round-trip
detector — it operates on raw daily prices and is orthogonal to this layer.
"""

from __future__ import annotations

from ._detect import audit_adjustments
from ._repair import repair_adjustments
from ._types import AdjustmentAuditReport, RepairPolicy, RepairResult

__all__ = [
    "AdjustmentAuditReport",
    "RepairPolicy",
    "RepairResult",
    "audit_adjustments",
    "repair_adjustments",
]
