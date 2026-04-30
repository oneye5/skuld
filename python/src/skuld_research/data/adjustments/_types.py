"""Public types and column schemas for the adjustments package.

Private module — import from :mod:`skuld_research.data.adjustments` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class RepairPolicy(str, Enum):  # noqa: UP042 - keep str+Enum for API compatibility
    """Controls which detected events are actually repaired."""

    OFF = "off"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


_EVENT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "ex_date",
    "kind",
    "severity",
    "observed_ratio",
    "expected_ratio",
    "residual",
    "adj_close_prev",
    "adj_close_ex",
    "raw_close_prev",
    "raw_close_ex",
    "corp_action_type",
    "corp_action_factor",
    "notes",
)

_REPAIR_COLUMNS: tuple[str, ...] = (
    "ticker",
    "ex_date",
    "kind",
    "action",
    "factor_applied",
    "range_start",
    "range_end",
)


@dataclass(frozen=True)
class AdjustmentAuditReport:
    """Long-form ledger of every detected adjustment-chain discrepancy.

    See :mod:`skuld_research.data.adjustments` module docstring for the
    per-row schema.
    """

    events: pd.DataFrame


@dataclass(frozen=True)
class RepairResult:
    """Result of :func:`repair_adjustments`.

    Attributes:
        prices: The (possibly mutated) adjusted-close panel.
        report: Audit report of all detections (including non-repaired).
        repairs: Long-form ledger of every repair actually applied.
    """

    prices: pd.DataFrame
    report: AdjustmentAuditReport
    repairs: pd.DataFrame


# ---------------------------------------------------------------------------
# Shared helpers used by both _detect and _repair
# ---------------------------------------------------------------------------


def _empty_dtype(column: str) -> str:
    if column in {"ticker", "kind", "severity", "corp_action_type", "notes", "action"}:
        return "object"
    if column in {"ex_date", "range_start", "range_end"}:
        return "datetime64[ns]"
    return "float64"


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype=_empty_dtype(col)) for col in _EVENT_COLUMNS}
    )


def _empty_repairs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype=_empty_dtype(col)) for col in _REPAIR_COLUMNS}
    )
