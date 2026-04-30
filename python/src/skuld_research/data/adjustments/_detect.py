"""Detection logic for adjustment-chain discrepancies.

Private module — import :func:`audit_adjustments` from
:mod:`skuld_research.data.adjustments` instead.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ._types import (
    _EVENT_COLUMNS,
    AdjustmentAuditReport,
    _empty_events_frame,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ratio_matches_any(
    observed: float, candidates: tuple[float, ...], tol: float
) -> tuple[bool, float]:
    """Return (match, matched_candidate). Tests both candidate and 1/candidate."""
    for c in candidates:
        if c == 0.0:
            continue
        if abs(observed / c - 1.0) <= tol:
            return True, c
    return False, float("nan")


def _new_event(
    ticker: str,
    ex_date: pd.Timestamp,
    kind: str,
    severity: str,
    *,
    observed_ratio: float = float("nan"),
    expected_ratio: float = float("nan"),
    residual: float = float("nan"),
    adj_close_prev: float = float("nan"),
    adj_close_ex: float = float("nan"),
    raw_close_prev: float = float("nan"),
    raw_close_ex: float = float("nan"),
    corp_action_type: str = "",
    corp_action_factor: float = float("nan"),
    notes: str = "",
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "ex_date": ex_date,
        "kind": kind,
        "severity": severity,
        "observed_ratio": observed_ratio,
        "expected_ratio": expected_ratio,
        "residual": residual,
        "adj_close_prev": adj_close_prev,
        "adj_close_ex": adj_close_ex,
        "raw_close_prev": raw_close_prev,
        "raw_close_ex": raw_close_ex,
        "corp_action_type": corp_action_type,
        "corp_action_factor": corp_action_factor,
        "notes": notes,
    }


def _ts(idx: pd.DatetimeIndex, i: int) -> pd.Timestamp:
    """Narrow ``idx[i]`` to ``pd.Timestamp`` for static type checkers."""
    return cast(pd.Timestamp, idx[i])


def _align_ex_date(
    series_index: pd.DatetimeIndex, ex_date: pd.Timestamp
) -> tuple[int, int] | None:
    """Map an ex_date to (prev_pos, ex_pos) trading-day positions.

    Returns ``None`` if the ex_date is outside the index (orphan), or if
    there is no prior trading day.
    """
    pos = int(series_index.searchsorted(ex_date, side="left"))  # type: ignore[arg-type]
    if pos >= len(series_index):
        return None  # orphan: past end
    if pos == 0:
        return None  # no prior obs
    return pos - 1, pos


def _group_actions(
    corporate_actions: pd.DataFrame,
) -> dict[str, list[tuple[pd.Timestamp, str, float]]]:
    out: dict[str, list[tuple[pd.Timestamp, str, float]]] = {}
    if corporate_actions is None or corporate_actions.empty:
        return out
    for _, row in corporate_actions.iterrows():
        t = str(row["ticker"])
        ex_ts = pd.Timestamp(cast(object, row["ex_date"]))  # type: ignore[arg-type]
        atype = str(row["type"])
        factor = float(cast(object, row["factor"]))  # type: ignore[arg-type]
        out.setdefault(t, []).append((ex_ts, atype, factor))
    return out


def _detect_duplicates(corporate_actions: pd.DataFrame) -> list[dict[str, object]]:
    if corporate_actions is None or corporate_actions.empty:
        return []
    grouped = corporate_actions.groupby(
        ["ticker", "ex_date", "type"], dropna=False
    ).size()
    dupes = grouped[grouped > 1]
    out: list[dict[str, object]] = []
    for key, count in dupes.items():  # type: ignore[union-attr]
        ticker, ex_date, atype = key  # type: ignore[misc]
        out.append(
            _new_event(
                str(ticker),
                pd.Timestamp(cast(object, ex_date)),  # type: ignore[arg-type]
                kind="duplicate_action",
                severity="warn",
                corp_action_type=str(atype),
                notes=f"{int(cast(object, count))} rows on same (ticker, ex_date, type)",  # type: ignore[arg-type]
            )
        )
    return out


def _raw_at_or_before(raw_series: pd.Series, ts: pd.Timestamp) -> float:
    pos = int(raw_series.index.searchsorted(ts, side="right")) - 1  # type: ignore[arg-type]
    if pos < 0:
        return float("nan")
    return float(raw_series.iloc[pos])


def _raw_at_or_after(raw_series: pd.Series, ts: pd.Timestamp) -> float:
    pos = int(raw_series.index.searchsorted(ts, side="left"))  # type: ignore[arg-type]
    if pos >= len(raw_series):
        return float("nan")
    return float(raw_series.iloc[pos])


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_adjustments(
    adj_close: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    raw_close: pd.DataFrame | None = None,
    dividend_residual_tol: float = 0.25,
    split_residual_tol: float = 0.05,
    missed_split_ratios: tuple[float, ...] = (
        0.5,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        0.1,
        0.2,
        0.25,
        1.0 / 3.0,
    ),
    unit_jump_ratios: tuple[float, ...] = (100.0, 0.01),
    unit_jump_tol: float = 0.02,
) -> AdjustmentAuditReport:
    """Detect discrepancies between ``adj_close`` and the corp-action ledger.

    See :mod:`skuld_research.data.adjustments` module docstring for detection
    categories and severities. Inputs are not mutated.

    Args:
        adj_close: Wide ``date × ticker`` adjusted-close panel.
        corporate_actions: Long-form frame with columns
            ``ticker, ex_date, type, factor``. ``type ∈ {"dividend", "split"}``.
        raw_close: Optional wide unadjusted-close panel. Required for
            ``bad_div_adjust`` detection. When omitted, dividend events are
            recorded with severity ``skipped_no_raw``.
        dividend_residual_tol: Tolerance for the dividend back-adjustment
            residual.
        split_residual_tol: Tolerance for split-ratio matching.
        missed_split_ratios: Candidate split factors and their reciprocals
            to test against unflagged ratio jumps.
        unit_jump_ratios: Candidate unit-confusion factors (typically 100,
            0.01).
        unit_jump_tol: Tolerance for unit-jump matching.

    Returns:
        :class:`AdjustmentAuditReport` with one row per detected event.
    """
    rows: list[dict[str, object]] = []
    actions_by_ticker = _group_actions(corporate_actions)

    # Pre-compute duplicates (groupby ticker, ex_date, type)
    dup_events = _detect_duplicates(corporate_actions)
    rows.extend(dup_events)

    for ticker in adj_close.columns:
        ticker = str(ticker)
        series = cast(pd.Series, adj_close[ticker]).dropna()
        if len(series) < 2:
            continue
        idx = cast(pd.DatetimeIndex, series.index)
        raw_series: pd.Series | None = None
        if raw_close is not None and ticker in raw_close.columns:
            raw_series = cast(pd.Series, raw_close[ticker]).dropna()

        ticker_actions = actions_by_ticker.get(ticker, [])

        # Track which (ex_pos) are explained by an action so missed_split
        # detection can skip them.
        explained_positions: set[int] = set()

        # First pass: actions referencing this ticker.
        for ex_date, atype, factor in ticker_actions:
            aligned = _align_ex_date(idx, ex_date)
            if aligned is None:
                # Orphan: ex_date before first or after last observation.
                rows.append(
                    _new_event(
                        ticker,
                        ex_date,
                        kind="orphan_action",
                        severity="info",
                        corp_action_type=atype,
                        corp_action_factor=float(factor),
                        notes="ex_date outside price index",
                    )
                )
                continue
            prev_pos, ex_pos = aligned
            # Mark a window around the ex_pos as explained for missed_split
            # suppression (±3 trading days per spec guidance).
            for off in range(-3, 4):
                p = ex_pos + off
                if 0 <= p < len(idx):
                    explained_positions.add(p)

            adj_prev = float(series.iloc[prev_pos])
            adj_ex = float(series.iloc[ex_pos])
            observed = adj_ex / adj_prev if adj_prev != 0.0 else float("nan")

            raw_prev = float("nan")
            raw_ex = float("nan")
            if raw_series is not None:
                raw_prev = _raw_at_or_before(raw_series, _ts(idx, prev_pos))
                raw_ex = _raw_at_or_after(raw_series, _ts(idx, ex_pos))

            if atype == "split":
                expected = 1.0 / float(factor) if factor != 0.0 else float("nan")
                resid = (
                    abs(observed / expected - 1.0)
                    if expected and not np.isnan(expected)
                    else float("nan")
                )
                if not np.isnan(resid) and resid > split_residual_tol:
                    rows.append(
                        _new_event(
                            ticker,
                            _ts(idx, ex_pos),
                            kind="split_mismatch",
                            severity="warn",
                            observed_ratio=observed,
                            expected_ratio=expected,
                            residual=resid,
                            adj_close_prev=adj_prev,
                            adj_close_ex=adj_ex,
                            raw_close_prev=raw_prev,
                            raw_close_ex=raw_ex,
                            corp_action_type=atype,
                            corp_action_factor=float(factor),
                            notes="observed ratio differs from 1/split_factor",
                        )
                    )
            elif atype == "dividend":
                if raw_series is None or ticker not in (
                    raw_close.columns if raw_close is not None else []
                ):
                    rows.append(
                        _new_event(
                            ticker,
                            _ts(idx, ex_pos),
                            kind="bad_div_adjust",
                            severity="skipped_no_raw",
                            observed_ratio=observed,
                            adj_close_prev=adj_prev,
                            adj_close_ex=adj_ex,
                            corp_action_type=atype,
                            corp_action_factor=float(factor),
                            notes="raw_close not provided",
                        )
                    )
                else:
                    if np.isnan(raw_prev) or np.isnan(raw_ex) or raw_prev == 0.0:
                        # Can't compute expected; skip.
                        continue
                    expected = (raw_ex - float(factor)) / raw_prev
                    resid = (
                        abs(observed / expected - 1.0)
                        if expected != 0.0
                        else float("nan")
                    )
                    if not np.isnan(resid) and resid > dividend_residual_tol:
                        rows.append(
                            _new_event(
                                ticker,
                                _ts(idx, ex_pos),
                                kind="bad_div_adjust",
                                severity="error",
                                observed_ratio=observed,
                                expected_ratio=expected,
                                residual=resid,
                                adj_close_prev=adj_prev,
                                adj_close_ex=adj_ex,
                                raw_close_prev=raw_prev,
                                raw_close_ex=raw_ex,
                                corp_action_type=atype,
                                corp_action_factor=float(factor),
                                notes=(
                                    "adj ratio inconsistent with price-drop "
                                    "model beyond tolerance"
                                ),
                            )
                        )

        # Second pass: scan every trading day for unexplained jumps.
        rets = series.pct_change().to_numpy()  # ratio - 1
        for i in range(1, len(series)):
            r = rets[i]
            if np.isnan(r):
                continue
            ratio = 1.0 + float(r)
            # Skip near-1 ratios.
            if abs(ratio - 1.0) < min(split_residual_tol, unit_jump_tol):
                continue
            if i in explained_positions:
                continue
            # Try unit_jump first (more specific candidate set).
            uj_match, uj_c = _ratio_matches_any(ratio, unit_jump_ratios, unit_jump_tol)
            ms_match, ms_c = _ratio_matches_any(
                ratio, missed_split_ratios, split_residual_tol
            )
            if uj_match:
                rows.append(
                    _new_event(
                        ticker,
                        _ts(idx, i),
                        kind="unit_jump",
                        severity="error",
                        observed_ratio=ratio,
                        expected_ratio=uj_c,
                        residual=abs(ratio / uj_c - 1.0),
                        adj_close_prev=float(series.iloc[i - 1]),
                        adj_close_ex=float(series.iloc[i]),
                        notes="unit-confusion (100x / 0.01x) jump",
                    )
                )
            elif ms_match:
                rows.append(
                    _new_event(
                        ticker,
                        _ts(idx, i),
                        kind="missed_split",
                        severity="error",
                        observed_ratio=ratio,
                        expected_ratio=ms_c,
                        residual=abs(ratio / ms_c - 1.0),
                        adj_close_prev=float(series.iloc[i - 1]),
                        adj_close_ex=float(series.iloc[i]),
                        notes="ratio matches a known split factor; no split row",
                    )
                )

    if rows:
        events = pd.DataFrame(rows, columns=list(_EVENT_COLUMNS))
        events["ex_date"] = pd.to_datetime(events["ex_date"])
    else:
        events = _empty_events_frame()
    return AdjustmentAuditReport(events=events)
