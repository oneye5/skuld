"""Repair logic for the adjusted-close panel.

Private module — import :func:`repair_adjustments` from
:mod:`skuld_research.data.adjustments` instead.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ._detect import _ts, audit_adjustments
from ._types import (
    _REPAIR_COLUMNS,
    RepairPolicy,
    RepairResult,
    _empty_repairs_frame,
)


def repair_adjustments(
    adj_close: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    *,
    raw_close: pd.DataFrame | None = None,
    policy: RepairPolicy = RepairPolicy.CONSERVATIVE,
    **audit_kwargs: object,
) -> RepairResult:
    """Audit then optionally repair the adjusted-close panel.

    Args:
        adj_close: Wide ``date × ticker`` adjusted-close panel.
        corporate_actions: Long-form corp-action frame.
        raw_close: Optional unadjusted-close panel. Required for AGGRESSIVE
            repair of ``bad_div_adjust`` events.
        policy: One of :class:`RepairPolicy`.
        **audit_kwargs: Forwarded to :func:`audit_adjustments`.

    Returns:
        :class:`RepairResult` with the (possibly repaired) prices, the
        audit report, and a ledger of repairs actually applied.
    """
    report = audit_adjustments(
        adj_close,
        corporate_actions,
        raw_close=raw_close,
        **audit_kwargs,  # type: ignore[arg-type]
    )

    if policy is RepairPolicy.OFF:
        return RepairResult(
            prices=adj_close.copy(),
            report=report,
            repairs=_empty_repairs_frame(),
        )

    repaired = adj_close.copy()
    repair_rows: list[dict[str, object]] = []

    # 1. Conservative back-scaling for missed_split and unit_jump.
    err_events = report.events[
        report.events["severity"] == "error"
    ].copy()
    for _, ev in err_events.iterrows():
        kind = str(ev["kind"])
        if kind not in {"missed_split", "unit_jump"}:
            continue
        ticker = str(ev["ticker"])
        ex_date = pd.Timestamp(cast(object, ev["ex_date"]))  # type: ignore[arg-type]
        factor = float(cast(object, ev["expected_ratio"]))  # type: ignore[arg-type]
        if ticker not in repaired.columns or factor == 0.0 or np.isnan(factor):
            continue
        col = repaired[ticker]
        # Back-scale: divide all values strictly before ex_date by factor.
        # Since observed ratio = adj_ex / adj_prev ≈ factor, dividing the
        # pre-ex segment by factor aligns it with the post-ex scale.
        pre_mask = col.index < ex_date
        # Identify the range of non-NaN pre-ex values for the ledger.
        pre_segment = col.loc[pre_mask].dropna()
        if pre_segment.empty:
            continue
        # Bring the pre-ex segment onto the post-ex scale by multiplying
        # by the observed jump factor (== expected_ratio for these kinds).
        repaired.loc[pre_mask, ticker] = col.loc[pre_mask] * factor
        repair_rows.append(
            {
                "ticker": ticker,
                "ex_date": ex_date,
                "kind": kind,
                "action": "back_scale",
                "factor_applied": factor,
                "range_start": pre_segment.index[0],
                "range_end": pre_segment.index[-1],
            }
        )

    # 2. AGGRESSIVE: re-derive dividend chain from raw_close.
    if policy is RepairPolicy.AGGRESSIVE and raw_close is not None:
        bad_div = report.events[
            (report.events["kind"] == "bad_div_adjust")
            & (report.events["severity"] == "error")
        ]
        for ticker in pd.unique(cast(pd.Series, bad_div["ticker"]).to_numpy()):
            ticker_str = str(ticker)
            if ticker_str not in raw_close.columns:
                continue
            raw_col = cast(pd.Series, raw_close[ticker_str])
            new_series = _rederive_adjusted_chain(
                raw_col, corporate_actions, ticker_str
            )
            if new_series is None:
                continue
            repaired[ticker_str] = new_series.reindex(repaired.index)
            # One repair-ledger row per ticker.
            non_null = new_series.dropna()
            if non_null.empty:
                continue
            repair_rows.append(
                {
                    "ticker": ticker_str,
                    "ex_date": non_null.index[-1],
                    "kind": "bad_div_adjust",
                    "action": "rederive_chain",
                    "factor_applied": float("nan"),
                    "range_start": non_null.index[0],
                    "range_end": non_null.index[-1],
                }
            )

    if repair_rows:
        repairs = pd.DataFrame(repair_rows, columns=list(_REPAIR_COLUMNS))
        repairs["ex_date"] = pd.to_datetime(repairs["ex_date"])
        repairs["range_start"] = pd.to_datetime(repairs["range_start"])
        repairs["range_end"] = pd.to_datetime(repairs["range_end"])
    else:
        repairs = _empty_repairs_frame()

    return RepairResult(prices=repaired, report=report, repairs=repairs)


def _rederive_adjusted_chain(
    raw_series: pd.Series,
    corporate_actions: pd.DataFrame,
    ticker: str,
) -> pd.Series | None:
    """Standard CRSP backward total-return chain for a single ticker.

    Walks the raw-close series last-to-first, maintaining an ``adj_factor``
    initialised to 1.0. On each visited day, applies the running factor to
    the raw close to produce the adjusted close. Then, before stepping to
    the prior day, updates the factor:

    * dividend D on visited day with prior-day raw P_prev → factor *= (1 - D/P_prev)
    * split factor F on visited day → factor /= F
    """
    series = raw_series.dropna()
    if series.empty:
        return None
    series_idx = cast(pd.DatetimeIndex, series.index)

    # Build per-day action lookup for this ticker.
    div_by_date: dict[pd.Timestamp, float] = {}
    split_by_date: dict[pd.Timestamp, float] = {}
    if corporate_actions is not None and not corporate_actions.empty:
        sub = corporate_actions[corporate_actions["ticker"] == ticker]
        for _, row in sub.iterrows():
            d = pd.Timestamp(cast(object, row["ex_date"]))  # type: ignore[arg-type]
            atype = str(row["type"])
            f = float(cast(object, row["factor"]))  # type: ignore[arg-type]
            # Align to the at-or-after trading day.
            pos = int(series_idx.searchsorted(d, side="left"))  # type: ignore[arg-type]
            if pos >= len(series_idx):
                continue
            aligned = _ts(series_idx, pos)
            if atype == "dividend":
                div_by_date[aligned] = div_by_date.get(aligned, 0.0) + f
            elif atype == "split":
                # Multiple splits same day: multiply factors.
                split_by_date[aligned] = split_by_date.get(aligned, 1.0) * f

    adj_values = np.empty(len(series), dtype=float)
    factor = 1.0
    for i in range(len(series) - 1, -1, -1):
        d = _ts(series_idx, i)
        adj_values[i] = float(series.iloc[i]) * factor
        # Update factor for prior days.
        if d in div_by_date:
            div = div_by_date[d]
            if i > 0:
                p_prev = float(series.iloc[i - 1])
                if p_prev != 0.0:
                    factor *= 1.0 - div / p_prev
        if d in split_by_date:
            f = split_by_date[d]
            if f != 0.0:
                factor /= f

    return pd.Series(adj_values, index=series.index, name=ticker)
