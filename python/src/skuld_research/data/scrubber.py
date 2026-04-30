"""Daily-price scrubber: detect and repair single-day round-trip anomalies.

Some raw price feeds occasionally print a daily close that is wildly out of
line with both the prior and next day (e.g. SKT.NZ 2010-01-04 close=5.05
between $32.10 and $32.61). When daily returns are compounded into monthly
returns via `(1 + r).prod() - 1`, such a print survives the monthly product
and inflates the apparent return.

This module exposes :func:`scrub_daily_prices`, which:

* computes per-ticker daily returns on each ticker's own observed days
  (avoiding cross-ticker NaN holes from the panel pivot),
* flags any day where ``|r_t| > threshold`` AND the next day's return
  reverses the move so that ``|(1 + r_t)(1 + r_{t+1}) - 1| < reversal_tolerance``,
* replaces the flagged price with the geometric mean of the surrounding
  prints (``sqrt(p_prev * p_next)``), which preserves the surrounding
  trajectory and is monotone-safe in borderline cases.

The function is opt-in for callers and idempotent: running it on already-
scrubbed prices is a no-op.

See also: :mod:`skuld_research.data.adjustments` for corporate-action
consistency checks (split/dividend cross-validation against ``adj_close``),
which is a complementary but independent concern from the round-trip print
scrubber implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_REPORT_COLUMNS = ("ticker", "date", "original", "replacement", "r_t", "r_next")


@dataclass(frozen=True)
class ScrubReport:
    """Structured audit ledger of every scrubbing event.

    ``events`` is a long-form ``DataFrame`` with one row per replaced cell
    and the columns declared in :data:`_REPORT_COLUMNS`.
    """

    events: pd.DataFrame


@dataclass(frozen=True)
class ScrubResult:
    """Output of :func:`scrub_daily_prices`."""

    prices: pd.DataFrame
    report: ScrubReport


def scrub_daily_prices(
    prices: pd.DataFrame,
    *,
    threshold: float = 0.30,
    reversal_tolerance: float = 0.10,
) -> ScrubResult:
    """Detect and repair single-day round-trip anomalies in a daily price panel.

    Args:
        prices: Wide ``date × ticker`` daily close prices. May contain NaN
            for cross-ticker holiday gaps.
        threshold: Minimum absolute daily return required to consider a
            print suspicious. Defaults to ``0.30``.
        reversal_tolerance: Maximum absolute compounded two-day return
            ``|(1 + r_t)(1 + r_{t+1}) - 1|`` for the move to be classified
            as a round-trip. Defaults to ``0.10``.

    Returns:
        :class:`ScrubResult` containing the cleaned prices and an audit
        ledger of every replaced cell.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be > 0, got {threshold!r}")
    if reversal_tolerance < 0.0:
        raise ValueError(
            f"reversal_tolerance must be >= 0, got {reversal_tolerance!r}"
        )

    cleaned = prices.copy()
    rows: list[dict[str, object]] = []

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 3:
            continue

        rets = series.pct_change()
        next_rets = rets.shift(-1)
        combined = (1.0 + rets) * (1.0 + next_rets) - 1.0
        mask = (rets.abs() > threshold) & (combined.abs() < reversal_tolerance)
        flagged_dates = series.index[mask.fillna(False)]

        for d in flagged_dates:
            i = series.index.get_loc(d)
            # Skip first/last (no surrounding pair to validate against).
            if i == 0 or i >= len(series) - 1:
                continue
            p_prev = float(series.iloc[i - 1])
            p_next = float(series.iloc[i + 1])
            replacement = float(np.sqrt(p_prev * p_next))
            rows.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "original": float(series.iloc[i]),
                    "replacement": replacement,
                    "r_t": float(rets.loc[d]),
                    "r_next": float(next_rets.loc[d]),
                }
            )
            cleaned.at[d, ticker] = replacement

    if rows:
        events = pd.DataFrame(rows, columns=list(_REPORT_COLUMNS))
    else:
        events = pd.DataFrame(
            {col: pd.Series(dtype=_empty_dtype(col)) for col in _REPORT_COLUMNS}
        )

    return ScrubResult(prices=cleaned, report=ScrubReport(events=events))


def _empty_dtype(column: str) -> str:
    if column == "ticker":
        return "object"
    if column == "date":
        return "datetime64[ns]"
    return "float64"
