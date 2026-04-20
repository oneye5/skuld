"""Data validation utilities.

All functions inspect data and return reports — they never mutate the input.
Consumers decide what to do with the report (log, raise, exclude rows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Result of a single validation check."""

    check_name: str
    issue_count: int = 0
    details: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return self.issue_count == 0


def detect_negative_prices(prices: pd.DataFrame) -> ValidationReport:
    """Flag any negative values in a prices DataFrame.

    Args:
        prices: index=date, columns=ticker, values=price

    Returns:
        Report listing affected (ticker, date) pairs.
    """
    report = ValidationReport(check_name="negative_prices")
    for ticker in prices.columns:
        neg_mask = prices[ticker] < 0
        if neg_mask.any():
            neg_dates = prices.index[neg_mask].strftime("%Y-%m-%d").tolist()
            report.details[ticker] = neg_dates
            report.issue_count += int(neg_mask.sum())
    return report


def detect_gaps(
    prices: pd.DataFrame, max_gap_days: int = 5
) -> ValidationReport:
    """Flag tickers with gaps of >max_gap_days consecutive business days.

    Args:
        prices: index=date (sorted), columns=ticker
        max_gap_days: threshold for gap detection

    Returns:
        Report listing affected tickers and gap periods.
    """
    report = ValidationReport(check_name="price_gaps")
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 2:
            continue
        dates = pd.DatetimeIndex(series.index).sort_values()
        dates_arr = dates.values.astype("datetime64[D]")
        bday_gaps = np.busday_count(dates_arr[:-1], dates_arr[1:])
        large_idx = np.where(bday_gaps > max_gap_days)[0]
        if len(large_idx) > 0:
            report.details[ticker] = [
                f"{dates[i].strftime('%Y-%m-%d')} → {dates[i + 1].strftime('%Y-%m-%d')} ({bday_gaps[i]} bdays)"
                for i in large_idx
            ]
            report.issue_count += len(large_idx)
    return report


def detect_stale_sources(
    source_latest: dict[str, pd.Timestamp],
    as_of: pd.Timestamp,
    max_age_days: int = 7,
) -> ValidationReport:
    """Flag sources whose latest data is older than threshold.

    Args:
        source_latest: mapping of source name → latest observation timestamp
        as_of: reference date (typically today)
        max_age_days: days before a source is considered stale

    Returns:
        Report listing stale sources with their age.
    """
    report = ValidationReport(check_name="stale_sources")
    for source, latest in source_latest.items():
        age = (as_of - latest).days
        if age > max_age_days:
            report.details[source] = [f"last data: {latest.strftime('%Y-%m-%d')}, age: {age} days"]
            report.issue_count += 1
    return report
