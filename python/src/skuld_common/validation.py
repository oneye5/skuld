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

    Vectorised: a single boolean comparison + stack, no per-ticker loop.

    Args:
        prices: index=date, columns=ticker, values=price

    Returns:
        Report listing affected (ticker, date) pairs.
    """
    report = ValidationReport(check_name="negative_prices")
    if prices.empty:
        return report

    neg_mask = prices.lt(0)  # NaN < 0 → False, so NaN is safe
    if not neg_mask.to_numpy().any():
        return report

    # Stack to (date, ticker) MultiIndex of True positions
    hits_idx = neg_mask.stack()
    hits_idx = hits_idx[hits_idx].index
    report.issue_count = int(len(hits_idx))

    dates = pd.DatetimeIndex(hits_idx.get_level_values(0)).strftime("%Y-%m-%d")
    tickers = hits_idx.get_level_values(1).to_numpy()
    pairs = pd.DataFrame({"ticker": tickers, "date": dates})
    report.details = {t: g["date"].tolist() for t, g in pairs.groupby("ticker", sort=False)}
    return report


def detect_gaps(
    prices: pd.DataFrame, max_gap_days: int = 5
) -> ValidationReport:
    """Flag tickers with gaps of >max_gap_days consecutive business days.

    Vectorised: a single `np.busday_count` over the long-form observations
    rather than per-ticker. Suitable for thousands of tickers.

    Args:
        prices: index=date (sorted), columns=ticker
        max_gap_days: threshold for gap detection

    Returns:
        Report listing affected tickers and gap periods.
    """
    report = ValidationReport(check_name="price_gaps")
    if prices.empty:
        return report

    # Long form: (date, ticker, value) for each non-null observation.
    long = prices.stack().reset_index()
    long.columns = ["date", "ticker", "value"]
    long = long.sort_values(["ticker", "date"], kind="stable")

    if len(long) < 2:
        return report

    dates = long["date"].to_numpy().astype("datetime64[D]")
    tickers = long["ticker"].to_numpy()

    # busday_count between each row and its predecessor; first row of each
    # ticker is meaningless so we mask those out via the same-ticker mask.
    prev_dates = np.empty_like(dates)
    prev_dates[1:] = dates[:-1]
    prev_dates[0] = dates[0]
    bday = np.busday_count(prev_dates, dates)

    same_ticker = np.empty(len(tickers), dtype=bool)
    same_ticker[0] = False
    same_ticker[1:] = tickers[1:] == tickers[:-1]

    big = same_ticker & (bday > max_gap_days)
    if not big.any():
        return report

    big_df = pd.DataFrame(
        {
            "ticker": tickers[big],
            "prev_date": pd.to_datetime(prev_dates[big]),
            "date": pd.to_datetime(dates[big]),
            "bdays": bday[big],
        }
    )
    big_df["msg"] = (
        big_df["prev_date"].dt.strftime("%Y-%m-%d")
        + " → "
        + big_df["date"].dt.strftime("%Y-%m-%d")
        + " ("
        + big_df["bdays"].astype(str)
        + " bdays)"
    )
    report.issue_count = int(len(big_df))
    report.details = {t: g["msg"].tolist() for t, g in big_df.groupby("ticker", sort=False)}
    return report


def detect_ohlc_inconsistencies(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> ValidationReport:
    """Flag rows where OHLC relationships are violated.

    Enforced relationships per (date, ticker):
    - high >= open, high >= close, high >= low (high is the maximum)
    - low  <= open, low  <= close              (low  is the minimum)

    All frames must share the same index and columns. NaN rows are skipped.

    Args:
        open_: date × ticker open prices
        high:  date × ticker high prices
        low:   date × ticker low prices
        close: date × ticker close prices

    Returns:
        Report keyed by ticker listing violation descriptions.
    """
    report = ValidationReport(check_name="ohlc_inconsistencies")
    if open_.empty or high.empty or low.empty or close.empty:
        return report

    # Reindex everything onto a common (date, ticker) surface; NaN means skip.
    h = high.reindex_like(close)
    l = low.reindex_like(close)  # noqa: E741
    o = open_.reindex_like(close)
    c = close

    violated = (
        h.lt(o) | h.lt(c) | h.lt(l)
        | l.gt(o) | l.gt(c)
    )
    # Only flag cells where all four values are non-null (can actually evaluate).
    all_present = o.notna() & h.notna() & l.notna() & c.notna()
    violated = violated & all_present

    if not violated.to_numpy().any():
        return report

    hits = violated.stack()
    hits = hits[hits].index
    report.issue_count = int(len(hits))

    dates = pd.DatetimeIndex(hits.get_level_values(0)).strftime("%Y-%m-%d")
    tickers = hits.get_level_values(1).to_numpy()
    df = pd.DataFrame({"ticker": tickers, "date": dates})
    report.details = {
        t: g["date"].tolist() for t, g in df.groupby("ticker", sort=False)
    }
    return report


def detect_nan_density(
    prices: pd.DataFrame,
    *,
    max_nan_fraction: float = 0.5,
    min_rows: int = 20,
) -> ValidationReport:
    """Flag tickers whose NaN fraction in the price series exceeds the threshold.

    A very high NaN fraction usually signals a stale, delisted, or misgrouped
    ticker rather than a few legitimate trading halts or thin-market gaps.

    Args:
        prices: date × ticker price frame (adj_close or similar).
        max_nan_fraction: Fraction of NaN values above which a ticker is flagged.
            Default 0.5 means >50% NaN is flagged.
        min_rows: Minimum number of rows before a ticker is evaluated. Tickers
            with fewer rows are skipped (expected for very new listings).

    Returns:
        Report keyed by ticker with a human-readable summary per ticker.
    """
    report = ValidationReport(check_name="nan_density")
    if prices.empty:
        return report

    n_rows = len(prices)
    if n_rows < min_rows:
        return report

    nan_fractions = prices.isna().mean()
    flagged = nan_fractions[nan_fractions > max_nan_fraction]
    if flagged.empty:
        return report

    report.issue_count = int(len(flagged))
    for ticker, frac in flagged.items():
        report.details[str(ticker)] = [f"NaN fraction: {frac:.1%} ({int(frac * n_rows)}/{n_rows} rows)"]
    return report


def detect_duplicate_observations(long_df: pd.DataFrame) -> ValidationReport:
    """Flag duplicate `(date, ticker, feature)` rows in the raw long-format frame.

    The CSV loader resolves duplicates with `pivot_table(..., aggfunc="last")`,
    which silently keeps one observation per cell. In a financial context that
    is usually the wrong default unless duplicates are expected and logged.
    Run this check on the raw frame *before* pivoting so duplicates are visible.

    Args:
        long_df: Raw long-format frame with columns at least
            `date` (or `timestamp`), `ticker`, `feature`. Extra columns are
            ignored.

    Returns:
        Report keyed by ticker (or "<macro>" for ticker-less rows) with a
        per-key list of human-readable duplicate descriptions.
    """
    report = ValidationReport(check_name="duplicate_observations")
    if long_df.empty:
        return report

    date_col = "date" if "date" in long_df.columns else "timestamp"
    if date_col not in long_df.columns or "feature" not in long_df.columns:
        return report
    ticker_col = "ticker" if "ticker" in long_df.columns else None
    if ticker_col is None:
        keys = [date_col, "feature"]
    else:
        keys = [date_col, ticker_col, "feature"]

    counts = long_df.groupby(keys, dropna=False).size()
    dups = counts[counts > 1]
    if dups.empty:
        return report

    report.issue_count = int(dups.sum() - len(dups))  # extra rows beyond the first
    for key, n in dups.items():
        if ticker_col is None:
            d, feat = key
            t = "<macro>"
        else:
            d, t, feat = key
            if t == "" or pd.isna(t):
                t = "<macro>"
        d_str = pd.Timestamp(d).strftime("%Y-%m-%d") if not pd.isna(d) else "<NaT>"
        report.details.setdefault(str(t), []).append(f"{d_str} {feat} x{n}")
    return report


def detect_invalid_corporate_actions(corporate_actions: pd.DataFrame) -> ValidationReport:
    """Flag corporate-action rows with non-positive `factor` values.

    Splits and dividends should always be strictly positive. Zero or negative
    factors indicate ingestion bugs or upstream data corruption that would
    otherwise propagate silently into return calculations.

    Args:
        corporate_actions: Frame with columns `ticker`, `ex_date`, `type`,
            `factor` (as produced by `csv_loader._build_corporate_actions`).

    Returns:
        Report keyed by ticker with a list of "<ex_date> <type>=<factor>" entries.
    """
    report = ValidationReport(check_name="invalid_corporate_actions")
    if corporate_actions.empty or "factor" not in corporate_actions.columns:
        return report

    bad = corporate_actions[
        corporate_actions["factor"].isna() | (corporate_actions["factor"] <= 0)
    ]
    if bad.empty:
        return report

    report.issue_count = int(len(bad))
    for ticker, group in bad.groupby("ticker"):
        msgs = [
            f"{pd.Timestamp(r.ex_date).strftime('%Y-%m-%d')} {r.type}={r.factor}"
            for r in group.itertuples()
        ]
        report.details[str(ticker)] = msgs
    return report


def detect_stale_fundamentals(
    fundamentals: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    max_age_days: int = 540,
    fields: tuple[str, ...] | None = None,
) -> ValidationReport:
    """Flag tickers whose latest fundamental publication is older than threshold.

    Default threshold is ~18 months, which catches share-count or earnings
    series that have stopped updating but are still being forward-filled into
    market-cap and other derived series.

    Args:
        fundamentals: MultiIndex `(ticker, publication_date)` frame as held on
            `PITSnapshot.fundamentals`.
        as_of: Reference timestamp (typically the snapshot's `asof`).
        max_age_days: Days beyond which the latest publication is considered stale.
        fields: Optional iterable of fundamental column names to restrict the
            check to. If None, the latest publication date across any field is
            used per ticker.

    Returns:
        Report keyed by ticker with `["last publication: YYYY-MM-DD, age: N days"]`.
    """
    report = ValidationReport(check_name="stale_fundamentals")
    if fundamentals.empty:
        return report

    if "publication_date" not in (fundamentals.index.names or []):
        return report

    asof_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of

    if fields is not None:
        cols = [c for c in fields if c in fundamentals.columns]
        if not cols:
            return report
        non_null = fundamentals[cols].dropna(how="all")
    else:
        non_null = fundamentals

    if non_null.empty:
        return report

    latest_per_ticker = (
        non_null.reset_index()
        .groupby("ticker")["publication_date"]
        .max()
    )
    for ticker, latest in latest_per_ticker.items():
        latest_ts = pd.Timestamp(latest)
        if latest_ts.tzinfo:
            latest_ts = latest_ts.tz_localize(None)
        age = (asof_naive - latest_ts).days
        if age > max_age_days:
            report.details[str(ticker)] = [
                f"last publication: {latest_ts.strftime('%Y-%m-%d')}, age: {age} days"
            ]
            report.issue_count += 1
    return report


def detect_stale_sources(
    source_latest: dict[str, pd.Timestamp],
    as_of: pd.Timestamp,
    max_age_days: int = 7,
) -> ValidationReport:
    """Flag sources whose latest data is older than threshold.

    Not vectorised: input is a small metadata dict (~tens of sources), so
    pandas overhead would dominate.

    Args:
        source_latest: mapping of source name → latest observation timestamp
        as_of: reference date (typically today)
        max_age_days: days before a source is considered stale

    Returns:
        Report listing stale sources with their age.
    """
    report = ValidationReport(check_name="stale_sources")
    for source, latest in source_latest.items():
        latest_ts = pd.Timestamp(latest) if not isinstance(latest, pd.Timestamp) else latest
        age = (as_of - latest_ts).days
        if age > max_age_days:
            report.details[source] = [f"last data: {latest_ts.strftime('%Y-%m-%d')}, age: {age} days"]
            report.issue_count += 1
    return report
