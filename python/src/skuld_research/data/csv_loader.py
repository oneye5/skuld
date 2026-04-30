"""Load the long-format CSV into typed DataFrames, split by feature/shape.

Categorisation is driven entirely by what the data *is*, not by where it
came from:

  * ticker present, feature in PRICE_FEATURES         → prices / volumes
  * ticker present, feature in CORPORATE_ACTIONS      → corporate_actions
  * ticker present, any other feature                 → fundamentals
  * ticker absent                                     → macro

The `src` column in the long CSV is intentionally ignored here — it is
provenance metadata for staleness/audit reporting, not a routing key.
No PIT filtering happens in this module; that is pit_loader's job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from skuld_common.features import (
    ADJ_CLOSE,
    CLOSE,
    CORPORATE_ACTIONS,
    HIGH,
    LOW,
    PRICE_FEATURES,
    VOLUME,
)
from skuld_research.data.adjustments import (
    AdjustmentAuditReport,
    RepairPolicy,
    audit_adjustments,
    repair_adjustments,
)
from skuld_research.data.scrubber import ScrubReport, scrub_daily_prices

if TYPE_CHECKING:
    from skuld_research.config.spec import AdjustmentSpec, ScrubbingSpec


_LOG = logging.getLogger(__name__)


def _empty_scrub_report() -> ScrubReport:
    columns = ("ticker", "date", "original", "replacement", "r_t", "r_next")
    dtypes = {
        "ticker": "object",
        "date": "datetime64[ns]",
        "original": "float64",
        "replacement": "float64",
        "r_t": "float64",
        "r_next": "float64",
    }
    return ScrubReport(
        events=pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in columns})
    )


@dataclass
class RawData:
    """All data from the CSV, categorised but unfiltered."""

    prices: pd.DataFrame  # index=date, columns=ticker, values=adj_close
    volumes: pd.DataFrame  # index=date, columns=ticker, values=volume
    fundamentals: pd.DataFrame  # MultiIndex (ticker, publication_date), columns=feature
    macro: pd.DataFrame  # index=date, columns=feature
    corporate_actions: pd.DataFrame  # columns: ticker, ex_date, type, factor
    scrub_report: ScrubReport = field(default_factory=_empty_scrub_report)
    adjustment_report: AdjustmentAuditReport | None = field(default=None)


def load_raw_csv(
    path: Path,
    *,
    scrub: ScrubbingSpec | None = None,
    adjustments: AdjustmentSpec | None = None,
) -> RawData:
    """Load long-format CSV and split into categorised DataFrames.

    Args:
        path: Path to data_long.csv.
        scrub: Optional :class:`ScrubbingSpec`. When provided and
            ``kind="round_trip"``, the adjusted-close price panel is
            scrubbed for single-day round-trip anomalies and the resulting
            audit ledger is attached to the returned :class:`RawData` as
            ``scrub_report``. Defaults to ``None`` (no scrubbing).
        adjustments: Optional :class:`AdjustmentSpec` controlling the
            corporate-action audit/repair layer. When provided and
            ``kind != "off"``, the layer runs *after* scrubbing using the
            spec's tolerance fields. ``kind="audit"`` attaches the
            :class:`AdjustmentAuditReport` to ``RawData.adjustment_report``
            without mutating prices. ``kind="repair"`` additionally replaces
            the price panel with the repaired output, mapping
            ``policy ∈ {"off","conservative","aggressive"}`` to
            :class:`RepairPolicy`. Defaults to ``None`` (no audit/repair).

    Returns:
        RawData with all observations categorised by feature/shape.
    """
    df = pd.read_csv(
        path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str},
        usecols=["timestamp", "ticker", "feature", "value"],
    )
    df["ticker"] = df["ticker"].fillna("")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)

    has_ticker = df["ticker"] != ""
    is_price_feature = df["feature"].isin(PRICE_FEATURES)
    is_corp_action = df["feature"].isin(CORPORATE_ACTIONS)

    prices = _pivot_ticker_feature(df, has_ticker & (df["feature"] == ADJ_CLOSE))
    scrub_report = _empty_scrub_report()
    if scrub is not None and scrub.kind == "round_trip" and not prices.empty:
        result = scrub_daily_prices(
            prices,
            threshold=scrub.threshold,
            reversal_tolerance=scrub.reversal_tolerance,
        )
        prices = result.prices
        scrub_report = result.report

    corporate_actions = _build_corporate_actions(df, has_ticker & is_corp_action)

    adjustment_report: AdjustmentAuditReport | None = None
    if adjustments is not None and adjustments.kind != "off" and not prices.empty:
        raw_close_panel = _pivot_ticker_feature(
            df, has_ticker & (df["feature"] == CLOSE)
        )
        raw_close_arg = raw_close_panel if not raw_close_panel.empty else None
        if adjustments.kind == "audit":
            adjustment_report = audit_adjustments(
                prices,
                corporate_actions,
                raw_close=raw_close_arg,
                dividend_residual_tol=adjustments.dividend_residual_tol,
                split_residual_tol=adjustments.split_residual_tol,
                unit_jump_tol=adjustments.unit_jump_tol,
            )
        else:  # kind == "repair"
            repair_result = repair_adjustments(
                prices,
                corporate_actions,
                raw_close=raw_close_arg,
                policy=RepairPolicy(adjustments.policy),
                dividend_residual_tol=adjustments.dividend_residual_tol,
                split_residual_tol=adjustments.split_residual_tol,
                unit_jump_tol=adjustments.unit_jump_tol,
            )
            prices = repair_result.prices
            adjustment_report = repair_result.report

    return RawData(
        prices=prices,
        volumes=_pivot_ticker_feature(df, has_ticker & (df["feature"] == VOLUME)),
        fundamentals=_build_fundamentals(df, has_ticker & ~is_price_feature & ~is_corp_action),
        macro=_build_macro(df, ~has_ticker),
        corporate_actions=corporate_actions,
        scrub_report=scrub_report,
        adjustment_report=adjustment_report,
    )


def _pivot_ticker_feature(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Pivot date×ticker for a single feature mask."""
    subset = df.loc[mask, ["date", "ticker", "value"]]
    if subset.empty:
        return pd.DataFrame()
    pivoted = subset.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
    pivoted.index.name = "date"
    return pivoted.sort_index()


def _build_fundamentals(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Build fundamentals with MultiIndex (ticker, publication_date)."""
    subset = df.loc[mask, ["ticker", "date", "feature", "value"]]
    if subset.empty:
        return pd.DataFrame(
            columns=pd.Index([], dtype=str),
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"]),
        )
    pivoted = subset.pivot_table(
        index=["ticker", "date"], columns="feature", values="value", aggfunc="last"
    )
    pivoted.index = pivoted.index.set_names(["ticker", "publication_date"])
    return pivoted


def _build_macro(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Build macro DataFrame: date × feature for ticker-less rows."""
    subset = df.loc[mask, ["date", "feature", "value"]]
    if subset.empty:
        return pd.DataFrame()
    pivoted = subset.pivot_table(index="date", columns="feature", values="value", aggfunc="last")
    pivoted.index.name = "date"
    return pivoted.sort_index()


def _build_corporate_actions(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Extract dividend and split rows into a flat event-shaped DataFrame."""
    subset = df.loc[mask, ["ticker", "date", "feature", "value"]].copy()
    subset = subset.rename(columns={"date": "ex_date", "feature": "type", "value": "factor"})
    return subset.reset_index(drop=True)


def load_raw_ohlc(
    path: Path,
    *,
    scrub: ScrubbingSpec | None = None,
    adjustments: AdjustmentSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw (unadjusted) high/low/close per ticker per date.

    Returned separately from `RawData` because OHLC is only required by the
    cost-modelling layer (Abdi-Ranaldo spread estimator). Keeping this off
    the core PIT contract avoids forcing every snapshot consumer to carry
    OHLC frames it doesn't need.

    Args:
        path: Path to data_long.csv.
        scrub: Optional :class:`ScrubbingSpec`. When provided and
            ``kind="round_trip"``, the *close* series is scrubbed for
            single-day round-trip anomalies. ``high`` and ``low`` are left
            untouched (they are microstructure inputs to the spread
            estimator and must not be reshaped to match a cleaned close).
        adjustments: Optional :class:`AdjustmentSpec`. Asymmetric with
            :func:`load_raw_csv`: this function returns a tuple with no
            attachment point for an :class:`AdjustmentAuditReport`, so
            ``kind="audit"`` is logged and ignored. ``kind="repair"``
            applies the repair to the ``close`` series only (``high`` and
            ``low`` are untouched, matching the scrub asymmetry); the
            audit report is discarded.

    Returns:
        (high, low, close) — each is a date x ticker DataFrame of raw
        (unadjusted) prices. Use these inputs ONLY for spread estimation
        and similar microstructure-derived metrics; use `RawData.prices`
        (adjusted close) for return calculations.
    """
    df = pd.read_csv(
        path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str},
        usecols=["timestamp", "ticker", "feature", "value"],
    )
    df["ticker"] = df["ticker"].fillna("")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    has_ticker = df["ticker"] != ""
    high = _pivot_ticker_feature(df, has_ticker & (df["feature"] == HIGH))
    low = _pivot_ticker_feature(df, has_ticker & (df["feature"] == LOW))
    close = _pivot_ticker_feature(df, has_ticker & (df["feature"] == CLOSE))
    if scrub is not None and scrub.kind == "round_trip" and not close.empty:
        close = scrub_daily_prices(
            close,
            threshold=scrub.threshold,
            reversal_tolerance=scrub.reversal_tolerance,
        ).prices
    if adjustments is not None and adjustments.kind != "off" and not close.empty:
        if adjustments.kind == "audit":
            _LOG.warning(
                "load_raw_ohlc received adjustments.kind='audit' but has no "
                "RawData attachment point; ignoring. Use load_raw_csv for "
                "audit reports."
            )
        else:  # kind == "repair"
            corporate_actions = _build_corporate_actions(
                df, has_ticker & df["feature"].isin(list(CORPORATE_ACTIONS))
            )
            close = repair_adjustments(
                close,
                corporate_actions,
                raw_close=close,
                policy=RepairPolicy(adjustments.policy),
                dividend_residual_tol=adjustments.dividend_residual_tol,
                split_residual_tol=adjustments.split_residual_tol,
                unit_jump_tol=adjustments.unit_jump_tol,
            ).prices
    return high, low, close
