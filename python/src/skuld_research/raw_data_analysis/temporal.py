from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from skuld_research.raw_data_analysis.models import AnalysisDataset

MACRO_SERIES_KEY = "__macro__"
STALE_VALUE_SUMMARY_COLUMNS = ["ticker", "feature", "max_repeat_run", "max_repeat_span_days"]
TEMPORAL_PATTERN_COLUMNS = [
    "feature",
    "observation_dates",
    "series_count",
    "gap_count",
    "median_gap_days",
    "gap_p90_days",
    "max_gap_days",
    "irregularity_ratio",
    "frequency_label",
]


def _label_frequency(median_gap_days: float) -> str:
    if median_gap_days <= 2:
        return "daily"
    if median_gap_days <= 10:
        return "weekly_or_irregular"
    if median_gap_days <= 45:
        return "monthly_or_slower"
    return "quarterly_or_slower"


def _with_series_key(rows: pd.DataFrame) -> pd.DataFrame:
    scoped_rows = rows.copy()
    scope_key = scoped_rows["ticker"].where(scoped_rows["ticker"] != "", MACRO_SERIES_KEY)
    scoped_rows["series_key"] = scoped_rows["source_name"] + "::" + scope_key
    return scoped_rows


def _with_source_series(rows: pd.DataFrame) -> pd.DataFrame:
    scoped_rows = rows.copy()
    scoped_rows["scope_key"] = scoped_rows["ticker"].where(
        scoped_rows["ticker"] != "",
        MACRO_SERIES_KEY,
    )
    return scoped_rows


def _collapse_same_date_numeric_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=["ticker", "feature", "source_name", "date", "numeric_value", "is_conflict"]
        )

    grouped = (
        _with_source_series(rows)
        .groupby(["ticker", "feature", "source_name", "date"], dropna=False)
        .agg(
            distinct_numeric_values=("numeric_value", "nunique"),
            numeric_value=("numeric_value", "min"),
        )
        .reset_index()
    )
    grouped["is_conflict"] = grouped["distinct_numeric_values"] > 1
    grouped.loc[grouped["is_conflict"], "numeric_value"] = np.nan
    return grouped.loc[
        :,
        ["ticker", "feature", "source_name", "date", "numeric_value", "is_conflict"],
    ]


def build_temporal_patterns(dataset: AnalysisDataset) -> pd.DataFrame:
    distinct_observations = (
        _with_series_key(dataset.rows)
        .loc[:, ["feature", "series_key", "date"]]
        .drop_duplicates()
    )

    records: list[dict[str, object]] = []
    for feature, feature_rows in distinct_observations.groupby("feature", dropna=False):
        series_gaps: list[pd.Series] = []
        for _, series_rows in feature_rows.groupby("series_key", dropna=False):
            ordered_dates = series_rows["date"].sort_values().reset_index(drop=True)
            gaps = ordered_dates.diff().dropna().dt.days.astype(float)
            if not gaps.empty:
                series_gaps.append(gaps)

        if series_gaps:
            all_gaps = pd.concat(series_gaps, ignore_index=True)
            median_gap_days = float(all_gaps.median())
            gap_p90_days = float(all_gaps.quantile(0.9))
            max_gap_days = float(all_gaps.max())
            irregularity_ratio = (
                float(all_gaps.std(ddof=0) / median_gap_days)
                if median_gap_days > 0
                else 0.0
            )
            gap_count = int(len(all_gaps))
            frequency_label = _label_frequency(median_gap_days)
        else:
            median_gap_days = np.nan
            gap_p90_days = np.nan
            max_gap_days = np.nan
            irregularity_ratio = 0.0
            gap_count = 0
            frequency_label = "singleton"

        records.append(
            {
                "feature": feature,
                "observation_dates": int(feature_rows["date"].nunique()),
                "series_count": int(feature_rows["series_key"].nunique()),
                "gap_count": gap_count,
                "median_gap_days": median_gap_days,
                "gap_p90_days": gap_p90_days,
                "max_gap_days": max_gap_days,
                "irregularity_ratio": irregularity_ratio,
                "frequency_label": frequency_label,
            }
        )
    if not records:
        return pd.DataFrame(columns=TEMPORAL_PATTERN_COLUMNS)
    return (
        pd.DataFrame.from_records(records, columns=TEMPORAL_PATTERN_COLUMNS)
        .sort_values("feature")
        .reset_index(drop=True)
    )


def build_stale_value_summary(dataset: AnalysisDataset) -> pd.DataFrame:
    numeric_rows = dataset.rows.loc[dataset.rows["numeric_value"].notna()].copy()
    if numeric_rows.empty:
        return pd.DataFrame(columns=STALE_VALUE_SUMMARY_COLUMNS)

    numeric_rows = _collapse_same_date_numeric_rows(numeric_rows).sort_values(
        ["ticker", "feature", "date"]
    )

    per_source_records: list[dict[str, object]] = []
    for group_key, group in numeric_rows.groupby(
        ["ticker", "feature", "source_name"],
        dropna=False,
    ):
        ticker, feature, _source_name = cast(tuple[object, object, object], group_key)
        ordered_group = group.sort_values("date").reset_index(drop=True)

        max_repeat_run = 0
        max_repeat_span_days = 0.0
        current_value: float | None = None
        run_start: pd.Timestamp | None = None
        run_length = 0

        for row in ordered_group.to_dict("records"):
            if bool(row["is_conflict"]) or pd.isna(row["numeric_value"]):
                current_value = None
                run_start = None
                run_length = 0
                continue

            numeric_value = float(row["numeric_value"])
            row_date = cast(pd.Timestamp, pd.Timestamp(row["date"]))

            if current_value is None or numeric_value != current_value:
                current_value = numeric_value
                run_start = row_date
                run_length = 1
                if run_length > max_repeat_run:
                    max_repeat_run = run_length
                continue

            run_length += 1
            assert run_start is not None
            repeat_span_days = float((row_date - run_start).days)
            if run_length > max_repeat_run or (
                run_length == max_repeat_run and repeat_span_days > max_repeat_span_days
            ):
                max_repeat_run = run_length
                max_repeat_span_days = repeat_span_days
        per_source_records.append(
            {
                "ticker": ticker,
                "feature": feature,
                "max_repeat_run": max_repeat_run,
                "max_repeat_span_days": max_repeat_span_days,
            }
        )

    records = (
        pd.DataFrame.from_records(per_source_records)
        .groupby(["ticker", "feature"], dropna=False)
        .agg(
            max_repeat_run=("max_repeat_run", "max"),
            max_repeat_span_days=("max_repeat_span_days", "max"),
        )
        .reset_index()
    )

    return (
        records.loc[:, STALE_VALUE_SUMMARY_COLUMNS]
        .sort_values(["ticker", "feature"])
        .reset_index(drop=True)
    )
