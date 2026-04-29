from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from skuld_research.raw_data_analysis.models import AnalysisDataset

DUPLICATE_FLAG_COLUMNS = ["ticker", "feature", "flag_type", "duplicate_count"]
OUTLIER_FLAG_COLUMNS = [
    "date",
    "ticker",
    "feature",
    "numeric_value",
    "flag_type",
    "score",
]
LEAKAGE_FLAG_COLUMNS = ["feature", "risk_level", "reason"]
FUNDAMENTAL_SOURCE_NAMES = {"yf_finances"}
ROBUST_ZSCORE_THRESHOLD = 8.0
LOW_SAMPLE_RATIO_THRESHOLD = 1_000.0
SLOW_MOVING_REVIEW_GAP_DAYS = 60.0


def build_duplicate_flags(dataset: AnalysisDataset) -> pd.DataFrame:
    duplicate_records: list[dict[str, object]] = []

    exact_duplicate_keys = ["timestamp", "ticker", "feature", "raw_value", "src"]
    exact_duplicate_groups = (
        dataset.rows.groupby(exact_duplicate_keys, dropna=False)
        .size()
        .reset_index()
        .rename(columns={0: "row_count"})
    )
    exact_duplicate_groups = exact_duplicate_groups.loc[
        exact_duplicate_groups["row_count"] > 1
    ].copy()
    if not exact_duplicate_groups.empty:
        exact_duplicate_groups["duplicate_count"] = exact_duplicate_groups["row_count"] - 1
        duplicate_summary = (
            exact_duplicate_groups.groupby(["ticker", "feature"], dropna=False)["duplicate_count"]
            .sum()
            .reset_index()
        )
        for row in duplicate_summary.itertuples(index=False):
            duplicate_records.append(
                {
                    "ticker": row.ticker,
                    "feature": row.feature,
                    "flag_type": "duplicate",
                    "duplicate_count": int(row.duplicate_count),
                }
            )

    same_day_rows = dataset.rows.assign(observation_date=dataset.rows["date"].dt.normalize())
    same_day_source_groups = (
        same_day_rows.groupby(["observation_date", "ticker", "feature", "src"], dropna=False)
        .agg(distinct_raw_values=("raw_value", "nunique"))
        .reset_index()
    )
    same_day_source_groups = same_day_source_groups.loc[
        same_day_source_groups["distinct_raw_values"] > 1
    ].copy()
    if not same_day_source_groups.empty:
        conflict_summary = (
            same_day_source_groups.groupby(["ticker", "feature"], dropna=False)
            .size()
            .reset_index(name="duplicate_count")
        )
        for row in conflict_summary.itertuples(index=False):
            duplicate_records.append(
                {
                    "ticker": row.ticker,
                    "feature": row.feature,
                    "flag_type": "conflict",
                    "duplicate_count": int(row.duplicate_count),
                }
            )

    if not duplicate_records:
        return pd.DataFrame(columns=DUPLICATE_FLAG_COLUMNS)

    duplicate_flags = pd.DataFrame.from_records(duplicate_records, columns=DUPLICATE_FLAG_COLUMNS)
    return duplicate_flags.sort_values(
        ["ticker", "feature", "flag_type", "duplicate_count"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def _append_mad_outliers(group: pd.DataFrame, records: list[dict[str, object]]) -> None:
    values = group["numeric_value"].astype(float)
    median_value = values.median()
    median = float(cast(float, median_value))
    mad_value = (values - median).abs().median()
    mad = float(cast(float, mad_value))
    if pd.isna(mad) or mad <= 0.0:
        return

    scores = 0.6745 * (values - median).abs() / mad
    flagged = group.loc[scores > ROBUST_ZSCORE_THRESHOLD].copy()
    if flagged.empty:
        return

    flagged_scores = scores.loc[flagged.index]
    for row_index, row in flagged.iterrows():
        records.append(
            {
                "date": row["date"],
                "ticker": row["ticker"],
                "feature": row["feature"],
                "numeric_value": row["numeric_value"],
                "flag_type": "robust_zscore",
                "score": float(flagged_scores.loc[row_index]),
            }
        )


def _append_low_sample_ratio_outliers(
    group: pd.DataFrame,
    records: list[dict[str, object]],
) -> None:
    unique_values = np.asarray(
        sorted(float(value) for value in group["numeric_value"].dropna().tolist()),
        dtype=float,
    )
    if unique_values.size < 2 or unique_values.size > 3:
        return
    if unique_values[0] <= 0.0:
        return

    ratio = float(unique_values[-1] / unique_values[0])
    if ratio < LOW_SAMPLE_RATIO_THRESHOLD:
        return

    highest_value = unique_values[-1]
    flagged = group.loc[group["numeric_value"].astype(float) == highest_value].copy()
    if flagged.empty:
        return

    latest_flagged = flagged.sort_values(
        ["date", "timestamp", "ticker"],
        ascending=[False, False, True],
    ).iloc[0]
    records.append(
        {
            "date": latest_flagged["date"],
            "ticker": latest_flagged["ticker"],
            "feature": latest_flagged["feature"],
            "numeric_value": latest_flagged["numeric_value"],
            "flag_type": "extreme_ratio",
            "score": ratio,
        }
    )


def build_numeric_outlier_flags(dataset: AnalysisDataset) -> pd.DataFrame:
    numeric_rows = dataset.rows.loc[dataset.rows["numeric_value"].notna()].copy()
    if numeric_rows.empty:
        return pd.DataFrame(columns=OUTLIER_FLAG_COLUMNS)

    numeric_rows = numeric_rows.sort_values(
        ["feature", "ticker", "source_name", "date", "timestamp"]
    )
    records: list[dict[str, object]] = []
    seen_keys: set[tuple[object, ...]] = set()
    for _, group in numeric_rows.groupby(["ticker", "feature", "source_name"], dropna=False):
        _append_mad_outliers(group, records)
        _append_low_sample_ratio_outliers(group, records)

    deduplicated_records: list[dict[str, object]] = []
    for record in records:
        key = (
            record["date"],
            record["ticker"],
            record["feature"],
            record["numeric_value"],
            record["flag_type"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated_records.append(record)

    if not deduplicated_records:
        return pd.DataFrame(columns=OUTLIER_FLAG_COLUMNS)
    return (
        pd.DataFrame.from_records(deduplicated_records, columns=OUTLIER_FLAG_COLUMNS)
        .sort_values(["feature", "date", "ticker", "flag_type"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )


def build_leakage_flags(dataset: AnalysisDataset) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for feature, group in dataset.rows.groupby("feature", dropna=False):
        source_names = set(group["source_name"])

        if source_names & FUNDAMENTAL_SOURCE_NAMES:
            records.append(
                {
                    "feature": feature,
                    "risk_level": "warning",
                    "reason": (
                        "Fundamental field uses period-end dates with unclear "
                        "publication timing; this creates conservative delay bias "
                        "rather than lookahead leakage, but is still unsafe by "
                        "default for raw feature engineering."
                    ),
                }
            )
            continue

        has_slow_series = False
        for _, series_group in group.groupby(["ticker", "source_name"], dropna=False):
            unique_dates = sorted(series_group["date"].drop_duplicates().tolist())
            unique_date_series = pd.Series(unique_dates, dtype="datetime64[ns]")
            gaps = unique_date_series.diff().dropna().dt.days.astype(float)
            median_gap_days = float(gaps.median()) if not gaps.empty else np.nan
            if pd.notna(median_gap_days) and median_gap_days > SLOW_MOVING_REVIEW_GAP_DAYS:
                has_slow_series = True
                break

        if has_slow_series:
            records.append(
                {
                    "feature": feature,
                    "risk_level": "review",
                    "reason": (
                        "Slow-moving field should be checked for publication timing "
                        "before use."
                    ),
                }
            )

    if not records:
        return pd.DataFrame(columns=LEAKAGE_FLAG_COLUMNS)
    return (
        pd.DataFrame.from_records(records, columns=LEAKAGE_FLAG_COLUMNS)
        .drop_duplicates()
        .sort_values(["risk_level", "feature"], ascending=[True, True])
        .reset_index(drop=True)
    )
