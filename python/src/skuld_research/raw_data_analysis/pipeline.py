from __future__ import annotations

import pandas as pd

from skuld_research.raw_data_analysis.anomalies import (
    build_duplicate_flags,
    build_leakage_flags,
    build_numeric_outlier_flags,
)
from skuld_research.raw_data_analysis.coverage import (
    build_dataset_overview,
    build_feature_inventory,
    build_feature_sparsity,
    build_source_inventory,
    build_ticker_sparsity,
)
from skuld_research.raw_data_analysis.models import AnalysisDataset, RawDataAnalysisResult
from skuld_research.raw_data_analysis.temporal import (
    build_stale_value_summary,
    build_temporal_patterns,
)

ANOMALY_FLAG_COLUMNS = [
    "date",
    "ticker",
    "feature",
    "flag_type",
    "duplicate_count",
    "numeric_value",
    "score",
]


def _combine_anomaly_flags(dataset: AnalysisDataset) -> pd.DataFrame:
    duplicate_flags = build_duplicate_flags(dataset)
    numeric_outlier_flags = build_numeric_outlier_flags(dataset)
    anomaly_flags = pd.concat(
        [duplicate_flags, numeric_outlier_flags],
        ignore_index=True,
        sort=False,
    )
    if anomaly_flags.empty:
        return pd.DataFrame(columns=ANOMALY_FLAG_COLUMNS)
    return anomaly_flags.reindex(columns=ANOMALY_FLAG_COLUMNS).sort_values(
        [
            "feature",
            "ticker",
            "flag_type",
            "date",
            "duplicate_count",
            "numeric_value",
            "score",
        ],
        ascending=[True, True, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def run_raw_data_analysis(dataset: AnalysisDataset) -> RawDataAnalysisResult:
    return RawDataAnalysisResult(
        dataset_overview=build_dataset_overview(dataset),
        source_inventory=build_source_inventory(dataset),
        feature_inventory=build_feature_inventory(dataset),
        sparsity_by_feature=build_feature_sparsity(dataset),
        sparsity_by_ticker=build_ticker_sparsity(dataset),
        temporal_patterns=build_temporal_patterns(dataset),
        stale_value_summary=build_stale_value_summary(dataset),
        anomaly_flags=_combine_anomaly_flags(dataset),
        leakage_flags=build_leakage_flags(dataset),
    )
