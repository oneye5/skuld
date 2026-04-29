from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_raw_rows(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        data_path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str, "src": "int64"},
    )


def _write_raw_rows(data_path: Path, rows: pd.DataFrame) -> None:
    rows.to_csv(data_path, index=False)


def test_build_duplicate_flags_detects_duplicate_rows(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.anomalies import build_duplicate_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    duplicates = build_duplicate_flags(dataset)

    assert not duplicates.empty
    duplicate_row = duplicates.loc[
        (duplicates["ticker"] == "ANZ.NZ") & (duplicates["feature"] == "adj_close")
    ].iloc[0]
    assert duplicate_row["flag_type"] == "duplicate"
    assert int(duplicate_row["duplicate_count"]) == 1


def test_build_duplicate_flags_separates_duplicate_and_same_day_conflict_counts(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_duplicate_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706832000000,
                "ticker": "ANZ.NZ",
                "feature": "volume",
                "value": "120000",
                "src": 6,
            },
            {
                "timestamp": 1706835600000,
                "ticker": "ANZ.NZ",
                "feature": "volume",
                "value": "125000",
                "src": 6,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    duplicates = build_duplicate_flags(dataset)

    volume_flags = duplicates.loc[
        (duplicates["ticker"] == "ANZ.NZ") & (duplicates["feature"] == "volume")
    ]
    assert set(volume_flags["flag_type"]) == {"conflict", "duplicate"}
    counts_by_type = dict(
        zip(volume_flags["flag_type"], volume_flags["duplicate_count"], strict=True)
    )
    assert counts_by_type == {"duplicate": 1, "conflict": 1}


def test_build_numeric_outlier_flags_detects_extreme_pageview_jump(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.anomalies import build_numeric_outlier_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    outliers = build_numeric_outlier_flags(dataset)

    page_view_row = outliers.loc[
        (outliers["ticker"] == "ANZ.NZ") & (outliers["feature"] == "page_views")
    ].iloc[0]
    assert page_view_row["flag_type"] == "extreme_ratio"
    assert float(page_view_row["numeric_value"]) == 9_999_999.0


def test_build_numeric_outlier_flags_uses_robust_zscore_within_single_series(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_numeric_outlier_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    rows = rows.loc[rows["feature"] != "stress_signal"].copy()
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "stress_signal",
                "value": "10",
                "src": 8,
            },
            {
                "timestamp": 1706745600000,
                "ticker": "ANZ.NZ",
                "feature": "stress_signal",
                "value": "11",
                "src": 8,
            },
            {
                "timestamp": 1706832000000,
                "ticker": "ANZ.NZ",
                "feature": "stress_signal",
                "value": "10",
                "src": 8,
            },
            {
                "timestamp": 1706918400000,
                "ticker": "ANZ.NZ",
                "feature": "stress_signal",
                "value": "11",
                "src": 8,
            },
            {
                "timestamp": 1707004800000,
                "ticker": "ANZ.NZ",
                "feature": "stress_signal",
                "value": "40",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    outliers = build_numeric_outlier_flags(dataset)

    stress_row = outliers.loc[
        (outliers["ticker"] == "ANZ.NZ") & (outliers["feature"] == "stress_signal")
    ].iloc[0]
    assert stress_row["flag_type"] == "robust_zscore"
    assert float(stress_row["numeric_value"]) == 40.0


def test_build_numeric_outlier_flags_scopes_detection_per_ticker_series(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_numeric_outlier_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    rows = rows.loc[rows["feature"] != "ticker_scaled_metric"].copy()
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "ticker_scaled_metric",
                "value": "10",
                "src": 8,
            },
            {
                "timestamp": 1706745600000,
                "ticker": "ANZ.NZ",
                "feature": "ticker_scaled_metric",
                "value": "10",
                "src": 8,
            },
            {
                "timestamp": 1706832000000,
                "ticker": "ANZ.NZ",
                "feature": "ticker_scaled_metric",
                "value": "10",
                "src": 8,
            },
            {
                "timestamp": 1706659200000,
                "ticker": "SPK.NZ",
                "feature": "ticker_scaled_metric",
                "value": "10000",
                "src": 8,
            },
            {
                "timestamp": 1706745600000,
                "ticker": "SPK.NZ",
                "feature": "ticker_scaled_metric",
                "value": "10000",
                "src": 8,
            },
            {
                "timestamp": 1706832000000,
                "ticker": "SPK.NZ",
                "feature": "ticker_scaled_metric",
                "value": "10000",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    outliers = build_numeric_outlier_flags(dataset)

    assert "ticker_scaled_metric" not in set(outliers["feature"])


def test_build_numeric_outlier_flags_scopes_detection_per_source_series(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_numeric_outlier_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    rows = rows.loc[rows["feature"] != "cross_source_metric"].copy()
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "cross_source_metric",
                "value": "10",
                "src": 6,
            },
            {
                "timestamp": 1706745600000,
                "ticker": "ANZ.NZ",
                "feature": "cross_source_metric",
                "value": "10",
                "src": 6,
            },
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "cross_source_metric",
                "value": "10000",
                "src": 8,
            },
            {
                "timestamp": 1706745600000,
                "ticker": "ANZ.NZ",
                "feature": "cross_source_metric",
                "value": "10000",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    outliers = build_numeric_outlier_flags(dataset)

    assert "cross_source_metric" not in set(outliers["feature"])


def test_build_numeric_outlier_flags_skips_low_sample_ratio_when_min_non_positive(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_numeric_outlier_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "net_flows",
                "value": "-1",
                "src": 8,
            },
            {
                "timestamp": 1706745600000,
                "ticker": "ANZ.NZ",
                "feature": "net_flows",
                "value": "1000000",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    outliers = build_numeric_outlier_flags(dataset)

    assert "net_flows" not in set(outliers["feature"])


def test_build_numeric_outlier_flags_uses_latest_max_row_for_low_sample_fallback(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_numeric_outlier_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "search_interest",
                "value": "5",
                "src": 8,
            },
            {
                "timestamp": 1706745600000,
                "ticker": "ANZ.NZ",
                "feature": "search_interest",
                "value": "6000",
                "src": 8,
            },
            {
                "timestamp": 1709251200000,
                "ticker": "ANZ.NZ",
                "feature": "search_interest",
                "value": "6000",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    outliers = build_numeric_outlier_flags(dataset)

    feature_rows = outliers.loc[outliers["feature"] == "search_interest"]
    assert len(feature_rows) == 1
    selected_row = feature_rows.iloc[0]
    assert selected_row["flag_type"] == "extreme_ratio"
    assert selected_row["date"] == pd.Timestamp("2024-03-01")


def test_build_numeric_outlier_flags_breaks_same_day_max_ties_by_timestamp(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_numeric_outlier_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    rows = rows.loc[rows["feature"] != "same_day_search_interest"].copy()
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "same_day_search_interest",
                "value": "5",
                "src": 8,
            },
            {
                "timestamp": 1709251200000,
                "ticker": "ANZ.NZ",
                "feature": "same_day_search_interest",
                "value": "6000",
                "src": 8,
            },
            {
                "timestamp": 1709294400000,
                "ticker": "ANZ.NZ",
                "feature": "same_day_search_interest",
                "value": "6000",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    outliers = build_numeric_outlier_flags(dataset)

    selected_row = outliers.loc[
        outliers["feature"] == "same_day_search_interest"
    ].iloc[0]
    assert selected_row["date"] == pd.Timestamp("2024-03-01")


def test_build_leakage_flags_marks_finance_fields_as_warning(raw_analysis_csv_path):
    from skuld_research.raw_data_analysis.anomalies import build_leakage_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    flags = build_leakage_flags(dataset)

    leakage_row = flags.loc[flags["feature"] == "annual_basic_average_shares"].iloc[0]
    assert leakage_row["risk_level"] == "warning"
    assert "unclear publication timing" in leakage_row["reason"]
    assert "period-end" in leakage_row["reason"]
    assert "conservative delay bias" in leakage_row["reason"]
    assert "unsafe by default" in leakage_row["reason"]


def test_build_leakage_flags_marks_slow_moving_non_finance_feature_for_review(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_leakage_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    rows = rows.loc[rows["feature"] != "slow_indicator"].copy()
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "slow_indicator",
                "value": "1",
                "src": 8,
            },
            {
                "timestamp": 1714521600000,
                "ticker": "ANZ.NZ",
                "feature": "slow_indicator",
                "value": "2",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    flags = build_leakage_flags(dataset)

    leakage_row = flags.loc[flags["feature"] == "slow_indicator"].iloc[0]
    assert leakage_row["risk_level"] == "review"
    assert "Slow-moving field" in leakage_row["reason"]


def test_build_leakage_flags_rolls_up_slow_series_despite_staggered_feature_dates(
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.anomalies import build_leakage_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    rows = rows.loc[rows["feature"] != "staggered_slow_indicator"].copy()
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "staggered_slow_indicator",
                "value": "1",
                "src": 8,
            },
            {
                "timestamp": 1714521600000,
                "ticker": "ANZ.NZ",
                "feature": "staggered_slow_indicator",
                "value": "2",
                "src": 8,
            },
            {
                "timestamp": 1709251200000,
                "ticker": "SPK.NZ",
                "feature": "staggered_slow_indicator",
                "value": "3",
                "src": 8,
            },
            {
                "timestamp": 1717113600000,
                "ticker": "SPK.NZ",
                "feature": "staggered_slow_indicator",
                "value": "4",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    flags = build_leakage_flags(dataset)

    leakage_row = flags.loc[flags["feature"] == "staggered_slow_indicator"].iloc[0]
    assert leakage_row["risk_level"] == "review"
    assert "Slow-moving field" in leakage_row["reason"]


def test_build_leakage_flags_respects_source_specific_cadence(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.anomalies import build_leakage_flags
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = raw_analysis_csv_path
    rows = _read_raw_rows(data_path)
    rows = rows.loc[rows["feature"] != "shared_release_metric"].copy()
    new_rows = pd.DataFrame(
        [
            {
                "timestamp": 1706659200000,
                "ticker": "ANZ.NZ",
                "feature": "shared_release_metric",
                "value": "1",
                "src": 6,
            },
            {
                "timestamp": 1719792000000,
                "ticker": "ANZ.NZ",
                "feature": "shared_release_metric",
                "value": "2",
                "src": 6,
            },
            {
                "timestamp": 1709251200000,
                "ticker": "ANZ.NZ",
                "feature": "shared_release_metric",
                "value": "10",
                "src": 8,
            },
            {
                "timestamp": 1711929600000,
                "ticker": "ANZ.NZ",
                "feature": "shared_release_metric",
                "value": "11",
                "src": 8,
            },
            {
                "timestamp": 1714521600000,
                "ticker": "ANZ.NZ",
                "feature": "shared_release_metric",
                "value": "12",
                "src": 8,
            },
        ]
    )
    _write_raw_rows(data_path, pd.concat([rows, new_rows], ignore_index=True))

    dataset = load_analysis_dataset(data_path)
    flags = build_leakage_flags(dataset)
    leakage_row = flags.loc[flags["feature"] == "shared_release_metric"].iloc[0]

    assert leakage_row["risk_level"] == "review"
