from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXPECTED_HEADINGS = [
    "# Raw Data Analysis Report",
    "## Dataset Overview",
    "## Source Inventory",
    "## Feature Inventory",
    "## Sparsity and Missingness",
    "## Temporal Behavior",
    "## Outliers and Anomalies",
    "## Leakage Risk Review",
    "## Research Implications",
]

EXPECTED_TABLE_FILES = {
    "anomaly_flags.csv",
    "dataset_overview.csv",
    "feature_inventory.csv",
    "leakage_flags.csv",
    "source_inventory.csv",
    "sparsity_by_feature.csv",
    "sparsity_by_ticker.csv",
    "stale_value_summary.csv",
    "temporal_patterns.csv",
}

EXPECTED_TABLE_FILE_ORDER = [
    "dataset_overview.csv",
    "source_inventory.csv",
    "feature_inventory.csv",
    "sparsity_by_feature.csv",
    "sparsity_by_ticker.csv",
    "temporal_patterns.csv",
    "stale_value_summary.csv",
    "anomaly_flags.csv",
    "leakage_flags.csv",
]

EXPECTED_RESEARCH_CATEGORY_HEADINGS = [
    "### Likely usable now",
    "### Usable only with safeguards",
    "### Unsafe until repaired or better timestamped",
    "### Too sparse to prioritize",
]


def test_run_raw_data_analysis_returns_populated_result(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    result = run_raw_data_analysis(dataset)

    assert int(result.dataset_overview.loc[0, "row_count"]) == len(dataset.rows)
    assert not result.source_inventory.empty
    assert not result.feature_inventory.empty
    assert not result.sparsity_by_feature.empty
    assert not result.sparsity_by_ticker.empty
    assert not result.temporal_patterns.empty
    assert not result.stale_value_summary.empty
    assert not result.anomaly_flags.empty
    assert not result.leakage_flags.empty
    assert set(result.anomaly_flags["flag_type"]) >= {"duplicate", "extreme_ratio"}


def test_write_raw_data_report_creates_stable_headings_in_order(
    tmp_path: Path,
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis
    from skuld_research.raw_data_analysis.report import write_raw_data_report

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    result = run_raw_data_analysis(dataset)
    out_dir = tmp_path / "raw_data_analysis" / "2026-04-29"

    report_path, _ = write_raw_data_report(result, out_dir)

    content = report_path.read_text(encoding="utf-8")
    heading_positions = [content.index(heading) for heading in EXPECTED_HEADINGS]

    assert heading_positions == sorted(heading_positions)
    for heading in EXPECTED_RESEARCH_CATEGORY_HEADINGS:
        assert heading in content
    assert "### Confirmed issues" in content
    assert "### Heuristic warnings" in content
    assert "No confirmed leakage issues were proven from raw timestamps alone." in content
    assert "### Heuristic leakage warnings" in content


def test_write_raw_data_report_lists_machine_readable_artifacts_in_stable_order(
    tmp_path: Path,
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis
    from skuld_research.raw_data_analysis.report import write_raw_data_report

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    result = run_raw_data_analysis(dataset)
    out_dir = tmp_path / "raw_data_analysis" / "2026-04-29"

    report_path, _ = write_raw_data_report(result, out_dir)

    content = report_path.read_text(encoding="utf-8")
    expected_links = [
        "- `summary.json`: [summary.json](summary.json)",
        *[
            f"- `{file_name}`: [tables/{file_name}](tables/{file_name})"
            for file_name in EXPECTED_TABLE_FILE_ORDER
        ],
    ]

    positions = [content.index(link) for link in expected_links]

    assert positions == sorted(positions)


def test_write_raw_data_report_writes_expected_table_files(
    tmp_path: Path,
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis
    from skuld_research.raw_data_analysis.report import write_raw_data_report

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    result = run_raw_data_analysis(dataset)
    out_dir = tmp_path / "raw_data_analysis" / "2026-04-29"

    write_raw_data_report(result, out_dir)

    tables_dir = out_dir / "tables"
    assert tables_dir.is_dir()
    assert {path.name for path in tables_dir.iterdir()} == EXPECTED_TABLE_FILES


def test_write_raw_data_report_writes_summary_json_with_stable_keys(
    tmp_path: Path,
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis
    from skuld_research.raw_data_analysis.report import write_raw_data_report

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    result = run_raw_data_analysis(dataset)
    out_dir = tmp_path / "raw_data_analysis" / "2026-04-29"

    report_path, summary_path = write_raw_data_report(result, out_dir)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary_path.exists()
    assert list(summary) == [
        "anomaly_flag_count",
        "date_max",
        "date_min",
        "feature_count",
        "issue_counts",
        "leakage_flag_count",
        "report_path",
        "research_implications",
        "row_count",
        "source_count",
        "table_paths",
        "top_findings",
        "ticker_count",
    ]
    assert summary["report_path"] == "report.md"
    assert set(summary["table_paths"]) == EXPECTED_TABLE_FILES
    assert list(summary["table_paths"]) == EXPECTED_TABLE_FILE_ORDER
    assert summary["table_paths"]["dataset_overview.csv"] == "tables/dataset_overview.csv"
    assert list(summary["issue_counts"]) == [
        "confirmed_anomaly_issues",
        "heuristic_anomaly_warnings",
        "heuristic_leakage_warnings",
    ]
    assert list(summary["research_implications"]) == [
        "Likely usable now",
        "Usable only with safeguards",
        "Unsafe until repaired or better timestamped",
        "Too sparse to prioritize",
    ]
    assert isinstance(summary["top_findings"], list)
    assert summary["top_findings"]


def test_write_raw_data_report_research_implication_buckets_are_mutually_exclusive(
    tmp_path: Path,
    raw_analysis_csv_path: Path,
):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis
    from skuld_research.raw_data_analysis.report import write_raw_data_report

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    result = run_raw_data_analysis(dataset)
    out_dir = tmp_path / "raw_data_analysis" / "2026-04-29"

    _, summary_path = write_raw_data_report(result, out_dir)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    buckets = summary["research_implications"]
    bucket_sets = {name: set(features) for name, features in buckets.items()}

    for left_name, left_features in bucket_sets.items():
        for right_name, right_features in bucket_sets.items():
            if left_name == right_name:
                continue
            assert left_features.isdisjoint(right_features)


def test_run_raw_data_analysis_sorts_anomaly_flags_with_total_deterministic_order(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.pipeline import run_raw_data_analysis

    def run_case(case_dir: Path, final_values: tuple[int, int]) -> pd.DataFrame:
        case_dir.mkdir(parents=True, exist_ok=True)
        data_path = case_dir / "data_long.csv"
        data_path.write_text(
            "timestamp,ticker,feature,value,src\n"
            "1706659200000,AAA.NZ,signal,10,8\n"
            "1706745600000,AAA.NZ,signal,11,8\n"
            "1706832000000,AAA.NZ,signal,10,8\n"
            "1706918400000,AAA.NZ,signal,11,8\n"
            f"1707004800000,AAA.NZ,signal,{final_values[0]},8\n"
            f"1707004800000,AAA.NZ,signal,{final_values[1]},8\n",
            encoding="utf-8",
        )
        (case_dir / "source_legend.csv").write_text(
            "id,name\n8,wikimedia_pageviews\n",
            encoding="utf-8",
        )
        dataset = load_analysis_dataset(data_path)
        return run_raw_data_analysis(dataset).anomaly_flags.loc[
            lambda frame: frame["feature"] == "signal"
        ].reset_index(drop=True)

    forward_flags = run_case(tmp_path / "forward", (50, 40))
    reverse_flags = run_case(tmp_path / "reverse", (40, 50))

    assert list(forward_flags["flag_type"]) == ["conflict", "robust_zscore", "robust_zscore"]
    assert list(forward_flags["date"]) == [
        pd.NaT,
        pd.Timestamp("2024-02-04"),
        pd.Timestamp("2024-02-04"),
    ]
    assert pd.isna(forward_flags.loc[0, "numeric_value"])
    assert list(forward_flags["numeric_value"].iloc[1:]) == [40.0, 50.0]
    assert pd.isna(reverse_flags.loc[0, "numeric_value"])
    assert list(reverse_flags["numeric_value"].iloc[1:]) == [40.0, 50.0]
    assert forward_flags.equals(reverse_flags)
