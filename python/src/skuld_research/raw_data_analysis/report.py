from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from skuld_research.raw_data_analysis.models import RawDataAnalysisResult

TABLE_FILE_ORDER = [
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


def _table_map(result: RawDataAnalysisResult) -> dict[str, pd.DataFrame]:
    return {
        "dataset_overview.csv": result.dataset_overview,
        "source_inventory.csv": result.source_inventory,
        "feature_inventory.csv": result.feature_inventory,
        "sparsity_by_feature.csv": result.sparsity_by_feature,
        "sparsity_by_ticker.csv": result.sparsity_by_ticker,
        "temporal_patterns.csv": result.temporal_patterns,
        "stale_value_summary.csv": result.stale_value_summary,
        "anomaly_flags.csv": result.anomaly_flags,
        "leakage_flags.csv": result.leakage_flags,
    }


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."

    display_frame = frame.copy()
    for column in display_frame.columns:
        display_frame[column] = display_frame[column].map(_cell_to_text)

    headers = [str(column) for column in display_frame.columns]
    rows = display_frame.values.tolist()
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        cells = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator, *[format_row(row) for row in rows]])


def _cell_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return ""
    return str(value)


def _timestamp_to_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    if isinstance(value, (str, int, float)):
        return pd.Timestamp(value).isoformat()
    return None


def _path_to_text(path: Path) -> str:
    return path.as_posix()


def _relative_artifact_path(path: Path, out_dir: Path) -> str:
    return _path_to_text(path.relative_to(out_dir))


def _feature_list_markdown(features: list[str]) -> str:
    if not features:
        return "- None"
    return "\n".join(f"- `{feature}`" for feature in features)


def _sorted_unique_strings(values: pd.Series) -> list[str]:
    return sorted(values.dropna().astype(str).unique().tolist())


def _build_research_implication_buckets(result: RawDataAnalysisResult) -> dict[str, list[str]]:
    all_features = sorted(
        result.feature_inventory["feature"].dropna().astype(str).unique().tolist()
    )

    warning_features = set(
        _sorted_unique_strings(
            result.leakage_flags.loc[result.leakage_flags["risk_level"] == "warning", "feature"]
        )
    )
    review_features = set(
        _sorted_unique_strings(
            result.leakage_flags.loc[result.leakage_flags["risk_level"] == "review", "feature"]
        )
    )
    anomaly_feature_values = cast(
        pd.Series,
        result.anomaly_flags["feature"]
        if "feature" in result.anomaly_flags.columns
        else pd.Series(dtype=str),
    )
    anomaly_features = set(_sorted_unique_strings(anomaly_feature_values))
    sparse_features = set(
        _sorted_unique_strings(
            cast(
                pd.Series,
                result.sparsity_by_feature.loc[
                    result.sparsity_by_feature["missing_fraction"].fillna(0.0) >= 0.75,
                    "feature",
                ],
            )
        )
    )

    unsafe_features = warning_features
    safeguards_features = (review_features | anomaly_features) - unsafe_features
    too_sparse_features = sparse_features - unsafe_features - safeguards_features
    likely_usable_features = sorted(
        set(all_features)
        - set(unsafe_features)
        - set(safeguards_features)
        - set(too_sparse_features)
    )

    return {
        "Likely usable now": likely_usable_features,
        "Usable only with safeguards": sorted(safeguards_features),
        "Unsafe until repaired or better timestamped": sorted(unsafe_features),
        "Too sparse to prioritize": sorted(too_sparse_features),
    }


def _select_rows(frame: pd.DataFrame, flag_types: set[str]) -> pd.DataFrame:
    if frame.empty or "flag_type" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[frame["flag_type"].isin(sorted(flag_types))].reset_index(drop=True)


def _build_top_findings(
    confirmed_anomaly_issues: pd.DataFrame,
    heuristic_anomaly_warnings: pd.DataFrame,
    leakage_flags: pd.DataFrame,
    research_buckets: dict[str, list[str]],
) -> list[str]:
    findings = [
        f"Confirmed anomaly issues: {len(confirmed_anomaly_issues)}",
        f"Heuristic anomaly warnings: {len(heuristic_anomaly_warnings)}",
        f"Heuristic leakage warnings: {len(leakage_flags)}",
        (
            "Unsafe until repaired or better timestamped: "
            + ", ".join(research_buckets["Unsafe until repaired or better timestamped"])
            if research_buckets["Unsafe until repaired or better timestamped"]
            else "Unsafe until repaired or better timestamped: none"
        ),
    ]
    return findings


def write_raw_data_report(result: RawDataAnalysisResult, out_dir: Path) -> tuple[Path, Path]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    table_map = _table_map(result)
    for file_name in TABLE_FILE_ORDER:
        table_map[file_name].to_csv(tables_dir / file_name, index=False)

    report_path = out_dir / "report.md"
    research_buckets = _build_research_implication_buckets(result)
    confirmed_anomaly_issues = _select_rows(result.anomaly_flags, {"duplicate", "conflict"})
    heuristic_anomaly_warnings = _select_rows(
        result.anomaly_flags,
        {"robust_zscore", "extreme_ratio"},
    )
    warning_leakage_flags = result.leakage_flags.loc[
        result.leakage_flags["risk_level"].isin(["warning", "review"])
    ].reset_index(drop=True)
    report_lines = [
        "# Raw Data Analysis Report",
        "",
        "## Dataset Overview",
        "",
        "### Machine-readable artifacts",
        "",
        "- `summary.json`: [summary.json](summary.json)",
        *[
            f"- `{file_name}`: [tables/{file_name}](tables/{file_name})"
            for file_name in TABLE_FILE_ORDER
        ],
        "",
        _frame_to_markdown(result.dataset_overview),
        "",
        "## Source Inventory",
        "",
        _frame_to_markdown(result.source_inventory),
        "",
        "## Feature Inventory",
        "",
        _frame_to_markdown(result.feature_inventory),
        "",
        "## Sparsity and Missingness",
        "",
        "Feature-level sparsity:",
        "",
        _frame_to_markdown(result.sparsity_by_feature),
        "",
        "Ticker-level sparsity:",
        "",
        _frame_to_markdown(result.sparsity_by_ticker),
        "",
        "## Temporal Behavior",
        "",
        "Temporal patterns:",
        "",
        _frame_to_markdown(result.temporal_patterns),
        "",
        "Stale value summary:",
        "",
        _frame_to_markdown(result.stale_value_summary),
        "",
        "## Outliers and Anomalies",
        "",
        "### Confirmed issues",
        "",
        _frame_to_markdown(confirmed_anomaly_issues),
        "",
        "### Heuristic warnings",
        "",
        _frame_to_markdown(heuristic_anomaly_warnings),
        "",
        "## Leakage Risk Review",
        "",
        "No confirmed leakage issues were proven from raw timestamps alone.",
        "",
        "### Heuristic leakage warnings",
        "",
        _frame_to_markdown(warning_leakage_flags),
        "",
        "## Research Implications",
        "",
        "### Likely usable now",
        "",
        _feature_list_markdown(research_buckets["Likely usable now"]),
        "",
        "### Usable only with safeguards",
        "",
        _feature_list_markdown(research_buckets["Usable only with safeguards"]),
        "",
        "### Unsafe until repaired or better timestamped",
        "",
        _feature_list_markdown(research_buckets["Unsafe until repaired or better timestamped"]),
        "",
        "### Too sparse to prioritize",
        "",
        _feature_list_markdown(research_buckets["Too sparse to prioritize"]),
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    overview_row = cast(pd.Series, result.dataset_overview.iloc[0])
    summary_path = out_dir / "summary.json"
    top_findings = _build_top_findings(
        confirmed_anomaly_issues=confirmed_anomaly_issues,
        heuristic_anomaly_warnings=heuristic_anomaly_warnings,
        leakage_flags=warning_leakage_flags,
        research_buckets=research_buckets,
    )
    summary = {
        "anomaly_flag_count": int(len(result.anomaly_flags)),
        "date_max": _timestamp_to_text(overview_row["date_max"]),
        "date_min": _timestamp_to_text(overview_row["date_min"]),
        "feature_count": int(cast(int, overview_row["unique_features"])),
        "issue_counts": {
            "confirmed_anomaly_issues": int(len(confirmed_anomaly_issues)),
            "heuristic_anomaly_warnings": int(len(heuristic_anomaly_warnings)),
            "heuristic_leakage_warnings": int(len(warning_leakage_flags)),
        },
        "leakage_flag_count": int(len(result.leakage_flags)),
        "report_path": _relative_artifact_path(report_path, out_dir),
        "research_implications": research_buckets,
        "row_count": int(cast(int, overview_row["row_count"])),
        "source_count": int(cast(int, overview_row["unique_sources"])),
        "table_paths": {
            file_name: _relative_artifact_path(tables_dir / file_name, out_dir)
            for file_name in TABLE_FILE_ORDER
        },
        "top_findings": top_findings,
        "ticker_count": int(cast(int, overview_row["unique_tickers"])),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path, summary_path
