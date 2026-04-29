from __future__ import annotations

from pathlib import Path

import pandas as pd

from skuld_research.raw_data_analysis.models import AnalysisDataset


def load_analysis_dataset(data_path: Path) -> AnalysisDataset:
    legend_path = data_path.with_name("source_legend.csv")
    rows = pd.read_csv(
        data_path,
        dtype={"timestamp": "int64", "ticker": str, "feature": str, "value": str, "src": "int64"},
        keep_default_na=False,
    )
    legend = pd.read_csv(legend_path, dtype={"id": "int64", "name": str}, keep_default_na=False)
    duplicate_ids = legend.loc[legend["id"].duplicated(), "id"].tolist()
    if duplicate_ids:
        duplicate_id_text = ", ".join(str(source_id) for source_id in duplicate_ids)
        raise ValueError(f"Duplicate source legend ids: {duplicate_id_text}")
    rows["ticker"] = rows["ticker"].fillna("")
    rows["raw_value"] = rows["value"]
    rows["numeric_value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows["date"] = (
        pd.to_datetime(rows["timestamp"], unit="ms", utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    rows = rows.merge(
        legend.rename(columns={"id": "src", "name": "source_name"}),
        on="src",
        how="left",
    )
    missing_source_ids = rows.loc[rows["source_name"].isna(), "src"].drop_duplicates().tolist()
    if missing_source_ids:
        missing_source_text = ", ".join(str(source_id) for source_id in missing_source_ids)
        raise ValueError(f"Missing source legend mappings for src ids: {missing_source_text}")
    return AnalysisDataset(rows=rows, source_legend=legend)
