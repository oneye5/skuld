from pathlib import Path

import pandas as pd
import pytest


def test_load_analysis_dataset_maps_source_names(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)

    assert dataset.rows.loc[0, "src"] == 6
    assert dataset.rows.loc[0, "source_name"] == "yf_prices"


def test_load_analysis_dataset_parses_timestamp_to_utc_naive(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)

    assert dataset.rows["date"].dtype == "datetime64[ns]"
    assert dataset.rows["date"].min() == pd.Timestamp("2024-01-31")


def test_load_analysis_dataset_preserves_raw_value_and_numeric_value(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    row = dataset.rows.loc[dataset.rows["feature"] == "annual_basic_average_shares"].iloc[0]

    assert row["raw_value"] == "1000000"
    assert row["numeric_value"] == 1_000_000.0


def test_load_analysis_dataset_raises_for_missing_source_mapping(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n1706659200000,ABC,close,12.5,99\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing source legend mappings for src ids: 99"):
        load_analysis_dataset(data_path)


def test_load_analysis_dataset_raises_for_duplicate_legend_ids(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n1706659200000,ABC,close,12.5,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n6,yf_prices\n6,duplicate_name\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate source legend ids: 6"):
        load_analysis_dataset(data_path)


def test_load_analysis_dataset_coerces_non_numeric_value_to_nan(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n1706659200000,ABC,label,not_a_number,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    row = dataset.rows.iloc[0]

    assert row["raw_value"] == "not_a_number"
    assert pd.isna(row["numeric_value"])


def test_load_analysis_dataset_preserves_na_like_raw_tokens(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n1706659200000,ABC,label,NA,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    row = dataset.rows.iloc[0]

    assert row["raw_value"] == "NA"
    assert pd.isna(row["numeric_value"])


def test_load_analysis_dataset_preserves_na_like_legend_names(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n1706659200000,ABC,label,12.5,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,NA\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)

    assert dataset.rows.loc[0, "source_name"] == "NA"
    assert dataset.source_legend.loc[0, "name"] == "NA"
