from pathlib import Path

import pytest


def test_build_dataset_overview_reports_core_counts(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.coverage import build_dataset_overview
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    overview = build_dataset_overview(dataset)

    assert int(overview.loc[0, "row_count"]) == len(dataset.rows)
    assert int(overview.loc[0, "date_count"]) == 6
    assert int(overview.loc[0, "unique_tickers"]) == 2
    assert int(overview.loc[0, "unique_sources"]) == 4
    assert int(overview.loc[0, "unique_features"]) >= 4
    assert 0.0 <= float(overview.loc[0, "numeric_parse_rate"]) <= 1.0


def test_build_source_inventory_uses_source_names(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.coverage import build_source_inventory
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    inventory = build_source_inventory(dataset)
    prices_row = inventory.loc[inventory["source_name"] == "yf_prices"].iloc[0]

    assert set(inventory["source_name"]) >= {"yf_prices", "yf_finances"}
    assert "dataset_row_share" in inventory.columns
    assert int(prices_row["row_count"]) == 5
    assert int(prices_row["date_count"]) == 3
    assert int(prices_row["ticker_count"]) == 2
    assert float(prices_row["dataset_row_share"]) == 0.5


def test_build_feature_sparsity_includes_missing_fraction(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.coverage import build_feature_sparsity
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    sparsity = build_feature_sparsity(dataset)
    shares_row = sparsity.loc[sparsity["feature"] == "annual_basic_average_shares"].iloc[0]

    assert "missing_fraction" in sparsity.columns
    assert (sparsity["missing_fraction"] >= 0.0).all()
    assert (sparsity["missing_fraction"] <= 1.0).all()
    assert float(shares_row["missing_fraction"]) == 0.75


def test_build_feature_sparsity_handles_mixed_macro_and_equity_scope(tmp_path: Path):
    from skuld_research.raw_data_analysis.coverage import build_feature_sparsity
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,mixed_feature,1,6\n"
        "1706745600000,,mixed_feature,2,10\n"
        "1706745600000,BBB.NZ,other_feature,3,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n6,yf_prices\n10,nz_business_confidence\n",
        encoding="utf-8",
    )

    dataset = load_analysis_dataset(data_path)
    sparsity = build_feature_sparsity(dataset)
    mixed_row = sparsity.loc[sparsity["feature"] == "mixed_feature"].iloc[0]

    assert 0.0 <= float(mixed_row["missing_fraction"]) <= 1.0
    assert float(mixed_row["missing_fraction"]) == pytest.approx(1.0 / 3.0)


def test_build_feature_sparsity_counts_missing_absent_tickers(tmp_path: Path):
    from skuld_research.raw_data_analysis.coverage import build_feature_sparsity
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,partial_feature,1,6\n"
        "1706745600000,AAA.NZ,partial_feature,2,6\n"
        "1706659200000,BBB.NZ,other_feature,3,6\n"
        "1706745600000,BBB.NZ,other_feature,4,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    sparsity = build_feature_sparsity(dataset)
    partial_row = sparsity.loc[sparsity["feature"] == "partial_feature"].iloc[0]

    assert float(partial_row["missing_fraction"]) == 0.5


def test_build_feature_inventory_groups_by_feature_and_source_name(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.coverage import build_feature_inventory
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    inventory = build_feature_inventory(dataset)

    page_views = inventory.loc[
        (inventory["feature"] == "page_views")
        & (inventory["source_name"] == "wikimedia_pageviews")
    ].iloc[0]

    assert page_views["row_count"] == 2
    assert 0.0 <= float(page_views["numeric_parse_rate"]) <= 1.0


def test_build_feature_inventory_keeps_same_feature_separate_by_source_name(tmp_path: Path):
    from skuld_research.raw_data_analysis.coverage import build_feature_inventory
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,shared_feature,1,6\n"
        "1706745600000,AAA.NZ,shared_feature,2,10\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n6,yf_prices\n10,nz_business_confidence\n",
        encoding="utf-8",
    )

    dataset = load_analysis_dataset(data_path)
    inventory = build_feature_inventory(dataset)
    shared_rows = inventory.loc[inventory["feature"] == "shared_feature"]

    assert len(shared_rows) == 2
    assert set(shared_rows["source_name"]) == {"yf_prices", "nz_business_confidence"}
    assert set(shared_rows["row_count"]) == {1}


def test_build_ticker_sparsity_excludes_empty_macro_ticker(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.coverage import build_ticker_sparsity
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    ticker_sparsity = build_ticker_sparsity(dataset)

    assert "" not in set(ticker_sparsity["ticker"])
    assert set(ticker_sparsity["ticker"]) >= {"ANZ.NZ", "SPK.NZ"}
