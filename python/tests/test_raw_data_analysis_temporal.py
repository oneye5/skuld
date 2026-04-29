from pathlib import Path


def test_infer_feature_temporal_patterns_labels_monthly_fundamentals(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,ANZ.NZ,annual_basic_average_shares,1000000,12\n"
        "1709251200000,ANZ.NZ,annual_basic_average_shares,1001000,12\n"
        "1709251200000,SPK.NZ,annual_basic_average_shares,2000000,12\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n12,yf_finances\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    patterns = build_temporal_patterns(dataset)
    shares_row = patterns.loc[patterns["feature"] == "annual_basic_average_shares"].iloc[0]

    assert shares_row["frequency_label"] in {"monthly_or_slower", "quarterly_or_slower"}
    assert shares_row["gap_count"] == 1


def test_temporal_patterns_reports_max_gap_days(raw_analysis_csv_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    dataset = load_analysis_dataset(raw_analysis_csv_path)
    patterns = build_temporal_patterns(dataset)

    assert "max_gap_days" in patterns.columns
    assert (patterns["max_gap_days"].dropna() >= 0).all()


def test_build_temporal_patterns_singleton_reports_nan_max_gap_days(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,,singleton_macro,100.5,10\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n10,nz_business_confidence\n",
        encoding="utf-8",
    )

    dataset = load_analysis_dataset(data_path)
    patterns = build_temporal_patterns(dataset)
    singleton_row = patterns.loc[patterns["feature"] == "singleton_macro"].iloc[0]

    assert singleton_row["frequency_label"] == "singleton"
    assert singleton_row["gap_count"] == 0
    assert singleton_row["max_gap_days"] != singleton_row["max_gap_days"]


def test_build_temporal_patterns_uses_per_series_intervals_not_feature_wide_union(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,monthly_feature,10,12\n"
        "1709337600000,AAA.NZ,monthly_feature,11,12\n"
        "1706745600000,BBB.NZ,monthly_feature,20,12\n"
        "1709424000000,BBB.NZ,monthly_feature,21,12\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n12,yf_finances\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    patterns = build_temporal_patterns(dataset)
    feature_row = patterns.loc[patterns["feature"] == "monthly_feature"].iloc[0]

    assert float(feature_row["median_gap_days"]) == 31.0
    assert feature_row["frequency_label"] == "monthly_or_slower"


def test_build_temporal_patterns_treats_same_calendar_day_timestamps_as_one_observation(
    tmp_path: Path,
):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1709251200000,AAA.NZ,intraday_feature,10,6\n"
        "1709294400000,AAA.NZ,intraday_feature,11,6\n"
        "1709337600000,AAA.NZ,intraday_feature,12,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    patterns = build_temporal_patterns(dataset)
    feature_row = patterns.loc[patterns["feature"] == "intraday_feature"].iloc[0]

    assert int(feature_row["observation_dates"]) == 2
    assert int(feature_row["gap_count"]) == 1
    assert float(feature_row["median_gap_days"]) == 1.0


def test_build_temporal_patterns_separates_same_feature_across_sources(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,shared_signal,10,6\n"
        "1707868800000,AAA.NZ,shared_signal,11,8\n"
        "1709337600000,AAA.NZ,shared_signal,12,6\n"
        "1710547200000,AAA.NZ,shared_signal,13,8\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n6,yf_prices\n8,wikimedia_pageviews\n",
        encoding="utf-8",
    )

    dataset = load_analysis_dataset(data_path)
    patterns = build_temporal_patterns(dataset)
    feature_row = patterns.loc[patterns["feature"] == "shared_signal"].iloc[0]

    assert int(feature_row["series_count"]) == 2
    assert float(feature_row["median_gap_days"]) == 31.0


def test_build_temporal_patterns_includes_gap_distribution_and_irregularity_columns(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,irregular_feature,1,6\n"
        "1706745600000,AAA.NZ,irregular_feature,2,6\n"
        "1707436800000,AAA.NZ,irregular_feature,3,6\n"
        "1709251200000,AAA.NZ,irregular_feature,4,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    patterns = build_temporal_patterns(dataset)
    feature_row = patterns.loc[patterns["feature"] == "irregular_feature"].iloc[0]

    assert set(patterns.columns) >= {
        "series_count",
        "gap_count",
        "gap_p90_days",
        "irregularity_ratio",
    }
    assert int(feature_row["series_count"]) == 1
    assert int(feature_row["gap_count"]) == 3
    assert float(feature_row["gap_p90_days"]) >= float(feature_row["median_gap_days"])
    assert float(feature_row["irregularity_ratio"]) > 0.0


def test_build_temporal_patterns_returns_empty_frame_with_stable_schema(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_temporal_patterns

    data_path = tmp_path / "data_long.csv"
    data_path.write_text("timestamp,ticker,feature,value,src\n", encoding="utf-8")
    (tmp_path / "source_legend.csv").write_text("id,name\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    patterns = build_temporal_patterns(dataset)

    assert list(patterns.columns) == [
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
    assert patterns.empty


def test_build_stale_value_summary_flags_repeated_values(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_stale_value_summary

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,adj_close,10,6\n"
        "1706745600000,AAA.NZ,adj_close,10,6\n"
        "1706832000000,AAA.NZ,adj_close,10,6\n"
        "1706918400000,AAA.NZ,adj_close,11,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    stale = build_stale_value_summary(dataset)
    stale_row = stale.loc[(stale["ticker"] == "AAA.NZ") & (stale["feature"] == "adj_close")].iloc[0]

    assert "max_repeat_run" in stale.columns
    assert "max_repeat_span_days" in stale.columns
    assert int(stale_row["max_repeat_run"]) == 3
    assert float(stale_row["max_repeat_span_days"]) >= 2.0


def test_build_stale_value_summary_includes_blank_ticker_macro_series(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_stale_value_summary

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,,oecd_bcicp,100.5,10\n"
        "1709251200000,,oecd_bcicp,100.5,10\n"
        "1711929600000,,oecd_bcicp,101.0,10\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n10,nz_business_confidence\n",
        encoding="utf-8",
    )

    dataset = load_analysis_dataset(data_path)
    stale = build_stale_value_summary(dataset)
    stale_row = stale.loc[(stale["ticker"] == "") & (stale["feature"] == "oecd_bcicp")].iloc[0]

    assert int(stale_row["max_repeat_run"]) == 2
    assert float(stale_row["max_repeat_span_days"]) == 30.0


def test_build_stale_value_summary_uses_distinct_observation_dates(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_stale_value_summary

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,adj_close,10,6\n"
        "1706659200000,AAA.NZ,adj_close,10,6\n"
        "1706745600000,AAA.NZ,adj_close,10,6\n"
        "1706832000000,AAA.NZ,adj_close,11,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    stale = build_stale_value_summary(dataset)
    stale_row = stale.loc[(stale["ticker"] == "AAA.NZ") & (stale["feature"] == "adj_close")].iloc[0]

    assert int(stale_row["max_repeat_run"]) == 2
    assert float(stale_row["max_repeat_span_days"]) == 1.0


def test_build_stale_value_summary_collapses_same_calendar_day_rows(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_stale_value_summary

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1709251200000,AAA.NZ,adj_close,10,6\n"
        "1709294400000,AAA.NZ,adj_close,10,6\n"
        "1709337600000,AAA.NZ,adj_close,10,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    stale = build_stale_value_summary(dataset)
    stale_row = stale.loc[(stale["ticker"] == "AAA.NZ") & (stale["feature"] == "adj_close")].iloc[0]

    assert int(stale_row["max_repeat_run"]) == 2
    assert float(stale_row["max_repeat_span_days"]) == 1.0


def test_build_stale_value_summary_treats_conflicting_same_date_duplicates_as_break(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_stale_value_summary

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,AAA.NZ,adj_close,10,6\n"
        "1706745600000,AAA.NZ,adj_close,10,6\n"
        "1706745600000,AAA.NZ,adj_close,11,6\n"
        "1706832000000,AAA.NZ,adj_close,10,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text("id,name\n6,yf_prices\n", encoding="utf-8")

    dataset = load_analysis_dataset(data_path)
    stale = build_stale_value_summary(dataset)
    stale_row = stale.loc[(stale["ticker"] == "AAA.NZ") & (stale["feature"] == "adj_close")].iloc[0]

    assert int(stale_row["max_repeat_run"]) == 1
    assert float(stale_row["max_repeat_span_days"]) == 0.0


def test_build_stale_value_summary_returns_empty_frame_with_stable_schema(tmp_path: Path):
    from skuld_research.raw_data_analysis.dataset import load_analysis_dataset
    from skuld_research.raw_data_analysis.temporal import build_stale_value_summary

    data_path = tmp_path / "data_long.csv"
    data_path.write_text(
        "timestamp,ticker,feature,value,src\n"
        "1706659200000,,macro_value,not_numeric,10\n"
        "1706745600000,AAA.NZ,text_feature,not_numeric,6\n",
        encoding="utf-8",
    )
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n6,yf_prices\n10,nz_business_confidence\n",
        encoding="utf-8",
    )

    dataset = load_analysis_dataset(data_path)
    stale = build_stale_value_summary(dataset)

    assert list(stale.columns) == ["ticker", "feature", "max_repeat_run", "max_repeat_span_days"]
    assert stale.empty
