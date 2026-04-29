from __future__ import annotations

from typing import cast

import pandas as pd

from skuld_research.raw_data_analysis.models import AnalysisDataset


def build_dataset_overview(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows
    equity_rows = rows.loc[rows["ticker"] != ""]
    numeric_parse_rate = cast(float, rows["numeric_value"].notna().mean())
    return pd.DataFrame(
        [
            {
                "row_count": len(rows),
                "date_count": rows["date"].nunique(),
                "date_min": rows["date"].min(),
                "date_max": rows["date"].max(),
                "unique_tickers": equity_rows["ticker"].nunique(),
                "unique_features": rows["feature"].nunique(),
                "unique_sources": rows["source_name"].nunique(),
                "numeric_parse_rate": float(numeric_parse_rate),
            }
        ]
    )


def build_source_inventory(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows
    inventory = (
        rows.groupby("source_name", dropna=False)
        .agg(
            row_count=("feature", "size"),
            date_count=("date", "nunique"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            ticker_count=("ticker", lambda values: values.replace("", pd.NA).dropna().nunique()),
            feature_count=("feature", "nunique"),
        )
        .reset_index()
    )
    inventory["dataset_row_share"] = inventory["row_count"] / len(rows)
    return (
        inventory.sort_values(["row_count", "source_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_feature_sparsity(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows
    scoped_rows = rows.assign(scope=rows["ticker"].where(rows["ticker"] != "", "macro"))
    scope_observation_counts = (
        scoped_rows.loc[:, ["scope", "date"]].drop_duplicates().groupby("scope").size()
    )
    equity_observation_count = int(
        scope_observation_counts.drop(index="macro", errors="ignore").sum()
    )
    macro_observation_count = int(scope_observation_counts.get("macro", 0))
    sparsity = (
        rows.groupby("feature", dropna=False)
        .agg(
            row_count=("feature", "size"),
            ticker_count=("ticker", lambda values: values.replace("", pd.NA).dropna().nunique()),
            date_count=("date", "nunique"),
            numeric_count=("numeric_value", lambda values: values.notna().sum()),
        )
        .reset_index()
    )
    feature_scope_counts = (
        scoped_rows.loc[:, ["feature", "scope", "date"]]
        .drop_duplicates()
        .groupby("feature")
        .agg(
            observed_scope_dates=("date", "size"),
            possible_scope_dates=(
                "scope",
                lambda scopes: int(
                    sum(scope_observation_counts.loc[scope] for scope in pd.Index(scopes).unique())
                ),
            ),
        )
    )
    feature_scope_flags = rows.groupby("feature", dropna=False).agg(
        has_equity=("ticker", lambda values: bool((values != "").any())),
        has_macro=("ticker", lambda values: bool((values == "").any())),
    )
    possible_scope_dates = pd.Series(
        feature_scope_flags["has_equity"].astype(int) * equity_observation_count
        + feature_scope_flags["has_macro"].astype(int) * macro_observation_count,
        index=feature_scope_flags.index,
    )
    possible_scope_dates_frame = possible_scope_dates.rename("possible_scope_dates").reset_index()
    possible_scope_dates_frame = possible_scope_dates_frame.rename(columns={"index": "feature"})
    sparsity = sparsity.merge(possible_scope_dates_frame, on="feature", how="left")
    sparsity["missing_fraction"] = 1.0 - (
        sparsity["feature"].map(feature_scope_counts["observed_scope_dates"]).astype(float)
        / sparsity["possible_scope_dates"].astype(float)
    )
    sparsity = sparsity.drop(columns=["possible_scope_dates"])
    return (
        sparsity.sort_values(["row_count", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_feature_inventory(dataset: AnalysisDataset) -> pd.DataFrame:
    inventory = (
        dataset.rows.groupby(["feature", "source_name"], dropna=False)
        .agg(
            row_count=("feature", "size"),
            ticker_count=("ticker", lambda values: values.replace("", pd.NA).dropna().nunique()),
            date_count=("date", "nunique"),
            numeric_parse_rate=("numeric_value", lambda values: float(values.notna().mean())),
        )
        .reset_index()
    )
    return (
        inventory.sort_values(
            ["row_count", "feature", "source_name"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    )


def build_ticker_sparsity(dataset: AnalysisDataset) -> pd.DataFrame:
    rows = dataset.rows.loc[dataset.rows["ticker"] != ""]
    sparsity = (
        rows.groupby("ticker", dropna=False)
        .agg(
            row_count=("feature", "size"),
            feature_count=("feature", "nunique"),
            date_count=("date", "nunique"),
        )
        .reset_index()
    )
    sparsity["row_share"] = sparsity["row_count"] / len(rows)
    return (
        sparsity.sort_values(["row_count", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )
