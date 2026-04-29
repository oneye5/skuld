from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AnalysisDataset:
    rows: pd.DataFrame
    source_legend: pd.DataFrame


@dataclass(frozen=True)
class RawDataAnalysisResult:
    dataset_overview: pd.DataFrame
    source_inventory: pd.DataFrame
    feature_inventory: pd.DataFrame
    sparsity_by_feature: pd.DataFrame
    sparsity_by_ticker: pd.DataFrame
    temporal_patterns: pd.DataFrame
    stale_value_summary: pd.DataFrame
    anomaly_flags: pd.DataFrame
    leakage_flags: pd.DataFrame
