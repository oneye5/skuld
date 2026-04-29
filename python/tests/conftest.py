"""Shared test fixtures for Skuld.

All fixtures produce synthetic data — no real CSV files needed.
Timestamps are chosen to exercise edge cases (future data, gaps, negatives).
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

# -- Synthetic CSV content -----------------------------------------------

# asof will be 2025-01-15 00:00 UTC = 1736899200000 ms
ASOF_MS = 1736899200000  # 2025-01-15T00:00:00Z
ASOF_TS = pd.Timestamp("2025-01-15", tz="UTC")

# 2025-01-13 = 1736726400000, 2025-01-14 = 1736812800000
DAY_13_MS = 1736726400000
DAY_14_MS = 1736812800000
# 2025-01-15 = 1736899200000 (== asof, should be excluded by PIT loader)
DAY_15_MS = 1736899200000
# 2024-06-30 = 1719705600000 (fundamental period-end)
FUND_MS = 1719705600000

SYNTHETIC_CSV = f"""\
timestamp,ticker,feature,value,src
{DAY_13_MS},ANZ.NZ,adj_close,50.0,6
{DAY_13_MS},ANZ.NZ,volume,100000,6
{DAY_14_MS},ANZ.NZ,adj_close,51.0,6
{DAY_14_MS},ANZ.NZ,volume,120000,6
{DAY_13_MS},SPK.NZ,adj_close,4.80,6
{DAY_13_MS},SPK.NZ,volume,200000,6
{DAY_14_MS},SPK.NZ,adj_close,4.85,6
{DAY_14_MS},SPK.NZ,volume,210000,6
{DAY_15_MS},ANZ.NZ,adj_close,52.0,6
{DAY_15_MS},ANZ.NZ,volume,130000,6
{DAY_13_MS},ANZ.NZ,adj_close,-5.0,6
{FUND_MS},ANZ.NZ,annual_net_income_common_stockholders,1000000,12
{DAY_13_MS},,oecd_bcicp,100.5,10
{DAY_14_MS},ANZ.NZ,dividend,0.50,6
{DAY_13_MS},SPK.NZ,split,2.0,6
"""

# Minimal source legend covering the IDs used in SYNTHETIC_CSV (6, 10, 12).
# Mirrors the schema of the real data/source_legend.csv.
SYNTHETIC_LEGEND = """\
id,name
6,yf_prices
10,nz_business_confidence
12,yf_finances
"""

RAW_ANALYSIS_CSV = """\
timestamp,ticker,feature,value,src
1706659200000,ANZ.NZ,adj_close,50.0,6
1706745600000,ANZ.NZ,adj_close,51.0,6
1706745600000,ANZ.NZ,adj_close,51.0,6
1706832000000,ANZ.NZ,volume,120000,6
1706659200000,SPK.NZ,adj_close,4.80,6
1709251200000,ANZ.NZ,annual_basic_average_shares,1000000,12
1709251200000,SPK.NZ,annual_basic_average_shares,0,12
1709251200000,,oecd_bcicp,100.5,10
1711929600000,ANZ.NZ,page_views,9999999,8
1714521600000,ANZ.NZ,page_views,10,8
"""


@pytest.fixture
def synthetic_csv_path(tmp_path: Path) -> Path:
    """Write synthetic CSV (and sibling legend) to a temp dir, return CSV path."""
    csv_file = tmp_path / "data_long.csv"
    csv_file.write_text(SYNTHETIC_CSV)
    (tmp_path / "source_legend.csv").write_text(SYNTHETIC_LEGEND)
    return csv_file


@pytest.fixture
def raw_analysis_csv_path(tmp_path: Path) -> Path:
    csv_file = tmp_path / "data_long.csv"
    csv_file.write_text(RAW_ANALYSIS_CSV, encoding="utf-8")
    (tmp_path / "source_legend.csv").write_text(
        "id,name\n6,yf_prices\n8,wikimedia_pageviews\n10,nz_business_confidence\n12,yf_finances\n",
        encoding="utf-8",
    )
    return csv_file


@pytest.fixture
def synthetic_raw(synthetic_csv_path: Path):
    """Loaded RawData for the synthetic CSV."""
    from skuld_research.data.csv_loader import load_raw_csv

    return load_raw_csv(synthetic_csv_path)


@pytest.fixture
def synthetic_csv_io() -> io.StringIO:
    """Return synthetic CSV as a StringIO for in-memory tests."""
    return io.StringIO(SYNTHETIC_CSV)


@pytest.fixture
def asof_timestamp() -> pd.Timestamp:
    return ASOF_TS
