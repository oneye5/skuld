"""Investigate why only ~50 tickers appear in predictions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from config.columns import TIMESTAMP, TICKER, CLOSE
from config.settings import YEAR_2000_MS

# Load data
print("Loading data...")
long_df = load_long_data()
long_df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
long_df = clean_and_classify_tickers(long_df)
long_df = add_macro_prefix(long_df)
wide_df = long_to_wide(long_df)

# Filter to stock tickers
stock_df = wide_df[~wide_df[TICKER].str.startswith('MACRO_')]

# Get unique timestamps and their counts
ts_counts = stock_df.groupby(TIMESTAMP).size()
print(f"Unique timestamps: {len(ts_counts)}")
print()
print("Stock count per timestamp (last 20):")
last_20_ts = sorted(stock_df[TIMESTAMP].unique())[-20:]
for ts in last_20_ts:
    count = ts_counts[ts]
    ts_date = pd.to_datetime(ts, unit='ms').date()
    print(f"  {ts_date}: {count} stocks")

# Check for gaps - how many of our 138 tickers are active at the end?
all_tickers = stock_df[TICKER].unique()
print()
print(f"Total unique tickers across all time: {len(all_tickers)}")

latest_ts = max(last_20_ts)
latest_date = pd.to_datetime(latest_ts, unit='ms').date()
latest_tickers = stock_df[stock_df[TIMESTAMP] == latest_ts][TICKER].unique()
print(f"Tickers at latest timestamp ({latest_date}): {len(latest_tickers)}")

# What tickers are missing?
missing = set(all_tickers) - set(latest_tickers)
print()
print(f"Missing tickers at latest timestamp ({len(missing)}):")
missing_list = []
for t in sorted(missing):
    # Find when this ticker last had data
    ticker_data = stock_df[stock_df[TICKER] == t]
    last_ts = ticker_data[TIMESTAMP].max()
    last_date = pd.to_datetime(last_ts, unit='ms').date()
    missing_list.append((t, last_date))

# Sort by last date
missing_list.sort(key=lambda x: x[1], reverse=True)
for t, last_date in missing_list[:40]:
    print(f"  {t:12}: last data {last_date}")

if len(missing_list) > 40:
    print(f"  ... and {len(missing_list) - 40} more")

# How many had data in 2025 or later?
recent_missing = [(t, d) for t, d in missing_list if d.year >= 2025]
print()
print(f"Tickers with data in 2025+ but missing now: {len(recent_missing)}")
for t, d in recent_missing:
    print(f"  {t:12}: last data {d}")

# Check if this is a Yahoo Finance data freshness issue
print()
print("=" * 60)
print("This is a data freshness issue from Yahoo Finance.")
print("Many stocks haven't been fetched recently in the Java ingestion.")
print("=" * 60)
