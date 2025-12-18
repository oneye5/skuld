"""Debug script to check data and windows."""

import pandas as pd
from utils.data_loader import load_long_data
from runnables.pipeline import prepare_wide_data
from runnables.rolling_window_runner import calculate_window_timestamps
from config.column_names import TIMESTAMP, TICKER, CLOSE
from config.model_config import MS_PER_DAY

print("Loading data...")
df = load_long_data()
print(f"Loaded {len(df):,} rows")

print("\nConverting to wide format...")
wide_df = prepare_wide_data(df)
print(f"Wide format: {len(wide_df):,} rows, {len(wide_df.columns)} columns")

max_ts = int(wide_df[TIMESTAMP].max())
min_ts = int(wide_df[TIMESTAMP].min())
print(f"\nTimestamp range:")
print(f"  Min: {min_ts} ({pd.to_datetime(min_ts, unit='ms')})")
print(f"  Max: {max_ts} ({pd.to_datetime(max_ts, unit='ms')})")

print("\nCalculating window timestamps...")
windows = calculate_window_timestamps(max_ts)
for i, (train_end, test_end) in enumerate(windows):
    print(f"\nWindow {i}:")
    print(f"  Train end: {train_end} ({pd.to_datetime(train_end, unit='ms')})")
    print(f"  Test end:  {test_end} ({pd.to_datetime(test_end, unit='ms')})")
    
    # Check data availability
    train_count = len(wide_df[wide_df[TIMESTAMP] < train_end])
    test_count = len(wide_df[(wide_df[TIMESTAMP] >= train_end) & (wide_df[TIMESTAMP] < test_end)])
    print(f"  Train rows: {train_count:,}")
    print(f"  Test rows: {test_count:,}")
