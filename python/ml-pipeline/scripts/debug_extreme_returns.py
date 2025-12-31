"""Debug extreme daily returns in the backtest."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# Load true daily returns
df = pd.read_csv('output/runs/ranking_20251231_125012_41ae8f93/true_daily_returns.csv')

# Find extreme returns
extreme = df[df['return'].abs() > 0.5].copy()
print(f"Extreme daily returns (|r| > 50%): {len(extreme)}")

# Load the predictions to check what tickers were held
predictions = pd.read_csv('output/runs/ranking_20251231_125012_41ae8f93/predictions.csv')
print(f"\nPredictions: {len(predictions)} rows")
print(f"Columns: {predictions.columns.tolist()}")

# Load price data
from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers

print("\nLoading price data...")
long_df = load_long_data()
wide_df = long_to_wide(add_macro_prefix(clean_and_classify_tickers(long_df)))
print(f"Wide data: {len(wide_df)} rows")

# Check one of the extreme timestamps
if len(extreme) > 0:
    bad_ts = extreme.iloc[0]['timestamp']
    bad_return = extreme.iloc[0]['return']
    print(f"\n=== Investigating timestamp {bad_ts} with return {bad_return:.2%} ===")
    
    # Convert to datetime for human readability
    from datetime import datetime
    dt = datetime.fromtimestamp(bad_ts / 1000)
    print(f"Date: {dt}")
    
    # Find the previous timestamp
    all_ts = sorted(df['timestamp'].unique())
    idx = all_ts.index(bad_ts)
    if idx > 0:
        prev_ts = all_ts[idx - 1]
        print(f"Previous timestamp: {datetime.fromtimestamp(prev_ts / 1000)}")
        
        # Check prices at these timestamps
        curr_prices = wide_df[wide_df['timestamp'] == bad_ts][['timestamp', 'ticker', 'Close']]
        prev_prices = wide_df[wide_df['timestamp'] == prev_ts][['timestamp', 'ticker', 'Close']]
        
        print(f"\nStocks at current timestamp: {len(curr_prices)}")
        print(f"Stocks at previous timestamp: {len(prev_prices)}")
        
        # Check for stocks that exist in both
        curr_tickers = set(curr_prices['ticker'])
        prev_tickers = set(prev_prices['ticker'])
        common = curr_tickers & prev_tickers
        print(f"Common tickers: {len(common)}")
        
        # Check for price changes
        merged = prev_prices.merge(curr_prices, on='ticker', suffixes=('_prev', '_curr'))
        merged['return'] = (merged['Close_curr'] - merged['Close_prev']) / merged['Close_prev']
        print(f"\nMerged (stocks in both): {len(merged)}")
        print(f"Return stats:")
        print(merged['return'].describe())
        
        # Check for extreme individual stock returns
        extreme_stocks = merged[merged['return'].abs() > 0.5]
        if len(extreme_stocks) > 0:
            print(f"\nExtreme individual stock returns:")
            print(extreme_stocks)
