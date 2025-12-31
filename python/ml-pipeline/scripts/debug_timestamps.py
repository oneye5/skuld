"""Understand the timestamp sampling issue."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.portfolio_simulator import run_portfolio_backtest, PortfolioConfig
from config.columns import TIMESTAMP, TICKER

run_dir = Path(__file__).parent.parent / "output/runs/ranking_20251230_224601_a8eb71e7"

predictions = pd.read_csv(run_dir / "predictions.csv")
timestamps = sorted(predictions['timestamp'].unique())

MS_PER_DAY = 86400000

print("=" * 70)
print("TIMESTAMP ANALYSIS")
print("=" * 70)

print(f"Total unique timestamps: {len(timestamps)}")

# Check spacing between consecutive timestamps
spacings = [(timestamps[i+1] - timestamps[i]) / MS_PER_DAY for i in range(len(timestamps)-1)]
print(f"\nTimestamp spacing (days):")
print(f"  Mean: {np.mean(spacings):.1f}")
print(f"  Min:  {np.min(spacings):.1f}")
print(f"  Max:  {np.max(spacings):.1f}")
print(f"  Median: {np.median(spacings):.1f}")

# OLD (buggy) way: every 365th INDEX
old_sampled_ts = timestamps[::365]  
print(f"\nOLD sampling (every 365th index): {len(old_sampled_ts)} periods")

# NEW (correct) way: every ~365 days
new_sample = []
last_ts = timestamps[0]
new_sample.append(last_ts)
horizon_ms = 365 * MS_PER_DAY
for ts in timestamps[1:]:
    if (ts - last_ts) >= horizon_ms:
        new_sample.append(ts)
        last_ts = ts

print(f"NEW sampling (every ~365 days): {len(new_sample)} periods")

# Now test with actual backtest
print("\n" + "=" * 70)
print("TESTING FIXED BACKTEST")
print("=" * 70)

config = PortfolioConfig(top_n=10, bottom_n=0, long_only=True)
result = run_portfolio_backtest(
    predictions.rename(columns={'timestamp': TIMESTAMP, 'ticker': TICKER}),
    config,
    score_col='predicted_score',
    return_col='actual_return',
    return_horizon_days=365,
)

print(f"\nBacktest results with FIXED sampling:")
print(f"  Number of periods: {result.num_rebalances}")
print(f"  Total return: {result.total_return:.2%}")
print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
print(f"  Max drawdown: {result.max_drawdown:.2%}")

# Show the returns
print(f"\nPeriod returns:")
for ts, ret in zip(result.daily_returns.index, result.daily_returns.values):
    date = pd.to_datetime(ts, unit='ms').strftime('%Y-%m-%d')
    print(f"  {date}: {ret:+.2%}")
