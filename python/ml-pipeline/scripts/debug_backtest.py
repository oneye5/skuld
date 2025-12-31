"""Debug script to investigate the zero drawdown issue."""

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = Path(__file__).parent.parent / "output/runs/ranking_20251230_224601_a8eb71e7"

print("=" * 70)
print("BACKTEST DATA INVESTIGATION")
print("=" * 70)

# Load the daily returns
daily_returns = pd.read_csv(run_dir / "daily_returns.csv")
daily_returns_pre_fee = pd.read_csv(run_dir / "daily_returns_pre_fee.csv")

print(f"\nDaily returns shape: {daily_returns.shape}")
print(daily_returns.head(10))

print(f"\nNumber of return observations: {len(daily_returns)}")
print(f"Min return: {daily_returns['return'].min():.4f}")
print(f"Max return: {daily_returns['return'].max():.4f}")
print(f"Mean return: {daily_returns['return'].mean():.4f}")

# Check: are returns always positive?
positive_returns = (daily_returns['return'] > 0).sum()
negative_returns = (daily_returns['return'] < 0).sum()
print(f"\nPositive returns: {positive_returns}")
print(f"Negative returns: {negative_returns}")
print(f"Zero returns: {(daily_returns['return'] == 0).sum()}")

# Compute cumulative returns
cum_returns = (1 + daily_returns['return']).cumprod() - 1
print(f"\nCumulative returns:")
print(cum_returns)

# Compute max drawdown properly
wealth = 1 + cum_returns
running_max = wealth.cummax()
drawdowns = (running_max - wealth) / running_max
max_dd = drawdowns.max()

print(f"\nMax drawdown (computed): {max_dd:.4f}")
print(f"Drawdown series:")
print(drawdowns)

# The issue: with only 6 data points and all positive returns, there's no drawdown
print("\n" + "=" * 70)
print("THE PROBLEM")
print("=" * 70)
print(f"""
With only {len(daily_returns)} return observations:
- If returns are mostly positive, cumulative wealth always increases
- Max drawdown = 0 because we never dip below previous high

This happens because:
1. With 365-day forward returns, we sample timestamps every 365 days
2. With 80 windows each producing ~25 test timestamps, but sampled every 365 days...
3. We only get ~{len(daily_returns)} non-overlapping periods

The Sharpe ratio is computed correctly for these {len(daily_returns)} periods,
but the drawdown statistic is meaningless with so few observations.
""")

# Check turnover
turnover = pd.read_csv(run_dir / "turnover.csv")
print("\n" + "=" * 70)
print("TURNOVER DATA")
print("=" * 70)
print(turnover)
print(f"\nAverage turnover: {turnover['turnover'].mean():.4f}")

# Load config to understand the sampling
import json
with open(run_dir / "config.json") as f:
    config = json.load(f)

print("\n" + "=" * 70)
print("CONFIG")
print("=" * 70)
print(f"forward_return_days: {config['forward_return_days']}")
print(f"num_windows: {config['num_windows']}")
print(f"windows_completed: {config['windows_completed']}")

# The portfolio_simulator.py samples every forward_return_days to avoid overlap
print(f"""
EXPLANATION:
- Forward returns = {config['forward_return_days']} days
- Portfolio simulator samples every {config['forward_return_days']} timestamps
- This ensures non-overlapping return periods
- But it means we only get {len(daily_returns)} rebalance points over 20 years!
""")

# Check the predictions file to understand temporal coverage
predictions = pd.read_csv(run_dir / "predictions.csv")
timestamps = sorted(predictions['timestamp'].unique())
print(f"\nPredictions cover {len(timestamps)} timestamps")

# Convert to dates
MS_PER_DAY = 86400000
first_date = pd.to_datetime(timestamps[0], unit='ms')
last_date = pd.to_datetime(timestamps[-1], unit='ms')
print(f"Date range: {first_date.date()} to {last_date.date()}")

# How many 365-day periods fit?
total_days = (timestamps[-1] - timestamps[0]) / MS_PER_DAY
n_periods = total_days / 365
print(f"Total span: {total_days:.0f} days = {n_periods:.1f} x 365-day periods")
print(f"Actual backtest periods: {len(daily_returns)}")
