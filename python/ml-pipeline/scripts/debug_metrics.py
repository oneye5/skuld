"""Debug the broken metrics (25867% drawdown, 3130% volatility)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# Check the most recent run output
output_dir = Path(__file__).resolve().parent.parent / "output" / "runs"
runs = sorted(output_dir.glob("ranking_*"))
if runs:
    latest = runs[-1]
    print(f"Latest run: {latest.name}")
    
    # Load predictions if available
    pred_file = latest / "predictions.csv"
    if pred_file.exists():
        df = pd.read_csv(pred_file)
        print(f"\nLoaded {len(df)} predictions")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check actual returns distribution
        if 'actual_return' in df.columns:
            returns = df['actual_return']
            print(f"\n=== Actual Returns Distribution ===")
            print(f"Min: {returns.min():.2%}")
            print(f"Max: {returns.max():.2%}")
            print(f"Mean: {returns.mean():.2%}")
            print(f"Std: {returns.std():.2%}")
            print(f"Count: {len(returns)}")
            
            # Check for extreme values
            extreme = df[returns.abs() > 2.0]  # More than 200% return
            print(f"\nExtreme returns (|r| > 200%): {len(extreme)}")
            if len(extreme) > 0:
                print(extreme[['timestamp', 'ticker', 'actual_return']].head(10))
        
        # Simulate the backtest calculation manually
        print("\n=== Manual Backtest Simulation ===")
        
        timestamps = sorted(df['timestamp'].unique())
        print(f"Total timestamps: {len(timestamps)}")
        
        # Sample timestamps like the backtest does
        return_horizon_days = 365
        MS_PER_DAY = 86_400_000
        
        first_ts = timestamps[0]
        rebalance_timestamps = [first_ts]
        last_ts = first_ts
        
        for ts in timestamps[1:]:
            days_since_last = (ts - last_ts) / MS_PER_DAY
            if days_since_last >= return_horizon_days:
                rebalance_timestamps.append(ts)
                last_ts = ts
        
        print(f"Rebalance timestamps: {len(rebalance_timestamps)}")
        
        # Compute period returns for each rebalance
        period_returns = []
        for ts in rebalance_timestamps:
            ts_df = df[df['timestamp'] == ts].sort_values('predicted_score', ascending=False)
            top_10 = ts_df.head(10)
            if len(top_10) > 0:
                avg_ret = top_10['actual_return'].mean()
                period_returns.append(avg_ret)
                print(f"  ts={ts}: top-10 avg return = {avg_ret:.2%}")
        
        returns_array = np.array(period_returns)
        print(f"\n=== Period Returns Stats ===")
        print(f"Count: {len(returns_array)}")
        print(f"Min: {returns_array.min():.2%}")
        print(f"Max: {returns_array.max():.2%}")
        print(f"Mean: {returns_array.mean():.2%}")
        print(f"Std: {returns_array.std():.2%}")
        
        # Compute cumulative returns
        cumulative = np.cumprod(1 + returns_array) - 1
        print(f"\n=== Cumulative Returns ===")
        print(f"Final: {cumulative[-1]:.2%}")
        
        # Compute max drawdown
        wealth = 1 + cumulative
        running_max = np.maximum.accumulate(wealth)
        drawdown = (running_max - wealth) / running_max
        max_dd = drawdown.max()
        print(f"\n=== Drawdown Calculation ===")
        print(f"Wealth series: {wealth}")
        print(f"Running max: {running_max}")
        print(f"Drawdowns: {drawdown}")
        print(f"Max drawdown: {max_dd:.2%}")
        
        # Compute volatility
        # Check time span
        first_ts = rebalance_timestamps[0] if rebalance_timestamps else 0
        last_ts = rebalance_timestamps[-1] if rebalance_timestamps else 0
        total_days = (last_ts - first_ts) / MS_PER_DAY
        total_years = max(total_days / 365.0, 0.01)
        periods_per_year = len(returns_array) / total_years
        
        vol = returns_array.std() * np.sqrt(periods_per_year)
        print(f"\n=== Volatility Calculation ===")
        print(f"Total years: {total_years:.2f}")
        print(f"Periods per year: {periods_per_year:.2f}")
        print(f"Period std: {returns_array.std():.2%}")
        print(f"Annualized vol: {vol:.2%}")
        
        # What if periods_per_year is wrong?
        print(f"\n=== Alternative Vol Calculations ===")
        print(f"If 1 period/year: {returns_array.std() * np.sqrt(1):.2%}")
        print(f"If 252 periods/year (daily): {returns_array.std() * np.sqrt(252):.2%}")
        print(f"If 12 periods/year (monthly): {returns_array.std() * np.sqrt(12):.2%}")
        
else:
    print("No runs found!")
