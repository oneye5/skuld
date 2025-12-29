"""Analyze quintile returns by year to understand when the model fails."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load latest 30-window results  
run_dir = Path(__file__).parent.parent / "output/runs/ranking_20251229_181811"
pred_df = pd.read_csv(run_dir / "predictions.csv")

# Convert timestamp to date
pred_df['date'] = pd.to_datetime(pred_df['timestamp'], unit='ms')
pred_df['year'] = pred_df['date'].dt.year

# Compute quintiles per timestamp
def assign_quintiles(group):
    group = group.copy()
    try:
        group['predicted_quintile'] = pd.qcut(group['predicted_score'].rank(method='first'), q=5, labels=[1,2,3,4,5])
    except:
        group['predicted_quintile'] = np.nan
    return group

pred_df = pred_df.groupby('timestamp', group_keys=False).apply(assign_quintiles)

print("=" * 70)
print("QUINTILE RETURNS BY YEAR")
print("=" * 70)
print(f"{'Year':<6} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'Q5':>8} {'Q5-Q4':>8} {'Mono':>6}")
print("-" * 70)

for year in sorted(pred_df['year'].unique()):
    year_df = pred_df[pred_df['year'] == year]
    quintile_ret = year_df.groupby('predicted_quintile')['actual_return'].mean()
    
    q1 = quintile_ret.get(1, np.nan)
    q2 = quintile_ret.get(2, np.nan)
    q3 = quintile_ret.get(3, np.nan)
    q4 = quintile_ret.get(4, np.nan)
    q5 = quintile_ret.get(5, np.nan)
    
    q5_vs_q4 = q5 - q4 if pd.notna(q5) and pd.notna(q4) else np.nan
    
    # Check monotonicity
    returns = [q1, q2, q3, q4, q5]
    valid_returns = [r for r in returns if pd.notna(r)]
    is_monotonic = all(valid_returns[i] <= valid_returns[i+1] for i in range(len(valid_returns)-1)) if len(valid_returns) > 1 else False
    
    print(f"{year:<6} {q1:>8.3f} {q2:>8.3f} {q3:>8.3f} {q4:>8.3f} {q5:>8.3f} {q5_vs_q4:>8.3f} {'Yes' if is_monotonic else 'No':>6}")

print("=" * 70)
print("\nKey insight: Years where Q5-Q4 < 0 indicate the lottery stock problem.")
print("Model picks extreme winners that mean-revert instead of continuing.")
