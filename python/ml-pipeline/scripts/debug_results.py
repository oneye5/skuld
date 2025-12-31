"""Debug script to investigate suspicious evaluation results."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Load the predictions - use absolute path
run_dir = Path(__file__).parent.parent / "output/runs/ranking_20251230_224601_a8eb71e7"
df = pd.read_csv(run_dir / "predictions.csv")

print("=" * 70)
print("DATA STRUCTURE ANALYSIS")
print("=" * 70)
print(f"Total rows: {len(df):,}")
print(f"Unique timestamps: {df['timestamp'].nunique()}")
print(f"Unique tickers: {df['ticker'].nunique()}")
print(f"Unique windows: {df['window_id'].nunique()}")

# Check for duplicate (timestamp, ticker) pairs
duplicates = df.groupby(['timestamp', 'ticker']).size()
duplicates_count = (duplicates > 1).sum()
print(f"\nDuplicate (timestamp, ticker) pairs: {duplicates_count}")

# If there are duplicates, this is a major issue!
if duplicates_count > 0:
    print("\n*** CRITICAL: Found duplicate predictions for same (timestamp, ticker)! ***")
    dup_examples = duplicates[duplicates > 1].head(10)
    print("Examples:")
    print(dup_examples)

# Analyze windows
print("\n" + "=" * 70)
print("WINDOW ANALYSIS")
print("=" * 70)
window_stats = df.groupby('window_id').agg({
    'timestamp': ['nunique', 'min', 'max'],
    'ticker': 'nunique',
    'actual_return': 'mean'
})
window_stats.columns = ['num_timestamps', 'min_ts', 'max_ts', 'num_tickers', 'mean_return']
print(window_stats)

# Check if windows overlap
print("\n--- Window Timestamp Ranges ---")
for window_id in sorted(df['window_id'].unique()):
    w_df = df[df['window_id'] == window_id]
    print(f"Window {window_id}: ts {w_df['timestamp'].min()} - {w_df['timestamp'].max()} "
          f"({w_df['timestamp'].nunique()} timestamps)")

# Check if same timestamp appears in multiple windows
print("\n--- Timestamp Overlap Between Windows ---")
ts_window_counts = df.groupby('timestamp')['window_id'].nunique()
overlapping = (ts_window_counts > 1).sum()
print(f"Timestamps appearing in multiple windows: {overlapping}")

if overlapping > 0:
    print("\n*** ISSUE: Some timestamps appear in multiple windows! ***")
    print("This causes double-counting in IC calculation!")
    overlap_examples = ts_window_counts[ts_window_counts > 1].head(10)
    print("Example timestamps with overlaps:")
    print(overlap_examples)

# Compute IC per timestamp (the way ranking_metrics does it)
print("\n" + "=" * 70)
print("IC ANALYSIS")
print("=" * 70)

def compute_ic_per_ts(group):
    """Compute IC for a single timestamp."""
    if len(group) < 5:
        return np.nan
    pred = group['predicted_score']
    actual = group['actual_return']
    if pred.std() == 0 or actual.std() == 0:
        return np.nan
    return stats.pearsonr(pred, actual)[0]

# Group by timestamp only (ignoring window - as the evaluation code does)
ic_by_ts = df.groupby('timestamp').apply(compute_ic_per_ts)
ic_clean = ic_by_ts.dropna()

print(f"Mean IC (all timestamps): {ic_clean.mean():.4f}")
print(f"Std IC: {ic_clean.std():.4f}")
print(f"Min IC: {ic_clean.min():.4f}")
print(f"Max IC: {ic_clean.max():.4f}")
print(f"IC > 0: {(ic_clean > 0).sum()} / {len(ic_clean)} ({(ic_clean > 0).mean():.1%})")

# Now check: does the IC get inflated by duplicates?
print("\n--- Checking if duplicates inflate IC ---")
# Deduplicate by taking mean predicted_score for duplicate (ts, ticker) pairs
df_dedup = df.groupby(['timestamp', 'ticker']).agg({
    'predicted_score': 'mean',
    'actual_return': 'first',  # Should be same for same ticker
}).reset_index()

print(f"Rows after deduplication: {len(df_dedup):,} (was {len(df):,})")

ic_by_ts_dedup = df_dedup.groupby('timestamp').apply(
    lambda g: stats.pearsonr(g['predicted_score'], g['actual_return'])[0]
    if len(g) >= 5 and g['predicted_score'].std() > 0 and g['actual_return'].std() > 0
    else np.nan
)
ic_clean_dedup = ic_by_ts_dedup.dropna()

print(f"\nMean IC after dedup: {ic_clean_dedup.mean():.4f}")
print(f"Std IC after dedup: {ic_clean_dedup.std():.4f}")

# Check correlation between predicted_score and actual_return
print("\n" + "=" * 70)
print("PREDICTION vs ACTUAL RELATIONSHIP")
print("=" * 70)

# Sample a few timestamps to check correlation patterns
sample_ts = sorted(df['timestamp'].unique())[:5]
print("\nSample IC values for first 5 timestamps:")
for ts in sample_ts:
    ts_df = df[df['timestamp'] == ts]
    ic = stats.pearsonr(ts_df['predicted_score'], ts_df['actual_return'])[0]
    print(f"  ts={ts}: IC={ic:.4f}, n={len(ts_df)}, "
          f"pred_std={ts_df['predicted_score'].std():.4f}, "
          f"ret_std={ts_df['actual_return'].std():.4f}")

# Check: are the actual_return values reasonable?
print("\n" + "=" * 70)
print("RETURN DISTRIBUTION")
print("=" * 70)
print(f"Mean actual_return: {df['actual_return'].mean():.4f}")
print(f"Std actual_return: {df['actual_return'].std():.4f}")
print(f"Min actual_return: {df['actual_return'].min():.4f}")
print(f"Max actual_return: {df['actual_return'].max():.4f}")

# Check winsorization
print(f"\nReturns at -50%: {(df['actual_return'] == -0.5).sum()}")
print(f"Returns at +50%: {(df['actual_return'] == 0.5).sum()}")

# Quintile analysis
print("\n" + "=" * 70)
print("QUINTILE CHECK")
print("=" * 70)

def assign_quintile(group):
    """Assign quintiles based on predicted score."""
    return pd.qcut(group['predicted_score'], q=5, labels=[1,2,3,4,5], duplicates='drop')

df_dedup['quintile'] = df_dedup.groupby('timestamp').apply(
    lambda g: pd.qcut(g['predicted_score'], q=5, labels=[1,2,3,4,5], duplicates='drop')
).reset_index(level=0, drop=True)

quintile_returns = df_dedup.groupby('quintile')['actual_return'].mean()
print("Quintile average returns:")
print(quintile_returns)
print(f"\nQ5 - Q1 spread: {quintile_returns[5] - quintile_returns[1]:.4f}")

# Final diagnosis
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

issues = []
if duplicates_count > 0:
    issues.append("DUPLICATE predictions for same (timestamp, ticker) - major issue!")
if overlapping > 0:
    issues.append("OVERLAPPING windows - timestamps counted multiple times in IC")
if ic_clean.mean() > 0.20:
    issues.append(f"IC is very high ({ic_clean.mean():.2f}) - possible leakage or issue")
if ic_clean_dedup.mean() < ic_clean.mean() * 0.9:
    issues.append("IC drops significantly after deduplication - duplicates inflating IC")

if issues:
    print("ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("No obvious issues found in data structure.")
