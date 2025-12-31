"""Deep investigation into why IC is so high."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

run_dir = Path(__file__).parent.parent / "output/runs/ranking_20251230_224601_a8eb71e7"
df = pd.read_csv(run_dir / "predictions.csv")

print("=" * 70)
print("DEEP IC INVESTIGATION")
print("=" * 70)

# The IC is computed per-timestamp. Let's see the distribution
def compute_ic_per_ts(group):
    if len(group) < 5:
        return np.nan
    pred = group['predicted_score']
    actual = group['actual_return']
    if pred.std() == 0 or actual.std() == 0:
        return np.nan
    return stats.pearsonr(pred, actual)[0]

ic_series = df.groupby('timestamp', group_keys=False).apply(compute_ic_per_ts)

print(f"\nIC Distribution:")
print(f"  Mean: {ic_series.mean():.4f}")
print(f"  Std:  {ic_series.std():.4f}")
print(f"  25%:  {ic_series.quantile(0.25):.4f}")
print(f"  50%:  {ic_series.quantile(0.50):.4f}")
print(f"  75%:  {ic_series.quantile(0.75):.4f}")

# Check: Is the actual_return truly the 365-day forward return?
# If it's calculated correctly, stocks that go up 365 days later should have positive returns
print("\n" + "=" * 70)
print("SANITY CHECK: IS actual_return THE CORRECT TARGET?")
print("=" * 70)

# The key question: is the model predicting something that's "easy" to predict?
# For instance, if actual_return is the SAME across all timestamps for a ticker,
# then we're just learning ticker-specific returns, not time-series prediction.

# Check variance of actual_return per ticker
ticker_return_stats = df.groupby('ticker')['actual_return'].agg(['mean', 'std', 'count'])
print("\nReturn variance by ticker:")
print(f"  Mean of ticker std: {ticker_return_stats['std'].mean():.4f}")
print(f"  Min ticker std:     {ticker_return_stats['std'].min():.4f}")
print(f"  Max ticker std:     {ticker_return_stats['std'].max():.4f}")

# If the std is low, the ticker's return is nearly constant across all timestamps
low_var_tickers = ticker_return_stats[ticker_return_stats['std'] < 0.1]
print(f"\nTickers with std < 0.1: {len(low_var_tickers)} / {len(ticker_return_stats)}")

# Check: what does the model actually predict?
# Compare predicted_score distribution across tickers
print("\n" + "=" * 70)
print("WHAT IS THE MODEL LEARNING?")
print("=" * 70)

# For each ticker, compute mean predicted_score and mean actual_return
ticker_avg = df.groupby('ticker').agg({
    'predicted_score': 'mean',
    'actual_return': 'mean',
}).reset_index()

# Correlation between average prediction and average return per ticker
ticker_corr = stats.pearsonr(ticker_avg['predicted_score'], ticker_avg['actual_return'])[0]
print(f"\nCorrelation between TICKER-AVERAGE prediction and return: {ticker_corr:.4f}")
print("(If high, model is learning which tickers are generally good/bad)")

# Now check: within each ticker, does the model predict well?
print("\n--- Per-ticker IC (does model predict TIMING?) ---")
ticker_ic_list = []
for ticker in df['ticker'].unique():
    ticker_df = df[df['ticker'] == ticker]
    if len(ticker_df) >= 5:
        pred = ticker_df['predicted_score']
        actual = ticker_df['actual_return']
        if pred.std() > 0 and actual.std() > 0:
            ic = stats.pearsonr(pred, actual)[0]
            ticker_ic_list.append(ic)

ticker_ic_series = pd.Series(ticker_ic_list)
print(f"Mean within-ticker IC: {ticker_ic_series.mean():.4f}")
print(f"Std within-ticker IC:  {ticker_ic_series.std():.4f}")
print(f"Tickers with positive IC: {(ticker_ic_series > 0).sum()} / {len(ticker_ic_series)}")

print("\n" + "=" * 70)
print("KEY INSIGHT: CROSS-SECTIONAL vs LONGITUDINAL")
print("=" * 70)
print("""
The model is evaluated CROSS-SECTIONALLY (ranking stocks within each timestamp).

If the model learns which TICKERS are generally good, and ticker-average returns
are stable, the cross-sectional IC will be high even without predicting timing.

This is NOT leakage - it's a valid signal. But it means the model may not be
predicting time-varying alpha, just capturing ticker effects.
""")

# Check stability of ticker returns across time
print("=" * 70)
print("TICKER RETURN STABILITY OVER TIME")
print("=" * 70)

# Split data into first half and second half by time
timestamps = sorted(df['timestamp'].unique())
mid_ts = timestamps[len(timestamps) // 2]

df_first_half = df[df['timestamp'] < mid_ts]
df_second_half = df[df['timestamp'] >= mid_ts]

# Compute ticker-average returns in each half
first_half_avg = df_first_half.groupby('ticker')['actual_return'].mean()
second_half_avg = df_second_half.groupby('ticker')['actual_return'].mean()

# Merge and compute correlation
merged = pd.DataFrame({
    'first_half': first_half_avg,
    'second_half': second_half_avg
}).dropna()

if len(merged) >= 3:
    persistence_corr = stats.pearsonr(merged['first_half'], merged['second_half'])[0]
    print(f"Ticker return persistence (first half vs second half): {persistence_corr:.4f}")
    print("(If high, ticker characteristics are stable, explaining high cross-sectional IC)")
else:
    print("Not enough data to compute persistence")

# Check: Does the model score vary much for each ticker over time?
print("\n" + "=" * 70)
print("PREDICTION VARIABILITY")
print("=" * 70)

pred_var_by_ticker = df.groupby('ticker')['predicted_score'].std()
print(f"Mean prediction std by ticker: {pred_var_by_ticker.mean():.4f}")
print(f"Min: {pred_var_by_ticker.min():.4f}, Max: {pred_var_by_ticker.max():.4f}")

# If predictions vary a lot within each ticker, the model is trying to predict timing
# If predictions are stable, model is just ranking tickers

# Compare to return variability
ret_var_by_ticker = df.groupby('ticker')['actual_return'].std()
print(f"\nMean return std by ticker: {ret_var_by_ticker.mean():.4f}")
print(f"Min: {ret_var_by_ticker.min():.4f}, Max: {ret_var_by_ticker.max():.4f}")

# Ratio of prediction variance to return variance
var_ratio = pred_var_by_ticker.mean() / ret_var_by_ticker.mean()
print(f"\nPrediction variance / Return variance: {var_ratio:.4f}")

print("\n" + "=" * 70)
print("LEAKAGE CHECK: Does predicted_score correlate with current return?")
print("=" * 70)

# This would be BAD - using future info
# actual_return is the 365-day FORWARD return
# predicted_score should not correlate with current/past returns if no leakage

# Let's check if predictions are suspiciously similar to actual returns
pred_actual_corr_overall = stats.pearsonr(df['predicted_score'], df['actual_return'])[0]
print(f"Overall correlation between predicted_score and actual_return: {pred_actual_corr_overall:.4f}")
print("(This is the pooled correlation, not cross-sectional IC)")

# If this is close to 1.0, there might be direct leakage
if pred_actual_corr_overall > 0.5:
    print("\n*** WARNING: Very high overall correlation may indicate leakage! ***")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Cross-sectional IC:           {ic_series.mean():.4f}
Ticker-average correlation:   {ticker_corr:.4f}
Within-ticker IC:             {ticker_ic_series.mean():.4f}
Return persistence:           {persistence_corr:.4f}
Overall pred-actual corr:     {pred_actual_corr_overall:.4f}

INTERPRETATION:
- High ticker-avg correlation ({ticker_corr:.2f}) means the model ranks tickers 
  by their AVERAGE return, not by time-varying alpha.
- High return persistence ({persistence_corr:.2f}) means ticker characteristics 
  are stable over time.
- The cross-sectional IC ({ic_series.mean():.2f}) reflects ticker selection, 
  not market timing.

This is NOT necessarily leakage - it's valid cross-sectional stock selection.
But it means the model may not generalize well if ticker characteristics change.
""")
