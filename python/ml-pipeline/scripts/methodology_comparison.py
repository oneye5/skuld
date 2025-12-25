"""
Comparison Investigation: Key Methodological Differences

This script compares possible methodology differences between 
nzx-predictor and skuld that could explain the Sharpe gap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from config.columns import TIMESTAMP, TICKER, CLOSE, TARGET, PREDICTION_PROB
from config.settings import (
    LOOKAHEAD_DAYS, MS_PER_DAY, GAIN_THRESHOLD_PCT, YEAR_2000_MS,
    PREDICTION_THRESHOLD
)
from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers


def ts_to_date(ts: int) -> str:
    return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")


def analyze_trades():
    """Analyze the trade results from the most recent run."""
    print("="*70)
    print("ANALYSIS: Trade Results from Recent Run")
    print("="*70)
    
    # Find most recent run
    output_dir = Path(__file__).parent.parent / "output" / "runs"
    runs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not runs:
        print("No runs found!")
        return
    
    latest_run = runs[0]
    trades_file = latest_run / "trades.csv"
    
    if not trades_file.exists():
        print(f"No trades.csv found in {latest_run}")
        return
    
    trades = pd.read_csv(trades_file)
    print(f"\nLoaded {len(trades)} trades from {latest_run.name}")
    
    # Basic stats
    print(f"\nTrade Statistics:")
    print(f"  Total trades: {len(trades)}")
    print(f"  Mean return: {trades['return_pct'].mean():.2f}%")
    print(f"  Median return: {trades['return_pct'].median():.2f}%")
    print(f"  Std return: {trades['return_pct'].std():.2f}%")
    
    # Sharpe (as calculated in the pipeline)
    sharpe = trades['return_pct'].mean() / trades['return_pct'].std() if trades['return_pct'].std() > 0 else 0
    print(f"  Sharpe (raw): {sharpe:.4f}")
    
    # Win rate
    win_rate = (trades['return_pct'] > 0).mean()
    print(f"  Win rate: {win_rate:.2%}")
    
    # Distribution of returns
    print(f"\nReturn distribution:")
    print(f"  Min: {trades['return_pct'].min():.2f}%")
    print(f"  25th percentile: {trades['return_pct'].quantile(0.25):.2f}%")
    print(f"  50th percentile: {trades['return_pct'].quantile(0.50):.2f}%")
    print(f"  75th percentile: {trades['return_pct'].quantile(0.75):.2f}%")
    print(f"  Max: {trades['return_pct'].max():.2f}%")
    
    # Check for outliers
    print(f"\nOutlier analysis:")
    huge_wins = trades[trades['return_pct'] > 100]
    huge_losses = trades[trades['return_pct'] < -50]
    print(f"  Returns > 100%: {len(huge_wins)} trades")
    print(f"  Returns < -50%: {len(huge_losses)} trades")
    
    # Calculate Sharpe without outliers
    trades_filtered = trades[(trades['return_pct'] > -50) & (trades['return_pct'] < 100)]
    sharpe_filtered = trades_filtered['return_pct'].mean() / trades_filtered['return_pct'].std() if len(trades_filtered) > 0 else 0
    print(f"\nSharpe without outliers: {sharpe_filtered:.4f} (n={len(trades_filtered)})")
    
    # Time analysis
    trades['buy_date'] = pd.to_datetime(trades['buy_timestamp'] / 1000, unit='s')
    trades['year'] = trades['buy_date'].dt.year
    
    print(f"\nReturn by year:")
    yearly = trades.groupby('year')['return_pct'].agg(['mean', 'std', 'count'])
    for year, row in yearly.iterrows():
        sharpe_yr = row['mean'] / row['std'] if row['std'] > 0 else 0
        print(f"  {year}: mean={row['mean']:+.2f}%, std={row['std']:.2f}%, n={int(row['count'])}, sharpe={sharpe_yr:.3f}")


def investigate_nzx_predictor_methodology():
    """
    Document what we know about nzx-predictor's methodology 
    from the code comments.
    """
    print("\n" + "="*70)
    print("KNOWN DIFFERENCES FROM CODE COMMENTS")
    print("="*70)
    
    print("""
From the skuld codebase comments referencing nzx-predictor:

1. SCALER FITTING:
   - scaler.py line 26: "Following nzx-predictor approach: fit on combined train+test data"
   - single_window.py line 172: "Note: nzx-predictor fit on combined, but fitting on train is more correct"
   
   ⚠️  This is a DIFFERENCE: skuld now fits on train only (line 175)
   
2. PREDICTION THRESHOLD:
   - settings.py line 17: PREDICTION_THRESHOLD = 0.75
   - Comment says: "(0.79 matches nzx-predictor)"
   
   ⚠️  Current threshold (0.75) is DIFFERENT from nzx-predictor (0.79)

3. MISSING VALUE HANDLING:
   - preprocessor.py: "Following nzx-predictor Java approach exactly"
   - MissingFlag pattern documented as matching CsvWriter.java
   
   ✓  Should be SAME approach

4. LABELING:
   - Lookahead days: {LOOKAHEAD_DAYS}
   - Gain threshold: {GAIN_THRESHOLD_PCT}%
   
   ❓  Unknown if nzx-predictor used same parameters

5. ROLLING WINDOWS:
   - Current: 25 rolling windows
   
   ❓  Did nzx-predictor use rolling windows or single train/test?
""".format(LOOKAHEAD_DAYS=LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT=GAIN_THRESHOLD_PCT))


def simulate_leaky_scenario():
    """
    Simulate what would happen if we intentionally add leakage.
    This helps understand how much leakage affects performance.
    """
    print("\n" + "="*70)
    print("SIMULATION: Impact of Scaler Leakage")
    print("="*70)
    
    # Load a sample of data
    long_df = load_long_data()
    long_df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS]
    
    # Just use a subset for speed
    tickers = long_df['ticker'].unique()[:20]
    long_df = long_df[long_df['ticker'].isin(tickers)]
    
    long_df = clean_and_classify_tickers(long_df)
    long_df = add_macro_prefix(long_df)
    wide_df = long_to_wide(long_df)
    
    print(f"Loaded {len(wide_df):,} samples for simulation")
    
    from sklearn.preprocessing import RobustScaler
    from core.labeler import create_labels
    from core.splitter import split_by_timestamp
    
    # Create labels
    labeled = create_labels(wide_df.copy(), LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT)
    
    # Split
    max_ts = labeled[TIMESTAMP].max()
    test_period_ms = 365 * MS_PER_DAY
    train_end_ts = max_ts - LOOKAHEAD_DAYS * MS_PER_DAY - test_period_ms
    test_end_ts = max_ts - LOOKAHEAD_DAYS * MS_PER_DAY
    
    split = split_by_timestamp(labeled, train_end_ts, test_end_ts)
    
    print(f"Train: {len(split.train):,} samples")
    print(f"Test: {len(split.test):,} samples")
    
    # Get numeric columns
    excluded = [TIMESTAMP, TICKER, TARGET]
    numeric_cols = [c for c in split.train.columns 
                   if c not in excluded 
                   and pd.api.types.is_numeric_dtype(split.train[c])
                   and split.train[c].std() > 0][:100]
    
    if not numeric_cols or len(split.test) == 0:
        print("Not enough data for simulation")
        return
    
    # Fill NaN for this test
    train_X = split.train[numeric_cols].fillna(0)
    test_X = split.test[numeric_cols].fillna(0)
    train_y = split.train[TARGET]
    test_y = split.test[TARGET]
    
    from lightgbm import LGBMClassifier
    
    # Scenario 1: Correct (fit scaler on train only)
    print("\n--- Scenario 1: Fit scaler on TRAIN only (correct) ---")
    scaler1 = RobustScaler()
    train_scaled1 = scaler1.fit_transform(train_X)
    test_scaled1 = scaler1.transform(test_X)
    
    model1 = LGBMClassifier(verbosity=-1, random_state=42)
    model1.fit(train_scaled1, train_y)
    proba1 = model1.predict_proba(test_scaled1)[:, 1]
    
    # Scenario 2: Leaky (fit scaler on combined)
    print("\n--- Scenario 2: Fit scaler on COMBINED (leaky) ---")
    combined = pd.concat([train_X, test_X])
    scaler2 = RobustScaler()
    scaler2.fit(combined)
    train_scaled2 = scaler2.transform(train_X)
    test_scaled2 = scaler2.transform(test_X)
    
    model2 = LGBMClassifier(verbosity=-1, random_state=42)
    model2.fit(train_scaled2, train_y)
    proba2 = model2.predict_proba(test_scaled2)[:, 1]
    
    # Compare
    from sklearn.metrics import roc_auc_score
    
    auc1 = roc_auc_score(test_y, proba1) if len(test_y.unique()) > 1 else 0.5
    auc2 = roc_auc_score(test_y, proba2) if len(test_y.unique()) > 1 else 0.5
    
    print(f"\nResults comparison:")
    print(f"  AUC (correct): {auc1:.4f}")
    print(f"  AUC (leaky):   {auc2:.4f}")
    print(f"  Difference:    {auc2 - auc1:+.4f}")
    
    # Calculate simple Sharpe-like metrics
    threshold = 0.75
    
    def simple_sharpe(proba, actual, threshold):
        signals = proba >= threshold
        if signals.sum() == 0:
            return 0, 0
        actual_at_signals = actual[signals]
        # Simplified: assume +10% for correct, -10% for wrong
        returns = actual_at_signals * 20 - 10  # +10 if target=1, -10 if target=0
        return returns.mean() / (returns.std() + 1e-6), signals.sum()
    
    sharpe1, n1 = simple_sharpe(proba1, test_y.values, threshold)
    sharpe2, n2 = simple_sharpe(proba2, test_y.values, threshold)
    
    print(f"\nSimplified Sharpe (threshold={threshold}):")
    print(f"  Sharpe (correct): {sharpe1:.4f} (n={n1})")
    print(f"  Sharpe (leaky):   {sharpe2:.4f} (n={n2})")
    
    print("\n⚠️  Note: Scaler leakage typically has SMALL effect on performance")
    print("    If nzx-predictor had 10x better Sharpe, it's likely from other causes")


def check_threshold_impact():
    """
    Check how prediction threshold affects number of trades and returns.
    """
    print("\n" + "="*70)
    print("ANALYSIS: Prediction Threshold Impact")
    print("="*70)
    
    output_dir = Path(__file__).parent.parent / "output" / "runs"
    runs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not runs:
        print("No runs found!")
        return
    
    latest_run = runs[0]
    trades_file = latest_run / "trades.csv"
    
    if not trades_file.exists():
        return
    
    trades = pd.read_csv(trades_file)
    
    print(f"\nCurrent threshold: {PREDICTION_THRESHOLD}")
    print(f"Total trades: {len(trades)}")
    print(f"Mean return: {trades['return_pct'].mean():.2f}%")
    
    # Note: We can't test different thresholds without re-running
    # But we can check the distribution of prediction probabilities
    
    print(f"""
To investigate threshold sensitivity:
1. Lower threshold → More trades, potentially lower quality
2. Higher threshold → Fewer trades, potentially higher quality

Current: threshold={PREDICTION_THRESHOLD}, trades={len(trades)}, mean_return={trades['return_pct'].mean():.2f}%

If nzx-predictor used threshold=0.79 (higher), it would have:
- Fewer trades (only high-confidence predictions)
- Potentially better average return per trade
- Different Sharpe calculation base
""")


def check_position_sizing_impact():
    """
    Check how position sizing affects Sharpe.
    """
    print("\n" + "="*70)
    print("ANALYSIS: Position Sizing Impact")
    print("="*70)
    
    from config.settings import INITIAL_CAPITAL, MAX_POSITION_SIZE_PCT
    
    print(f"""
Current settings:
- INITIAL_CAPITAL: ${INITIAL_CAPITAL:,.0f}
- MAX_POSITION_SIZE_PCT: {MAX_POSITION_SIZE_PCT}%
- Position size: ${INITIAL_CAPITAL * MAX_POSITION_SIZE_PCT / 100:,.0f}

Sharpe ratio in current implementation:
- Calculated as: mean(return_pct) / std(return_pct)
- This is the return % per trade, not accounting for capital allocation

IMPORTANT: If position sizing differs between pipelines, Sharpe isn't comparable!

Standard Sharpe for portfolios would be:
- Portfolio return = sum(position_weight * return)
- Sharpe = mean(portfolio_return) / std(portfolio_return) * sqrt(N)
  where N = number of periods per year

Current approach treats each trade independently.
If nzx-predictor used portfolio-level Sharpe, results differ significantly.
""")


if __name__ == "__main__":
    analyze_trades()
    investigate_nzx_predictor_methodology()
    simulate_leaky_scenario()
    check_threshold_impact()
    check_position_sizing_impact()
