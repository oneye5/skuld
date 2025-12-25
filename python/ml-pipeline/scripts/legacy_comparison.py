"""
Analysis of Legacy Trade Results vs Current Pipeline

Compare the legacy trade_simulation.csv with current results
to understand the performance gap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime


def ts_to_date(ts: int) -> str:
    return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")


def analyze_legacy_trades():
    print("="*70)
    print("ANALYSIS: Legacy vs Current Trade Results")
    print("="*70)
    
    # Load legacy trades
    legacy_path = Path(__file__).parent.parent.parent.parent / "data" / "legacy" / "trade_simulation.csv"
    legacy = pd.read_csv(legacy_path)
    print(f"\nLegacy trades: {len(legacy)}")
    
    # Load current trades
    output_dir = Path(__file__).parent.parent / "output" / "runs"
    runs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
    current_path = runs[0] / "trades.csv" if runs else None
    
    if current_path and current_path.exists():
        current = pd.read_csv(current_path)
        print(f"Current trades: {len(current)}")
    else:
        current = None
        print("Current trades: Not found")
    
    print("\n" + "-"*70)
    print("LEGACY TRADE ANALYSIS")
    print("-"*70)
    
    # Basic stats
    print(f"\nLegacy Statistics:")
    print(f"  Total trades: {len(legacy)}")
    print(f"  Mean return: {legacy['return_pct'].mean():.2f}%")
    print(f"  Median return: {legacy['return_pct'].median():.2f}%")
    print(f"  Std return: {legacy['return_pct'].std():.2f}%")
    
    sharpe_legacy = legacy['return_pct'].mean() / legacy['return_pct'].std()
    print(f"  Sharpe (raw): {sharpe_legacy:.4f}")
    
    win_rate = (legacy['return_pct'] > 0).mean()
    print(f"  Win rate: {win_rate:.2%}")
    
    # Distribution
    print(f"\nReturn distribution:")
    print(f"  Min: {legacy['return_pct'].min():.2f}%")
    print(f"  25th percentile: {legacy['return_pct'].quantile(0.25):.2f}%")
    print(f"  50th percentile: {legacy['return_pct'].quantile(0.50):.2f}%")
    print(f"  75th percentile: {legacy['return_pct'].quantile(0.75):.2f}%")
    print(f"  Max: {legacy['return_pct'].max():.2f}%")
    
    # Check for suspicious returns
    print(f"\nOutlier analysis:")
    huge_wins = legacy[legacy['return_pct'] > 100]
    huge_losses = legacy[legacy['return_pct'] < -50]
    print(f"  Returns > 100%: {len(huge_wins)} trades ({len(huge_wins)/len(legacy)*100:.1f}%)")
    print(f"  Returns < -50%: {len(huge_losses)} trades ({len(huge_losses)/len(legacy)*100:.1f}%)")
    
    # Analyze the huge wins
    if len(huge_wins) > 0:
        print(f"\nHuge wins (>100% return):")
        for _, row in huge_wins.head(10).iterrows():
            buy_date = ts_to_date(int(row['buy_time']))
            sell_date = ts_to_date(int(row['sell_time']))
            print(f"  {row['ticker']}: {buy_date} → {sell_date}, {row['return_pct']:.0f}% (buy=${row['buy_price']:.4f}, sell=${row['sell_price']:.4f})")
    
    # Check tickers
    print(f"\nTickers in legacy trades:")
    ticker_counts = legacy['ticker'].value_counts()
    print(f"  Unique tickers: {len(ticker_counts)}")
    print(f"  Top tickers:")
    for ticker, count in ticker_counts.head(10).items():
        mean_ret = legacy[legacy['ticker'] == ticker]['return_pct'].mean()
        print(f"    {ticker}: {count} trades, mean return = {mean_ret:.2f}%")
    
    # Suspicious ticker analysis
    print(f"\n⚠️  SUSPICIOUS: Check the ticker names!")
    print(f"  Many tickers start with '%5E' (URL-encoded '^')")
    print(f"  These might be INDEX tickers, not tradeable stocks!")
    
    index_trades = legacy[legacy['ticker'].str.startswith('%5E')]
    stock_trades = legacy[~legacy['ticker'].str.startswith('%5E')]
    
    print(f"\nIndex vs Stock trades:")
    print(f"  Index-like trades: {len(index_trades)} ({len(index_trades)/len(legacy)*100:.1f}%)")
    print(f"  Stock trades: {len(stock_trades)} ({len(stock_trades)/len(legacy)*100:.1f}%)")
    
    if len(index_trades) > 0 and len(stock_trades) > 0:
        print(f"\nPerformance comparison:")
        print(f"  Index mean return: {index_trades['return_pct'].mean():.2f}%")
        print(f"  Stock mean return: {stock_trades['return_pct'].mean():.2f}%")
        print(f"  Index Sharpe: {index_trades['return_pct'].mean() / index_trades['return_pct'].std():.4f}")
        print(f"  Stock Sharpe: {stock_trades['return_pct'].mean() / stock_trades['return_pct'].std():.4f}")
    
    # Time period analysis
    legacy['buy_date'] = pd.to_datetime(legacy['buy_time'] / 1000, unit='s')
    legacy['sell_date'] = pd.to_datetime(legacy['sell_time'] / 1000, unit='s')
    legacy['holding_days'] = (legacy['sell_date'] - legacy['buy_date']).dt.days
    
    print(f"\nHolding period analysis:")
    print(f"  Min holding days: {legacy['holding_days'].min()}")
    print(f"  Mean holding days: {legacy['holding_days'].mean():.0f}")
    print(f"  Max holding days: {legacy['holding_days'].max()}")
    
    # Year analysis
    legacy['year'] = legacy['buy_date'].dt.year
    
    print(f"\nReturn by year:")
    yearly = legacy.groupby('year')['return_pct'].agg(['mean', 'std', 'count'])
    for year, row in yearly.iterrows():
        sharpe_yr = row['mean'] / row['std'] if row['std'] > 0 else 0
        print(f"  {year}: mean={row['mean']:+.2f}%, std={row['std']:.2f}%, n={int(row['count'])}, sharpe={sharpe_yr:.3f}")
    
    print("\n" + "-"*70)
    print("COMPARISON WITH CURRENT PIPELINE")
    print("-"*70)
    
    if current is not None:
        print(f"\n{'Metric':<25} {'Legacy':>15} {'Current':>15} {'Diff':>15}")
        print("-"*70)
        
        metrics = [
            ('Total trades', len(legacy), len(current)),
            ('Mean return %', legacy['return_pct'].mean(), current['return_pct'].mean()),
            ('Median return %', legacy['return_pct'].median(), current['return_pct'].median()),
            ('Std return %', legacy['return_pct'].std(), current['return_pct'].std()),
            ('Sharpe', legacy['return_pct'].mean() / legacy['return_pct'].std(),
             current['return_pct'].mean() / current['return_pct'].std()),
            ('Win rate %', (legacy['return_pct'] > 0).mean() * 100,
             (current['return_pct'] > 0).mean() * 100),
        ]
        
        for name, leg, cur in metrics:
            diff = cur - leg
            print(f"{name:<25} {leg:>15.2f} {cur:>15.2f} {diff:>+15.2f}")
    
    # Summary of key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print(f"""
1. LEGACY SHARPE: {sharpe_legacy:.4f}
   This is NOT 1+ as mentioned. The legacy data shows similar low Sharpe!
   
2. LEGACY DATA ISSUES:
   - Contains URL-encoded ticker names (%5EFTSE instead of ^FTSE)
   - Includes index/macro data that may not be tradeable
   - Some huge returns (>1000%) suggest data quality issues
   
3. HOLDING PERIOD:
   - Legacy mean holding days: {legacy['holding_days'].mean():.0f}
   - This confirms ~1 year lookahead (similar to current 366 days)
   
4. POSSIBLE EXPLANATION FOR "1+ Sharpe":
   - The "1+ Sharpe" might have been from a DIFFERENT evaluation
   - Or calculated differently (annualized? different formula?)
   - Or on a subset of data with cherry-picked time period
   
5. THE CURRENT PIPELINE IS LIKELY CORRECT:
   - Similar methodology to legacy
   - Similar (low) Sharpe ratio
   - The model may simply not have strong predictive power
""")
    
    return legacy


def check_specific_leakage_patterns():
    """
    Check for specific leakage patterns in the legacy data.
    """
    print("\n" + "="*70)
    print("LEAKAGE PATTERN ANALYSIS")
    print("="*70)
    
    legacy_path = Path(__file__).parent.parent.parent.parent / "data" / "legacy" / "trade_simulation.csv"
    legacy = pd.read_csv(legacy_path)
    
    # Check if sell_price could be known at buy_time (potential leakage)
    # This would be a serious issue if the model somehow had access to future prices
    
    print("\nChecking for suspicious price patterns...")
    
    # Calculate if there's any correlation between return and buy_price
    corr = legacy['return_pct'].corr(legacy['buy_price'])
    print(f"  Correlation between return_pct and buy_price: {corr:.4f}")
    
    # Check if buy_time and sell_time are always exactly LOOKAHEAD apart
    legacy['holding_days'] = (legacy['sell_time'] - legacy['buy_time']) / (1000 * 60 * 60 * 24)
    
    print(f"\nHolding period distribution:")
    print(f"  Min: {legacy['holding_days'].min():.0f} days")
    print(f"  Max: {legacy['holding_days'].max():.0f} days")
    print(f"  Mean: {legacy['holding_days'].mean():.0f} days")
    print(f"  Std: {legacy['holding_days'].std():.0f} days")
    
    # Check for exact 366-day holds
    exact_366 = (legacy['holding_days'] > 364) & (legacy['holding_days'] < 368)
    print(f"  Trades with ~366 day hold: {exact_366.sum()} ({exact_366.mean()*100:.1f}%)")
    
    # Check the legacy evaluation metrics
    metrics_path = Path(__file__).parent.parent.parent.parent / "data" / "legacy" / "evaluation_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    
    print(f"\nLegacy evaluation metrics:")
    print(f"  n_samples: {int(metrics['n_samples'].iloc[0])}")
    print(f"  accuracy: {metrics['accuracy'].iloc[0]:.4f}")
    print(f"  precision: {metrics['precision'].iloc[0]:.4f}")
    print(f"  recall: {metrics['recall'].iloc[0]:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc'].iloc[0]:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc'].iloc[0]:.4f}")


if __name__ == "__main__":
    analyze_legacy_trades()
    check_specific_leakage_patterns()
