"""
Deep Investigation of Suspicious High-Correlation Features

The first investigation found features with 0.99+ correlation to the target:
- trailingDividendIncome: +0.9989
- trailingFeesandCommissionExpense: +0.9717
- trailingDilutedNIAvailtoComStockholders: +0.8654

This script investigates WHY these correlations are so high and whether
they represent actual leakage or spurious correlations from sparse data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from config.columns import TIMESTAMP, TICKER, CLOSE, TARGET
from config.settings import LOOKAHEAD_DAYS, MS_PER_DAY, GAIN_THRESHOLD_PCT, YEAR_2000_MS
from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from core.labeler import create_labels


def ts_to_date(ts: int) -> str:
    return datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")


def investigate_high_correlation_features():
    print("="*70)
    print("DEEP INVESTIGATION: Why do some features have ~0.99 target correlation?")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    long_df = load_long_data()
    long_df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS]
    long_df = clean_and_classify_tickers(long_df)
    long_df = add_macro_prefix(long_df)
    wide_df = long_to_wide(long_df)
    print(f"Wide format: {len(wide_df):,} rows, {len(wide_df.columns)} columns")
    
    # Create labels
    labeled = create_labels(wide_df.copy(), LOOKAHEAD_DAYS, GAIN_THRESHOLD_PCT)
    print(f"Labeled samples: {len(labeled):,}")
    
    # The suspicious features
    suspicious_features = [
        'trailingDividendIncome',
        'trailingFeesandCommissionExpense', 
        'trailingDilutedNIAvailtoComStockholders',
        'trailingGainOnSaleOfSecurity',
        'trailingFeesAndCommissions'
    ]
    
    print("\n" + "="*70)
    print("ANALYSIS 1: Sparsity of suspicious features")
    print("="*70)
    
    for feat in suspicious_features:
        if feat in labeled.columns:
            non_null = labeled[feat].notna().sum()
            non_zero = (labeled[feat] != 0).sum()
            total = len(labeled)
            
            print(f"\n{feat}:")
            print(f"  Non-null: {non_null:,} / {total:,} ({non_null/total*100:.2f}%)")
            print(f"  Non-zero: {non_zero:,} / {total:,} ({non_zero/total*100:.2f}%)")
            
            # Check correlation only among non-null/non-zero values
            mask = labeled[feat].notna() & (labeled[feat] != 0)
            if mask.sum() > 10:
                subset = labeled[mask]
                corr = subset[feat].corr(subset[TARGET])
                print(f"  Correlation (among {mask.sum()} non-zero samples): {corr:.4f}")
                print(f"  Target distribution in non-zero subset: {dict(subset[TARGET].value_counts())}")
            else:
                print(f"  Too few non-zero samples to compute correlation")
    
    print("\n" + "="*70)
    print("ANALYSIS 2: Which tickers have these features?")
    print("="*70)
    
    for feat in suspicious_features[:3]:  # Just check first 3
        if feat not in labeled.columns:
            continue
            
        tickers_with_feat = labeled[labeled[feat].notna() & (labeled[feat] != 0)][TICKER].unique()
        print(f"\n{feat}:")
        print(f"  Tickers with this feature: {len(tickers_with_feat)}")
        if len(tickers_with_feat) <= 10:
            print(f"  Tickers: {list(tickers_with_feat)}")
    
    print("\n" + "="*70)
    print("ANALYSIS 3: Are these features actually predictive or spuriously correlated?")
    print("="*70)
    
    # A feature with 99% correlation could be:
    # 1. Actual leakage (future info)
    # 2. Spurious correlation from very sparse data (only a few samples have it)
    # 3. Legitimate strong predictor
    
    print("\nChecking if high correlation is due to data sparsity...")
    
    for feat in suspicious_features[:3]:
        if feat not in labeled.columns:
            continue
            
        print(f"\n{feat}:")
        
        # Get samples with this feature
        mask = labeled[feat].notna() & (labeled[feat] != 0)
        n_samples = mask.sum()
        
        # Compare target rate in samples WITH vs WITHOUT this feature
        target_rate_with = labeled[mask][TARGET].mean() if n_samples > 0 else 0
        target_rate_without = labeled[~mask][TARGET].mean()
        
        print(f"  Samples with feature:    {n_samples:,}, target rate = {target_rate_with:.2%}")
        print(f"  Samples without feature: {(~mask).sum():,}, target rate = {target_rate_without:.2%}")
        
        # This is the key insight:
        # If the feature is extremely sparse but the samples that HAVE it 
        # also happen to have a different target rate, the correlation 
        # will be artificially high, but not predictive for new data.
        
        if n_samples < 1000:
            print(f"  ⚠️  VERY SPARSE: Only {n_samples} samples have this feature")
            print(f"     High correlation is likely spurious due to small sample size!")
    
    print("\n" + "="*70)
    print("ANALYSIS 4: Time distribution of suspicious features")
    print("="*70)
    
    feat = 'trailingDividendIncome'
    if feat in labeled.columns:
        mask = labeled[feat].notna() & (labeled[feat] != 0)
        subset = labeled[mask].copy()
        subset['year'] = pd.to_datetime(subset[TIMESTAMP] / 1000, unit='s').dt.year
        
        print(f"\n{feat} by year:")
        yearly = subset.groupby('year').agg({
            TARGET: ['count', 'mean']
        })
        yearly.columns = ['count', 'target_rate']
        for year, row in yearly.iterrows():
            print(f"  {year}: {int(row['count']):,} samples, target rate = {row['target_rate']:.2%}")
    
    print("\n" + "="*70)
    print("CONCLUSION: What's causing the performance gap?")
    print("="*70)
    
    print("""
The high correlations (0.99+) are likely caused by SPARSE FEATURES:
- Features like 'trailingDividendIncome' have very few non-zero values
- When only a small subset of samples has a feature, and that subset
  happens to have a different target distribution, the correlation
  appears artificially high
- This is NOT real predictive power and NOT leakage per se

HOWEVER, the key question remains: Why did nzx-predictor achieve 1+ Sharpe?

Possible explanations for the performance GAP:
1. NZX-predictor fitted scaler on train+test (documented in comments)
   - This would help during the test period but not explain 1+ Sharpe
   
2. NZX-predictor may have had DIFFERENT train/test split methodology
   - Check: Did it use a single split or rolling windows?
   - Check: How was the test period defined?
   
3. NZX-predictor may have used DIFFERENT evaluation metrics
   - Sharpe calculation differences
   - Position sizing differences
   - Holding period differences
   
4. NZX-predictor may have had TRUE LEAKAGE that we fixed
   - If the old pipeline had leakage, that's WHY it performed better
   - Removing the leakage reveals the TRUE (lower) performance
   
5. Different feature sets or preprocessing
   - The old pipeline may have had more/fewer features
   - Different handling of missing values
""")
    
    return labeled


def investigate_sharpe_calculation():
    """
    Compare Sharpe calculation approaches to ensure they're equivalent.
    """
    print("\n" + "="*70)
    print("ANALYSIS 5: Sharpe Ratio Calculation Methodology")
    print("="*70)
    
    # Current pipeline Sharpe calculation
    print("""
Current pipeline Sharpe calculation (from simulator.py):
- Mean return across all trades
- Std of returns across all trades  
- Sharpe = mean / std (no annualization adjustment shown)

If nzx-predictor used annualized Sharpe:
- Sharpe_annual = Sharpe * sqrt(252) for daily returns
- Or Sharpe * sqrt(12) for monthly returns

A daily Sharpe of 0.086 annualized would be: 0.086 * sqrt(252) = 1.36
A monthly Sharpe of 0.086 annualized would be: 0.086 * sqrt(12) = 0.30

QUESTION: Is the nzx-predictor Sharpe of 1+ an ANNUALIZED figure?
""")
    
    print("\nTo verify, we need to know:")
    print("1. What time period was each trade? (1 day? 1 month? 1 year?)")
    print(f"2. Current LOOKAHEAD_DAYS = {LOOKAHEAD_DAYS} (about 1 year)")
    print("3. If trades are annual, no annualization multiplier needed")
    print("4. If the old pipeline used different lookahead, Sharpe isn't comparable")


def investigate_feature_removal_impact():
    """
    What happens if we remove the sparse/suspicious features?
    """
    print("\n" + "="*70)
    print("ANALYSIS 6: Feature Sparsity Summary")
    print("="*70)
    
    long_df = load_long_data()
    long_df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS]
    long_df = clean_and_classify_tickers(long_df)
    long_df = add_macro_prefix(long_df)
    wide_df = long_to_wide(long_df)
    
    excluded = [TIMESTAMP, TICKER]
    feature_cols = [c for c in wide_df.columns if c not in excluded]
    
    sparsity_summary = []
    for col in feature_cols:
        non_zero_rate = (wide_df[col] != 0).mean()
        non_null_rate = wide_df[col].notna().mean()
        sparsity_summary.append({
            'column': col,
            'non_zero_rate': non_zero_rate,
            'non_null_rate': non_null_rate
        })
    
    df_summary = pd.DataFrame(sparsity_summary)
    
    print(f"\nFeature sparsity distribution:")
    print(f"  Features > 90% zero: {(df_summary['non_zero_rate'] < 0.10).sum()}")
    print(f"  Features > 80% zero: {(df_summary['non_zero_rate'] < 0.20).sum()}")
    print(f"  Features > 50% zero: {(df_summary['non_zero_rate'] < 0.50).sum()}")
    print(f"  Features < 50% zero: {(df_summary['non_zero_rate'] >= 0.50).sum()}")
    
    print("\nMost sparse features (>99% zero):")
    very_sparse = df_summary[df_summary['non_zero_rate'] < 0.01].sort_values('non_zero_rate')
    for _, row in very_sparse.head(20).iterrows():
        print(f"  {row['column'][:50]:50s}: {row['non_zero_rate']*100:.2f}% non-zero")
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Very sparse (>99% zero): {len(very_sparse)}")
    print(f"\n⚠️  These {len(very_sparse)} sparse features may cause spurious correlations!")


if __name__ == "__main__":
    investigate_high_correlation_features()
    investigate_sharpe_calculation()
    investigate_feature_removal_impact()
