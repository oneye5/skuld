"""Quick debug script to test anomaly detection on real data."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from core.preprocessor import (
    detect_price_anomalies,
    filter_anomalous_data,
    get_anomaly_summary,
)
from config.columns import TIMESTAMP, TICKER
from config.settings import YEAR_2000_MS


def main():
    print("=" * 60)
    print("ANOMALY DETECTION TEST ON REAL DATA")
    print("(Trims OLD data before discontinuity, keeps newer series)")
    print("=" * 60)
    
    # Load and convert data
    print("\n1. Loading data...")
    long_df = load_long_data()
    
    # Filter to post-2000
    long_df = long_df[long_df[TIMESTAMP] >= YEAR_2000_MS].copy()
    print(f"   Loaded {len(long_df):,} rows (post-2000)")
    
    # Clean and convert to wide
    print("\n2. Converting to wide format...")
    long_df = clean_and_classify_tickers(long_df)
    long_df = add_macro_prefix(long_df)
    wide_df = long_to_wide(long_df)
    
    # Filter to non-macro tickers
    non_macro = wide_df[~wide_df[TICKER].str.startswith('MACRO_')]
    print(f"   Wide format: {len(wide_df):,} rows, {wide_df[TICKER].nunique()} tickers")
    print(f"   Non-macro: {len(non_macro):,} rows, {non_macro[TICKER].nunique()} tickers")
    
    # Detect anomalies
    print("\n3. Detecting price anomalies (threshold: 200%)...")
    result = detect_price_anomalies(
        non_macro,
        price_col='Close',
        return_threshold=2.0,  # 200%
    )
    
    # Get summary
    summary = get_anomaly_summary(result)
    print(f"\n=== ANOMALY SUMMARY ===")
    print(f"   Total rows:          {summary['total_rows']:,}")
    print(f"   Anomaly points:      {summary['anomaly_rows']:,}")
    print(f"   Rows to TRIM:        {summary['rows_to_trim']:,}")
    print(f"   Trim percentage:     {summary['trim_pct']:.2f}%")
    print(f"   Affected tickers:    {summary['n_affected_tickers']}")
    
    if summary['n_affected_tickers'] > 0:
        print(f"\n   Affected tickers: {summary['affected_tickers']}")
        
        if 'max_return' in summary:
            print(f"\n   Max extreme return:  {summary['max_return']:.1%}")
            print(f"   Min extreme return:  {summary['min_return']:.1%}")
    
    # Show what data will be trimmed for each ticker
    if summary['n_affected_tickers'] > 0:
        print("\n4. Per-ticker trim details:")
        for ticker in summary['affected_tickers'][:10]:  # Limit to 10
            ticker_data = result[result[TICKER] == ticker]
            anomaly_ts = ticker_data['_anomaly_timestamp'].iloc[0]
            before = len(ticker_data[ticker_data[TIMESTAMP] < anomaly_ts])
            after = len(ticker_data[ticker_data[TIMESTAMP] >= anomaly_ts])
            anomaly_date = pd.to_datetime(anomaly_ts, unit='ms').date()
            print(f"   {ticker}: trim {before} rows before {anomaly_date}, keep {after} rows")
    
    # Filter and show impact
    print("\n5. Filtering (trimming old data)...")
    filtered, removed = filter_anomalous_data(result, trim_before_anomaly=True)
    
    print(f"   Before filtering:  {len(result):,} rows")
    print(f"   After filtering:   {len(filtered):,} rows")
    print(f"   Removed:           {len(removed):,} rows ({len(removed)/len(result)*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
