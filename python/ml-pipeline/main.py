"""Main entry point for running the ML pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import load_long_data
from runnables import run_rolling_windows, print_summary


def main():
    """Run the complete rolling window pipeline."""
    print("Loading data...")
    long_df = load_long_data()
    print(f"Loaded {len(long_df):,} rows")
    
    print("\nStarting rolling window pipeline...")
    results = run_rolling_windows(long_df)
    
    print_summary(results)
    
    return results


if __name__ == "__main__":
    main()
