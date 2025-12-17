from pathlib import Path
import pandas as pd
from src.config.config import *
from src.tests.utils import print_sample_data
from src.utils.csv_utils import load_csv, save_csv


def long_to_wide_and_impute(long_csv_path: str, imputed_csv_path: str):
    """
    Convert long-format CSV to wide-format, forward-fill missing values,
    fill remaining missing values with 0, and add _present columns for each feature.
    - Macro data (no ticker) forward-filled globally
    - Ticker-specific data forward-filled per ticker
    - Macro features merged onto ticker rows using as-of alignment
    """
    df = load_csv(long_csv_path)

    # Split macro and ticker data - use boolean indexing with .loc for clarity
    is_macro = df['ticker'].isna()
    df_macro = df.loc[is_macro].sort_values(TIMESTAMP_COL)
    df_ticker = df.loc[~is_macro].sort_values([TICKER_COL, TIMESTAMP_COL])

    # ===== MACRO =====
    df_macro_wide = df_macro.pivot_table(
        index=TIMESTAMP_COL,
        columns='feature',
        values='value',
        aggfunc='first'  # Explicit aggregation function
    )

    # Create present flags before fill (more efficient)
    df_macro_present = df_macro_wide.notna().astype('int8').add_suffix('_present')

    # Use fillna with method parameter (slightly faster than chaining)
    df_macro_wide = df_macro_wide.fillna(method='ffill').fillna(0)

    # Reset for as-of merge
    df_macro_wide.reset_index(inplace=True)
    df_macro_present.reset_index(inplace=True)

    # ===== TICKER =====
    df_ticker_wide = df_ticker.pivot_table(
        index=[TIMESTAMP_COL, TICKER_COL],
        columns='feature',
        values='value',
        aggfunc='first'
    )

    # Create present flags before ffill
    df_ticker_present = df_ticker_wide.notna().astype('int8').add_suffix('_present')

    # Forward fill per ticker using transform (faster than groupby + ffill)
    df_ticker_wide = df_ticker_wide.groupby(level=TICKER_COL, group_keys=False).apply(
        lambda x: x.fillna(method='ffill')
    ).fillna(0)

    # Reset multi-index for merging
    df_ticker_wide.reset_index(inplace=True)
    df_ticker_present.reset_index(inplace=True)

    # Sort only once after reset (already sorted by construction, but ensure)
    df_ticker_wide.sort_values(TIMESTAMP_COL, inplace=True)
    df_ticker_present.sort_values(TIMESTAMP_COL, inplace=True)

    # ===== AS-OF MERGE =====
    df_final = pd.merge_asof(
        df_ticker_wide,
        df_macro_wide,
        on=TIMESTAMP_COL,
        direction='backward'
    )

    # Merge present flags directly instead of separate merge + loop
    df_final = pd.merge_asof(
        df_final,
        df_macro_present,
        on=TIMESTAMP_COL,
        direction='backward',
        suffixes=('', '_macro_present')
    )

    # Add ticker present flags using join (faster than loop)
    present_cols = [col for col in df_ticker_present.columns
                    if col not in [TIMESTAMP_COL, TICKER_COL]]
    df_final = df_final.merge(
        df_ticker_present[[TIMESTAMP_COL, TICKER_COL] + present_cols],
        on=[TIMESTAMP_COL, TICKER_COL],
        how='left'
    )

    # Remove column index name
    df_final.columns.name = None

    print_sample_data(df_final)
    save_csv(df_final, imputed_csv_path)
    print(f"Imputed wide CSV saved to {imputed_csv_path}")


if __name__ == "__main__":
    print("Loading from:", LONG_CSV_PATH)
    print("Saving to:   ", WIDE_CSV_PATH)

    long_to_wide_and_impute(str(LONG_CSV_PATH), str(WIDE_CSV_PATH))