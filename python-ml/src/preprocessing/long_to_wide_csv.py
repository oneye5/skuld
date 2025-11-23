from pathlib import Path
import pandas as pd
from src.config.config import *
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

    # Split macro and ticker data
    df_macro = df[df['ticker'].isna()].sort_values(TIMESTAMP_COL)
    df_ticker = df[df['ticker'].notna()].sort_values([TICKER_COL, TIMESTAMP_COL])

    # ===== MACRO =====
    df_macro_wide = df_macro.pivot_table(
        index=TIMESTAMP_COL,
        columns='feature',
        values='value'
    )
    df_macro_present = df_macro_wide.notna().astype(int).add_suffix('_present')
    df_macro_wide = df_macro_wide.ffill().fillna(0)  # global ffill

    # Reset for as-of merge
    df_macro_wide = df_macro_wide.reset_index().sort_values(TIMESTAMP_COL)
    df_macro_present = df_macro_present.reset_index().sort_values(TIMESTAMP_COL)

    # ===== TICKER =====
    df_ticker_wide = df_ticker.pivot_table(
        index=[TIMESTAMP_COL, TICKER_COL],
        columns='feature',
        values='value'
    )
    df_ticker_present = df_ticker_wide.notna().astype(int).add_suffix('_present')

    # fill per ticker
    df_ticker_wide = df_ticker_wide.groupby(level=TICKER_COL).ffill().fillna(0)

    # Reset multi-index for merging
    df_ticker_wide = df_ticker_wide.reset_index().sort_values(TIMESTAMP_COL)
    df_ticker_present = df_ticker_present.reset_index().sort_values(TIMESTAMP_COL)

    # ===== AS-OF MERGE =====
    df_final = pd.merge_asof(
        df_ticker_wide,
        df_macro_wide,
        on=TIMESTAMP_COL,
        direction='backward'
    )

    df_present_final = pd.merge_asof(
        df_ticker_present,
        df_macro_present,
        on=TIMESTAMP_COL,
        direction='backward'
    )

    # Add present flags into df_final
    for col in df_present_final.columns:
        if col.endswith('_present'):
            df_final[col] = df_present_final[col]

    # Remove column index name
    df_final.columns.name = None

    save_csv(df_final, imputed_csv_path)
    print(f"Imputed wide CSV saved to {imputed_csv_path}")


if __name__ == "__main__":
    print("Loading from:", LONG_CSV_PATH)
    print("Saving to:   ", WIDE_CSV_PATH)

    long_to_wide_and_impute(str(LONG_CSV_PATH), str(WIDE_CSV_PATH))
