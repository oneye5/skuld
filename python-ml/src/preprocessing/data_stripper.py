"""Utility to strip predictions to essential columns for output."""
from src.config.config import *
from src.utils.csv_utils import load_csv, save_csv


def strip_data(in_csv_path: str, out_csv_path: str) -> None:
    """
    Keep only essential columns for final predictions output.
    
    Retains: timestamp, prediction probability, and ticker.
    Discards all other features.
    
    Args:
        in_csv_path: Path to input CSV with all columns.
        out_csv_path: Path to save stripped CSV.
    """
    df = load_csv(in_csv_path)
    keep_columns = [TIMESTAMP_COL, PREDICTION_COL, TICKER_COL]

    # Keep only the specified columns
    df_stripped = df[keep_columns].copy()

    save_csv(df_stripped, out_csv_path)

    print(f"Stripped data saved to: {out_csv_path}")
    print(f"Kept {len(keep_columns)} columns: {keep_columns}")