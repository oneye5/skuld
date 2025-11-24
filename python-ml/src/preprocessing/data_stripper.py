from src.config.config import *
from src.utils.csv_utils import *


def strip_data(in_csv_path: str, out_csv_path: str):
    df = load_csv(in_csv_path)
    keep_columns = [TIMESTAMP_COL, PREDICTION_COL, TICKER_COL]

    # Keep only the specified columns
    df_stripped = df[keep_columns].copy()

    save_csv(df_stripped, out_csv_path)

    print(f"Stripped data saved to: {out_csv_path}")
    print(f"Kept {len(keep_columns)} columns: {keep_columns}")