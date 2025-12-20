"""Module for scaling features - ticker features per-ticker, macro globally."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config.column_names import TIMESTAMP, TICKER, TARGET, MACRO_PREFIX


@dataclass
class ScalerSet:
    """Container for fitted scalers."""
    macro_scaler: StandardScaler | None
    ticker_scalers: dict[str, StandardScaler]
    macro_columns: list[str]
    ticker_columns: list[str]


def get_macro_columns(df: pd.DataFrame) -> list[str]:
    """Get columns that are macro features (start with MACRO_)."""
    return [col for col in df.columns if col.startswith(MACRO_PREFIX)]


def get_ticker_columns(df: pd.DataFrame) -> list[str]:
    """Get columns that are ticker features (don't start with MACRO_)."""
    exclude = {TIMESTAMP, TICKER, TARGET}
    return [
        col for col in df.columns
        if not col.startswith(MACRO_PREFIX)
        and col not in exclude
        and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]


def fit_scalers(train_df: pd.DataFrame) -> ScalerSet:
    """
    Fit scalers on training data.
    
    Macro features are scaled globally, ticker features are scaled per-ticker.
    
    Args:
        train_df: Training DataFrame with features to scale.
    
    Returns:
        ScalerSet containing fitted scalers.
    """
    macro_cols = get_macro_columns(train_df)
    ticker_cols = get_ticker_columns(train_df)
    
    # Fit macro scaler globally
    macro_scaler = None
    if macro_cols:
        macro_data = train_df[macro_cols].values
        # Handle NaN - fit on non-NaN values only
        valid_mask = ~np.isnan(macro_data).any(axis=1)
        if valid_mask.any():
            macro_scaler = StandardScaler()
            macro_scaler.fit(macro_data[valid_mask])
    
    # Fit ticker scalers per-ticker
    ticker_scalers = {}
    if ticker_cols:
        for ticker in train_df[TICKER].unique():
            ticker_data = train_df[train_df[TICKER] == ticker][ticker_cols].values
            valid_mask = ~np.isnan(ticker_data).any(axis=1)
            if valid_mask.any() and len(ticker_data[valid_mask]) > 0:
                scaler = StandardScaler()
                scaler.fit(ticker_data[valid_mask])
                ticker_scalers[ticker] = scaler
    
    return ScalerSet(
        macro_scaler=macro_scaler,
        ticker_scalers=ticker_scalers,
        macro_columns=macro_cols,
        ticker_columns=ticker_cols,
    )


def transform_data(df: pd.DataFrame, scaler_set: ScalerSet) -> pd.DataFrame:
    """
    Transform data using fitted scalers. Note: Modifies df inplace.
    
    Args:
        df: DataFrame with features to scale.
        scaler_set: ScalerSet containing fitted scalers.
    
    Returns:
        DataFrame with scaled features (same object as input).
    """
    # Get columns that exist in this DataFrame
    macro_cols = [c for c in scaler_set.macro_columns if c in df.columns]
    ticker_cols = [c for c in scaler_set.ticker_columns if c in df.columns]
    
    # Scale macro features globally
    if scaler_set.macro_scaler is not None and macro_cols:
        macro_data = df[macro_cols].values
        # Handle rows with NaN
        valid_mask = ~np.isnan(macro_data).any(axis=1)
        if valid_mask.any():
            scaled_macro = np.full_like(macro_data, np.nan)
            scaled_macro[valid_mask] = scaler_set.macro_scaler.transform(macro_data[valid_mask])
            df[macro_cols] = scaled_macro
    
    # Scale ticker features per-ticker
    if ticker_cols:
        for ticker in df[TICKER].unique():
            mask = df[TICKER] == ticker
            ticker_data = df.loc[mask, ticker_cols].values
            
            if ticker in scaler_set.ticker_scalers:
                scaler = scaler_set.ticker_scalers[ticker]
                valid_mask = ~np.isnan(ticker_data).any(axis=1)
                if valid_mask.any():
                    scaled_ticker = np.full_like(ticker_data, np.nan)
                    scaled_ticker[valid_mask] = scaler.transform(ticker_data[valid_mask])
                    df.loc[mask, ticker_cols] = scaled_ticker
    
    return df


def save_scalers(scaler_set: ScalerSet, output_dir: Path, window_id: int) -> None:
    """Save scalers to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if scaler_set.macro_scaler is not None:
        path = output_dir / f"macro_window{window_id}_scaler.pkl"
        with open(path, 'wb') as f:
            pickle.dump(scaler_set.macro_scaler, f)
    
    for ticker, scaler in scaler_set.ticker_scalers.items():
        # Sanitize ticker name for filename
        safe_ticker = ticker.replace('/', '_').replace('\\', '_')
        path = output_dir / f"{safe_ticker}_window{window_id}_scaler.pkl"
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)


def load_scalers(
    output_dir: Path,
    window_id: int,
    macro_columns: list[str],
    ticker_columns: list[str],
    tickers: list[str],
) -> ScalerSet:
    """Load scalers from disk."""
    macro_scaler = None
    macro_path = output_dir / f"macro_window{window_id}_scaler.pkl"
    if macro_path.exists():
        with open(macro_path, 'rb') as f:
            macro_scaler = pickle.load(f)
    
    ticker_scalers = {}
    for ticker in tickers:
        safe_ticker = ticker.replace('/', '_').replace('\\', '_')
        path = output_dir / f"{safe_ticker}_window{window_id}_scaler.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                ticker_scalers[ticker] = pickle.load(f)
    
    return ScalerSet(
        macro_scaler=macro_scaler,
        ticker_scalers=ticker_scalers,
        macro_columns=macro_columns,
        ticker_columns=ticker_columns,
    )
