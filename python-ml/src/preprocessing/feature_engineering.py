"""Feature engineering and data scaling operations.

Provides RobustScaler-based scaling that:
- Prevents data leakage by fitting only on training data
- Preserves binary and categorical columns
- Validates feature alignment between train/test
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Optional, List, Tuple

from src.config.config import *


def scale_data_with_scaler(
    df: pd.DataFrame,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = False,
    continuous_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[RobustScaler], List[str]]:
    """Scale continuous features using RobustScaler (resistant to outliers).
    
    Preserves binary columns (0/1) and non-numeric columns unchanged.
    Can either fit a new scaler (training) or apply existing one (testing).
    
    Args:
        df: Input DataFrame.
        scaler: Pre-fitted RobustScaler. Required if fit_scaler=False.
        fit_scaler: Whether to fit scaler on this data (training mode).
                   If False, applies provided scaler (test mode).
        continuous_cols: List of columns to scale. If None, infers from data.
    
    Returns:
        Tuple[pd.DataFrame, RobustScaler, List[str]]: Scaled data, fitted scaler,
        and list of columns that were scaled.
    
    Raises:
        ValueError: If fit_scaler=False but scaler is None, or if column count mismatch.
    """
    if not fit_scaler and scaler is None:
        raise ValueError("Must provide fitted scaler when fit_scaler=False")
    
    df = df.copy()
    
    # Identify binary columns (only scale if explicitly numeric)
    true_binary_cols = []
    for col in df.columns:
        if df[col].dtype in ['int8', 'uint8', 'int16', 'uint16']:
            try:
                unique_vals = set(df[col].dropna().unique())
                if len(unique_vals) <= 2 and unique_vals.issubset({0, 1}):
                    true_binary_cols.append(col)
            except:
                pass

    # Define continuous columns: exclude special columns and binary features
    inferred_continuous_cols = [
        col for col in df.columns
        if col not in true_binary_cols
        and col not in [TIMESTAMP_COL, LABEL_COL, CLOSE_COL, TIMESTAMP_SCALED_COL]
        and not col.startswith(TICKER_PREFIX)
        and df[col].dtype in ['int32', 'int64', 'float32', 'float64']
    ]

    # Use provided columns or inferred ones
    if continuous_cols is not None:
        missing = [c for c in continuous_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Provided columns missing from DataFrame: {missing}")
        continuous_cols = list(continuous_cols)
    else:
        continuous_cols = inferred_continuous_cols

    if not continuous_cols:
        return df, scaler, continuous_cols

    # Ensure floating point
    df[continuous_cols] = df[continuous_cols].astype(float)

    if fit_scaler:
        scaler = RobustScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols].values)
    else:
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(continuous_cols):
            raise ValueError(
                f"Scaler expects {scaler.n_features_in_} features, got {len(continuous_cols)}"
            )
        df[continuous_cols] = scaler.transform(df[continuous_cols].values)

    return df, scaler, continuous_cols


def scale_continuous_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale continuous features using RobustScaler.
    
    WARNING: Only use this before train/test split. For proper handling
    after split, use scale_data_with_scaler() with fit_scaler control.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with scaled continuous features.
    """
    df_scaled, _, _ = scale_data_with_scaler(df, scaler=None, fit_scaler=True)
    return df_scaled


def to_feature_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pre-split feature engineering transformations.
    Adds financial ratios, growth rates, per-share metrics, labor/population ratios, interest coverage, macro spreads, and rolling/statistical features.
    Scaling is NOT applied here to prevent data leakage.
    Scaling happens separately in post_split_preprocessing.
    Args:
        df: Input DataFrame.
    Returns:
        pd.DataFrame: Feature engineered DataFrame (no scaling).
    """
    df = df.copy()

    # --- Financial Ratios ---
    if ANNUAL_BASIC_EPS_COL in df.columns and ANNUAL_DILUTED_EPS_COL in df.columns:
        df['eps_ratio'] = df[ANNUAL_BASIC_EPS_COL] / df[ANNUAL_DILUTED_EPS_COL].replace(0, np.nan)
    if TRAILING_BASIC_EPS_COL in df.columns and TRAILING_DILUTED_EPS_COL in df.columns:
        df['trailing_eps_ratio'] = df[TRAILING_BASIC_EPS_COL] / df[TRAILING_DILUTED_EPS_COL].replace(0, np.nan)
    if ANNUAL_NET_INCOME_COL in df.columns and ANNUAL_TOTAL_REVENUE_COL in df.columns:
        df['profit_margin'] = df[ANNUAL_NET_INCOME_COL] / df[ANNUAL_TOTAL_REVENUE_COL].replace(0, np.nan)
    if TRAILING_NET_INCOME_COL in df.columns and TRAILING_TOTAL_REVENUE_COL in df.columns:
        df['trailing_profit_margin'] = df[TRAILING_NET_INCOME_COL] / df[TRAILING_TOTAL_REVENUE_COL].replace(0, np.nan)
    if ANNUAL_TOTAL_UNUSUAL_ITEMS_COL in df.columns and ANNUAL_TOTAL_REVENUE_COL in df.columns:
        df['unusual_items_ratio'] = df[ANNUAL_TOTAL_UNUSUAL_ITEMS_COL] / df[ANNUAL_TOTAL_REVENUE_COL].replace(0, np.nan)
    if TRAILING_TOTAL_UNUSUAL_ITEMS_COL in df.columns and TRAILING_TOTAL_REVENUE_COL in df.columns:
        df['trailing_unusual_items_ratio'] = df[TRAILING_TOTAL_UNUSUAL_ITEMS_COL] / df[TRAILING_TOTAL_REVENUE_COL].replace(0, np.nan)
    if ANNUAL_GA_EXPENSE_COL in df.columns and ANNUAL_TOTAL_REVENUE_COL in df.columns:
        df['ga_expense_ratio'] = df[ANNUAL_GA_EXPENSE_COL] / df[ANNUAL_TOTAL_REVENUE_COL].replace(0, np.nan)
    if TRAILING_GA_EXPENSE_COL in df.columns and TRAILING_TOTAL_REVENUE_COL in df.columns:
        df['trailing_ga_expense_ratio'] = df[TRAILING_GA_EXPENSE_COL] / df[TRAILING_TOTAL_REVENUE_COL].replace(0, np.nan)

    # --- Growth Rates ---
    if ANNUAL_NET_INCOME_COL in df.columns and TRAILING_NET_INCOME_COL in df.columns:
        df['net_income_growth'] = (df[ANNUAL_NET_INCOME_COL] - df[TRAILING_NET_INCOME_COL]) / df[TRAILING_NET_INCOME_COL].replace(0, np.nan)
    if ANNUAL_TOTAL_REVENUE_COL in df.columns and TRAILING_TOTAL_REVENUE_COL in df.columns:
        df['revenue_growth'] = (df[ANNUAL_TOTAL_REVENUE_COL] - df[TRAILING_TOTAL_REVENUE_COL]) / df[TRAILING_TOTAL_REVENUE_COL].replace(0, np.nan)
    if ANNUAL_EBITDA_COL in df.columns and TRAILING_EBITDA_COL in df.columns:
        df['ebitda_growth'] = (df[ANNUAL_EBITDA_COL] - df[TRAILING_EBITDA_COL]) / df[TRAILING_EBITDA_COL].replace(0, np.nan)

    # --- Per-Share Metrics ---
    if ANNUAL_NET_INCOME_COL in df.columns and ANNUAL_DILUTED_AVG_SHARES_COL in df.columns:
        df['net_income_per_share'] = df[ANNUAL_NET_INCOME_COL] / df[ANNUAL_DILUTED_AVG_SHARES_COL].replace(0, np.nan)
    if ANNUAL_EBITDA_COL in df.columns and ANNUAL_DILUTED_AVG_SHARES_COL in df.columns:
        df['ebitda_per_share'] = df[ANNUAL_EBITDA_COL] / df[ANNUAL_DILUTED_AVG_SHARES_COL].replace(0, np.nan)

    # --- Labor/Population Ratios ---
    if NZL_EMP_Y15T64_T_COL in df.columns and NZL_POP_Y15T64_T_COL in df.columns:
        df['employment_rate'] = df[NZL_EMP_Y15T64_T_COL] / df[NZL_POP_Y15T64_T_COL].replace(0, np.nan)
    if NZL_LF_Y15T64_T_COL in df.columns and NZL_POP_Y15T64_T_COL in df.columns:
        df['labor_force_participation'] = df[NZL_LF_Y15T64_T_COL] / df[NZL_POP_Y15T64_T_COL].replace(0, np.nan)

    # --- Interest Coverage ---
    if ANNUAL_EBIT_COL in df.columns and ANNUAL_INTEREST_EXPENSE_COL in df.columns:
        df['interest_coverage'] = df[ANNUAL_EBIT_COL] / df[ANNUAL_INTEREST_EXPENSE_COL].replace(0, np.nan)

    # --- Macro/Market Features ---
    if LONG_TERM_RATE_COL in df.columns and SHORT_TERM_RATE_COL in df.columns:
        df['interest_rate_spread'] = df[LONG_TERM_RATE_COL] - df[SHORT_TERM_RATE_COL]

    # --- Rolling/statistical features (only for ticker rows) ---
    if TICKER_COL in df.columns and CLOSE_COL in df.columns:
        # Sort for rolling operations
        df = df.sort_values([TICKER_COL, TIMESTAMP_COL])
        group = df.groupby(TICKER_COL, group_keys=False)

        # Rolling window size (e.g., 5, 10, 20 periods)
        windows = [5, 10, 20]
        for w in windows:
            # Rolling mean, std, min, max of Close
            df[f'close_mean_{w}'] = group[CLOSE_COL].transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f'close_std_{w}'] = group[CLOSE_COL].transform(lambda x: x.rolling(w, min_periods=1).std())
            df[f'close_min_{w}'] = group[CLOSE_COL].transform(lambda x: x.rolling(w, min_periods=1).min())
            df[f'close_max_{w}'] = group[CLOSE_COL].transform(lambda x: x.rolling(w, min_periods=1).max())

            # Rolling returns (percentage change)
            df[f'return_{w}'] = group[CLOSE_COL].transform(lambda x: x.pct_change(w).fillna(0))

            # Rolling volatility (std of returns)
            df[f'volatility_{w}'] = group[CLOSE_COL].transform(lambda x: x.pct_change().rolling(w, min_periods=1).std().fillna(0))

        # Momentum (rate of change over 10 periods)
        df['momentum_10'] = group[CLOSE_COL].transform(lambda x: x.pct_change(periods=10).fillna(0))

        # Lag features (previous 1, 2, 3 closes)
        for lag in [1, 2, 3]:
            df[f'close_lag_{lag}'] = group[CLOSE_COL].transform(lambda x: x.shift(lag))

        # Optional: rolling skew/kurtosis
        for w in [10, 20]:
            df[f'close_skew_{w}'] = group[CLOSE_COL].transform(lambda x: x.rolling(w, min_periods=1).skew())
            df[f'close_kurt_{w}'] = group[CLOSE_COL].transform(lambda x: x.rolling(w, min_periods=1).kurt())

    return df