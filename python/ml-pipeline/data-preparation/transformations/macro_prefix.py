"""Module for adding MACRO_ prefix to macro (empty ticker) features."""

import pandas as pd

from config.column_names import TICKER, FEATURE, MACRO_PREFIX


def add_macro_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACRO_ prefix to features where ticker is empty.
    
    Macro data is identified by having an empty ticker value.
    This prefixing allows differentiation during scaling (macro features 
    are scaled globally, ticker features are scaled per-ticker).
    
    Args:
        df: Long format DataFrame with ticker and feature columns.
    
    Returns:
        DataFrame with MACRO_ prefix added to macro features.
    """
    df = df.copy()
    
    is_macro = df[TICKER] == ""
    df.loc[is_macro, FEATURE] = MACRO_PREFIX + df.loc[is_macro, FEATURE]
    
    return df
