"""Feature configuration for experimentation.

This module defines feature sets that can be toggled on/off for A/B testing.
Each feature set is self-contained and can be independently enabled.

Usage:
    from features.feature_config import FEATURE_SETS, get_enabled_features
    
    # Get all enabled feature functions
    feature_funcs = get_enabled_features(["base", "extended_momentum"])
    
    # Apply features
    for func in feature_funcs:
        df = func(df)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import pandas as pd


@dataclass
class FeatureSet:
    """Definition of a feature set for experimentation."""
    name: str
    description: str
    enabled: bool = True
    # List of feature column patterns this set produces
    produces: list[str] = field(default_factory=list)


# =============================================================================
# FEATURE SET DEFINITIONS
# =============================================================================

FEATURE_SETS = {
    # BASE FEATURES (always on)
    "base_technical": FeatureSet(
        name="base_technical",
        description="Core technical indicators: RSI, MACD, ATR, Bollinger",
        enabled=True,
        produces=["RSI_14", "MACD_*", "ATR_14", "NATR_14", "BB_Width_20"],
    ),
    
    "base_momentum": FeatureSet(
        name="base_momentum",
        description="Short-term momentum: ROC_10, lagged returns",
        enabled=True,
        produces=["ROC_10", "Ret_Lag_*"],
    ),
    
    "base_trend": FeatureSet(
        name="base_trend",
        description="Trend indicators: MA distances, 52-week position",
        enabled=True,
        produces=["Dist_MA_*", "Pos_52w_Range", "Dist_52w_High"],
    ),
    
    "base_volatility": FeatureSet(
        name="base_volatility",
        description="Volatility: rolling std, price ratios",
        enabled=True,
        produces=["Vol_20", "Vol_252"],
    ),
    
    "base_cross_sectional": FeatureSet(
        name="base_cross_sectional",
        description="Cross-sectional ranks of key features",
        enabled=True,
        produces=["Rank_*"],
    ),
    
    # EXTENDED FEATURES (experimental)
    "extended_momentum": FeatureSet(
        name="extended_momentum",
        description="Long-term momentum: ROC_126, ROC_252, ROC_378",
        enabled=False,  # Disabled by default - test manually
        produces=["ROC_126", "ROC_252", "ROC_378"],
    ),
    
    "volatility_adjusted_momentum": FeatureSet(
        name="volatility_adjusted_momentum",
        description="Risk-adjusted momentum: returns/volatility",
        enabled=False,
        produces=["Risk_Adj_ROC_*", "Momentum_Sharpe_*"],
    ),
    
    "value_features": FeatureSet(
        name="value_features",
        description="Mean-reversion: distance from 52w mid, 5Y mean",
        enabled=False,
        produces=["Dist_52w_Mid", "Reversion_Score"],
    ),
    
    "interaction_features": FeatureSet(
        name="interaction_features",
        description="Feature interactions: momentum*volatility, etc.",
        enabled=False,
        produces=["Mom_x_Vol", "RSI_x_Pos52w", "MA_Trend_Strength"],
    ),
}


# =============================================================================
# EXPERIMENTAL FEATURE IMPLEMENTATIONS
# =============================================================================

def add_extended_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extended momentum features (6m, 12m, 18m).
    
    Research suggests longer-term momentum is predictive for annual horizons.
    """
    from config.columns import CLOSE, TICKER, TIMESTAMP
    from config.settings import EPSILON
    
    if TICKER not in df.columns or CLOSE not in df.columns:
        return df
    
    df = df.sort_values([TICKER, TIMESTAMP])
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        
        if len(ticker_df) < 380:  # Need ~18 months of data
            result_dfs.append(ticker_df)
            continue
        
        # Extended ROC
        for window in [126, 252, 378]:  # 6m, 12m, 18m
            ticker_df[f"ROC_{window}"] = ticker_df[CLOSE].pct_change(periods=window) * 100
        
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        return df
    
    return pd.concat(result_dfs).sort_values(TIMESTAMP).reset_index(drop=True)


def add_volatility_adjusted_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-adjusted momentum features.
    
    Risk-adjusted momentum (like Sharpe ratio for momentum) may be more stable.
    """
    from config.columns import CLOSE, TICKER, TIMESTAMP
    from config.settings import EPSILON
    
    if TICKER not in df.columns or CLOSE not in df.columns:
        return df
    
    df = df.sort_values([TICKER, TIMESTAMP])
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        
        if len(ticker_df) < 260:
            result_dfs.append(ticker_df)
            continue
        
        returns = ticker_df[CLOSE].pct_change()
        
        # Rolling Sharpe-like momentum
        for window in [63, 126, 252]:
            rolling_ret = returns.rolling(window).mean()
            rolling_vol = returns.rolling(window).std()
            
            ticker_df[f"Risk_Adj_ROC_{window}"] = rolling_ret / (rolling_vol + EPSILON)
            ticker_df[f"Momentum_Sharpe_{window}"] = (
                rolling_ret * (252 ** 0.5) / (rolling_vol + EPSILON)
            )
        
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        return df
    
    return pd.concat(result_dfs).sort_values(TIMESTAMP).reset_index(drop=True)


def add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add value/mean-reversion features.
    
    Distance from historical midpoints may predict reversion.
    """
    from config.columns import CLOSE, HIGH, LOW, TICKER, TIMESTAMP
    from config.settings import EPSILON
    
    if TICKER not in df.columns or CLOSE not in df.columns:
        return df
    
    df = df.sort_values([TICKER, TIMESTAMP])
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        
        if len(ticker_df) < 260:
            result_dfs.append(ticker_df)
            continue
        
        # 52-week midpoint distance
        rolling_high = ticker_df[HIGH].rolling(252).max()
        rolling_low = ticker_df[LOW].rolling(252).min()
        midpoint = (rolling_high + rolling_low) / 2
        range_size = rolling_high - rolling_low
        
        ticker_df["Dist_52w_Mid"] = (ticker_df[CLOSE] - midpoint) / (range_size + EPSILON)
        
        # Reversion score: high past return + high volatility = expect reversion
        roc_252 = ticker_df[CLOSE].pct_change(252)
        vol_252 = ticker_df[CLOSE].pct_change().rolling(252).std()
        ticker_df["Reversion_Score"] = -roc_252 * vol_252
        
        result_dfs.append(ticker_df)
    
    if not result_dfs:
        return df
    
    return pd.concat(result_dfs).sort_values(TIMESTAMP).reset_index(drop=True)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature interaction terms.
    
    Interactions may capture non-linear patterns missed by individual features.
    """
    from config.settings import EPSILON
    
    # Momentum × Volatility (high momentum + high vol = stronger signal or noise?)
    if "ROC_252" in df.columns and "Vol_252" in df.columns:
        df["Mom_x_Vol"] = df["ROC_252"] * df["Vol_252"]
    
    # RSI × 52-week Position (overbought at highs is more bearish)
    if "RSI_14" in df.columns and "Pos_52w_Range" in df.columns:
        df["RSI_x_Pos52w"] = df["RSI_14"] * df["Pos_52w_Range"]
    
    # MA Trend Alignment (are short and long trends aligned?)
    if "Dist_MA_50" in df.columns and "Dist_MA_200" in df.columns:
        df["MA_Trend_Strength"] = df["Dist_MA_50"] * df["Dist_MA_200"]
    
    return df


# =============================================================================
# FEATURE REGISTRY
# =============================================================================

# Map feature set names to their implementation functions
FEATURE_FUNCTIONS = {
    "extended_momentum": add_extended_momentum_features,
    "volatility_adjusted_momentum": add_volatility_adjusted_momentum,
    "value_features": add_value_features,
    "interaction_features": add_interaction_features,
}


def get_enabled_features() -> list[str]:
    """Get list of enabled feature set names."""
    return [name for name, fs in FEATURE_SETS.items() if fs.enabled]


def enable_feature_set(name: str) -> None:
    """Enable a feature set."""
    if name in FEATURE_SETS:
        FEATURE_SETS[name].enabled = True
    else:
        raise ValueError(f"Unknown feature set: {name}")


def disable_feature_set(name: str) -> None:
    """Disable a feature set."""
    if name in FEATURE_SETS:
        FEATURE_SETS[name].enabled = False
    else:
        raise ValueError(f"Unknown feature set: {name}")


def apply_experimental_features(df: pd.DataFrame, feature_sets: list[str]) -> pd.DataFrame:
    """Apply specified experimental feature sets to DataFrame.
    
    Args:
        df: DataFrame with base features already applied.
        feature_sets: List of feature set names to apply.
        
    Returns:
        DataFrame with experimental features added.
    """
    for name in feature_sets:
        if name in FEATURE_FUNCTIONS:
            print(f"  Adding feature set: {name}")
            df = FEATURE_FUNCTIONS[name](df)
    
    return df
