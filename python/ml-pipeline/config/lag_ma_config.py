"""Configuration for lag and moving average feature generation.

This module defines which features should have lags, moving averages,
and momentum (rate of change) computed, along with the specific windows
for each.

Structure:
    Each entry specifies:
    - feature_pattern: Regex or exact name to match input columns
    - output_prefix: Prefix for generated feature names (None = use original)
    - lags: List of lag periods (in days)
    - mas: List of moving average windows (in days)
    - momentum: List of momentum windows (% change over period)
    - scope: "ticker" (compute per-ticker) or "global" (timestamp-level only)
    - enabled: Whether this config is active

Example:
    For feature "Wiki_Views" with lags=[7,14], mas=[7], momentum=[7]:
    Generates:
        - Wiki_Views_Lag_7, Wiki_Views_Lag_14
        - Wiki_Views_MA_7
        - Wiki_Views_Mom_7
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple


@dataclass
class FeatureLagMAConfig:
    """Configuration for a single feature or feature pattern."""
    
    feature_pattern: str
    """Regex pattern or exact column name to match."""
    
    output_prefix: Optional[str] = None
    """Prefix for output columns. None uses original feature name."""
    
    lags: List[int] = field(default_factory=list)
    """Lag periods in days. E.g., [7, 14, 28] for 1/2/4 week lags."""
    
    mas: List[int] = field(default_factory=list)
    """Moving average windows in days."""
    
    momentum: List[int] = field(default_factory=list)
    """Momentum (% change) windows in days."""
    
    diffs: List[Tuple[int, int]] = field(default_factory=list)
    """Difference pairs as (window1, window2). Use 0 for current value.
    E.g., [(0, 252), (252, 756)] generates:
        - Current - MA_252 (deviation from 1-year mean)
        - MA_252 - MA_756 (1-year vs 3-year trend difference)
    Research: Mean reversion and trend regime signals."""
    
    scope: Literal["ticker", "global"] = "ticker"
    """'ticker' for per-ticker computation, 'global' for timestamp-level."""
    
    enabled: bool = True
    """Whether this config is active."""
    
    min_periods_ratio: float = 0.5
    """Minimum periods as ratio of window size for MA calculations."""
    
    include_spike: bool = False
    """Whether to add a spike indicator (current / MA_longest)."""
    
    include_volatility: bool = False
    """Whether to add rolling volatility."""
    
    volatility_window: int = 14
    """Window for volatility calculation if include_volatility=True."""


# =============================================================================
# TICKER-LEVEL FEATURES (computed per-ticker using groupby)
# =============================================================================

TICKER_VOLUME_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^DollarVolume$",
    output_prefix="DolVol",
    lags=[63, 126],             # Reduced: removed 252-day lag
    mas=[21, 63, 126, 252, 504, 1008],     # Reduced: removed 756, 1260-day MAs
    momentum=[63, 126],         # Medium-term volume momentum
    diffs=[(0, 126)], # Reduced: keep only top 2 diffs
    scope="ticker",
    include_spike=True,
    include_volatility=True,
    volatility_window=21,       # 21-day vol showed high importance (299)
)

TICKER_AMIHUD_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^Amihud_21d$|^Amihud_63d$",
    output_prefix="Illiq",
    lags=[21, 63],             
    mas=[21, 63, 252, 504, 1008],        
    momentum=[63, 252],              
    diffs=[(0, 63)],            
    scope="ticker",
    include_spike=True,
)

TICKER_VOLATILITY_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^Vol_252$|^Vol_20$",
    output_prefix="VolDyn",
    lags=[126],             
    mas=[63, 126, 252],         
    momentum=[63, 126],        
    diffs=[(0, 252)],           
    scope="ticker",
)

TICKER_TREND_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^TrendPersist_252d$|^Trend_RSq_",
    output_prefix="Trend",
    lags=[63],                  
    mas=[63, 126, 252, 504, 1004],             
    momentum=[63],             
    diffs=[(0, 126)],           
    scope="ticker",
)

TICKER_HIGHER_MOMENTS_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^Skew_126d$|^Kurt_126d$|^Skew_60d$|^Kurt_60d$",
    output_prefix="Moment",
    lags=[63],                  
    mas=[63, 252, 504, 1008],              
    momentum=[63],              
    diffs=[(0, 126)],           
    scope="ticker",
)

TICKER_DIVIDEND_YIELD_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^TrailingDivYield_252d$",
    output_prefix="DivYld",
    lags=[252],             
    mas=[63, 126, 252, 504, 1008],         
    momentum=[1008],             
    diffs=[(0, 252), (0, 1008)],          
    scope="ticker",
)

TICKER_ATR_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^ATR_|^NATR_",
    output_prefix="ATR",
    lags=[63, 126],            # Track lagged volatility regimes
    mas=[63, 126, 252],        # Medium to long-term volatility averages
    momentum=[63, 126],        # Volatility acceleration/deceleration
    diffs=[(0, 126), (0, 252)],  # Deviation from 6mo/12mo mean volatility
    scope="ticker",
    include_spike=True,        # Detect volatility spikes
)


# =============================================================================
# MACRO FEATURES (global, timestamp-level only)
# MEMORY OPTIMIZED: Reduced all windows significantly
# =============================================================================
MACRO_OECD_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_OECD_",
    output_prefix=None,
    lags=[126, 252],            
    mas=[126, 252],             
    momentum=[126],             
    diffs=[(0, 252)],           
    scope="global",
    include_spike=True,
)

MACRO_INTEREST_RATES_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"interest rate|Interest rate",
    output_prefix=None,
    lags=[126, 252],           
    mas=[126, 252],            
    momentum=[126],            
    diffs=[(0, 252)],          
    scope="global",
    include_spike=True,
    include_volatility=True,
    volatility_window=63,
)

MACRO_BOP_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_BOP",
    output_prefix=None,
    lags=[126, 252],           
    mas=[126, 252],            
    momentum=[126],            
    scope="global",
)

MACRO_INDEX_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^\^FTSE|^000001\.SS|^\^TNX",
    output_prefix=None,
    lags=[63, 126],             
    mas=[63, 126, 252],         
    momentum=[63],              
    diffs=[(0, 252)],           
    scope="global",
    include_spike=True,
    include_volatility=True,
    volatility_window=63,
)

MACRO_COMMODITY_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"=F_",    
    output_prefix=None,
    lags=[63, 126],            
    mas=[63, 126],             
    momentum=[63],             
    diffs=[(0, 126)],          
    scope="global",
    include_spike=True,
)

MACRO_FX_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_NZD.*=X_",
    output_prefix=None,
    lags=[63, 126],            
    mas=[63, 126],             
    momentum=[63],             
    scope="global",
    include_volatility=True,
    volatility_window=63,
)

MACRO_NZ_GDP_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_NZL_",
    output_prefix=None,
    lags=[126, 252],           
    mas=[126, 252],            
    momentum=[126],            
    scope="global",
)

MACRO_FAO_FOOD_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_FAO_",
    output_prefix=None,
    lags=[63, 126, 252],       
    mas=[63, 126, 252,1200],   
    momentum=[63, 126, 365],   
    diffs=[(0, 126), (0, 252)],
    scope="global",
    include_spike=True,        
    include_volatility=True,   
    volatility_window=63,      
)

# =============================================================================
# AGGREGATE CONFIGS
# =============================================================================

# All ticker-level configs
TICKER_CONFIGS: List[FeatureLagMAConfig] = [
    TICKER_VOLUME_CONFIG,
    TICKER_AMIHUD_CONFIG,
    TICKER_VOLATILITY_CONFIG,
    TICKER_TREND_CONFIG,
    TICKER_HIGHER_MOMENTS_CONFIG,
    TICKER_DIVIDEND_YIELD_CONFIG,
    TICKER_ATR_CONFIG,
]

# All macro/global configs  
MACRO_CONFIGS: List[FeatureLagMAConfig] = [
    MACRO_OECD_CONFIG,
    MACRO_INTEREST_RATES_CONFIG,
    MACRO_BOP_CONFIG,
    MACRO_INDEX_CONFIG,
    MACRO_COMMODITY_CONFIG,
    MACRO_FX_CONFIG,
    MACRO_NZ_GDP_CONFIG,
    MACRO_FAO_FOOD_CONFIG,
]

# Combined list of all configs
ALL_LAG_MA_CONFIGS: List[FeatureLagMAConfig] = TICKER_CONFIGS + MACRO_CONFIGS


def get_enabled_configs() -> List[FeatureLagMAConfig]:
    """Return only enabled configurations."""
    return [cfg for cfg in ALL_LAG_MA_CONFIGS if cfg.enabled]


def get_ticker_configs() -> List[FeatureLagMAConfig]:
    """Return enabled ticker-level configurations."""
    return [cfg for cfg in TICKER_CONFIGS if cfg.enabled]


def get_macro_configs() -> List[FeatureLagMAConfig]:
    """Return enabled macro/global configurations."""
    return [cfg for cfg in MACRO_CONFIGS if cfg.enabled]
