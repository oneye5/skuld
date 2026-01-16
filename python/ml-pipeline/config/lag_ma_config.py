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

# Wikipedia attention - for 365-day horizon, need longer windows
# Research: Da, Engelberg, Gao (2011) - "In Search of Attention"
# Note: Attention features showing low importance (~33) - simplify to focus on
# long-term smoothing relevant for annual predictions
TICKER_ATTENTION_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^Wiki_Views$|^Wiki_Views_Desktop$|^Wiki_Views_Mobile$",
    output_prefix="Attn",
    lags=[63, 126, 252],       # Quarterly, semi-annual, annual lags
    mas=[63, 126, 252, 756, 1260],  # Up to 5-year MA for market cycles
    momentum=[63, 126],        # Longer momentum windows
    diffs=[(0, 252), (0, 756), (252, 756)],  # Deviation from 1yr/3yr mean, trend regime
    scope="ticker",
    include_spike=True,
    include_volatility=True,
    volatility_window=63,      # Quarterly volatility for stability
)

# Dollar volume - tracks liquidity trends
# Research: Amihud (2002) - illiquidity predicts returns
# VERY HIGH IMPORTANCE: DolVol_MA_126 = 807, DolVol_MA_63 = 439, DolVol_MA_21 = 230
# Focus on the configurations that showed highest importance
TICKER_VOLUME_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^DollarVolume$",
    output_prefix="DolVol",
    lags=[63, 126, 252],       # Quarterly, semi-annual, annual lags
    mas=[21, 63, 126, 252, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Medium-term volume momentum
    diffs=[(0, 126), (0, 252), (63, 252), (252, 756)],  # Volume vs historical norms
    scope="ticker",
    include_spike=True,
    include_volatility=True,
    volatility_window=21,      # 21-day vol showed high importance (299)
)

# Amihud illiquidity - one of the top features, add temporal dynamics
# HIGH IMPORTANCE: Illiq_MA_63 = 394, Illiq_Lag_63 = 289, Illiq_MA_21 = 202
# Extend to 126-day windows for annual horizon
TICKER_AMIHUD_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^Amihud_21d$|^Amihud_63d$",
    output_prefix="Illiq",
    lags=[21, 63, 126],        # Add 126-day lag
    mas=[21, 63, 126, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Medium to long-term illiquidity momentum
    diffs=[(0, 63), (0, 126), (63, 252)],  # Illiquidity vs historical
    scope="ticker",
    include_spike=True,
)

# Volatility features - Vol_252 is top feature, track its dynamics
# VERY HIGH IMPORTANCE: VolDyn_Lag_126 = 450, VolDyn_MA_126 = 374, VolDyn_Mom_126 = 364
# VolDyn_MA_63 = 299, Vol_252 raw = 370
# 126-day windows dominate - add 252-day for full annual cycle
TICKER_VOLATILITY_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^Vol_252$|^Vol_20$",
    output_prefix="VolDyn",
    lags=[63, 126, 252],       # Focus on semi-annual and annual lags
    mas=[63, 126, 252, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126, 252],   # Add annual momentum
    diffs=[(0, 252), (0, 756), (126, 756), (252, 1260)],  # Vol regime changes
    scope="ticker",
)

# Trend persistence - 3rd most important feature
# IMPORTANCE: TrendPersist_252d = 294, Trend_MA_63 = 444, Trend_Lag_63 = 350
# Trend_Mom_63 = 141 - extend to 126-day windows
TICKER_TREND_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^TrendPersist_252d$|^Trend_RSq_",
    output_prefix="Trend",
    lags=[63, 126],            # Track trend quality shifts over quarters
    mas=[63, 126, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Trend quality momentum
    diffs=[(0, 126), (0, 756), (126, 756)],  # Trend quality vs historical
    scope="ticker",
)

# Higher moments features - HIGH IMPORTANCE
# Skew_126d = 435, Kurt_126d = 390, Skew_60d = 151, Kurt_60d = 138
# Add dynamics for these important features
TICKER_HIGHER_MOMENTS_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^Skew_126d$|^Kurt_126d$|^Skew_60d$|^Kurt_60d$",
    output_prefix="Moment",
    lags=[63, 126],            # Track how moments have shifted
    mas=[63, 126, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Momentum of distribution shape changes
    diffs=[(0, 126), (0, 756), (126, 756)],  # Distribution shape vs historical
    scope="ticker",
)

# Trailing Dividend Yield - 2nd highest feature (650 importance)
# Add lag/MA dynamics since yield changes predict returns
TICKER_DIVIDEND_YIELD_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^TrailingDivYield_252d$",
    output_prefix="DivYld",
    lags=[63, 126, 252],       # How yield has changed over time
    mas=[63, 126, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Yield momentum (increasing/decreasing dividends)
    diffs=[(0, 252), (0, 756), (252, 1260)],  # Yield vs historical norms
    scope="ticker",
)


# =============================================================================
# MACRO FEATURES (global, timestamp-level only)
# =============================================================================

# OECD Confidence indicators - top macro features (CCICP: 102, BCICP: 72 importance)
# Monthly releases, focus on longer lags for annual horizon
# Research: Stock & Watson (2003) - leading indicators for business cycles
MACRO_OECD_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_OECD_",
    output_prefix=None,
    lags=[63, 126, 252],       # Quarterly, semi-annual, annual (252 showed ~102 importance)
    mas=[63, 126, 252, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Medium-term confidence shifts
    diffs=[(0, 252), (0, 756), (252, 756)],  # Confidence vs historical norms
    scope="global",
    include_spike=True,        # Detect sudden confidence shifts
)

# Interest rates - highly important (ranks 28-29 in feature importance)
# Research: Fama & French (1989) - term structure predicts returns
# Long-term rate L252 = 58, Short-term L252 = 54 - annual lags important
MACRO_INTEREST_RATES_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"interest rate|Interest rate",
    output_prefix=None,
    lags=[63, 126, 252],       # Focus on longer periods for annual prediction
    mas=[63, 126, 252, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Rate direction over quarters
    diffs=[(0, 252), (0, 756), (252, 756), (252, 1260)],  # Rate vs historical levels
    scope="global",
    include_spike=True,
    include_volatility=True,   # Interest rate volatility is predictive
    volatility_window=63,      # Quarterly volatility
)

# Road fatalities - economic activity proxy (importance: 38 at L126)
# Simplify - not a top driver
MACRO_FATALITIES_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"road_fatalities",
    output_prefix=None,
    lags=[63, 126],            # Quarterly to semi-annual lags
    mas=[63, 126, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Activity momentum
    scope="global",
)

# BOP (Balance of Payments) - quarterly data, need longer windows
# Multiple BOP features showing ~5-14 importance - moderate value
MACRO_BOP_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_BOP",
    output_prefix=None,
    lags=[126, 252],           # Semi-annual, annual (longer is better for BOP)
    mas=[126, 252, 756, 1260], # Up to 5-year MA
    momentum=[126, 252],       # Long-term BOP trends
    scope="global",
)

# International indices (FTSE: importance ~89-105, Shanghai: ~33-72)
# FTSE Volume = 105, FTSE Open = 102 - strong predictive value
# Research: Rapach et al. (2013) - international return predictability
MACRO_INDEX_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^\^FTSE|^000001\.SS|^\^TNX",
    output_prefix=None,
    lags=[63, 126, 252],       # Focus on longer lags for annual horizon
    mas=[63, 126, 252, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # International momentum
    diffs=[(0, 252), (0, 756), (252, 756)],  # Index vs historical levels
    scope="global",
    include_spike=True,
    include_volatility=True,
    volatility_window=63,      # Quarterly volatility
)

# Commodity futures - track commodity cycles
# Research: Hong & Yogo (2012) - commodity fundamentals predict returns
# Various commodities showing 4-36 importance, focus on longer windows
MACRO_COMMODITY_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"=F_",    # Futures pattern
    output_prefix=None,
    lags=[63, 126],            # Quarterly, semi-annual for cycles
    mas=[63, 126, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Commodity momentum
    scope="global",
    include_spike=True,
)

# NZ Dollar exchange rates - currency dynamics affect NZX
# Moderate importance - simplify configuration
MACRO_FX_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_NZD.*=X_",
    output_prefix=None,
    lags=[63, 126],            # Currency cycles over quarters
    mas=[63, 126, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # FX momentum
    scope="global",
    include_volatility=True,
    volatility_window=63,      # Quarterly FX volatility
)

# NZ GDP/Government expenditure data - quarterly, need long lags
# NZL features showing 7-15 importance at 252-day windows
MACRO_NZ_GDP_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"^MACRO_NZL_",
    output_prefix=None,
    lags=[126, 252],           # Semi-annual to annual
    mas=[126, 252, 756, 1260], # Up to 5-year MA
    momentum=[126, 252],       # Long-term GDP trends
    scope="global",
)

# Fear/sentiment Wikipedia indicators - market psychology
# Research: Da, Engelberg, Gao (2015) - "The Sum of All FEARS"
# Limited appearance in top features - simplify to longer windows
MACRO_WIKI_FEAR_CONFIG = FeatureLagMAConfig(
    feature_pattern=r"MACRO_.*_(Recession|Financial_crisis|Stock_market_crash|"
                    r"Bear_market|Credit_crunch|Bankruptcy|Panic_selling|"
                    r"Economic_bubble|Inflation|Unemployment)_Wiki_Views$",
    output_prefix=None,
    lags=[63, 126],            # Quarterly to semi-annual fear lags
    mas=[63, 126, 756, 1260],  # Up to 5-year MA
    momentum=[63, 126],        # Fear momentum over longer periods
    scope="global",
    include_spike=True,        # Sudden fear spikes are predictive
)


# =============================================================================
# AGGREGATE CONFIGS
# =============================================================================

# All ticker-level configs
TICKER_CONFIGS: List[FeatureLagMAConfig] = [
    TICKER_ATTENTION_CONFIG,
    TICKER_VOLUME_CONFIG,
    TICKER_AMIHUD_CONFIG,
    TICKER_VOLATILITY_CONFIG,
    TICKER_TREND_CONFIG,
    TICKER_HIGHER_MOMENTS_CONFIG,
    TICKER_DIVIDEND_YIELD_CONFIG,
]

# All macro/global configs  
MACRO_CONFIGS: List[FeatureLagMAConfig] = [
    MACRO_OECD_CONFIG,
    MACRO_INTEREST_RATES_CONFIG,
    MACRO_FATALITIES_CONFIG,
    MACRO_BOP_CONFIG,
    MACRO_INDEX_CONFIG,
    MACRO_COMMODITY_CONFIG,
    MACRO_FX_CONFIG,
    MACRO_NZ_GDP_CONFIG,
    MACRO_WIKI_FEAR_CONFIG,
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
