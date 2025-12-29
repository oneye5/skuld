"""Centralized settings for the ML pipeline.

All configurable parameters should be defined here. No magic numbers elsewhere.
"""

from enum import Enum


class ReturnType(str, Enum):
    """Type of return calculation."""
    SIMPLE = "simple"  # (P_t+n - P_t) / P_t
    LOG = "log"        # ln(P_t+n / P_t)


# =============================================================================
# TARGET DEFINITION (Ranking Pipeline)
# =============================================================================
FORWARD_RETURN_DAYS: int = 365
"""Number of days ahead to calculate returns for ranking target."""

RETURN_TYPE: ReturnType = ReturnType.SIMPLE
"""Type of returns to calculate: SIMPLE for arithmetic, LOG for logarithmic."""

WINSORIZE_LIMITS: tuple[float, float] | None = (-0.5, 0.5)
"""Clip extreme returns to this range. None to disable winsorization.
E.g., (-0.5, 0.5) clips returns to [-50%, +50%]."""


# =============================================================================
# ROLLING WINDOW SETTINGS
# =============================================================================
NUM_ROLLING_WINDOWS: int = 30
"""Number of rolling windows for backtesting."""

ROLLING_WINDOW_MOVEMENT_YEARS: float = 0.6666666666
"""How far back (in years) each window moves from the previous."""

TEST_PERIOD_YEARS: float = 0.6666666666
"""Length of test period in each window (in years)."""


# =============================================================================
# RANKING MODEL (LightGBM LGBMRanker)
# =============================================================================
RANKER_N_ESTIMATORS: int = 100
"""Number of boosting iterations for the ranking model.
Reduced from 150 to prevent overfitting with 365-day horizon."""

RANKER_LEARNING_RATE: float = 0.05
"""Learning rate (shrinkage) for the ranking model."""

RANKER_NUM_LEAVES: int = 31
"""Maximum number of leaves per tree. 31 is LightGBM default."""

RANKER_MAX_DEPTH: int = -1
"""Maximum tree depth. -1 means no limit (use num_leaves to control)."""

RANKER_MIN_CHILD_SAMPLES: int = 50
"""Minimum number of samples required in a leaf.
Increased from default 20 for more robust splits with noisy data."""

RANKER_SUBSAMPLE: float = 0.8
"""Fraction of samples to use for each boosting iteration (bagging)."""

RANKER_COLSAMPLE_BYTREE: float = 0.8
"""Fraction of features to use for each tree."""

RANKER_EVAL_AT: tuple[int, ...] = (5, 10, 20)
"""Evaluation positions for NDCG metric."""

MIN_STOCKS_PER_TIMESTAMP: int = 10
"""Minimum stocks required per timestamp for valid ranking. 
LGBMRanker needs sufficient group size for meaningful ranking."""


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================
PORTFOLIO_TOP_N: int = 10
"""Number of top-ranked stocks for long portfolio."""

PORTFOLIO_BOTTOM_N: int = 10
"""Number of bottom-ranked stocks for short portfolio."""

TRANSACTION_COST_BPS: float = 190.0
"""Round-trip transaction cost in basis points.
Default reflects Sharesies NZX fee of 1.9% (190 bps) per trade."""

SLIPPAGE_BPS: float = 15.0
"""Slippage in basis points per trade (15 bps = 0.15%).
Slippage models the difference between expected and executed price due to
market impact, bid-ask spread, and order timing.
NZX typically has wider spreads than US markets due to lower liquidity."""

LONG_ONLY: bool = True
"""If True, only take long positions (no shorting)."""

INITIAL_CAPITAL: float = 100_000.0
"""Starting capital for portfolio simulation."""


# =============================================================================
# EVALUATION
# =============================================================================
MIN_STOCKS_FOR_IC: int = 5
"""Minimum stocks per timestamp to compute IC."""

TOP_N_FOR_HIT_RATE: int = 10
"""Number of top predictions to consider for hit rate calculation."""

PERIODS_PER_YEAR: int = 252
"""Trading days per year, used for annualizing ICIR."""


# =============================================================================
# CONSTANTS
# =============================================================================
MS_PER_DAY: int = 86_400_000
"""Milliseconds per day."""

YEAR_2000_MS: int = 946684800000
"""Unix timestamp for 2000-01-01 00:00:00 UTC in milliseconds."""

EPSILON: float = 1e-6
"""Small value to avoid division by zero in ratio calculations."""

CLIP_THRESHOLD: float = 10.0
"""Clip scaled feature values to [-CLIP_THRESHOLD, CLIP_THRESHOLD] after scaling."""

