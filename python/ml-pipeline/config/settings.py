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

RETURN_PRICE_COLUMN: str = "AdjClose"
"""Price column to use for forward return calculations.
Use 'AdjClose' for total return (includes dividends, adjusted for splits).
Use 'Close' for price-only returns (ignores dividends).
Recommended: 'AdjClose' for accurate evaluation of stock performance."""


# =============================================================================
# ROLLING WINDOW SETTINGS
# =============================================================================
NUM_ROLLING_WINDOWS: int = 20
"""Number of rolling windows for backtesting. Use a smaller value for testing, 20 windows takes a long time"""

ROLLING_WINDOW_MOVEMENT_YEARS: float = 1
"""How far back (in years) each window moves from the previous."""

TEST_PERIOD_YEARS: float = 0.1
"""This effectively does not do anything when test period is less than double the window movement."""


# =============================================================================
# RANKING MODEL (LightGBM LGBMRanker)
# =============================================================================
RANKER_N_ESTIMATORS: int = 150
"""Number of boosting iterations for the ranking model.
Optimized for 365-day horizon (increased from 100 based on grid search)."""

RANKER_LEARNING_RATE: float = 0.05
"""Learning rate (shrinkage) for the ranking model."""

RANKER_NUM_LEAVES: int = 31
"""Maximum number of leaves per tree."""

RANKER_MAX_DEPTH: int = -1
"""Maximum tree depth. -1 means no limit."""

RANKER_MIN_CHILD_SAMPLES: int = 20
"""Minimum number of samples required in a leaf."""

RANKER_SUBSAMPLE: float = 0.8
"""Fraction of samples to use for each boosting iteration (bagging)."""

RANKER_COLSAMPLE_BYTREE: float = 0.8
"""Fraction of features to use for each tree."""

RANKER_EVAL_AT: tuple[int, ...] = (5, 10, 20)
"""Evaluation positions for NDCG metric."""

RANKER_DEVICE: str = "gpu"
"""Device for training: 'cpu' or 'gpu'. GPU requires CUDA and GPU-enabled LightGBM.
Note: GPU provides ~1.5x speedup for our data size. Falls back to CPU if GPU unavailable."""

RANKER_EARLY_STOPPING_ROUNDS: int | None = None
"""Early stopping rounds. None to disable early stopping.
IMPORTANT: Early stopping was tested but found to HURT performance significantly
(IC dropped from 0.27 to 0.00). This is because ranking with long-horizon targets
(365 days) needs many iterations to learn, and validation NDCG plateaus early
leading to premature stopping. Keep this as None."""

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
# DATA QUALITY / ANOMALY DETECTION
# =============================================================================
ANOMALY_RETURN_THRESHOLD: float = 2.0
"""Daily return threshold for flagging anomalous data (default 2.0 = 200%).
Returns exceeding this threshold (positive or negative) indicate potential
unadjusted stock splits, ticker recycling, or data errors."""

FILTER_ANOMALIES: bool = True
"""Whether to filter out anomalous data points before training.
If True, rows with extreme daily returns are removed.
If False, anomalies are flagged but kept in the data."""

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

