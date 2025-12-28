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
FORWARD_RETURN_DAYS: int = 5
"""Number of days to compute forward return. Common values: 1, 5, 20."""

RETURN_TYPE: ReturnType = ReturnType.SIMPLE
"""Type of returns to calculate: SIMPLE for arithmetic, LOG for logarithmic."""

WINSORIZE_LIMITS: tuple[float, float] | None = (-0.5, 0.5)
"""Clip extreme returns to this range. None to disable winsorization.
E.g., (-0.5, 0.5) clips returns to [-50%, +50%]."""


# =============================================================================
# ROLLING WINDOW SETTINGS
# =============================================================================
NUM_ROLLING_WINDOWS: int = 25
"""Number of rolling windows for backtesting."""

ROLLING_WINDOW_MOVEMENT_YEARS: float = 0.3652423 * 2.2
"""How far back (in years) each window moves from the previous."""

TEST_PERIOD_YEARS: float = 1.0 / 12.0
"""Length of test period in each window (in years)."""


# =============================================================================
# RANKING MODEL (LightGBM LGBMRanker)
# =============================================================================
RANKER_N_ESTIMATORS: int = 100
"""Number of boosting iterations for the ranking model."""

RANKER_LEARNING_RATE: float = 0.05
"""Learning rate (shrinkage) for the ranking model."""

RANKER_NUM_LEAVES: int = 31
"""Maximum number of leaves per tree."""

RANKER_MAX_DEPTH: int = -1
"""Maximum tree depth. -1 means no limit."""

RANKER_MIN_CHILD_SAMPLES: int = 20
"""Minimum number of samples required in a leaf."""

RANKER_SUBSAMPLE: float = 0.8
"""Fraction of samples to use for each boosting iteration."""

RANKER_COLSAMPLE_BYTREE: float = 0.8
"""Fraction of features to use for each tree."""

RANKER_EVAL_AT: tuple[int, ...] = (5, 10, 20)
"""Evaluation positions for NDCG metric."""

LABEL_GAIN: list[float] | None = None
"""Custom label gains for NDCG. None = use default (2^label - 1)."""

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

TRANSACTION_COST_BPS: float = 10.0
"""Round-trip transaction cost in basis points (10 bps = 0.1%)."""

LONG_ONLY: bool = False
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
# DEBUG
# =============================================================================
SAVE_DEBUG_SAMPLES: bool = True
"""Save preprocessed data samples to output/debug/ for inspection."""

DEBUG_SAMPLE_SIZE: int = 500
"""Number of rows to sample for debug output."""


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


# =============================================================================
# CONFIG EXPORT
# =============================================================================
def get_config_dict() -> dict:
    """Export current configuration as dictionary for logging.
    
    Returns:
        Dictionary with all configuration values.
    """
    return {
        "target": {
            "forward_return_days": FORWARD_RETURN_DAYS,
            "return_type": RETURN_TYPE.value,
            "winsorize_limits": WINSORIZE_LIMITS,
        },
        "rolling_window": {
            "num_windows": NUM_ROLLING_WINDOWS,
            "movement_years": ROLLING_WINDOW_MOVEMENT_YEARS,
            "test_period_years": TEST_PERIOD_YEARS,
        },
        "ranking_model": {
            "n_estimators": RANKER_N_ESTIMATORS,
            "learning_rate": RANKER_LEARNING_RATE,
            "num_leaves": RANKER_NUM_LEAVES,
            "max_depth": RANKER_MAX_DEPTH,
        },
        "portfolio": {
            "top_n": PORTFOLIO_TOP_N,
            "bottom_n": PORTFOLIO_BOTTOM_N,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "initial_capital": INITIAL_CAPITAL,
            "long_only": LONG_ONLY,
        },
        "evaluation": {
            "min_stocks_for_ic": MIN_STOCKS_FOR_IC,
            "min_stocks_per_timestamp": MIN_STOCKS_PER_TIMESTAMP,
        },
    }

