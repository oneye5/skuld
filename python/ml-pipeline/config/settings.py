"""Centralized settings for the ML pipeline.

All configurable parameters should be defined here. No magic numbers elsewhere.
"""


# =============================================================================
# TARGET DEFINITION
# =============================================================================
LOOKAHEAD_DAYS: int = 366
"""Number of days to look ahead for price change prediction."""

GAIN_THRESHOLD_PCT: float = 13.0
"""Minimum percentage gain for positive class (1 = will gain >= X%)."""

PREDICTION_THRESHOLD: float = 0.79
"""Probability threshold for buy signal (0.79 matches nzx-predictor)."""


# =============================================================================
# ROLLING WINDOW SETTINGS
# =============================================================================
NUM_ROLLING_WINDOWS: int = 25
"""Number of rolling windows for backtesting."""

ROLLING_WINDOW_MOVEMENT_YEARS: float = 2.0
"""How far back (in years) each window moves from the previous."""

TEST_PERIOD_YEARS: float = 1.0/12.0
"""Length of test period in each window (in years)."""


# =============================================================================
# TRADING SIMULATION
# =============================================================================
INITIAL_CAPITAL: float = 100_000.0
"""Starting capital for trading simulation."""

TRANSACTION_COST_PCT: float = 0.0
"""Transaction cost as percentage (0.0 = no cost)."""

MAX_POSITION_SIZE_PCT: float = 5.0
"""Maximum position size as percentage of initial capital."""


# =============================================================================
# CONSTANTS
# =============================================================================
MS_PER_DAY: int = 86_400_000
"""Milliseconds per day."""

YEAR_2000_MS: int = 946684800000
"""Unix timestamp for 2000-01-01 00:00:00 UTC in milliseconds."""

EPSILON: float = 1e-6
"""Small value to avoid division by zero in ratio calculations."""


# =============================================================================
# MODEL DEFINITION
# =============================================================================
def create_model():
    """Create and return the model instance.
    
    Edit this function to change the model architecture.
    Any sklearn-compatible classifier with fit() and predict_proba() works.
    
    Returns:
        A classifier instance.
    """
    from lightgbm import LGBMClassifier
    
    return LGBMClassifier(
        verbosity=-1,
        random_state=42,
    )


# =============================================================================
# CONFIG EXPORT
# =============================================================================
def get_config_dict() -> dict:
    """Export current configuration as dictionary for logging.
    
    Returns:
        Dictionary with all configuration values.
    """
    model = create_model()
    model_params = model.get_params() if hasattr(model, "get_params") else {}
    
    return {
        "target": {
            "lookahead_days": LOOKAHEAD_DAYS,
            "gain_threshold_pct": GAIN_THRESHOLD_PCT,
        },
        "rolling_window": {
            "num_windows": NUM_ROLLING_WINDOWS,
            "movement_years": ROLLING_WINDOW_MOVEMENT_YEARS,
            "test_period_years": TEST_PERIOD_YEARS,
        },
        "trading": {
            "prediction_threshold": PREDICTION_THRESHOLD,
            "initial_capital": INITIAL_CAPITAL,
            "transaction_cost_pct": TRANSACTION_COST_PCT,
            "max_position_size_pct": MAX_POSITION_SIZE_PCT,
        },
        "model": {
            "type": type(model).__name__,
            "params": model_params,
        },
    }
