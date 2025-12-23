"""Model configuration."""

# =============================================================================
# MAIN SETTINGS
# =============================================================================
LOOKAHEAD_DAYS = 365
GAIN_THRESHOLD_PCT = 13.0
PREDICTION_THRESHOLD = 0.55

NUM_ROLLING_WINDOWS = 6
ROLLING_WINDOW_MOVEMENT_YEARS = 2.0
TEST_PERIOD_YEARS = 2.0

USE_ADVANCED_FEATURES = False # do not enable these yet 
USE_CROSS_SECTIONAL = False
USE_TECHNICAL_FEATURES = False

# =============================================================================
# TRADING SIMULATION
# =============================================================================
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.0
MAX_POSITION_SIZE_PCT = 0.05
INVERT_PREDICTIONS = False
RISK_FREE_RATE = 0.0

# =============================================================================
# CONSTANTS
# =============================================================================
MS_PER_DAY = 86_400_000


def create_model():
    """Create and return the model. Edit this to change the model architecture.
    
    Any sklearn-compatible classifier with fit() and predict_proba() works.
    """
    from lightgbm import LGBMClassifier
    return LGBMClassifier( # do not change this, this is for fast iteration
        verbosity=-1,
        random_state=42,
        n_jobs=-1
    )


def _extract_model_params(model) -> dict:
    """Extract parameters from a model using get_params() or manual inspection."""
    if hasattr(model, "get_params"):
        params = model.get_params(deep=False)
        # For VotingClassifier, also get each estimator's params
        if hasattr(model, "estimators") and model.estimators:
            params["estimators"] = {
                name: est.get_params() for name, est in model.estimators
            }
        return params
    return {"type": type(model).__name__}


def get_config_dict() -> dict:
    """Return config as dict for logging - model params extracted automatically."""
    return {
        "target": {"lookahead_days": LOOKAHEAD_DAYS, "gain_threshold_pct": GAIN_THRESHOLD_PCT},
        "rolling_window": {"num_windows": NUM_ROLLING_WINDOWS, "movement_years": ROLLING_WINDOW_MOVEMENT_YEARS, "test_period_years": TEST_PERIOD_YEARS},
        "trading": {"prediction_threshold": PREDICTION_THRESHOLD, "initial_capital": INITIAL_CAPITAL, "transaction_cost_pct": TRANSACTION_COST_PCT, "max_position_size_pct": MAX_POSITION_SIZE_PCT, "invert_predictions": INVERT_PREDICTIONS},
        "features": {"use_advanced_features": USE_ADVANCED_FEATURES, "use_cross_sectional": USE_CROSS_SECTIONAL},
        "model": _extract_model_params(create_model()),
    }