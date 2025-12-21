"""Model and pipeline configuration constants."""

# Target variable configuration
LOOKAHEAD_DAYS = 90  # 3-month horizon (more predictable)
GAIN_THRESHOLD_PCT = 3.0  # 3% gain in 3 months

# Rolling window configuration
NUM_ROLLING_WINDOWS = 3
ROLLING_WINDOW_MOVEMENT_YEARS = 2.0
TEST_PERIOD_YEARS = 1.5

# Trading simulation configuration
PREDICTION_THRESHOLD = 0.52  # Slightly above 0.5 to filter low-confidence
INVERT_PREDICTIONS = False  # Normal predictions
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.1
RISK_FREE_RATE = 0.0
MAX_POSITION_SIZE_PCT = 0.08  # 8% per position

# XGBoost - simpler model, focus on generalization
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 2,  # Very shallow - learn only strongest signals
    "learning_rate": 0.05,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "subsample": 0.7,
    "colsample_bytree": 0.5,  # Use only half the features per tree
    "colsample_bylevel": 0.7,
    "min_child_weight": 50,  # Very conservative - need 50 samples per leaf
    "gamma": 0.5,  # High threshold for splits
    "reg_alpha": 2.0,  # Strong L1 regularization
    "reg_lambda": 5.0,  # Very strong L2 regularization
    "scale_pos_weight": 1.0,
}

# Data processing
MS_PER_DAY = 86_400_000  # Milliseconds per day (timestamps are in ms)


def get_config_dict() -> dict:
    """Return current configuration as a dictionary for logging."""
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
            "invert_predictions": INVERT_PREDICTIONS,
            "initial_capital": INITIAL_CAPITAL,
            "transaction_cost_pct": TRANSACTION_COST_PCT,
            "max_position_size_pct": MAX_POSITION_SIZE_PCT,
        },
        "xgboost": XGBOOST_PARAMS.copy(),
    }


def calculate_class_weight(y) -> float:
    """Calculate scale_pos_weight for imbalanced data."""
    import numpy as np
    y_arr = np.asarray(y)
    n_neg = (y_arr == 0).sum()
    n_pos = (y_arr == 1).sum()
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def initialize_model(params: dict | None = None, class_weight: float | None = None):
    """Initialize XGBoost model with optional class weight adjustment."""
    from xgboost import XGBClassifier
    
    model_params = (params or XGBOOST_PARAMS).copy()
    
    # Adjust class weight if provided
    if class_weight is not None:
        model_params["scale_pos_weight"] = class_weight
    
    return XGBClassifier(random_state=42, **model_params)