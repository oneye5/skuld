"""Model and pipeline configuration constants."""

# Target variable configuration
LOOKAHEAD_DAYS = 365  # How far into the future to predict
GAIN_THRESHOLD_PCT = 2.0  # Minimum % gain for positive class

# Rolling window configuration
NUM_ROLLING_WINDOWS = 5
ROLLING_WINDOW_MOVEMENT_YEARS = 1.3333  # How far to move window back in time

# Trading simulation configuration
PREDICTION_THRESHOLD = 0.7  # Minimum probability to trigger buy
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.1  # Cost per trade (buy or sell)
RISK_FREE_RATE = 0.0  # For Sharpe ratio calculation
MAX_POSITION_SIZE_PCT = 5.0  # Maximum % of capital per position

# XGBoost tuned configuration for better generalization
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,  # Reduced from 6 to prevent overfitting
    "learning_rate": 0.05,  # Reduced for better generalization
    "objective": "binary:logistic",
    "eval_metric": "auc",  # Changed from logloss to auc for ranking
    "subsample": 0.8,  # Row sampling to reduce overfitting
    "colsample_bytree": 0.8,  # Feature sampling per tree
    "colsample_bylevel": 0.8,  # Feature sampling per level
    "min_child_weight": 5,  # Minimum sum of instance weight in child
    "gamma": 0.1,  # Minimum loss reduction for split
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
    "scale_pos_weight": 1.0,  # Will be adjusted based on class ratio
}

# Data processing
MS_PER_DAY = 86_400_000  # Milliseconds per day (timestamps are in ms)


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