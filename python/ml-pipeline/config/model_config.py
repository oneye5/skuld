"""Model and pipeline configuration constants.

Best performing configuration (as of Dec 2025):
- 365-day lookahead, 13% gain threshold, 0.79 prediction threshold
- 5-model ensemble: RF, ExtraTrees, HistGradientBoosting, XGBoost, LightGBM
- StandardScaler (NOT RobustScaler - tested worse: -30% vs +177% return)
- One-hot encoded tickers
- Results: 177.63% return, 0.3422 Sharpe, 416 trades
"""

# Target variable configuration
# Research: 1-year horizon matches data granularity (quarterly/annual fundamentals)
# Per nzx-predictor: shorter periods = predicting on 'stale' data
LOOKAHEAD_DAYS = 365  # 1-year horizon
GAIN_THRESHOLD_PCT = 13.0  # 13% threshold - higher selectivity = better precision

# Rolling window configuration
NUM_ROLLING_WINDOWS = 10
ROLLING_WINDOW_MOVEMENT_YEARS = 0.5  # More overlap = more robust evaluation
TEST_PERIOD_YEARS = 1.0

# Trading simulation configuration
# Higher threshold = more selective = better precision (trades off recall)
# Per nzx-predictor: 0.79 threshold achieved Sharpe 1.453
PREDICTION_THRESHOLD = 0.79  # High threshold for quality over quantity
INVERT_PREDICTIONS = False
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.1
RISK_FREE_RATE = 0.0
MAX_POSITION_SIZE_PCT = 2.0  # 2% per position

# Model selection
USE_ENSEMBLE = True  # Use ensemble of XGBoost + LightGBM + CatBoost
USE_ADVANCED_FEATURES = True  # Add ATR, ADX, Stochastic, etc.
USE_CROSS_SECTIONAL = True  # Add market-relative features

# XGBoost - optimized for financial time series with moderate regularization
# Balance between underfitting and overfitting
XGBOOST_PARAMS = {
    "n_estimators": 600,  # More trees than original
    "max_depth": 5,  # Slightly shallower than original (6)
    "learning_rate": 0.015,  # Moderate learning rate
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "subsample": 0.65,
    "colsample_bytree": 0.55,
    "colsample_bylevel": 0.55,
    "min_child_weight": 75,  # Moderate (was 50 orig, 100 too high)
    "gamma": 0.3,  # Moderate split threshold
    "reg_alpha": 0.75,  # Moderate L1
    "reg_lambda": 3.0,  # Moderate L2 (was 2.0 orig, 5.0 too high)
    "scale_pos_weight": 1.0,  # Adjusted by calculate_class_weight()
    "tree_method": "hist",
}

# LightGBM params (used when USE_ENSEMBLE=True)
# Moderate regularization matching XGBoost
LIGHTGBM_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "learning_rate": 0.015,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "subsample": 0.65,
    "colsample_bytree": 0.55,
    "min_child_samples": 75,
    "reg_alpha": 0.75,
    "reg_lambda": 3.0,
    "class_weight": "balanced",
    "verbosity": -1,
    "min_gain_to_split": 0.05,  # Lower than before
}

# CatBoost params (used when USE_ENSEMBLE=True)
# Moderate regularization
CATBOOST_PARAMS = {
    "iterations": 600,
    "depth": 5,
    "learning_rate": 0.015,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "l2_leaf_reg": 5.0,  # Moderate L2
    "random_strength": 1.5,
    "bagging_temperature": 0.7,
    "auto_class_weights": "Balanced",
    "verbose": False,
    "min_data_in_leaf": 75,
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
        "model": {
            "use_ensemble": USE_ENSEMBLE,
            "use_advanced_features": USE_ADVANCED_FEATURES,
            "use_cross_sectional": USE_CROSS_SECTIONAL,
        },
        "xgboost": XGBOOST_PARAMS.copy(),
        "lightgbm": LIGHTGBM_PARAMS.copy() if USE_ENSEMBLE else None,
        "catboost": CATBOOST_PARAMS.copy() if USE_ENSEMBLE else None,
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