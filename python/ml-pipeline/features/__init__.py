"""Features package - feature engineering transformers."""

from features.ratios import add_financial_ratios
from features.technical import add_technical_features
from features.cross_sectional import add_cross_sectional_features
from features.alpha_factors import add_alpha_factors, add_seasonality_features
from features.attention_features import add_aggregate_attention_features
from features.lag_ma_features import (
    add_lag_ma_features,
    add_ticker_lag_ma_features,
    add_macro_lag_ma_features,
    create_custom_config,
)
from config.lag_ma_config import (
    FeatureLagMAConfig,
    get_enabled_configs,
    get_ticker_configs,
    get_macro_configs,
)

__all__ = [
    "add_financial_ratios",
    "add_technical_features",
    "add_cross_sectional_features",
    "add_alpha_factors",
    "add_seasonality_features",
    "add_aggregate_attention_features",
    "add_lag_ma_features",
    "add_ticker_lag_ma_features",
    "add_macro_lag_ma_features",
    "create_custom_config",
    "FeatureLagMAConfig",
    "get_enabled_configs",
    "get_ticker_configs",
    "get_macro_configs",
]
