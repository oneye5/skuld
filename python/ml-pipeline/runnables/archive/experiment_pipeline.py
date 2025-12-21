"""Experimental pipeline for model performance optimization.

This script tests various configurations to achieve target Sharpe ratio >= 0.8.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
import json
import gc

import pandas as pd
import numpy as np

# Add paths for hyphenated directories
_ml_pipeline = Path(__file__).parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "data-preparation"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "transformations"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "long-to-wide"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "data-splitting" / "train-test"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "labeling"))
sys.path.insert(0, str(_ml_pipeline / "evaluation"))
sys.path.insert(0, str(_ml_pipeline / "evaluation" / "model-evaluation"))
sys.path.insert(0, str(_ml_pipeline / "evaluation" / "trade-simulation"))

from config.column_names import (
    TIMESTAMP, TICKER, TARGET, CLOSE, OPEN, HIGH, LOW, VOLUME,
    DAY_OF_YEAR_SIN, DAY_OF_YEAR_COS, DAY_OF_WEEK_SIN, DAY_OF_WEEK_COS,
    MONTH_SIN, MONTH_COS,
)
from config.model_config import MS_PER_DAY

from utils.data_loader import load_long_data
from converter import long_to_wide
from splitter import split_by_timestamp
from labeler import create_labels

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from simulator import run_trading_simulation, run_baseline_simulation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass  
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    lookahead_days: int = 90
    gain_threshold_pct: float = 3.0
    test_period_years: float = 1.5
    prediction_threshold: float = 0.5
    max_position_size_pct: float = 0.10
    initial_capital: float = 100_000.0
    transaction_cost_pct: float = 0.1
    
    # Feature engineering
    use_technical_features: bool = True
    use_macro_features: bool = False  # Disable noisy macro data
    use_cyclical_time: bool = True
    
    # Core technical feature windows
    return_windows: list = None  # Default: [5, 20, 60]
    volatility_windows: list = None  # Default: [10, 20]
    sma_windows: list = None  # Default: [20, 50]
    
    # Scaling
    scaler_type: str = "standard"  # standard, robust, minmax
    
    # Model
    model_type: str = "xgboost"  # xgboost, lightgbm, random_forest
    model_params: dict = None
    
    def __post_init__(self):
        if self.return_windows is None:
            self.return_windows = [5, 20, 60]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20]
        if self.sma_windows is None:
            self.sma_windows = [20, 50]
        if self.model_params is None:
            self.model_params = {}


# ============================================================================
# SIMPLIFIED DATA PREPARATION
# ============================================================================

def prepare_simple_wide_data(long_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Convert to wide format keeping only essential price columns."""
    # Filter to only price data (no macro)
    if not config.use_macro_features:
        # Keep only rows with ticker (non-macro data)
        long_df = long_df[long_df['ticker'].notna() & (long_df['ticker'] != '')]
    
    # Filter to only OHLCV features
    ohlcv_features = {'Open', 'High', 'Low', 'Close', 'Volume'}
    long_df = long_df[long_df['feature'].isin(ohlcv_features)]
    
    # Convert to wide
    wide_df = long_to_wide(long_df)
    
    return wide_df


def add_focused_technical_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Add technical features that actually correlate with momentum."""
    if not config.use_technical_features:
        return df
    
    if CLOSE not in df.columns:
        return df
    
    df = df.sort_values([TICKER, TIMESTAMP])
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].copy()
        ticker_df = ticker_df.sort_values(TIMESTAMP).reset_index(drop=True)
        
        close = ticker_df[CLOSE].values
        n = len(close)
        
        if n < 2:
            result_dfs.append(ticker_df)
            continue
        
        # Returns at different horizons
        for window in config.return_windows:
            ret = np.full(n, np.nan)
            if n > window:
                ret[window:] = (close[window:] - close[:-window]) / np.maximum(np.abs(close[:-window]), 0.0001) * 100
            ticker_df[f'return_{window}d'] = ret.astype('float32')
        
        # Volatility (std of daily returns)
        daily_ret = np.zeros(n)
        daily_ret[1:] = (close[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 0.0001) * 100
        
        for window in config.volatility_windows:
            vol = pd.Series(daily_ret).rolling(window, min_periods=max(1, window//2)).std().values
            ticker_df[f'volatility_{window}d'] = (vol * 100).astype('float32')
        
        # Moving averages and price relative to MA
        for window in config.sma_windows:
            sma = pd.Series(close).rolling(window, min_periods=max(1, window//2)).mean().values
            ticker_df[f'sma_{window}'] = sma.astype('float32')
            ticker_df[f'price_to_sma_{window}'] = ((close / np.maximum(sma, 0.0001) - 1) * 100).astype('float32')
        
        # RSI (14-day)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
        
        rs = avg_gain / np.maximum(avg_loss, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        ticker_df['rsi_14'] = rsi.astype('float32')
        
        # 52-week high/low relative position
        rolling_high = pd.Series(close).rolling(252, min_periods=20).max().values
        rolling_low = pd.Series(close).rolling(252, min_periods=20).min().values
        
        ticker_df['pct_from_52w_high'] = ((close - rolling_high) / np.maximum(rolling_high, 0.0001) * 100).astype('float32')
        ticker_df['pct_from_52w_low'] = ((close - rolling_low) / np.maximum(rolling_low, 0.0001) * 100).astype('float32')
        
        # Volume features (if available)
        if VOLUME in ticker_df.columns:
            vol = ticker_df[VOLUME].values.astype(float)
            vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().values
            ticker_df['volume_ratio'] = (vol / np.maximum(vol_sma, 1)).astype('float32')
        
        # Range (High-Low relative to Close)
        if HIGH in ticker_df.columns and LOW in ticker_df.columns:
            high = ticker_df[HIGH].values
            low = ticker_df[LOW].values
            ticker_df['range_pct'] = ((high - low) / np.maximum(np.abs(close), 0.0001) * 100).astype('float32')
        
        result_dfs.append(ticker_df)
    
    return pd.concat(result_dfs, ignore_index=True)


def add_cyclical_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Add cyclical time features."""
    if not config.use_cyclical_time:
        return df
    
    dt = pd.to_datetime(df[TIMESTAMP], unit='ms')
    
    # Day of year (seasonality)
    day_of_year = dt.dt.dayofyear
    df[DAY_OF_YEAR_SIN] = np.sin(2 * np.pi * day_of_year / 365.25).astype('float32')
    df[DAY_OF_YEAR_COS] = np.cos(2 * np.pi * day_of_year / 365.25).astype('float32')
    
    # Day of week (weekly pattern)
    day_of_week = dt.dt.dayofweek
    df[DAY_OF_WEEK_SIN] = np.sin(2 * np.pi * day_of_week / 7).astype('float32')
    df[DAY_OF_WEEK_COS] = np.cos(2 * np.pi * day_of_week / 7).astype('float32')
    
    # Month
    month = dt.dt.month
    df[MONTH_SIN] = np.sin(2 * np.pi * month / 12).astype('float32')
    df[MONTH_COS] = np.cos(2 * np.pi * month / 12).astype('float32')
    
    return df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: ExperimentConfig):
    """Scale features using specified scaler type."""
    feature_cols = [c for c in train_df.columns 
                   if c not in [TIMESTAMP, TICKER, TARGET] 
                   and train_df[c].dtype in ['float32', 'float64', 'int32', 'int64']]
    
    # Create scaler
    if config.scaler_type == "robust":
        scaler = RobustScaler()
    elif config.scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    # Scale per ticker
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    for ticker in train_df[TICKER].unique():
        train_mask = train_df[TICKER] == ticker
        test_mask = test_df[TICKER] == ticker
        
        train_data = train_df.loc[train_mask, feature_cols].values
        
        # Handle NaN
        valid_train = ~np.isnan(train_data).any(axis=1)
        if not valid_train.any():
            continue
        
        scaler.fit(train_data[valid_train])
        
        # Transform train
        scaled_train = np.full_like(train_data, np.nan)
        scaled_train[valid_train] = scaler.transform(train_data[valid_train])
        train_scaled.loc[train_mask, feature_cols] = scaled_train
        
        # Transform test
        if test_mask.any() and ticker in test_df[TICKER].values:
            test_data = test_df.loc[test_mask, feature_cols].values
            valid_test = ~np.isnan(test_data).any(axis=1)
            if valid_test.any():
                scaled_test = np.full_like(test_data, np.nan)
                scaled_test[valid_test] = scaler.transform(test_data[valid_test])
                test_scaled.loc[test_mask, feature_cols] = scaled_test
    
    return train_scaled, test_scaled, feature_cols


def create_model(config: ExperimentConfig, class_weight: float = 1.0):
    """Create model based on configuration."""
    if config.model_type == "lightgbm":
        default_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": class_weight,
            "verbose": -1,
        }
        params = {**default_params, **config.model_params}
        return LGBMClassifier(random_state=42, **params)
    
    elif config.model_type == "random_forest":
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_split": 50,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
        }
        params = {**default_params, **config.model_params}
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    
    elif config.model_type == "gradient_boosting":
        default_params = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_samples_split": 50,
            "min_samples_leaf": 20,
            "subsample": 0.8,
        }
        params = {**default_params, **config.model_params}
        return GradientBoostingClassifier(random_state=42, **params)
    
    else:  # xgboost
        default_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 30,
            "gamma": 0.1,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "scale_pos_weight": class_weight,
        }
        params = {**default_params, **config.model_params}
        return XGBClassifier(random_state=42, **params)


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(config: ExperimentConfig) -> dict:
    """Run a single experiment with given configuration."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    long_df = load_long_data()
    
    # Prepare wide data
    print("Preparing wide format...")
    wide_df = prepare_simple_wide_data(long_df, config)
    del long_df
    gc.collect()
    
    print(f"Wide shape: {wide_df.shape}")
    print(f"Tickers: {wide_df[TICKER].nunique()}")
    
    # Add technical features
    print("Adding technical features...")
    wide_df = add_focused_technical_features(wide_df, config)
    
    # Calculate split timestamps
    max_ts = wide_df[TIMESTAMP].max()
    lookahead_ms = config.lookahead_days * MS_PER_DAY
    test_period_ms = int(config.test_period_years * 365.25 * MS_PER_DAY)
    
    test_end_ts = max_ts - lookahead_ms
    train_end_ts = test_end_ts - test_period_ms
    
    print(f"Train end: {pd.to_datetime(train_end_ts, unit='ms')}")
    print(f"Test end: {pd.to_datetime(test_end_ts, unit='ms')}")
    
    # Split
    split = split_by_timestamp(wide_df, train_end_ts, test_end_ts=test_end_ts)
    print(f"Train shape: {split.train.shape}, Test shape: {split.test.shape}")
    
    # Create labels
    print("Creating labels...")
    train_labeled = create_labels(split.train, config.lookahead_days, config.gain_threshold_pct)
    test_labeled = create_labels(
        split.test, config.lookahead_days, config.gain_threshold_pct,
        price_lookup_df=wide_df
    )
    
    del wide_df
    gc.collect()
    
    print(f"Train labeled: {len(train_labeled)}, Test labeled: {len(test_labeled)}")
    print(f"Train target rate: {train_labeled[TARGET].mean():.4f}")
    print(f"Test target rate: {test_labeled[TARGET].mean():.4f}")
    
    # Add cyclical features
    train_labeled = add_cyclical_features(train_labeled, config)
    test_labeled = add_cyclical_features(test_labeled, config)
    
    # Drop raw price columns (keep only derived features)
    drop_cols = [c for c in [CLOSE, OPEN, HIGH, LOW, VOLUME] + 
                 [f'sma_{w}' for w in config.sma_windows] if c in train_labeled.columns]
    train_features = train_labeled.drop(columns=drop_cols)
    test_features = test_labeled.drop(columns=drop_cols)
    
    # Scale features
    print("Scaling...")
    train_scaled, test_scaled, feature_cols = scale_features(train_features, test_features, config)
    
    # Handle NaN (fill with 0 for model)
    for col in feature_cols:
        train_scaled[col] = train_scaled[col].fillna(0)
        test_scaled[col] = test_scaled[col].fillna(0)
    
    # Prepare for training
    X_train = train_scaled[feature_cols].values
    y_train = train_scaled[TARGET].values
    X_test = test_scaled[feature_cols].values
    y_test = test_scaled[TARGET].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    # Calculate class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_weight = n_neg / max(n_pos, 1)
    print(f"Class weight: {class_weight:.2f}")
    
    # Train model
    print("Training model...")
    model = create_model(config, class_weight)
    model.fit(X_train, y_train)
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        imp = dict(zip(feature_cols, model.feature_importances_))
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 features:")
        for f, i in sorted_imp[:10]:
            print(f"  {f}: {i:.4f}")
    
    # Predict
    probs = model.predict_proba(X_test)[:, 1]
    
    print(f"\nPrediction distribution:")
    print(f"  Mean: {probs.mean():.4f}")
    print(f"  Std: {probs.std():.4f}")
    print(f"  Min: {probs.min():.4f}")
    print(f"  Max: {probs.max():.4f}")
    
    # Build predictions DataFrame
    predictions = test_scaled[[TIMESTAMP, TICKER]].copy()
    predictions['prediction'] = probs
    
    # Build actuals DataFrame  
    actuals = test_scaled[[TIMESTAMP, TICKER, TARGET]].copy()
    actuals = actuals.rename(columns={TARGET: 'actual'})
    
    # Get close prices for trading simulation
    test_close = test_labeled[[TIMESTAMP, TICKER, CLOSE]].copy()
    
    # Evaluate classification
    y_pred = (probs >= config.prediction_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    
    print(f"\nClassification metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    # Trading simulation
    # Rename prediction column to match expected name
    predictions['prediction_probability'] = predictions['prediction']
    
    trading, trades = run_trading_simulation(
        predictions_df=predictions,
        price_data=test_close,
        lookahead_days=config.lookahead_days,
        threshold=config.prediction_threshold,
        initial_capital=config.initial_capital,
        transaction_cost_pct=config.transaction_cost_pct,
        max_position_pct=config.max_position_size_pct,
    )
    
    # Baseline - get start and end timestamps
    start_ts = test_close[TIMESTAMP].min()
    end_ts = test_close[TIMESTAMP].max()
    
    baseline, _ = run_baseline_simulation(
        price_data=test_close,
        start_ts=start_ts,
        end_ts=end_ts,
        lookahead_days=config.lookahead_days,
        initial_capital=config.initial_capital,
        transaction_cost_pct=config.transaction_cost_pct,
    )
    
    print(f"\nTrading metrics:")
    print(f"  Total return: {trading.total_return_pct:.2f}%")
    print(f"  Sharpe ratio: {trading.sharpe_ratio:.4f}")
    print(f"  Num trades: {trading.num_trades}")
    print(f"  Median return: {trading.median_return_pct:.2f}%")
    
    print(f"\nBaseline metrics:")
    print(f"  Total return: {baseline.total_return_pct:.2f}%")
    print(f"  Sharpe ratio: {baseline.sharpe_ratio:.4f}")
    print(f"  Num trades: {baseline.num_trades}")
    
    return {
        "config": config.name,
        "classification": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        },
        "trading": {
            "total_return_pct": trading.total_return_pct,
            "sharpe_ratio": trading.sharpe_ratio,
            "num_trades": trading.num_trades,
            "median_return_pct": trading.median_return_pct,
        },
        "baseline": {
            "total_return_pct": baseline.total_return_pct,
            "sharpe_ratio": baseline.sharpe_ratio,
            "num_trades": baseline.num_trades,
        },
    }


def main():
    """Run multiple experiments to find optimal configuration."""
    results = []
    
    # Experiment 1: Baseline with focused features, XGBoost
    exp1 = ExperimentConfig(
        name="exp1_xgb_focused_features",
        use_macro_features=False,
        scaler_type="standard",
        model_type="xgboost",
    )
    results.append(run_experiment(exp1))
    
    # Experiment 2: LightGBM
    exp2 = ExperimentConfig(
        name="exp2_lgbm_focused",
        use_macro_features=False,
        scaler_type="standard", 
        model_type="lightgbm",
    )
    results.append(run_experiment(exp2))
    
    # Experiment 3: RobustScaler + XGBoost
    exp3 = ExperimentConfig(
        name="exp3_xgb_robust_scaler",
        use_macro_features=False,
        scaler_type="robust",
        model_type="xgboost",
    )
    results.append(run_experiment(exp3))
    
    # Experiment 4: Higher threshold
    exp4 = ExperimentConfig(
        name="exp4_higher_threshold",
        use_macro_features=False,
        scaler_type="standard",
        model_type="xgboost",
        prediction_threshold=0.6,
    )
    results.append(run_experiment(exp4))
    
    # Experiment 5: Shorter lookahead (60 days, 2% gain)
    exp5 = ExperimentConfig(
        name="exp5_shorter_horizon",
        lookahead_days=60,
        gain_threshold_pct=2.0,
        use_macro_features=False,
        scaler_type="standard",
        model_type="xgboost",
    )
    results.append(run_experiment(exp5))
    
    # Experiment 6: XGBoost with more trees, less depth
    exp6 = ExperimentConfig(
        name="exp6_xgb_more_trees",
        use_macro_features=False,
        scaler_type="standard",
        model_type="xgboost",
        model_params={
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.01,
            "min_child_weight": 100,
        }
    )
    results.append(run_experiment(exp6))
    
    # Experiment 7: Random Forest
    exp7 = ExperimentConfig(
        name="exp7_random_forest",
        use_macro_features=False,
        scaler_type="standard",
        model_type="random_forest",
    )
    results.append(run_experiment(exp7))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"\n{r['config']}:")
        print(f"  Sharpe: {r['trading']['sharpe_ratio']:.4f}")
        print(f"  Return: {r['trading']['total_return_pct']:.2f}%")
        print(f"  ROC AUC: {r['classification']['roc_auc']:.4f}")
        print(f"  Trades: {r['trading']['num_trades']}")
    
    # Save results
    output_path = _ml_pipeline / "output" / "experiment_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
