"""Focused experiments to achieve target Sharpe ratio >= 0.8."""

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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from simulator import run_trading_simulation, run_baseline_simulation


@dataclass  
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    lookahead_days: int = 60  # Shorter horizon is more predictable
    gain_threshold_pct: float = 2.0  # Lower threshold, easier to predict
    test_period_years: float = 1.5
    prediction_threshold: float = 0.65  # Higher confidence trades
    max_position_size_pct: float = 0.05  # Smaller positions
    initial_capital: float = 100_000.0
    transaction_cost_pct: float = 0.1
    
    # Feature engineering
    use_momentum_filter: bool = False  # Only buy stocks with positive momentum
    use_volatility_filter: bool = False  # Avoid high volatility stocks
    
    # Technical feature config
    return_windows: list = None
    volatility_windows: list = None
    sma_windows: list = None
    
    # Scaling
    scaler_type: str = "standard"
    
    # Model
    model_type: str = "xgboost"
    model_params: dict = None
    
    def __post_init__(self):
        if self.return_windows is None:
            self.return_windows = [5, 10, 20, 60]
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20]
        if self.sma_windows is None:
            self.sma_windows = [10, 20, 50]
        if self.model_params is None:
            self.model_params = {}


def prepare_simple_wide_data(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convert to wide format keeping only essential price columns."""
    # Keep only rows with ticker (non-macro data)
    long_df = long_df[long_df['ticker'].notna() & (long_df['ticker'] != '')]
    
    # Filter to only OHLCV features
    ohlcv_features = {'Open', 'High', 'Low', 'Close', 'Volume'}
    long_df = long_df[long_df['feature'].isin(ohlcv_features)]
    
    return long_to_wide(long_df)


def add_technical_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Add technical features."""
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
        
        # Daily returns for volatility calc
        daily_ret = np.zeros(n)
        daily_ret[1:] = (close[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 0.0001) * 100
        
        # Volatility
        for window in config.volatility_windows:
            vol = pd.Series(daily_ret).rolling(window, min_periods=max(1, window//2)).std().values
            ticker_df[f'volatility_{window}d'] = (vol * 100).astype('float32')
        
        # SMAs and price relative to MA
        for window in config.sma_windows:
            sma = pd.Series(close).rolling(window, min_periods=max(1, window//2)).mean().values
            ticker_df[f'sma_{window}'] = sma.astype('float32')
            ticker_df[f'price_to_sma_{window}'] = ((close / np.maximum(sma, 0.0001) - 1) * 100).astype('float32')
        
        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
        rs = avg_gain / np.maximum(avg_loss, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        ticker_df['rsi_14'] = rsi.astype('float32')
        
        # 52-week relative position
        rolling_high = pd.Series(close).rolling(252, min_periods=20).max().values
        rolling_low = pd.Series(close).rolling(252, min_periods=20).min().values
        ticker_df['pct_from_52w_high'] = ((close - rolling_high) / np.maximum(rolling_high, 0.0001) * 100).astype('float32')
        ticker_df['pct_from_52w_low'] = ((close - rolling_low) / np.maximum(rolling_low, 0.0001) * 100).astype('float32')
        
        # Volume features
        if VOLUME in ticker_df.columns:
            vol = ticker_df[VOLUME].values.astype(float)
            vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().values
            ticker_df['volume_ratio'] = (vol / np.maximum(vol_sma, 1)).astype('float32')
        
        # Range
        if HIGH in ticker_df.columns and LOW in ticker_df.columns:
            high = ticker_df[HIGH].values
            low = ticker_df[LOW].values
            ticker_df['range_pct'] = ((high - low) / np.maximum(np.abs(close), 0.0001) * 100).astype('float32')
        
        # Momentum score (for filtering)
        if config.use_momentum_filter:
            # Combine short and medium term momentum
            ret_5 = ticker_df.get('return_5d', pd.Series(np.zeros(n))).values
            ret_20 = ticker_df.get('return_20d', pd.Series(np.zeros(n))).values
            ticker_df['momentum_score'] = ((ret_5 + ret_20) / 2).astype('float32')
        
        result_dfs.append(ticker_df)
    
    return pd.concat(result_dfs, ignore_index=True)


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features."""
    dt = pd.to_datetime(df[TIMESTAMP], unit='ms')
    
    day_of_year = dt.dt.dayofyear
    df[DAY_OF_YEAR_SIN] = np.sin(2 * np.pi * day_of_year / 365.25).astype('float32')
    df[DAY_OF_YEAR_COS] = np.cos(2 * np.pi * day_of_year / 365.25).astype('float32')
    
    day_of_week = dt.dt.dayofweek
    df[DAY_OF_WEEK_SIN] = np.sin(2 * np.pi * day_of_week / 7).astype('float32')
    df[DAY_OF_WEEK_COS] = np.cos(2 * np.pi * day_of_week / 7).astype('float32')
    
    month = dt.dt.month
    df[MONTH_SIN] = np.sin(2 * np.pi * month / 12).astype('float32')
    df[MONTH_COS] = np.cos(2 * np.pi * month / 12).astype('float32')
    
    return df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: ExperimentConfig):
    """Scale features per-ticker."""
    feature_cols = [c for c in train_df.columns 
                   if c not in [TIMESTAMP, TICKER, TARGET] 
                   and train_df[c].dtype in ['float32', 'float64', 'int32', 'int64']]
    
    scaler = RobustScaler() if config.scaler_type == "robust" else StandardScaler()
    
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    for ticker in train_df[TICKER].unique():
        train_mask = train_df[TICKER] == ticker
        test_mask = test_df[TICKER] == ticker
        
        train_data = train_df.loc[train_mask, feature_cols].values
        valid_train = ~np.isnan(train_data).any(axis=1)
        
        if not valid_train.any():
            continue
        
        scaler.fit(train_data[valid_train])
        
        scaled_train = np.full_like(train_data, np.nan)
        scaled_train[valid_train] = scaler.transform(train_data[valid_train])
        train_scaled.loc[train_mask, feature_cols] = scaled_train
        
        if test_mask.any():
            test_data = test_df.loc[test_mask, feature_cols].values
            valid_test = ~np.isnan(test_data).any(axis=1)
            if valid_test.any():
                scaled_test = np.full_like(test_data, np.nan)
                scaled_test[valid_test] = scaler.transform(test_data[valid_test])
                test_scaled.loc[test_mask, feature_cols] = scaled_test
    
    return train_scaled, test_scaled, feature_cols


def create_model(config: ExperimentConfig, class_weight: float = 1.0):
    """Create model based on config."""
    if config.model_type == "lightgbm":
        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.03,
            "num_leaves": 20,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "scale_pos_weight": class_weight,
            "verbose": -1,
            **config.model_params,
        }
        return LGBMClassifier(random_state=42, **params)
    else:
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 50,
            "gamma": 0.2,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "scale_pos_weight": class_weight,
            **config.model_params,
        }
        return XGBClassifier(random_state=42, **params)


def apply_trading_filters(predictions: pd.DataFrame, test_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Apply additional filters to trading signals."""
    if not config.use_momentum_filter and not config.use_volatility_filter:
        return predictions
    
    # Merge with feature data
    merged = predictions.merge(
        test_df[[TIMESTAMP, TICKER, 'momentum_score', 'volatility_20d']] if 'momentum_score' in test_df.columns else test_df[[TIMESTAMP, TICKER]],
        on=[TIMESTAMP, TICKER],
        how='left'
    )
    
    # Apply momentum filter
    if config.use_momentum_filter and 'momentum_score' in merged.columns:
        merged.loc[merged['momentum_score'] < 0, 'prediction_probability'] = 0
    
    # Apply volatility filter (reduce probability for high vol stocks)
    if config.use_volatility_filter and 'volatility_20d' in merged.columns:
        high_vol_mask = merged['volatility_20d'] > merged['volatility_20d'].quantile(0.75)
        merged.loc[high_vol_mask, 'prediction_probability'] *= 0.5
    
    return merged[[TIMESTAMP, TICKER, 'prediction', 'prediction_probability']]


def run_experiment(config: ExperimentConfig) -> dict:
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*60}")
    
    # Load and prepare data
    long_df = load_long_data()
    wide_df = prepare_simple_wide_data(long_df)
    del long_df
    gc.collect()
    
    print(f"Wide shape: {wide_df.shape}, Tickers: {wide_df[TICKER].nunique()}")
    
    # Add technical features
    wide_df = add_technical_features(wide_df, config)
    
    # Calculate split timestamps
    max_ts = wide_df[TIMESTAMP].max()
    lookahead_ms = config.lookahead_days * MS_PER_DAY
    test_period_ms = int(config.test_period_years * 365.25 * MS_PER_DAY)
    
    test_end_ts = max_ts - lookahead_ms
    train_end_ts = test_end_ts - test_period_ms
    
    # Split
    split = split_by_timestamp(wide_df, train_end_ts, test_end_ts=test_end_ts)
    
    # Create labels
    train_labeled = create_labels(split.train, config.lookahead_days, config.gain_threshold_pct)
    test_labeled = create_labels(
        split.test, config.lookahead_days, config.gain_threshold_pct,
        price_lookup_df=wide_df
    )
    
    del wide_df
    gc.collect()
    
    print(f"Train: {len(train_labeled)}, Test: {len(test_labeled)}")
    print(f"Train target rate: {train_labeled[TARGET].mean():.4f}")
    print(f"Test target rate: {test_labeled[TARGET].mean():.4f}")
    
    # Add cyclical features
    train_labeled = add_cyclical_features(train_labeled)
    test_labeled = add_cyclical_features(test_labeled)
    
    # Keep close prices for trading
    test_close = test_labeled[[TIMESTAMP, TICKER, CLOSE]].copy()
    
    # Drop raw price columns, keep derived features
    drop_cols = [c for c in [CLOSE, OPEN, HIGH, LOW, VOLUME] + 
                 [f'sma_{w}' for w in config.sma_windows] if c in train_labeled.columns]
    train_features = train_labeled.drop(columns=drop_cols)
    test_features = test_labeled.drop(columns=drop_cols)
    
    # Scale
    train_scaled, test_scaled, feature_cols = scale_features(train_features, test_features, config)
    
    # Handle NaN
    for col in feature_cols:
        train_scaled[col] = train_scaled[col].fillna(0)
        test_scaled[col] = test_scaled[col].fillna(0)
    
    X_train = train_scaled[feature_cols].values
    y_train = train_scaled[TARGET].values
    X_test = test_scaled[feature_cols].values
    y_test = test_scaled[TARGET].values
    
    print(f"Features: {len(feature_cols)}")
    
    # Class weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_weight = n_neg / max(n_pos, 1)
    
    # Train
    model = create_model(config, class_weight)
    model.fit(X_train, y_train)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
        print("\nTop 5 features:", imp[:5])
    
    # Predict
    probs = model.predict_proba(X_test)[:, 1]
    
    print(f"\nPredictions: mean={probs.mean():.4f}, std={probs.std():.4f}")
    
    # Classification metrics
    y_pred = (probs >= config.prediction_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # Trading simulation
    predictions = test_scaled[[TIMESTAMP, TICKER]].copy()
    predictions['prediction'] = probs
    predictions['prediction_probability'] = probs
    
    # Apply filters
    predictions = apply_trading_filters(predictions, test_features, config)
    
    trading, trades = run_trading_simulation(
        predictions_df=predictions,
        price_data=test_close,
        lookahead_days=config.lookahead_days,
        threshold=config.prediction_threshold,
        initial_capital=config.initial_capital,
        transaction_cost_pct=config.transaction_cost_pct,
        max_position_pct=config.max_position_size_pct,
    )
    
    # Baseline
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
    
    print(f"\nTrading: Return={trading.total_return_pct:.2f}%, Sharpe={trading.sharpe_ratio:.4f}, Trades={trading.num_trades}")
    print(f"Median trade return: {trading.median_return_pct:.2f}%")
    print(f"Baseline: Return={baseline.total_return_pct:.2f}%, Sharpe={baseline.sharpe_ratio:.4f}")
    
    return {
        "config": config.name,
        "settings": {
            "lookahead_days": config.lookahead_days,
            "gain_threshold_pct": config.gain_threshold_pct,
            "prediction_threshold": config.prediction_threshold,
            "max_position_size_pct": config.max_position_size_pct,
        },
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
        "baseline_sharpe": baseline.sharpe_ratio,
    }


def main():
    """Run focused experiments."""
    results = []
    
    # Exp 1: Baseline - 60 day, 2%, threshold 0.65
    results.append(run_experiment(ExperimentConfig(
        name="baseline_60d_2pct_t065",
        lookahead_days=60,
        gain_threshold_pct=2.0,
        prediction_threshold=0.65,
    )))
    
    # Exp 2: Higher threshold 0.70
    results.append(run_experiment(ExperimentConfig(
        name="higher_threshold_070",
        lookahead_days=60,
        gain_threshold_pct=2.0,
        prediction_threshold=0.70,
    )))
    
    # Exp 3: Even higher threshold 0.75
    results.append(run_experiment(ExperimentConfig(
        name="very_high_threshold_075",
        lookahead_days=60,
        gain_threshold_pct=2.0,
        prediction_threshold=0.75,
    )))
    
    # Exp 4: With momentum filter
    results.append(run_experiment(ExperimentConfig(
        name="momentum_filter",
        lookahead_days=60,
        gain_threshold_pct=2.0,
        prediction_threshold=0.65,
        use_momentum_filter=True,
    )))
    
    # Exp 5: Smaller positions (2%)
    results.append(run_experiment(ExperimentConfig(
        name="small_positions_2pct",
        lookahead_days=60,
        gain_threshold_pct=2.0,
        prediction_threshold=0.65,
        max_position_size_pct=0.02,
    )))
    
    # Exp 6: 30-day horizon, 1.5% gain
    results.append(run_experiment(ExperimentConfig(
        name="short_30d_15pct",
        lookahead_days=30,
        gain_threshold_pct=1.5,
        prediction_threshold=0.60,
    )))
    
    # Exp 7: LightGBM with high threshold
    results.append(run_experiment(ExperimentConfig(
        name="lgbm_high_threshold",
        lookahead_days=60,
        gain_threshold_pct=2.0,
        prediction_threshold=0.70,
        model_type="lightgbm",
    )))
    
    # Exp 8: Very conservative - tiny positions, high threshold
    results.append(run_experiment(ExperimentConfig(
        name="ultra_conservative",
        lookahead_days=60,
        gain_threshold_pct=2.0,
        prediction_threshold=0.80,
        max_position_size_pct=0.01,
    )))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Sorted by Sharpe Ratio")
    print("="*80)
    
    sorted_results = sorted(results, key=lambda x: x['trading']['sharpe_ratio'], reverse=True)
    
    for r in sorted_results:
        print(f"\n{r['config']}:")
        print(f"  Sharpe: {r['trading']['sharpe_ratio']:.4f} (baseline: {r['baseline_sharpe']:.4f})")
        print(f"  Return: {r['trading']['total_return_pct']:.2f}%")
        print(f"  Trades: {r['trading']['num_trades']}, Median: {r['trading']['median_return_pct']:.2f}%")
        print(f"  ROC AUC: {r['classification']['roc_auc']:.4f}")
    
    # Save
    output_path = _ml_pipeline / "output" / "focused_experiment_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
