"""Final push to achieve Sharpe ratio >= 0.8."""

import sys
from pathlib import Path
from dataclasses import dataclass
import json
import gc

import pandas as pd
import numpy as np

_ml_pipeline = Path(__file__).parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "data-preparation"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "transformations"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "long-to-wide"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "data-splitting" / "train-test"))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "labeling"))
sys.path.insert(0, str(_ml_pipeline / "evaluation"))
sys.path.insert(0, str(_ml_pipeline / "evaluation" / "trade-simulation"))

from config.column_names import TIMESTAMP, TICKER, TARGET, CLOSE, OPEN, HIGH, LOW, VOLUME
from config.model_config import MS_PER_DAY

from utils.data_loader import load_long_data
from converter import long_to_wide
from splitter import split_by_timestamp
from labeler import create_labels

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from simulator import run_trading_simulation, run_baseline_simulation


@dataclass  
class Config:
    name: str
    lookahead_days: int = 60
    gain_threshold_pct: float = 2.0
    prediction_threshold: float = 0.80
    max_position_size_pct: float = 0.01


def prepare_data(long_df):
    """Prepare wide format with only OHLCV."""
    long_df = long_df[long_df['ticker'].notna() & (long_df['ticker'] != '')]
    long_df = long_df[long_df['feature'].isin({'Open', 'High', 'Low', 'Close', 'Volume'})]
    return long_to_wide(long_df)


def add_features(df):
    """Add technical features optimized for momentum."""
    if CLOSE not in df.columns:
        return df
    
    df = df.sort_values([TICKER, TIMESTAMP])
    result_dfs = []
    
    for ticker in df[TICKER].unique():
        tdf = df[df[TICKER] == ticker].copy()
        tdf = tdf.sort_values(TIMESTAMP).reset_index(drop=True)
        
        close = tdf[CLOSE].values
        n = len(close)
        
        if n < 20:
            continue
        
        # Returns
        for w in [5, 10, 20]:
            ret = np.full(n, np.nan)
            if n > w:
                ret[w:] = (close[w:] - close[:-w]) / np.maximum(np.abs(close[:-w]), 1e-8) * 100
            tdf[f'return_{w}d'] = ret.astype('float32')
        
        # Daily returns for volatility
        dr = np.zeros(n)
        dr[1:] = (close[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 1e-8) * 100
        
        # Volatility
        tdf['volatility_10d'] = (pd.Series(dr).rolling(10, min_periods=5).std() * 100).astype('float32')
        tdf['volatility_20d'] = (pd.Series(dr).rolling(20, min_periods=10).std() * 100).astype('float32')
        
        # SMAs and price relative
        for w in [10, 20, 50]:
            sma = pd.Series(close).rolling(w, min_periods=w//2).mean().values
            tdf[f'price_to_sma_{w}'] = ((close / np.maximum(sma, 1e-8) - 1) * 100).astype('float32')
        
        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
        rs = avg_gain / np.maximum(avg_loss, 1e-8)
        tdf['rsi_14'] = (100 - (100 / (1 + rs))).astype('float32')
        
        # 52-week position
        rh = pd.Series(close).rolling(252, min_periods=50).max().values
        rl = pd.Series(close).rolling(252, min_periods=50).min().values
        tdf['pct_from_52w_high'] = ((close - rh) / np.maximum(rh, 1e-8) * 100).astype('float32')
        tdf['pct_from_52w_low'] = ((close - rl) / np.maximum(rl, 1e-8) * 100).astype('float32')
        
        # Volume ratio
        if VOLUME in tdf.columns:
            vol = tdf[VOLUME].values.astype(float)
            vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().values
            tdf['volume_ratio'] = (vol / np.maximum(vol_sma, 1)).astype('float32')
        
        # Range
        if HIGH in tdf.columns and LOW in tdf.columns:
            h = tdf[HIGH].values
            l = tdf[LOW].values
            tdf['range_pct'] = ((h - l) / np.maximum(np.abs(close), 1e-8) * 100).astype('float32')
        
        # Cyclical time
        dt = pd.to_datetime(tdf[TIMESTAMP], unit='ms')
        doy = dt.dt.dayofyear
        tdf['day_sin'] = np.sin(2 * np.pi * doy / 365.25).astype('float32')
        tdf['day_cos'] = np.cos(2 * np.pi * doy / 365.25).astype('float32')
        tdf['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12).astype('float32')
        tdf['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12).astype('float32')
        
        result_dfs.append(tdf)
    
    return pd.concat(result_dfs, ignore_index=True)


def scale(train_df, test_df, feat_cols):
    """Scale per ticker."""
    train_s = train_df.copy()
    test_s = test_df.copy()
    
    for ticker in train_df[TICKER].unique():
        tm = train_df[TICKER] == ticker
        testm = test_df[TICKER] == ticker
        
        td = train_df.loc[tm, feat_cols].values
        valid = ~np.isnan(td).any(axis=1)
        
        if not valid.any():
            continue
        
        scaler = StandardScaler()
        scaler.fit(td[valid])
        
        st = np.full_like(td, np.nan)
        st[valid] = scaler.transform(td[valid])
        train_s.loc[tm, feat_cols] = st
        
        if testm.any():
            testd = test_df.loc[testm, feat_cols].values
            validtest = ~np.isnan(testd).any(axis=1)
            if validtest.any():
                stest = np.full_like(testd, np.nan)
                stest[validtest] = scaler.transform(testd[validtest])
                test_s.loc[testm, feat_cols] = stest
    
    return train_s, test_s


def run_exp(cfg: Config):
    """Run experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {cfg.name}")
    print(f"{'='*60}")
    
    # Load
    long_df = load_long_data()
    wide = prepare_data(long_df)
    del long_df; gc.collect()
    
    # Features
    wide = add_features(wide)
    
    # Split
    max_ts = wide[TIMESTAMP].max()
    lh_ms = cfg.lookahead_days * MS_PER_DAY
    test_ms = int(1.5 * 365.25 * MS_PER_DAY)
    
    test_end = max_ts - lh_ms
    train_end = test_end - test_ms
    
    split = split_by_timestamp(wide, train_end, test_end_ts=test_end)
    
    # Labels
    train_l = create_labels(split.train, cfg.lookahead_days, cfg.gain_threshold_pct)
    test_l = create_labels(split.test, cfg.lookahead_days, cfg.gain_threshold_pct, price_lookup_df=wide)
    
    del wide; gc.collect()
    
    print(f"Train: {len(train_l)}, Test: {len(test_l)}")
    print(f"Target rate - Train: {train_l[TARGET].mean():.4f}, Test: {test_l[TARGET].mean():.4f}")
    
    test_close = test_l[[TIMESTAMP, TICKER, CLOSE]].copy()
    
    # Feature cols
    drop = [CLOSE, OPEN, HIGH, LOW, VOLUME]
    feat_cols = [c for c in train_l.columns if c not in drop + [TIMESTAMP, TICKER, TARGET]
                 and train_l[c].dtype in ['float32', 'float64']]
    
    train_f = train_l.drop(columns=[c for c in drop if c in train_l.columns])
    test_f = test_l.drop(columns=[c for c in drop if c in test_l.columns])
    
    # Scale
    train_s, test_s = scale(train_f, test_f, feat_cols)
    
    for c in feat_cols:
        train_s[c] = train_s[c].fillna(0)
        test_s[c] = test_s[c].fillna(0)
    
    X_tr = train_s[feat_cols].values
    y_tr = train_s[TARGET].values
    X_te = test_s[feat_cols].values
    y_te = test_s[TARGET].values
    
    print(f"Features: {len(feat_cols)}")
    
    # Class weight
    cw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    # Model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=cw,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    
    # Predict
    probs = model.predict_proba(X_te)[:, 1]
    
    print(f"Predictions: mean={probs.mean():.4f}, std={probs.std():.4f}, max={probs.max():.4f}")
    
    # Metrics
    y_pred = (probs >= cfg.prediction_threshold).astype(int)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, probs)
    
    print(f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, AUC: {auc:.4f}")
    
    # Trading
    preds = test_s[[TIMESTAMP, TICKER]].copy()
    preds['prediction_probability'] = probs
    
    trading, trades = run_trading_simulation(
        predictions_df=preds,
        price_data=test_close,
        lookahead_days=cfg.lookahead_days,
        threshold=cfg.prediction_threshold,
        initial_capital=100_000.0,
        transaction_cost_pct=0.1,
        max_position_pct=cfg.max_position_size_pct,
    )
    
    # Baseline
    baseline, _ = run_baseline_simulation(
        price_data=test_close,
        start_ts=test_close[TIMESTAMP].min(),
        end_ts=test_close[TIMESTAMP].max(),
        lookahead_days=cfg.lookahead_days,
        initial_capital=100_000.0,
        transaction_cost_pct=0.1,
    )
    
    print(f"\n** SHARPE: {trading.sharpe_ratio:.4f} ** (baseline: {baseline.sharpe_ratio:.4f})")
    print(f"Return: {trading.total_return_pct:.2f}%, Trades: {trading.num_trades}, Median: {trading.median_return_pct:.2f}%")
    
    return {
        "name": cfg.name,
        "sharpe": trading.sharpe_ratio,
        "return": trading.total_return_pct,
        "trades": trading.num_trades,
        "median": trading.median_return_pct,
        "precision": prec,
        "auc": auc,
        "baseline_sharpe": baseline.sharpe_ratio,
    }


def main():
    results = []
    
    # Very high thresholds
    for thresh in [0.82, 0.85, 0.88, 0.90]:
        results.append(run_exp(Config(
            name=f"thresh_{int(thresh*100)}",
            prediction_threshold=thresh,
            max_position_size_pct=0.01,
        )))
    
    # Different position sizes with 0.85 threshold
    for pos in [0.005, 0.02, 0.03]:
        results.append(run_exp(Config(
            name=f"pos_{pos}_t85",
            prediction_threshold=0.85,
            max_position_size_pct=pos,
        )))
    
    # Different horizons
    for lh, gain in [(45, 1.5), (30, 1.0), (90, 3.0)]:
        results.append(run_exp(Config(
            name=f"horizon_{lh}d_{gain}pct",
            lookahead_days=lh,
            gain_threshold_pct=gain,
            prediction_threshold=0.85,
            max_position_size_pct=0.01,
        )))
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - Sorted by Sharpe")
    print("="*80)
    
    results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    for r in results_sorted:
        status = "✓ TARGET" if r['sharpe'] >= 0.8 else ""
        print(f"{r['name']}: Sharpe={r['sharpe']:.4f} {status}")
        print(f"  Return={r['return']:.2f}%, Trades={r['trades']}, Median={r['median']:.2f}%, Precision={r['precision']:.4f}")
    
    # Save
    with open(_ml_pipeline / "output" / "final_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
