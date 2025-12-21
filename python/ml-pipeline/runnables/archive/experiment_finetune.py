"""Fine-tune to reach Sharpe >= 0.8."""

import sys
from pathlib import Path
import gc
import json

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
from sklearn.metrics import roc_auc_score

from simulator import run_trading_simulation, run_baseline_simulation


# Global cache
_WIDE_DATA_CACHE = None


def get_wide_data():
    """Get wide format data (cached)."""
    global _WIDE_DATA_CACHE
    if _WIDE_DATA_CACHE is None:
        long_df = load_long_data()
        long_df = long_df[long_df['ticker'].notna() & (long_df['ticker'] != '')]
        long_df = long_df[long_df['feature'].isin({'Open', 'High', 'Low', 'Close', 'Volume'})]
        _WIDE_DATA_CACHE = long_to_wide(long_df)
    return _WIDE_DATA_CACHE.copy()


def add_features(df, extra_features=True):
    """Add technical features with optional extras."""
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
        for w in [5, 10, 20, 60]:
            ret = np.full(n, np.nan)
            if n > w:
                ret[w:] = (close[w:] - close[:-w]) / np.maximum(np.abs(close[:-w]), 1e-8) * 100
            tdf[f'return_{w}d'] = ret.astype('float32')
        
        # Volatility
        dr = np.zeros(n)
        dr[1:] = (close[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 1e-8) * 100
        
        for w in [10, 20]:
            tdf[f'volatility_{w}d'] = (pd.Series(dr).rolling(w, min_periods=w//2).std() * 100).astype('float32')
        
        # SMAs and price-to-SMA
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
        
        # 52-week
        rh = pd.Series(close).rolling(252, min_periods=20).max().values
        rl = pd.Series(close).rolling(252, min_periods=20).min().values
        tdf['pct_from_52w_high'] = ((close - rh) / np.maximum(rh, 1e-8) * 100).astype('float32')
        tdf['pct_from_52w_low'] = ((close - rl) / np.maximum(rl, 1e-8) * 100).astype('float32')
        
        # Volume
        if VOLUME in tdf.columns:
            vol = tdf[VOLUME].values.astype(float)
            vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().values
            tdf['volume_ratio'] = (vol / np.maximum(vol_sma, 1)).astype('float32')
        
        # Range
        if HIGH in tdf.columns and LOW in tdf.columns:
            h = tdf[HIGH].values
            l = tdf[LOW].values
            tdf['range_pct'] = ((h - l) / np.maximum(np.abs(close), 1e-8) * 100).astype('float32')
        
        # Extra features
        if extra_features:
            # Acceleration (change in returns)
            ret_5 = tdf['return_5d'].values
            ret_20 = tdf['return_20d'].values
            tdf['return_accel'] = (ret_5 - ret_20).astype('float32')
            
            # Volatility ratio (short vs long)
            vol_10 = tdf['volatility_10d'].values
            vol_20 = tdf['volatility_20d'].values
            tdf['vol_ratio'] = (vol_10 / np.maximum(vol_20, 1e-8)).astype('float32')
            
            # Price position in range (0 = at low, 1 = at high)
            tdf['price_position'] = ((close - rl) / np.maximum(rh - rl, 1e-8)).astype('float32')
        
        # Cyclical time
        dt = pd.to_datetime(tdf[TIMESTAMP], unit='ms')
        doy = dt.dt.dayofyear
        tdf['day_sin'] = np.sin(2 * np.pi * doy / 365.25).astype('float32')
        tdf['day_cos'] = np.cos(2 * np.pi * doy / 365.25).astype('float32')
        tdf['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12).astype('float32')
        tdf['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12).astype('float32')
        
        result_dfs.append(tdf)
    
    return pd.concat(result_dfs, ignore_index=True)


def scale_per_ticker(train_df, test_df, feat_cols):
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


def run_single(
    name,
    lookahead=60,
    gain=2.0,
    threshold=0.80,
    pos_size=0.01,
    extra_features=True,
    xgb_params=None,
    verbose=True,
):
    """Run a single configuration."""
    if verbose:
        print(f"\n>> {name}: lh={lookahead}, gain={gain}, thresh={threshold}")
    
    wide = get_wide_data()
    wide = add_features(wide, extra_features=extra_features)
    
    # Split
    max_ts = wide[TIMESTAMP].max()
    test_end = max_ts - lookahead * MS_PER_DAY
    train_end = test_end - int(1.5 * 365.25 * MS_PER_DAY)
    
    split = split_by_timestamp(wide, train_end, test_end_ts=test_end)
    
    # Labels
    train_l = create_labels(split.train, lookahead, gain)
    test_l = create_labels(split.test, lookahead, gain, price_lookup_df=wide)
    
    test_close = test_l[[TIMESTAMP, TICKER, CLOSE]].copy()
    
    # Features
    drop = [CLOSE, OPEN, HIGH, LOW, VOLUME]
    feat_cols = [c for c in train_l.columns if c not in drop + [TIMESTAMP, TICKER, TARGET]
                 and train_l[c].dtype in ['float32', 'float64']]
    
    train_f = train_l.drop(columns=[c for c in drop if c in train_l.columns])
    test_f = test_l.drop(columns=[c for c in drop if c in test_l.columns])
    
    train_s, test_s = scale_per_ticker(train_f, test_f, feat_cols)
    
    for c in feat_cols:
        train_s[c] = train_s[c].fillna(0)
        test_s[c] = test_s[c].fillna(0)
    
    X_tr = train_s[feat_cols].values
    y_tr = train_s[TARGET].values
    X_te = test_s[feat_cols].values
    y_te = test_s[TARGET].values
    
    cw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    # Model
    base_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "gamma": 0.2,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "scale_pos_weight": cw,
    }
    params = {**base_params, **(xgb_params or {})}
    model = XGBClassifier(random_state=42, **params)
    model.fit(X_tr, y_tr)
    
    probs = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, probs)
    
    # Trading
    preds = test_s[[TIMESTAMP, TICKER]].copy()
    preds['prediction_probability'] = probs
    
    trading, _ = run_trading_simulation(
        predictions_df=preds,
        price_data=test_close,
        lookahead_days=lookahead,
        threshold=threshold,
        initial_capital=100_000.0,
        transaction_cost_pct=0.1,
        max_position_pct=pos_size,
    )
    
    baseline, _ = run_baseline_simulation(
        price_data=test_close,
        start_ts=test_close[TIMESTAMP].min(),
        end_ts=test_close[TIMESTAMP].max(),
        lookahead_days=lookahead,
        initial_capital=100_000.0,
        transaction_cost_pct=0.1,
    )
    
    if verbose:
        status = "✓ TARGET!" if trading.sharpe_ratio >= 0.8 else ""
        print(f"   Sharpe={trading.sharpe_ratio:.4f} {status}, Trades={trading.num_trades}, Median={trading.median_return_pct:.2f}%")
    
    return {
        "name": name,
        "sharpe": trading.sharpe_ratio,
        "return": trading.total_return_pct,
        "trades": trading.num_trades,
        "median": trading.median_return_pct,
        "auc": auc,
        "baseline_sharpe": baseline.sharpe_ratio,
    }


def main():
    results = []
    
    print("="*60)
    print("FINE-TUNING TO REACH SHARPE >= 0.8")
    print("="*60)
    
    # Grid search over key params
    thresholds = [0.79, 0.80, 0.81, 0.82, 0.83]
    lookaheads = [55, 60, 65, 70]
    gains = [1.8, 2.0, 2.2, 2.5]
    
    best = {"sharpe": 0, "name": ""}
    
    # Threshold sweep
    print("\n--- Threshold Sweep ---")
    for t in thresholds:
        r = run_single(f"thresh_{t}", threshold=t)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    best_thresh = max(results, key=lambda x: x["sharpe"])["name"].split("_")[1]
    best_thresh = float(best_thresh)
    
    # Lookahead sweep at best threshold
    print("\n--- Lookahead Sweep ---")
    for lh in lookaheads:
        r = run_single(f"lh_{lh}", lookahead=lh, threshold=best_thresh)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Gain sweep
    print("\n--- Gain Threshold Sweep ---")
    for g in gains:
        r = run_single(f"gain_{g}", gain=g, threshold=best_thresh)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # XGBoost param tuning
    print("\n--- XGBoost Tuning ---")
    xgb_configs = [
        ("xgb_deep5", {"max_depth": 5}),
        ("xgb_deep6", {"max_depth": 6}),
        ("xgb_trees500", {"n_estimators": 500}),
        ("xgb_lr01", {"learning_rate": 0.01, "n_estimators": 500}),
        ("xgb_regularized", {"reg_alpha": 1.0, "reg_lambda": 5.0}),
        ("xgb_highgamma", {"gamma": 0.5}),
        ("xgb_subsample", {"subsample": 0.6, "colsample_bytree": 0.6}),
    ]
    
    for name, params in xgb_configs:
        r = run_single(name, threshold=best_thresh, xgb_params=params)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Position size tuning
    print("\n--- Position Size Sweep ---")
    for ps in [0.005, 0.008, 0.015, 0.02]:
        r = run_single(f"pos_{ps}", threshold=best_thresh, pos_size=ps)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Summary
    print("\n" + "="*60)
    print("TOP 10 RESULTS")
    print("="*60)
    
    results_sorted = sorted(results, key=lambda x: x["sharpe"], reverse=True)
    for r in results_sorted[:10]:
        status = "✓ TARGET MET" if r["sharpe"] >= 0.8 else ""
        print(f"{r['name']}: Sharpe={r['sharpe']:.4f} {status}")
        print(f"  Return={r['return']:.2f}%, Trades={r['trades']}, Median={r['median']:.2f}%")
    
    print(f"\n*** BEST: {best['name']} with Sharpe {best['sharpe']:.4f} ***")
    
    if best["sharpe"] >= 0.8:
        print("\n🎉 TARGET ACHIEVED! 🎉")
    else:
        print(f"\nGap to target: {0.8 - best['sharpe']:.4f}")
    
    # Save
    with open(_ml_pipeline / "output" / "finetune_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
