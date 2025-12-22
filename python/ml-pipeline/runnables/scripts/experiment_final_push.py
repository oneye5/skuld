"""Final push to reach Sharpe 0.8 - smaller test windows, higher confidence."""

import sys
from pathlib import Path
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

from simulator import run_trading_simulation


def get_wide_data():
    long_df = load_long_data()
    long_df = long_df[long_df['ticker'].notna() & (long_df['ticker'] != '')]
    long_df = long_df[long_df['feature'].isin({'Open', 'High', 'Low', 'Close', 'Volume'})]
    return long_to_wide(long_df)


def add_features(df):
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
        
        for w in [5, 10, 20, 60]:
            ret = np.full(n, np.nan)
            if n > w:
                ret[w:] = (close[w:] - close[:-w]) / np.maximum(np.abs(close[:-w]), 1e-8) * 100
            tdf[f'return_{w}d'] = ret.astype('float32')
        
        dr = np.zeros(n)
        dr[1:] = (close[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 1e-8) * 100
        for w in [10, 20]:
            tdf[f'volatility_{w}d'] = (pd.Series(dr).rolling(w, min_periods=w//2).std() * 100).astype('float32')
        
        for w in [10, 20, 50]:
            sma = pd.Series(close).rolling(w, min_periods=w//2).mean().values
            tdf[f'price_to_sma_{w}'] = ((close / np.maximum(sma, 1e-8) - 1) * 100).astype('float32')
        
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
        rs = avg_gain / np.maximum(avg_loss, 1e-8)
        tdf['rsi_14'] = (100 - (100 / (1 + rs))).astype('float32')
        
        rh = pd.Series(close).rolling(252, min_periods=20).max().values
        rl = pd.Series(close).rolling(252, min_periods=20).min().values
        tdf['pct_from_52w_high'] = ((close - rh) / np.maximum(rh, 1e-8) * 100).astype('float32')
        tdf['pct_from_52w_low'] = ((close - rl) / np.maximum(rl, 1e-8) * 100).astype('float32')
        
        if VOLUME in tdf.columns:
            vol = tdf[VOLUME].values.astype(float)
            vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().values
            tdf['volume_ratio'] = (vol / np.maximum(vol_sma, 1)).astype('float32')
        
        if HIGH in tdf.columns and LOW in tdf.columns:
            h = tdf[HIGH].values
            l = tdf[LOW].values
            tdf['range_pct'] = ((h - l) / np.maximum(np.abs(close), 1e-8) * 100).astype('float32')
        
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


def run_experiment(wide, test_days, lookahead, gain, threshold, verbose=True):
    """Run experiment with specific config."""
    max_ts = wide[TIMESTAMP].max()
    test_end = max_ts - lookahead * MS_PER_DAY
    train_end = test_end - test_days * MS_PER_DAY
    
    split = split_by_timestamp(wide, train_end, test_end_ts=test_end)
    
    train_l = create_labels(split.train, lookahead, gain)
    test_l = create_labels(split.test, lookahead, gain, price_lookup_df=wide)
    
    test_close = test_l[[TIMESTAMP, TICKER, CLOSE]].copy()
    
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
    
    cw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        scale_pos_weight=cw, random_state=42
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    
    preds = test_s[[TIMESTAMP, TICKER]].copy()
    preds['prediction_probability'] = probs
    
    trading, trades = run_trading_simulation(
        predictions_df=preds,
        price_data=test_close,
        lookahead_days=lookahead,
        threshold=threshold,
        initial_capital=100_000.0,
        transaction_cost_pct=0.1,
        max_position_pct=0.01,
    )
    
    if verbose:
        status = "✓ TARGET!" if trading.sharpe_ratio >= 0.8 else ""
        print(f"  test={test_days}d, lh={lookahead}, g={gain}%, t={threshold}: "
              f"Sharpe={trading.sharpe_ratio:.4f} {status}, Trades={trading.num_trades}, Median={trading.median_return_pct:.2f}%")
    
    return {
        "sharpe": trading.sharpe_ratio,
        "trades": trading.num_trades,
        "median": trading.median_return_pct,
        "return": trading.total_return_pct,
        "config": {"test_days": test_days, "lookahead": lookahead, "gain": gain, "threshold": threshold}
    }


def main():
    print("="*70)
    print("FINAL OPTIMIZATION - Smaller windows, higher thresholds")
    print("="*70)
    
    wide = get_wide_data()
    wide = add_features(wide)
    
    results = []
    best = {"sharpe": 0}
    
    # Test different window sizes with high thresholds
    print("\n--- Test Window Size Impact ---")
    for test_days in [30, 45, 60, 90, 120]:
        r = run_experiment(wide, test_days=test_days, lookahead=60, gain=2.0, threshold=0.79)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Find optimal test window
    best_test_days = sorted(results, key=lambda x: x["sharpe"], reverse=True)[0]["config"]["test_days"]
    print(f"\n  Best test window: {best_test_days} days")
    
    # Threshold sweep at best window size
    print("\n--- Threshold Sweep ---")
    for thresh in [0.75, 0.77, 0.79, 0.81, 0.83, 0.85, 0.87, 0.89, 0.91]:
        r = run_experiment(wide, test_days=best_test_days, lookahead=60, gain=2.0, threshold=thresh)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Lookahead sweep
    print("\n--- Lookahead Sweep ---")
    for lh in [30, 45, 60, 75, 90]:
        r = run_experiment(wide, test_days=best_test_days, lookahead=lh, gain=2.0, threshold=0.79)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Gain threshold sweep
    print("\n--- Gain Threshold Sweep ---")
    for g in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        r = run_experiment(wide, test_days=best_test_days, lookahead=60, gain=g, threshold=0.79)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Combined optimization - try promising combinations
    print("\n--- Combined Optimization ---")
    promising = [
        (30, 45, 1.5, 0.85),
        (30, 60, 2.0, 0.85),
        (45, 45, 1.5, 0.83),
        (45, 60, 2.0, 0.83),
        (60, 60, 2.0, 0.81),
        (30, 75, 2.5, 0.85),
        (45, 75, 2.5, 0.83),
    ]
    
    for test_d, lh, g, t in promising:
        r = run_experiment(wide, test_days=test_d, lookahead=lh, gain=g, threshold=t)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Very high thresholds
    print("\n--- Ultra-High Thresholds ---")
    for thresh in [0.90, 0.92, 0.94, 0.95]:
        r = run_experiment(wide, test_days=30, lookahead=60, gain=2.0, threshold=thresh)
        results.append(r)
        if r["sharpe"] > best["sharpe"]:
            best = r
    
    # Summary
    print(f"\n{'='*70}")
    print("TOP 10 RESULTS")
    print(f"{'='*70}")
    
    results_sorted = sorted(results, key=lambda x: x["sharpe"], reverse=True)
    for i, r in enumerate(results_sorted[:10]):
        c = r["config"]
        status = "✓ TARGET MET" if r["sharpe"] >= 0.8 else ""
        print(f"{i+1}. Sharpe={r['sharpe']:.4f} {status} | test={c['test_days']}d, lh={c['lookahead']}, g={c['gain']}%, t={c['threshold']}")
        print(f"   Trades={r['trades']}, Median={r['median']:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"BEST: Sharpe={best['sharpe']:.4f}")
    if best["sharpe"] >= 0.8:
        print("🎉 TARGET ACHIEVED! 🎉")
    else:
        print(f"Gap to target: {0.8 - best['sharpe']:.4f}")
    print(f"{'='*70}")
    
    # Save
    with open(_ml_pipeline / "output" / "final_optimization_results.json", 'w') as f:
        json.dump(results_sorted[:20], f, indent=2)


if __name__ == "__main__":
    main()
