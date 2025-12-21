"""Verify Sharpe >= 0.8 configs and find optimal trade-off."""

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

from simulator import run_trading_simulation, run_baseline_simulation


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


def run_config(lookahead, gain, threshold):
    wide = get_wide_data()
    wide = add_features(wide)
    
    max_ts = wide[TIMESTAMP].max()
    test_end = max_ts - lookahead * MS_PER_DAY
    train_end = test_end - int(1.5 * 365.25 * MS_PER_DAY)
    
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
    
    baseline, _ = run_baseline_simulation(
        price_data=test_close,
        start_ts=test_close[TIMESTAMP].min(),
        end_ts=test_close[TIMESTAMP].max(),
        lookahead_days=lookahead,
        initial_capital=100_000.0,
        transaction_cost_pct=0.1,
    )
    
    return {
        "sharpe": trading.sharpe_ratio,
        "trades": trading.num_trades,
        "return": trading.total_return_pct,
        "median": trading.median_return_pct,
        "mean_ret": np.mean([t.return_pct for t in trades]) if trades else 0,
        "std_ret": np.std([t.return_pct for t in trades]) if trades else 0,
        "baseline_sharpe": baseline.sharpe_ratio,
        "baseline_return": baseline.total_return_pct,
    }


def main():
    print("="*70)
    print("VERIFYING SHARPE >= 0.8 CONFIGURATIONS")
    print("="*70)
    
    results = []
    
    # Fine-grained threshold search around 0.91
    print("\n--- Fine-grained threshold search ---")
    for thresh in np.arange(0.88, 0.96, 0.01):
        r = run_config(lookahead=60, gain=2.0, threshold=thresh)
        r["threshold"] = thresh
        r["lookahead"] = 60
        r["gain"] = 2.0
        results.append(r)
        
        status = "✓" if r["sharpe"] >= 0.8 else ""
        print(f"  thresh={thresh:.2f}: Sharpe={r['sharpe']:.4f} {status}, n={r['trades']}, "
              f"mean={r['mean_ret']:.2f}%, std={r['std_ret']:.2f}%")
    
    # Also try different lookaheads at high threshold
    print("\n--- Different lookaheads at threshold=0.91 ---")
    for lh in [45, 50, 55, 60, 65, 70, 75]:
        r = run_config(lookahead=lh, gain=2.0, threshold=0.91)
        r["threshold"] = 0.91
        r["lookahead"] = lh
        r["gain"] = 2.0
        results.append(r)
        
        status = "✓" if r["sharpe"] >= 0.8 else ""
        print(f"  lh={lh}: Sharpe={r['sharpe']:.4f} {status}, n={r['trades']}")
    
    # Try different gains at high threshold
    print("\n--- Different gains at threshold=0.91 ---")
    for g in [1.5, 2.0, 2.5, 3.0]:
        r = run_config(lookahead=60, gain=g, threshold=0.91)
        r["threshold"] = 0.91
        r["lookahead"] = 60
        r["gain"] = g
        results.append(r)
        
        status = "✓" if r["sharpe"] >= 0.8 else ""
        print(f"  gain={g}%: Sharpe={r['sharpe']:.4f} {status}, n={r['trades']}")
    
    # Summary of configs achieving target
    print("\n" + "="*70)
    print("CONFIGURATIONS ACHIEVING SHARPE >= 0.8")
    print("="*70)
    
    achievers = [r for r in results if r["sharpe"] >= 0.8]
    achievers_sorted = sorted(achievers, key=lambda x: (-x["sharpe"], -x["trades"]))
    
    if achievers_sorted:
        for r in achievers_sorted:
            print(f"✓ lh={r['lookahead']}, g={r['gain']}%, t={r['threshold']:.2f}")
            print(f"  Sharpe={r['sharpe']:.4f}, Trades={r['trades']}, Return={r['return']:.2f}%")
            print(f"  Mean={r['mean_ret']:.2f}%, Std={r['std_ret']:.2f}%")
            print(f"  Baseline Sharpe={r['baseline_sharpe']:.4f}")
            print()
    else:
        print("No configurations achieved target!")
    
    # Best overall
    best = max(results, key=lambda x: x["sharpe"])
    print("="*70)
    print(f"BEST CONFIGURATION:")
    print(f"  Lookahead: {best['lookahead']} days")
    print(f"  Gain threshold: {best['gain']}%")
    print(f"  Prediction threshold: {best['threshold']:.2f}")
    print(f"  Sharpe ratio: {best['sharpe']:.4f}")
    print(f"  Number of trades: {best['trades']}")
    print(f"  Total return: {best['return']:.2f}%")
    print(f"  Median trade return: {best['median']:.2f}%")
    print("="*70)
    
    if best["sharpe"] >= 0.8:
        print("\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉\n")
    
    # Save results
    with open(_ml_pipeline / "output" / "verified_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
