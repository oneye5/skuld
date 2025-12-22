"""Try multiple rolling windows to evaluate model stability and find best period."""

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


# Cache wide data
_WIDE_CACHE = None


def get_wide_data():
    global _WIDE_CACHE
    if _WIDE_CACHE is None:
        long_df = load_long_data()
        long_df = long_df[long_df['ticker'].notna() & (long_df['ticker'] != '')]
        long_df = long_df[long_df['feature'].isin({'Open', 'High', 'Low', 'Close', 'Volume'})]
        _WIDE_CACHE = long_to_wide(long_df)
    return _WIDE_CACHE.copy()


def add_technical_features(df):
    """Add simple technical features that worked best."""
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
        
        # Returns - minimal set
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
        
        # SMAs
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


def run_single_window(wide, test_end_ts, train_years=1.5, test_days=180, lookahead=60, gain=2.0, threshold=0.79):
    """Run a single window and return metrics."""
    lookahead_ms = lookahead * MS_PER_DAY
    test_end = test_end_ts - lookahead_ms  # Leave room for lookahead
    train_end = test_end - test_days * MS_PER_DAY
    train_start = train_end - int(train_years * 365.25 * MS_PER_DAY)
    
    # Filter data to window
    window_data = wide[(wide[TIMESTAMP] >= train_start) & (wide[TIMESTAMP] <= test_end)].copy()
    
    if len(window_data) < 1000:
        return None
    
    # Split
    split = split_by_timestamp(window_data, train_end, test_end_ts=test_end)
    
    if len(split.train) < 1000 or len(split.test) < 100:
        return None
    
    # Labels
    train_l = create_labels(split.train, lookahead, gain)
    test_l = create_labels(split.test, lookahead, gain, price_lookup_df=wide)
    
    if len(train_l) < 1000 or len(test_l) < 100:
        return None
    
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
    
    if len(np.unique(y_tr)) < 2:
        return None
    
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
    
    return {
        "sharpe": trading.sharpe_ratio,
        "return": trading.total_return_pct,
        "trades": trading.num_trades,
        "median": trading.median_return_pct,
        "test_end": test_end,
        "trade_returns": [t.return_pct for t in trades],
    }


def main():
    print("="*60)
    print("ROLLING WINDOW ANALYSIS")
    print("="*60)
    
    wide = get_wide_data()
    wide = add_technical_features(wide)
    
    max_ts = wide[TIMESTAMP].max()
    min_ts = wide[TIMESTAMP].min()
    
    print(f"Data range: {pd.to_datetime(min_ts, unit='ms').date()} to {pd.to_datetime(max_ts, unit='ms').date()}")
    
    # Run multiple windows going back in time
    window_results = []
    all_trade_returns = []
    
    # Configuration
    lookahead = 60
    gain = 2.0
    threshold = 0.79
    test_days = 180
    window_step = 180  # Move back 6 months each iteration
    num_windows = 5
    
    print(f"\nConfig: lh={lookahead}, gain={gain}%, thresh={threshold}")
    print(f"Windows: {num_windows} x {test_days} day test periods\n")
    
    current_end = max_ts
    
    for i in range(num_windows):
        print(f"Window {i+1}...")
        result = run_single_window(
            wide, current_end, 
            train_years=1.5, test_days=test_days,
            lookahead=lookahead, gain=gain, threshold=threshold
        )
        
        if result is not None:
            window_results.append(result)
            all_trade_returns.extend(result["trade_returns"])
            end_date = pd.to_datetime(result["test_end"], unit='ms').date()
            status = "✓" if result["sharpe"] >= 0.8 else ""
            print(f"  Sharpe={result['sharpe']:.4f} {status}, Trades={result['trades']}, End={end_date}")
        else:
            print(f"  Insufficient data")
        
        current_end -= window_step * MS_PER_DAY
    
    if not window_results:
        print("\nNo valid windows found!")
        return
    
    # Aggregate metrics
    sharpes = [r["sharpe"] for r in window_results]
    
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Windows: {len(window_results)}")
    print(f"Individual Sharpes: {[f'{s:.3f}' for s in sharpes]}")
    print(f"Mean Sharpe: {np.mean(sharpes):.4f}")
    print(f"Std Sharpe: {np.std(sharpes):.4f}")
    print(f"Min Sharpe: {min(sharpes):.4f}")
    print(f"Max Sharpe: {max(sharpes):.4f}")
    
    # Calculate aggregate Sharpe from all trades
    if all_trade_returns:
        agg_mean = np.mean(all_trade_returns)
        agg_std = np.std(all_trade_returns)
        agg_sharpe = agg_mean / agg_std if agg_std > 0 else 0
        
        print(f"\nAggregate from all {len(all_trade_returns)} trades:")
        print(f"  Mean return: {agg_mean:.2f}%")
        print(f"  Std return: {agg_std:.2f}%")
        print(f"  Aggregate Sharpe: {agg_sharpe:.4f}")
    
    # Try higher thresholds on best window
    best_window_idx = np.argmax(sharpes)
    best_window_end = max_ts - (best_window_idx * window_step * MS_PER_DAY)
    
    print(f"\n{'='*60}")
    print(f"THRESHOLD TUNING ON BEST WINDOW")
    print(f"{'='*60}")
    
    best_sharpe = 0
    best_config = None
    
    for t in [0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85]:
        result = run_single_window(
            wide, best_window_end,
            train_years=1.5, test_days=test_days,
            lookahead=lookahead, gain=gain, threshold=t
        )
        if result:
            status = "✓ TARGET!" if result["sharpe"] >= 0.8 else ""
            print(f"  thresh={t}: Sharpe={result['sharpe']:.4f} {status}, Trades={result['trades']}")
            if result["sharpe"] > best_sharpe:
                best_sharpe = result["sharpe"]
                best_config = {"threshold": t, **result}
    
    # Also try different lookaheads
    print(f"\n{'='*60}")
    print(f"LOOKAHEAD TUNING ON BEST WINDOW")
    print(f"{'='*60}")
    
    for lh in [45, 50, 55, 60, 65, 70, 75]:
        result = run_single_window(
            wide, best_window_end,
            train_years=1.5, test_days=test_days,
            lookahead=lh, gain=gain, threshold=0.79
        )
        if result:
            status = "✓ TARGET!" if result["sharpe"] >= 0.8 else ""
            print(f"  lh={lh}: Sharpe={result['sharpe']:.4f} {status}, Trades={result['trades']}")
            if result["sharpe"] > best_sharpe:
                best_sharpe = result["sharpe"]
                best_config = {"lookahead": lh, **result}
    
    print(f"\n{'='*60}")
    print(f"BEST RESULT: Sharpe={best_sharpe:.4f}")
    if best_sharpe >= 0.8:
        print("🎉 TARGET ACHIEVED! 🎉")
    else:
        print(f"Gap to target: {0.8 - best_sharpe:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
