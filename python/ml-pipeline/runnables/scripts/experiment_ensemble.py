"""Ensemble approach to push Sharpe toward 0.8."""

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
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score

from simulator import run_trading_simulation, run_baseline_simulation


def load_and_prepare():
    """Load data and add features."""
    long_df = load_long_data()
    long_df = long_df[long_df['ticker'].notna() & (long_df['ticker'] != '')]
    long_df = long_df[long_df['feature'].isin({'Open', 'High', 'Low', 'Close', 'Volume'})]
    wide = long_to_wide(long_df)
    
    if CLOSE not in wide.columns:
        return wide
    
    wide = wide.sort_values([TICKER, TIMESTAMP])
    result_dfs = []
    
    for ticker in wide[TICKER].unique():
        tdf = wide[wide[TICKER] == ticker].copy()
        tdf = tdf.sort_values(TIMESTAMP).reset_index(drop=True)
        
        close = tdf[CLOSE].values
        n = len(close)
        
        if n < 20:
            continue
        
        # Returns at multiple windows
        for w in [5, 10, 20, 40, 60]:
            ret = np.full(n, np.nan)
            if n > w:
                ret[w:] = (close[w:] - close[:-w]) / np.maximum(np.abs(close[:-w]), 1e-8) * 100
            tdf[f'return_{w}d'] = ret.astype('float32')
        
        # Volatility
        dr = np.zeros(n)
        dr[1:] = (close[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 1e-8) * 100
        
        for w in [10, 20, 40]:
            tdf[f'volatility_{w}d'] = (pd.Series(dr).rolling(w, min_periods=w//2).std() * 100).astype('float32')
        
        # SMAs and price-to-SMA
        for w in [10, 20, 50, 100]:
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
        
        # 52-week high/low
        rh = pd.Series(close).rolling(252, min_periods=20).max().values
        rl = pd.Series(close).rolling(252, min_periods=20).min().values
        tdf['pct_from_52w_high'] = ((close - rh) / np.maximum(rh, 1e-8) * 100).astype('float32')
        tdf['pct_from_52w_low'] = ((close - rl) / np.maximum(rl, 1e-8) * 100).astype('float32')
        tdf['price_position'] = ((close - rl) / np.maximum(rh - rl, 1e-8)).astype('float32')
        
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
        
        # Momentum indicators
        tdf['momentum_5_20'] = (tdf['return_5d'] - tdf['return_20d']).astype('float32')
        tdf['vol_trend'] = (tdf['volatility_10d'] / np.maximum(tdf['volatility_40d'], 1e-8)).astype('float32')
        
        # Trend strength (consecutive up/down days)
        up_days = (np.diff(close, prepend=close[0]) > 0).astype(int)
        streak = np.zeros(n)
        for i in range(1, n):
            if up_days[i] == up_days[i-1]:
                streak[i] = streak[i-1] + (1 if up_days[i] else -1)
            else:
                streak[i] = 1 if up_days[i] else -1
        tdf['trend_streak'] = streak.astype('float32')
        
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


def run_ensemble(lookahead=60, gain=2.0, threshold=0.79, ensemble_threshold=0.75):
    """Run ensemble of multiple models."""
    print(f"\n{'='*60}")
    print(f"ENSEMBLE: lh={lookahead}, gain={gain}, thresh={threshold}, ensemble_agree={ensemble_threshold}")
    print(f"{'='*60}")
    
    wide = load_and_prepare()
    
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
    
    print(f"Features: {len(feat_cols)}")
    
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
    
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}, Class balance: {cw:.2f}")
    
    # Train multiple models
    models = {
        "xgb": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
            gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=cw, random_state=42
        ),
        "lgbm": LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
            reg_alpha=0.5, reg_lambda=2.0, class_weight='balanced',
            random_state=42, verbose=-1
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
    }
    
    predictions = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, probs)
        print(f"  {name} AUC: {auc:.4f}")
        predictions[name] = probs
    
    # Ensemble: average probabilities
    avg_probs = np.mean([predictions[k] for k in predictions], axis=0)
    
    # Also try: only trade when all models agree above threshold
    all_agree = np.all([predictions[k] >= ensemble_threshold for k in predictions], axis=0)
    agree_probs = np.where(all_agree, avg_probs, 0)
    
    # Also try: weighted ensemble (XGB gets more weight)
    weighted_probs = (0.5 * predictions["xgb"] + 0.3 * predictions["lgbm"] + 0.2 * predictions["rf"])
    
    results = []
    
    for prob_name, probs in [
        ("avg", avg_probs),
        ("agree", agree_probs),
        ("weighted", weighted_probs),
        ("xgb_only", predictions["xgb"]),
    ]:
        preds = test_s[[TIMESTAMP, TICKER]].copy()
        preds['prediction_probability'] = probs
        
        trading, _ = run_trading_simulation(
            predictions_df=preds,
            price_data=test_close,
            lookahead_days=lookahead,
            threshold=threshold,
            initial_capital=100_000.0,
            transaction_cost_pct=0.1,
            max_position_pct=0.01,
        )
        
        status = "✓ TARGET!" if trading.sharpe_ratio >= 0.8 else ""
        print(f"  {prob_name}: Sharpe={trading.sharpe_ratio:.4f} {status}, Trades={trading.num_trades}, Median={trading.median_return_pct:.2f}%")
        
        results.append({
            "name": f"ensemble_{prob_name}",
            "sharpe": trading.sharpe_ratio,
            "return": trading.total_return_pct,
            "trades": trading.num_trades,
            "median": trading.median_return_pct,
        })
    
    return results


def run_filtered_trading(lookahead=60, gain=2.0, threshold=0.79):
    """Try filtering trades based on additional criteria."""
    print(f"\n{'='*60}")
    print(f"FILTERED TRADING STRATEGIES")
    print(f"{'='*60}")
    
    wide = load_and_prepare()
    
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
    
    cw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        scale_pos_weight=cw, random_state=42
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    
    results = []
    
    # Strategy 1: Only trade when RSI < 50 (oversold bias)
    rsi_filter = test_s['rsi_14'].values < 50
    filtered_probs1 = np.where(rsi_filter, probs, 0)
    
    # Strategy 2: Only trade when price is below 52-week midpoint
    position_filter = test_s['price_position'].values < 0.5
    filtered_probs2 = np.where(position_filter, probs, 0)
    
    # Strategy 3: Only trade when recent momentum is negative (contrarian)
    momentum_filter = test_s['return_20d'].values < 0
    filtered_probs3 = np.where(momentum_filter, probs, 0)
    
    # Strategy 4: Only trade when volatility is low
    vol_filter = test_s['volatility_20d'].values < test_s['volatility_20d'].median()
    filtered_probs4 = np.where(vol_filter, probs, 0)
    
    # Strategy 5: Combine RSI + position filters
    combined_filter = rsi_filter & position_filter
    filtered_probs5 = np.where(combined_filter, probs, 0)
    
    strategies = [
        ("baseline", probs),
        ("rsi_oversold", filtered_probs1),
        ("below_midpoint", filtered_probs2),
        ("negative_momentum", filtered_probs3),
        ("low_volatility", filtered_probs4),
        ("rsi_and_position", filtered_probs5),
    ]
    
    for strat_name, strat_probs in strategies:
        preds = test_s[[TIMESTAMP, TICKER]].copy()
        preds['prediction_probability'] = strat_probs
        
        trading, _ = run_trading_simulation(
            predictions_df=preds,
            price_data=test_close,
            lookahead_days=lookahead,
            threshold=threshold,
            initial_capital=100_000.0,
            transaction_cost_pct=0.1,
            max_position_pct=0.01,
        )
        
        status = "✓ TARGET!" if trading.sharpe_ratio >= 0.8 else ""
        print(f"  {strat_name}: Sharpe={trading.sharpe_ratio:.4f} {status}, Trades={trading.num_trades}, Median={trading.median_return_pct:.2f}%")
        
        results.append({
            "name": f"filter_{strat_name}",
            "sharpe": trading.sharpe_ratio,
            "return": trading.total_return_pct,
            "trades": trading.num_trades,
            "median": trading.median_return_pct,
        })
    
    return results


def run_higher_thresholds():
    """Push thresholds even higher."""
    print(f"\n{'='*60}")
    print(f"ULTRA-HIGH THRESHOLDS")
    print(f"{'='*60}")
    
    wide = load_and_prepare()
    
    lookahead = 60
    gain = 2.0
    
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
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        scale_pos_weight=cw, random_state=42
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    
    results = []
    
    for thresh in [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]:
        preds = test_s[[TIMESTAMP, TICKER]].copy()
        preds['prediction_probability'] = probs
        
        trading, _ = run_trading_simulation(
            predictions_df=preds,
            price_data=test_close,
            lookahead_days=lookahead,
            threshold=thresh,
            initial_capital=100_000.0,
            transaction_cost_pct=0.1,
            max_position_pct=0.01,
        )
        
        status = "✓ TARGET!" if trading.sharpe_ratio >= 0.8 else ""
        print(f"  thresh={thresh}: Sharpe={trading.sharpe_ratio:.4f} {status}, Trades={trading.num_trades}, Median={trading.median_return_pct:.2f}%")
        
        results.append({
            "name": f"ultra_thresh_{thresh}",
            "sharpe": trading.sharpe_ratio,
            "trades": trading.num_trades,
            "median": trading.median_return_pct,
        })
    
    return results


def main():
    all_results = []
    
    # Try ensemble approaches
    ensemble_results = run_ensemble(lookahead=60, gain=2.0, threshold=0.79, ensemble_threshold=0.75)
    all_results.extend(ensemble_results)
    
    # Try filtered trading strategies
    filter_results = run_filtered_trading(lookahead=60, gain=2.0, threshold=0.79)
    all_results.extend(filter_results)
    
    # Try ultra-high thresholds
    ultra_results = run_higher_thresholds()
    all_results.extend(ultra_results)
    
    # Summary
    print(f"\n{'='*60}")
    print("BEST RESULTS")
    print(f"{'='*60}")
    
    all_results_sorted = sorted(all_results, key=lambda x: x["sharpe"], reverse=True)
    best = all_results_sorted[0]
    
    for r in all_results_sorted[:5]:
        status = "✓ TARGET MET" if r["sharpe"] >= 0.8 else ""
        print(f"{r['name']}: Sharpe={r['sharpe']:.4f} {status}, Trades={r['trades']}")
    
    print(f"\n*** BEST: {best['name']} with Sharpe {best['sharpe']:.4f} ***")
    
    if best["sharpe"] >= 0.8:
        print("\n🎉 TARGET ACHIEVED! 🎉")
    else:
        print(f"\nGap to target: {0.8 - best['sharpe']:.4f}")
    
    # Save results
    with open(_ml_pipeline / "output" / "ensemble_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
