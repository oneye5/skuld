"""Data quality analysis script - analyses each stage of the pipeline."""
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'python' / 'src'))

# ─────────────────────────────────────────────
# STAGE 0: Raw CSV
# ─────────────────────────────────────────────
print("=" * 60)
print("STAGE 0: RAW CSV (data/data_long.csv)")
print("=" * 60)

df = pd.read_csv('data/data_long.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nNulls per column:\n{df.isnull().sum()}")
print(f"\nDtypes:\n{df.dtypes}")

print(f"\n--- Source distribution ---")
print(df['src'].value_counts())

print(f"\n--- Feature distribution (top 30) ---")
print(df['feature'].value_counts().head(30))

print(f"\n--- Date range ---")
print(f"Min: {df['timestamp'].min()}, Max: {df['timestamp'].max()}")

dups = df.duplicated(subset=['timestamp', 'ticker', 'feature']).sum()
print(f"\n--- Duplicates (timestamp, ticker, feature) ---")
print(f"Count: {dups}")

# Value column as numeric
df['value_num'] = pd.to_numeric(df['value'], errors='coerce')
print(f"\n--- Value parse failures (non-numeric) ---")
print(f"Count: {df['value_num'].isnull().sum()}")

# Tickers
nzx = df[df['ticker'].notna() & df['ticker'].str.endswith('.NZ', na=False)]
print(f"\n--- NZX tickers ---")
print(f"Distinct NZX tickers: {nzx['ticker'].nunique()}")
print(f"Non-NZX tickers: {df[df['ticker'].notna() & ~df['ticker'].str.endswith('.NZ', na=False)]['ticker'].nunique()}")

# ─────────────────────────────────────────────
# STAGE 1: CSV Loader output
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 1: CSV LOADER -> RawData")
print("=" * 60)

from skuld_research.data.csv_loader import load_raw_csv
raw = load_raw_csv(Path('data/data_long.csv'))

print(f"\n--- prices ({raw.prices.shape}) ---")
print(f"Date range: {raw.prices.index.min()} to {raw.prices.index.max()}")
nan_pct = raw.prices.isnull().mean()
print(f"Tickers with >50% NaN: {(nan_pct > 0.5).sum()} / {len(nan_pct)}")
print(f"Tickers with >80% NaN: {(nan_pct > 0.8).sum()} / {len(nan_pct)}")
print(f"Overall NaN pct: {raw.prices.isnull().mean().mean():.2%}")
neg_prices = (raw.prices < 0).sum().sum()
print(f"Negative prices: {neg_prices}")
zero_prices = (raw.prices == 0).sum().sum()
print(f"Zero prices: {zero_prices}")

print(f"\n--- volumes ({raw.volumes.shape}) ---")
nan_pct_v = raw.volumes.isnull().mean()
print(f"Overall NaN pct: {raw.volumes.isnull().mean().mean():.2%}")
print(f"Tickers with >80% NaN: {(nan_pct_v > 0.8).sum()} / {len(nan_pct_v)}")
zero_vol = (raw.volumes == 0).sum().sum()
print(f"Zero volumes: {zero_vol}")

print(f"\n--- fundamentals ({raw.fundamentals.shape}) ---")
if raw.fundamentals is not None and not raw.fundamentals.empty:
    funds = raw.fundamentals
    print(f"Features: {funds.columns.tolist()}")
    nan_pct_f = funds.isnull().mean()
    print(f"NaN by feature:")
    print(nan_pct_f.sort_values(ascending=False).to_string())
    tickers_with_funds = funds.index.get_level_values('ticker').nunique()
    print(f"Tickers with any fundamental data: {tickers_with_funds}")
else:
    print("Empty or None")

print(f"\n--- corporate_actions ---")
if raw.corporate_actions is not None and not raw.corporate_actions.empty:
    ca = raw.corporate_actions
    print(f"Shape: {ca.shape}")
    print(f"Columns: {ca.columns.tolist()}")
    date_col = 'ex_date' if 'ex_date' in ca.columns else 'date'
    type_col = 'type' if 'type' in ca.columns else 'feature'
    print(f"Date range: {ca[date_col].min()} to {ca[date_col].max()}")
    print(f"Tickers with splits: {ca[ca[type_col]=='split']['ticker'].nunique()}")
    print(f"Tickers with dividends: {ca[ca[type_col]=='dividend']['ticker'].nunique()}")
else:
    print("Empty or None")

print(f"\n--- macro ---")
if raw.macro is not None and not raw.macro.empty:
    print(f"Shape: {raw.macro.shape}")
    print(f"Features: {raw.macro.columns.tolist()}")
    nan_pct_m = raw.macro.isnull().mean()
    print(f"NaN by feature:")
    print(nan_pct_m.sort_values(ascending=False).to_string())
else:
    print("Empty or None")

# ─────────────────────────────────────────────
# STAGE 2: PIT Snapshot (as of latest available)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2: PIT LOADER (as_of latest)")
print("=" * 60)

from skuld_research.data.pit_loader import PITLoader
import datetime

asof = pd.Timestamp(raw.prices.index.max()) - pd.offsets.BDay(1)
pit = PITLoader(raw).as_of(asof)
print(f"as_of: {asof.date()}")
print(f"prices shape: {pit.prices.shape}")
print(f"volumes shape: {pit.volumes.shape}")

# Check for survivorship - how many tickers have recent prices
recent_cutoff = asof - pd.DateOffset(months=6)
recent_prices = pit.prices.loc[pit.prices.index >= recent_cutoff]
active_tickers = recent_prices.columns[recent_prices.notna().any()].tolist()
print(f"Tickers with any price data in last 6 months: {len(active_tickers)} / {pit.prices.shape[1]}")

# ─────────────────────────────────────────────
# STAGE 3: PreparedPanel
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 3: PREPARED PANEL")
print("=" * 60)

from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.config.spec import AnomalyFilterSpec, UniverseSpec

uspec = UniverseSpec()
panel = build_prepared_panel(pit, nzx_only=True)

print(f"\n--- returns_daily ({panel.returns_daily.shape}) ---")
rd = panel.returns_daily
print(f"NaN pct: {rd.isnull().mean().mean():.2%}")
inf_count = np.isinf(rd.values).sum()
print(f"Inf values: {inf_count}")
extreme = (rd.abs() > 0.5).sum().sum()
print(f"Returns > 50% (single day): {extreme}")
extreme_99 = rd.stack().quantile(0.999)
extreme_01 = rd.stack().quantile(0.001)
print(f"99.9th percentile daily return: {extreme_99:.2%}")
print(f"0.1th percentile daily return: {extreme_01:.2%}")

print(f"\n--- returns_monthly ({panel.returns_monthly.shape}) ---")
rm = panel.returns_monthly
print(f"NaN pct: {rm.isnull().mean().mean():.2%}")
extreme_m = (rm.abs() > 1.0).sum().sum()
print(f"Returns > 100% single month: {extreme_m}")

print(f"\n--- universe_mask at latest rebalance ---")
universe_mask = panel.universe_mask
rebal_dates = universe_mask.index
print(f"Total rebalance dates: {len(rebal_dates)}")
last_rebal = rebal_dates[-1]
universe_sizes = universe_mask.sum(axis=1)
print(f"Universe size at {last_rebal}: {universe_mask.loc[last_rebal].sum()}")
print(f"Universe size over time - min={universe_sizes.min()}, median={universe_sizes.median():.0f}, max={universe_sizes.max()}")
tiny = (universe_sizes < 5).sum()
print(f"Rebalance periods with <5 tickers: {tiny}")

print(f"\n--- market_cap ---")
if panel.market_cap is not None:
    mc = panel.market_cap
    print(f"Shape: {mc.shape}")
    print(f"NaN pct: {mc.isnull().mean().mean():.2%}")
    print(f"Zero market_cap count: {(mc == 0).sum().sum()}")

# ─────────────────────────────────────────────
# STAGE 3b: Filtered panel (anomaly filter applied)
# ─────────────────────────────────────────────
print("\n--- returns_daily WITH anomaly filter (daily_abs_return_threshold=0.5, chronic_ticker_max_extreme_days=5) ---")
afilter = AnomalyFilterSpec(
    kind="mask_extremes",
    daily_abs_return_threshold=0.5,
    monthly_abs_return_threshold=1.0,
    chronic_ticker_max_extreme_days=5,
)
panel_f = build_prepared_panel(pit, nzx_only=True, anomaly_filter=afilter)
rd_f = panel_f.returns_daily
extreme_f = (rd_f.abs() > 0.5).sum().sum()
extreme_f99 = rd_f.stack().quantile(0.999)
print(f"Returns > 50% (single day): {extreme_f}  (was {(panel.returns_daily.abs() > 0.5).sum().sum()})")
print(f"99.9th percentile daily return: {extreme_f99:.2%}  (was {panel.returns_daily.stack().quantile(0.999):.2%})")
rm_f = panel_f.returns_monthly
extreme_fm = (rm_f.abs() > 1.0).sum().sum()
print(f"Returns > 100% single month: {extreme_fm}  (was {(panel.returns_monthly.abs() > 1.0).sum().sum()})")

# ─────────────────────────────────────────────
# STAGE 4-5: Factor signals and combiner
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 4: FACTOR SIGNALS (using anomaly-filtered panel)")
print("=" * 60)

from skuld_research.factors.momentum import MomentumFactor
from skuld_research.factors.low_volatility import LowVolatilityFactor
from skuld_research.factors.dividend_yield import DividendYieldFactor
from skuld_research.factors.size import SizeFactor

# Use last rebalance date
t = last_rebal
if hasattr(panel_f, 'universe_mask'):
    universe = panel_f.universe_mask.loc[t] if t in panel_f.universe_mask.index else None

factors = {
    'momentum': MomentumFactor(),
    'low_vol': LowVolatilityFactor(),
    'dividend_yield': DividendYieldFactor(),
    'size': SizeFactor(),
}

# universe is list[str] at last_rebal
universe_list = list(universe_mask.loc[last_rebal][universe_mask.loc[last_rebal]].index) if last_rebal in universe_mask.index else []
print(f"Universe at last rebal ({last_rebal}): {len(universe_list)} tickers")

results = {}
for name, factor in factors.items():
    try:
        scores = factor.score(panel_f, last_rebal, universe_list)
        n_valid = scores.notna().sum()
        n_nan = scores.isna().sum()
        if n_valid > 0:
            desc = scores.dropna().describe()
            skew = scores.dropna().skew()
            kurt = scores.dropna().kurtosis()
            results[name] = {
                'n_valid': int(n_valid), 'n_nan': int(n_nan),
                'mean': round(float(desc['mean']), 4), 'std': round(float(desc['std']), 4),
                'min': round(float(desc['min']), 4), 'max': round(float(desc['max']), 4),
                'skew': round(float(skew), 4), 'kurt': round(float(kurt), 4),
            }
        else:
            results[name] = {'n_valid': 0, 'n_nan': int(n_nan), 'error': 'all NaN'}
    except Exception as e:
        results[name] = {'error': str(e)}

for name, r in results.items():
    print(f"\n  {name}: {r}")

print("\n" + "=" * 60)
print("STAGE 5: COMBINED SCORES")
print("=" * 60)

from skuld_research.factors.combiner import combine_signals
from skuld_research.config.spec import FactorSpec, MomentumFactorSpec, LowVolatilityFactorSpec

try:
    raw_signals = {}
    for name, factor in factors.items():
        try:
            raw_signals[name] = factor.score(panel_f, last_rebal, universe_list)
        except Exception as e:
            print(f"  {name} failed: {e}")

    if raw_signals:
        sector = panel_f.sector if hasattr(panel_f, 'sector') else pd.Series(dtype='str')
        combined = combine_signals(raw_signals, universe_list, sector, last_rebal)
        cs = combined.scores
        print(f"Combined scores - n_valid: {cs.notna().sum()}, NaN: {cs.isna().sum()}")
        print(f"Distribution: mean={cs.mean():.3f}, std={cs.std():.3f}, min={cs.min():.3f}, max={cs.max():.3f}")
        print(f"Skew: {cs.skew():.3f}, Kurt: {cs.kurtosis():.3f}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("VALIDATION MODULE CHECKS")
print("=" * 60)

from skuld_common.validation import (
    detect_negative_prices,
    detect_gaps,
    detect_nan_density,
    detect_duplicate_observations,
    detect_stale_fundamentals,
    detect_stale_sources,
)

neg = detect_negative_prices(raw.prices)
print(f"\nnegative_prices: {neg.issue_count} issues, clean={neg.is_clean}")
if not neg.is_clean: print(neg.details)

gaps = detect_gaps(raw.prices)
print(f"\ngaps (>5 consecutive missing bdays): {gaps.issue_count} issues, clean={gaps.is_clean}")
if not gaps.is_clean: print(str(gaps.details)[:500])

nan_dense = detect_nan_density(raw.prices)
print(f"\nnan_density (>50% NaN tickers): {nan_dense.issue_count} issues, clean={nan_dense.is_clean}")
if not nan_dense.is_clean: print(str(nan_dense.details)[:500])

dups2 = detect_duplicate_observations(df)
print(f"\nduplicate_observations: {dups2.issue_count} issues, clean={dups2.is_clean}")
if not dups2.is_clean: print(str(dups2.details)[:200])

stale = detect_stale_fundamentals(raw.fundamentals, asof)
print(f"\nstale_fundamentals: {stale.issue_count} issues, clean={stale.is_clean}")
if not stale.is_clean: print(str(stale.details)[:500])

source_latest = (
    df.groupby("src")["timestamp"]
    .max()
    .apply(lambda ts: pd.Timestamp(ts, unit="ms") if not isinstance(ts, pd.Timestamp) else ts)
    .to_dict()
)
stale_src = detect_stale_sources(source_latest, asof)
print(f"\nstale_sources: {stale_src.issue_count} issues, clean={stale_src.is_clean}")
if not stale_src.is_clean: print(str(stale_src.details))

print("\n=== ANALYSIS COMPLETE ===")
