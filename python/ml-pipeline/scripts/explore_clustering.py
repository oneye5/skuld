"""Explore unsupervised clustering for industry/sector definition."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load data
data_path = Path(__file__).parent.parent.parent.parent / "data" / "data_long.csv"
df = pd.read_csv(data_path)

# Get NZ tickers only
nz_tickers = [t for t in df['ticker'].dropna().unique() 
              if isinstance(t, str) and t.endswith('.NZ')]
print(f"NZX tickers: {len(nz_tickers)}")

# Build price matrix
price_df = df[(df['feature'] == 'Close') & (df['ticker'].isin(nz_tickers))].copy()
pivot = price_df.pivot_table(index='timestamp', columns='ticker', values='value', aggfunc='first')
returns = pivot.pct_change(fill_method=None).iloc[1:]

print(f"Returns matrix: {returns.shape}")

# Calculate ticker characteristics for clustering
# Use recent 2 years of data
recent_ts = returns.index[-500:]
recent_returns = returns.loc[recent_ts]

# Build feature matrix for each ticker
stats = []
for ticker in recent_returns.columns:
    r = recent_returns[ticker].dropna()
    if len(r) > 100:
        stats.append({
            'ticker': ticker,
            'volatility': r.std() * np.sqrt(252),
            'mean_return': r.mean() * 252,
            'skewness': r.skew(),
            'kurtosis': r.kurtosis(),
            'autocorr': r.autocorr(1) if len(r) > 10 else 0,
            'pos_days_pct': (r > 0).mean(),
        })

stats_df = pd.DataFrame(stats)
print(f"Tickers with sufficient data: {len(stats_df)}")

# Normalize features
features = ['volatility', 'mean_return', 'skewness', 'kurtosis', 'autocorr', 'pos_days_pct']
X = stats_df[features].fillna(0).values
X_scaled = StandardScaler().fit_transform(X)

# K-means clustering
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
stats_df['cluster'] = kmeans.fit_predict(X_scaled)

print("\n" + "="*60)
print("FEATURE-BASED CLUSTERING RESULTS")
print("="*60)

for c in range(n_clusters):
    cluster_df = stats_df[stats_df['cluster'] == c]
    print(f"\nCluster {c} ({len(cluster_df)} stocks):")
    print(f"  Avg Volatility: {cluster_df['volatility'].mean():.1%}")
    print(f"  Avg Return: {cluster_df['mean_return'].mean():.1%}")
    print(f"  Avg Skewness: {cluster_df['skewness'].mean():.2f}")
    print(f"  Members: {', '.join(cluster_df['ticker'].tolist()[:8])}")
    if len(cluster_df) > 8:
        print(f"           ... and {len(cluster_df) - 8} more")

# Save cluster assignments
output_path = Path(__file__).parent.parent / "output" / "debug" / "cluster_assignments.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
stats_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
