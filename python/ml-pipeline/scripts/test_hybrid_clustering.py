"""Test hybrid clustering approaches for NZX stocks."""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers

# Known pairs for validation (should cluster together)
KNOWN_PAIRS = [
    ('MFT.NZ', 'MOV.NZ', 'Logistics'),
    ('ARG.NZ', 'GMT.NZ', 'Property'),
    ('CEN.NZ', 'MCY.NZ', 'Utilities'),
    ('FPH.NZ', 'RYM.NZ', 'Healthcare'),
    ('SPK.NZ', 'SKT.NZ', 'Telco'),
    ('ANZ.NZ', 'HGH.NZ', 'Banks'),
    ('KMD.NZ', 'WHS.NZ', 'Retail'),
]


def load_and_prepare():
    """Load data and prepare returns matrix."""
    print("Loading data...")
    long_df = load_long_data()
    wide_df = long_to_wide(add_macro_prefix(clean_and_classify_tickers(long_df)))
    nz_mask = wide_df['ticker'].str.endswith('.NZ', na=False)
    wide_df = wide_df[nz_mask]
    
    # Build price matrix - last 2 years
    timestamps = sorted(wide_df['timestamp'].unique())[-500:]
    df = wide_df[wide_df['timestamp'].isin(timestamps)]
    pivot = df.pivot_table(index='timestamp', columns='ticker', values='Close', aggfunc='first')
    returns = pivot.pct_change(fill_method=None).iloc[1:]
    
    # Filter tickers with enough data
    min_obs = 200
    valid = returns.columns[returns.notna().sum() >= min_obs].tolist()
    returns = returns[valid].clip(-1, 1)  # Clip anomalies at 100%
    
    return returns, valid


def compute_features(returns, valid):
    """Compute ticker characteristics."""
    features = []
    for ticker in valid:
        r = returns[ticker].dropna()
        if len(r) < 50:
            continue
        features.append({
            'ticker': ticker,
            'vol': r.std() * np.sqrt(252),
            'ret': r.mean() * 252,
            'skew': r.skew(),
            'kurt': r.kurtosis(),
            'autocorr': r.autocorr(1) if len(r) > 10 else 0,
            'pos_pct': (r > 0).mean(),
            'vol_of_vol': r.rolling(20).std().std() * np.sqrt(252) if len(r) > 20 else 0,
            'max_dd': (r.cumsum() - r.cumsum().cummax()).min(),
        })
    
    feat_df = pd.DataFrame(features).set_index('ticker')
    return feat_df


def test_clustering(dist_matrix, valid, name, n_clusters=12):
    """Test a clustering approach and validate against known pairs."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='precomputed', 
        linkage='average'
    )
    labels = model.fit_predict(dist_matrix)
    cluster_map = dict(zip(valid, labels))
    
    print(f"\n{'='*60}")
    print(f"{name} ({n_clusters} clusters)")
    print('='*60)
    
    # Check known pairs
    correct = 0
    for t1, t2, group in KNOWN_PAIRS:
        if t1 in cluster_map and t2 in cluster_map:
            same = cluster_map[t1] == cluster_map[t2]
            correct += same
            status = "SAME" if same else "DIFF"
            print(f"  {group:12s}: {t1}(C{cluster_map[t1]}) vs {t2}(C{cluster_map[t2]}) = {status}")
        else:
            print(f"  {group:12s}: Missing data")
    
    print(f"\n  Pair accuracy: {correct}/{len(KNOWN_PAIRS)}")
    
    # Cluster sizes
    sizes = Counter(labels)
    print(f"  Cluster sizes: {sorted(sizes.values(), reverse=True)}")
    
    # Show largest cluster members
    largest_cluster = sizes.most_common(1)[0][0]
    largest_members = [t for t, c in cluster_map.items() if c == largest_cluster]
    print(f"  Largest cluster ({len(largest_members)}): {largest_members[:10]}...")
    
    return cluster_map


def main():
    returns, valid = load_and_prepare()
    print(f"Valid tickers: {len(valid)}")
    
    # Compute features
    feat_df = compute_features(returns, valid)
    X_feat = StandardScaler().fit_transform(feat_df.fillna(0))
    
    # Distance matrices
    # 1. Correlation distance
    corr = returns.corr(min_periods=50).fillna(0)
    dist_corr = 1 - corr
    dist_corr = np.clip(dist_corr.values, 0, 2)
    
    # 2. Feature distance (euclidean)
    feat_dist = pairwise_distances(X_feat, metric='euclidean')
    feat_dist_norm = feat_dist / feat_dist.max()
    
    # 3. Combined distance
    combined_dist = 0.5 * dist_corr + 0.5 * feat_dist_norm
    
    # Test different approaches
    test_clustering(dist_corr, valid, "Pure Correlation", n_clusters=12)
    test_clustering(feat_dist_norm, valid, "Pure Features", n_clusters=12)
    test_clustering(combined_dist, valid, "Combined (50/50)", n_clusters=12)
    
    # Test with more clusters
    test_clustering(combined_dist, valid, "Combined (50/50)", n_clusters=16)
    test_clustering(combined_dist, valid, "Combined (50/50)", n_clusters=20)
    
    # Also test K-means on features (faster)
    print(f"\n{'='*60}")
    print("K-Means on Features (12 clusters)")
    print('='*60)
    
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_feat)
    cluster_map = dict(zip(feat_df.index, labels))
    
    correct = 0
    for t1, t2, group in KNOWN_PAIRS:
        if t1 in cluster_map and t2 in cluster_map:
            same = cluster_map[t1] == cluster_map[t2]
            correct += same
            status = "SAME" if same else "DIFF"
            print(f"  {group:12s}: {t1}(C{cluster_map[t1]}) vs {t2}(C{cluster_map[t2]}) = {status}")
    
    print(f"\n  Pair accuracy: {correct}/{len(KNOWN_PAIRS)}")
    sizes = Counter(labels)
    print(f"  Cluster sizes: {sorted(sizes.values(), reverse=True)}")


if __name__ == "__main__":
    main()
