"""Benchmark different clustering algorithms for NZX sector definition.

Goals:
1. Speed comparison (critical for pipeline integration)
2. Cluster quality validation against known NZX groupings
3. Leakage-safe implementation patterns
"""

import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import load_long_data
from core.long_to_wide import long_to_wide, add_macro_prefix, clean_and_classify_tickers
from config.columns import TIMESTAMP, TICKER

# =============================================================================
# KNOWN NZX SECTOR GROUPINGS (for validation)
# =============================================================================

# Based on actual NZX industry knowledge
KNOWN_GROUPS = {
    "Logistics/Transport": ["MFT.NZ", "MOV.NZ", "AIR.NZ", "FRE.NZ"],
    "Property/REITs": ["ARG.NZ", "GMT.NZ", "KPG.NZ", "PCT.NZ", "PFI.NZ", "VHP.NZ"],
    "Banks/Finance": ["ANZ.NZ", "HGH.NZ", "NZK.NZ"],
    "Utilities": ["CEN.NZ", "GNE.NZ", "MCY.NZ", "MEL.NZ", "VCT.NZ"],
    "Healthcare": ["AFT.NZ", "EBO.NZ", "FPH.NZ", "RYM.NZ", "SUM.NZ", "OCA.NZ"],
    "Tech/Software": ["ERD.NZ", "IKE.NZ", "PLX.NZ", "SER.NZ", "SKO.NZ", "XRO.NZ"],
    "Consumer/Retail": ["HLG.NZ", "KMD.NZ", "MHJ.NZ", "SKL.NZ", "WHS.NZ"],
    "Agriculture/Dairy": ["ATM.NZ", "FCG.NZ", "NZM.NZ", "SAN.NZ", "SML.NZ"],
    "Building/Construction": ["ARV.NZ", "FBU.NZ", "SKT.NZ", "STU.NZ"],
    "Telco/Media": ["SPK.NZ", "SKT.NZ"],
    "Investment Trusts": ["AFI.NZ", "ALF.NZ", "KFL.NZ", "LIC.NZ", "MHM.NZ"],
}


def load_data():
    """Load and prepare data for clustering."""
    print("Loading data...")
    long_df = load_long_data()
    wide_df = long_to_wide(add_macro_prefix(clean_and_classify_tickers(long_df)))
    
    # Filter to NZ stocks only
    nz_mask = wide_df[TICKER].str.endswith('.NZ', na=False)
    wide_df = wide_df[nz_mask].copy()
    
    return wide_df


def compute_return_matrix(wide_df: pd.DataFrame, lookback_days: int = 750) -> tuple[pd.DataFrame, list]:
    """Compute return matrix for clustering.
    
    Returns returns matrix and list of valid tickers.
    """
    # Get recent timestamps only
    timestamps = sorted(wide_df[TIMESTAMP].unique())
    recent_ts = timestamps[-lookback_days:]
    df = wide_df[wide_df[TIMESTAMP].isin(recent_ts)].copy()
    
    # Pivot to get price matrix
    price_pivot = df.pivot_table(index=TIMESTAMP, columns=TICKER, values='Close', aggfunc='first')
    
    # Calculate returns
    returns = price_pivot.pct_change(fill_method=None).iloc[1:]
    
    # Filter tickers with enough data
    min_obs = 250
    valid_mask = returns.notna().sum() >= min_obs
    valid_tickers = returns.columns[valid_mask].tolist()
    
    return returns[valid_tickers], valid_tickers


def compute_features(returns: pd.DataFrame, max_daily_return: float = 1.0) -> tuple[np.ndarray, list]:
    """Compute clustering features from returns.
    
    Args:
        returns: DataFrame with ticker columns and return values
        max_daily_return: Clip returns beyond this to handle data anomalies
    """
    features = []
    tickers = returns.columns.tolist()
    
    for ticker in tickers:
        r = returns[ticker].dropna()
        if len(r) < 100:
            continue
        
        # Clip extreme returns (data errors, unadjusted splits)
        r_clean = r.clip(-max_daily_return, max_daily_return)
        anomaly_rate = (r.abs() > max_daily_return).mean()
        
        features.append({
            'ticker': ticker,
            'volatility': r_clean.std() * np.sqrt(252),
            'mean_return': r_clean.mean() * 252,
            'skewness': r_clean.skew(),
            'kurtosis': r_clean.kurtosis(),
            'autocorr': r_clean.autocorr(1) if len(r_clean) > 10 else 0,
            'pos_days': (r_clean > 0).mean(),
            'anomaly_rate': anomaly_rate,
        })
    
    df = pd.DataFrame(features)
    feature_cols = ['volatility', 'mean_return', 'skewness', 'kurtosis', 'autocorr', 'pos_days']
    X = df[feature_cols].fillna(0).values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X, df['ticker'].tolist()


def compute_correlation_distance(returns: pd.DataFrame, tickers: list) -> np.ndarray:
    """Compute correlation-based distance matrix."""
    corr = returns[tickers].corr(min_periods=50)
    corr = corr.fillna(0)
    dist = 1 - corr.values
    # Ensure symmetry and non-negative
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)
    return dist


def validate_clusters(cluster_map: dict, method_name: str) -> dict:
    """Validate cluster quality against known groupings."""
    results = {
        'method': method_name,
        'group_scores': {},
        'specific_checks': {},
    }
    
    # Check each known group
    for group_name, tickers in KNOWN_GROUPS.items():
        available = [t for t in tickers if t in cluster_map]
        if len(available) < 2:
            continue
        
        clusters = [cluster_map[t] for t in available]
        # Score: fraction of pairs in same cluster
        same_cluster = sum(1 for i in range(len(clusters)) 
                         for j in range(i+1, len(clusters)) 
                         if clusters[i] == clusters[j])
        total_pairs = len(clusters) * (len(clusters) - 1) // 2
        score = same_cluster / total_pairs if total_pairs > 0 else 0
        results['group_scores'][group_name] = {
            'score': score,
            'tickers': available,
            'clusters': clusters,
        }
    
    # Specific checks
    # MFT and MOV (logistics)
    if 'MFT.NZ' in cluster_map and 'MOV.NZ' in cluster_map:
        results['specific_checks']['MFT_MOV_together'] = (
            cluster_map['MFT.NZ'] == cluster_map['MOV.NZ']
        )
    
    # Property stocks together
    prop_tickers = ['ARG.NZ', 'GMT.NZ', 'KPG.NZ']
    prop_available = [t for t in prop_tickers if t in cluster_map]
    if len(prop_available) >= 2:
        prop_clusters = [cluster_map[t] for t in prop_available]
        results['specific_checks']['Property_together'] = len(set(prop_clusters)) == 1
    
    # Utilities together
    util_tickers = ['CEN.NZ', 'GNE.NZ', 'MCY.NZ', 'MEL.NZ']
    util_available = [t for t in util_tickers if t in cluster_map]
    if len(util_available) >= 2:
        util_clusters = [cluster_map[t] for t in util_available]
        results['specific_checks']['Utilities_together'] = len(set(util_clusters)) == 1
    
    return results


def benchmark_algorithm(name: str, fit_func, X: np.ndarray, tickers: list, 
                       n_runs: int = 3) -> dict:
    """Benchmark a clustering algorithm."""
    times = []
    labels = None
    
    for _ in range(n_runs):
        start = time.perf_counter()
        labels = fit_func(X)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    cluster_map = dict(zip(tickers, labels))
    validation = validate_clusters(cluster_map, name)
    
    # Count clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
    
    return {
        'name': name,
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'n_clusters': n_clusters,
        'cluster_map': cluster_map,
        'validation': validation,
    }


def main():
    # Load data
    wide_df = load_data()
    print(f"Loaded {len(wide_df)} rows")
    
    # Compute features
    returns, valid_tickers = compute_return_matrix(wide_df)
    print(f"Valid tickers: {len(valid_tickers)}")
    
    X_raw, tickers = compute_features(returns)
    print(f"Feature matrix: {X_raw.shape}")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Correlation distance matrix (for hierarchical)
    dist_matrix = compute_correlation_distance(returns, tickers)
    
    # ==========================================================================
    # BENCHMARK ALGORITHMS
    # ==========================================================================
    
    n_clusters = 8  # Target number of clusters
    results = []
    
    print("\n" + "="*70)
    print("CLUSTERING ALGORITHM BENCHMARK")
    print("="*70)
    
    # 1. K-Means (feature-based)
    results.append(benchmark_algorithm(
        "KMeans (features)",
        lambda X: KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X),
        X, tickers
    ))
    
    # 2. K-Means++ (better init)
    results.append(benchmark_algorithm(
        "KMeans++ (features)",
        lambda X: KMeans(n_clusters=n_clusters, random_state=42, n_init=1, init='k-means++').fit_predict(X),
        X, tickers
    ))
    
    # 3. Hierarchical (Ward linkage on features)
    results.append(benchmark_algorithm(
        "Hierarchical-Ward (features)",
        lambda X: AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X),
        X, tickers
    ))
    
    # 4. Hierarchical (Average linkage on correlation distance)
    results.append(benchmark_algorithm(
        "Hierarchical-Corr (correlation)",
        lambda X: AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='precomputed', 
            linkage='average'
        ).fit_predict(dist_matrix),
        dist_matrix, tickers
    ))
    
    # 5. Gaussian Mixture Model
    results.append(benchmark_algorithm(
        "GMM (features)",
        lambda X: GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(X),
        X, tickers
    ))
    
    # 6. DBSCAN (density-based, auto clusters)
    results.append(benchmark_algorithm(
        "DBSCAN (features)",
        lambda X: DBSCAN(eps=1.0, min_samples=3).fit_predict(X),
        X, tickers
    ))
    
    # 7. Mini-batch K-Means (fastest for large data)
    from sklearn.cluster import MiniBatchKMeans
    results.append(benchmark_algorithm(
        "MiniBatch-KMeans (features)",
        lambda X: MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3).fit_predict(X),
        X, tickers
    ))
    
    # ==========================================================================
    # RESULTS SUMMARY
    # ==========================================================================
    
    print("\n" + "="*70)
    print("SPEED COMPARISON")
    print("="*70)
    print(f"{'Algorithm':<35} {'Time (ms)':<12} {'Clusters':<10}")
    print("-"*60)
    for r in sorted(results, key=lambda x: x['avg_time_ms']):
        print(f"{r['name']:<35} {r['avg_time_ms']:>8.2f}    {r['n_clusters']:<10}")
    
    print("\n" + "="*70)
    print("VALIDATION: KNOWN GROUP COHESION")
    print("="*70)
    
    # Show specific checks
    print("\nKey Checks (should be TRUE):")
    print("-"*60)
    for r in results:
        checks = r['validation']['specific_checks']
        mft_mov = checks.get('MFT_MOV_together', 'N/A')
        prop = checks.get('Property_together', 'N/A')
        util = checks.get('Utilities_together', 'N/A')
        print(f"{r['name']:<35} MFT+MOV: {str(mft_mov):<6} Prop: {str(prop):<6} Util: {str(util):<6}")
    
    # Detailed group scores
    print("\n" + "="*70)
    print("GROUP COHESION SCORES (1.0 = all in same cluster)")
    print("="*70)
    
    # Find best method for each group
    for group_name in KNOWN_GROUPS.keys():
        scores = []
        for r in results:
            if group_name in r['validation']['group_scores']:
                score = r['validation']['group_scores'][group_name]['score']
                scores.append((r['name'], score))
        
        if scores:
            best = max(scores, key=lambda x: x[1])
            print(f"{group_name:<25}: Best = {best[0]} ({best[1]:.2f})")
    
    # ==========================================================================
    # DETAILED CLUSTER MEMBERSHIP FOR BEST METHOD
    # ==========================================================================
    
    # Find method with highest validation score
    def overall_score(r):
        checks = r['validation']['specific_checks']
        return sum(1 for v in checks.values() if v is True)
    
    best_result = max(results, key=overall_score)
    print("\n" + "="*70)
    print(f"BEST METHOD: {best_result['name']}")
    print("="*70)
    
    cluster_map = best_result['cluster_map']
    clusters_by_id = {}
    for ticker, cluster in cluster_map.items():
        if cluster not in clusters_by_id:
            clusters_by_id[cluster] = []
        clusters_by_id[cluster].append(ticker)
    
    for cluster_id in sorted(clusters_by_id.keys()):
        members = sorted(clusters_by_id[cluster_id])
        print(f"\nCluster {cluster_id} ({len(members)} stocks):")
        # Show in rows of 8
        for i in range(0, len(members), 8):
            print(f"  {', '.join(members[i:i+8])}")
    
    # ==========================================================================
    # LEAKAGE-SAFE IMPLEMENTATION NOTES
    # ==========================================================================
    
    print("\n" + "="*70)
    print("LEAKAGE-SAFE IMPLEMENTATION")
    print("="*70)
    print("""
To prevent leakage in production:

1. TRAIN-TIME CLUSTERING:
   - Compute clusters using ONLY training period data
   - Use lookback window ending at train cutoff date
   
2. TEST-TIME APPLICATION:
   - Apply pre-computed cluster assignments to test stocks
   - For new stocks not in training: assign to nearest cluster or -1
   
3. ROLLING WINDOW APPROACH:
   - Recompute clusters at each window using only historical data
   - Cluster assignments may change over time (this is OK)

4. FEATURE COMPUTATION:
   - Rank_InCluster and Dist_Cluster_Mean are computed per-timestamp
   - These are safe as they use only current-timestamp cross-sectional data
""")

    return results


if __name__ == "__main__":
    results = main()
