"""Statistical sector/cluster assignment for NZX stocks.

This module creates pseudo-sectors using unsupervised clustering since
official NZX sector classifications are sparse. Clusters are based on:
1. Return correlation patterns (stocks that move together)
2. Risk/return characteristics (volatility, skewness, etc.)

Usage:
    from features.ticker_clusters import assign_clusters, add_cluster_features
    
    # Get cluster assignments
    cluster_map = assign_clusters(wide_df)
    
    # Add cluster-relative features to dataframe
    wide_df = add_cluster_features(wide_df, cluster_map)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import Optional
import logging

from config.columns import TIMESTAMP, TICKER

logger = logging.getLogger(__name__)


def compute_ticker_characteristics(
    wide_df: pd.DataFrame,
    lookback_days: int = 500,
    min_obs: int = 100,
    max_daily_return: float = 1.0,
) -> pd.DataFrame:
    """Compute risk/return characteristics for each ticker.
    
    Args:
        wide_df: Wide-format DataFrame with TIMESTAMP, TICKER, Close columns
        lookback_days: Number of recent days to use for statistics
        min_obs: Minimum observations required per ticker
        max_daily_return: Maximum allowed daily return (filter anomalies)
            Returns exceeding this are clipped. Default 1.0 = 100%.
        
    Returns:
        DataFrame with ticker-level statistics
    """
    # Filter to NZ tickers only
    nz_mask = wide_df[TICKER].str.endswith('.NZ', na=False)
    df = wide_df[nz_mask].copy()
    
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")
    
    # Get recent data
    recent_ts = sorted(df[TIMESTAMP].unique())[-lookback_days:]
    df = df[df[TIMESTAMP].isin(recent_ts)]
    
    # Calculate returns per ticker
    stats = []
    for ticker in df[TICKER].unique():
        ticker_df = df[df[TICKER] == ticker].sort_values(TIMESTAMP)
        prices = ticker_df['Close'].values
        
        if len(prices) < min_obs:
            continue
            
        # Daily returns
        returns = np.diff(prices) / prices[:-1]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < min_obs:
            continue
        
        # Filter extreme returns (likely data errors/unadjusted splits)
        # Count anomalies before filtering
        n_anomalies = np.sum(np.abs(returns) > max_daily_return)
        anomaly_rate = n_anomalies / len(returns)
        
        # Clip extreme returns instead of removing (preserves time alignment)
        returns_clean = np.clip(returns, -max_daily_return, max_daily_return)
        
        stats.append({
            'ticker': ticker,
            'volatility': np.std(returns_clean) * np.sqrt(252),
            'mean_return': np.mean(returns_clean) * 252,
            'skewness': pd.Series(returns_clean).skew(),
            'kurtosis': pd.Series(returns_clean).kurtosis(),
            'autocorr': np.corrcoef(returns_clean[:-1], returns_clean[1:])[0, 1] if len(returns_clean) > 10 else 0,
            'pos_days_pct': (returns_clean > 0).mean(),
            'max_drawdown': _calculate_max_drawdown(prices),
            'n_obs': len(returns_clean),
            'anomaly_rate': anomaly_rate,
        })
    
    return pd.DataFrame(stats)


def _calculate_max_drawdown(prices: np.ndarray) -> float:
    """Calculate maximum drawdown from price series."""
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    return float(np.min(drawdown))


def assign_clusters(
    wide_df: pd.DataFrame,
    n_clusters: int = 6,
    method: str = 'kmeans',
    lookback_days: int = 500,
) -> dict[str, int]:
    """Assign each ticker to a statistical cluster.
    
    Args:
        wide_df: Wide-format DataFrame
        n_clusters: Number of clusters to create
        method: 'kmeans' or 'hierarchical'
        lookback_days: Days of history to use for clustering
        
    Returns:
        Dictionary mapping ticker -> cluster_id
    """
    # Compute characteristics
    stats_df = compute_ticker_characteristics(wide_df, lookback_days)
    
    if len(stats_df) < n_clusters:
        logger.warning(f"Only {len(stats_df)} tickers available, reducing clusters")
        n_clusters = max(2, len(stats_df) // 5)
    
    # Prepare features
    features = ['volatility', 'mean_return', 'skewness', 'kurtosis', 
                'autocorr', 'pos_days_pct', 'max_drawdown']
    X = stats_df[features].fillna(0).values
    
    # Handle infinite values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    
    labels = model.fit_predict(X_scaled)
    stats_df['cluster'] = labels
    
    # Log cluster info
    for c in range(n_clusters):
        cluster_df = stats_df[stats_df['cluster'] == c]
        logger.info(
            f"Cluster {c}: {len(cluster_df)} stocks, "
            f"vol={cluster_df['volatility'].mean():.1%}, "
            f"ret={cluster_df['mean_return'].mean():.1%}"
        )
    
    return dict(zip(stats_df['ticker'], stats_df['cluster']))


def add_cluster_features(
    wide_df: pd.DataFrame,
    cluster_map: Optional[dict[str, int]] = None,
    n_clusters: int = 6,
) -> pd.DataFrame:
    """Add cluster-relative features to the DataFrame.
    
    Features added:
    - Cluster: cluster assignment (0 to n_clusters-1)
    - Cluster_Size: number of stocks in the cluster at each timestamp
    - Rank_Close_InCluster: rank of Close price within cluster
    - Rank_Return_InCluster: rank of return within cluster (if Return column exists)
    - Dist_Cluster_Mean_Close: distance from cluster mean Close
    
    Args:
        wide_df: Wide-format DataFrame
        cluster_map: Pre-computed cluster assignments. If None, will compute.
        n_clusters: Number of clusters (used if cluster_map is None)
        
    Returns:
        DataFrame with cluster features added
    """
    df = wide_df.copy()
    
    # Compute clusters if not provided
    if cluster_map is None:
        cluster_map = assign_clusters(df, n_clusters)
    
    # Add cluster column
    df['Cluster'] = df[TICKER].map(cluster_map).fillna(-1).astype(int)
    
    # Add cluster-relative features per timestamp
    results = []
    for ts, group in df.groupby(TIMESTAMP):
        group = group.copy()
        
        # Cluster size
        cluster_sizes = group.groupby('Cluster').size()
        group['Cluster_Size'] = group['Cluster'].map(cluster_sizes)
        
        # Rank within cluster for Close
        if 'Close' in group.columns:
            group['Rank_Close_InCluster'] = group.groupby('Cluster')['Close'].rank(pct=True)
            
            # Distance from cluster mean
            cluster_means = group.groupby('Cluster')['Close'].transform('mean')
            group['Dist_Cluster_Mean_Close'] = (group['Close'] - cluster_means) / cluster_means
        
        results.append(group)
    
    return pd.concat(results, ignore_index=True)


def get_cluster_summary(
    wide_df: pd.DataFrame,
    cluster_map: dict[str, int],
) -> pd.DataFrame:
    """Get summary statistics for each cluster.
    
    Returns DataFrame with cluster characteristics for interpretation.
    """
    stats_df = compute_ticker_characteristics(wide_df)
    stats_df['cluster'] = stats_df['ticker'].map(cluster_map)
    
    summary = stats_df.groupby('cluster').agg({
        'ticker': 'count',
        'volatility': 'mean',
        'mean_return': 'mean',
        'skewness': 'mean',
        'kurtosis': 'mean',
        'max_drawdown': 'mean',
    }).rename(columns={'ticker': 'n_stocks'})
    
    return summary


# =============================================================================
# CLUSTER REPORTING (for NZX-focused annual predictions)
# =============================================================================

def get_cluster_membership_report(
    cluster_map: dict[str, int],
    stats_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Generate detailed cluster membership report.
    
    Args:
        cluster_map: Dictionary mapping ticker -> cluster_id.
        stats_df: Optional DataFrame with ticker characteristics (from compute_ticker_characteristics).
    
    Returns:
        Dictionary with cluster details:
        {
            'clusters': {
                0: {
                    'label': 'Defensive/Quality',
                    'tickers': ['FPH.NZ', 'EBO.NZ', ...],
                    'n_stocks': 8,
                    'characteristics': {'volatility': 0.18, 'mean_return': 0.05, ...}
                },
                ...
            },
            'ticker_lookup': {'FPH.NZ': {'cluster': 0, 'label': 'Defensive/Quality'}, ...}
        }
    """
    # Build cluster -> tickers mapping
    clusters_to_tickers: dict[int, list] = {}
    for ticker, cluster in cluster_map.items():
        if cluster not in clusters_to_tickers:
            clusters_to_tickers[cluster] = []
        clusters_to_tickers[cluster].append(ticker)
    
    # Sort tickers within each cluster
    for cluster in clusters_to_tickers:
        clusters_to_tickers[cluster] = sorted(clusters_to_tickers[cluster])
    
    # Get characteristics per cluster if stats available
    cluster_chars = {}
    if stats_df is not None:
        stats_with_cluster = stats_df.copy()
        stats_with_cluster['cluster'] = stats_with_cluster['ticker'].map(cluster_map)
        
        for cluster in clusters_to_tickers:
            cluster_stats = stats_with_cluster[stats_with_cluster['cluster'] == cluster]
            if len(cluster_stats) > 0:
                cluster_chars[cluster] = {
                    'volatility': float(cluster_stats['volatility'].mean()),
                    'mean_return': float(cluster_stats['mean_return'].mean()),
                    'skewness': float(cluster_stats['skewness'].mean()),
                    'max_drawdown': float(cluster_stats['max_drawdown'].mean()),
                    'pos_days_pct': float(cluster_stats['pos_days_pct'].mean()),
                }
    
    # Build report
    report = {'clusters': {}, 'ticker_lookup': {}}
    
    for cluster, tickers in sorted(clusters_to_tickers.items()):
        # Get cluster label based on characteristics
        chars = cluster_chars.get(cluster, {})
        label = interpret_cluster(
            chars.get('volatility', 0.3),
            chars.get('mean_return', 0)
        )
        
        report['clusters'][cluster] = {
            'label': label,
            'tickers': tickers,
            'n_stocks': len(tickers),
            'characteristics': chars,
        }
        
        # Build ticker lookup
        for ticker in tickers:
            report['ticker_lookup'][ticker] = {
                'cluster': cluster,
                'label': label,
            }
    
    return report


def format_cluster_report_text(report: dict) -> str:
    """Format cluster report as human-readable text.
    
    Args:
        report: Output from get_cluster_membership_report().
    
    Returns:
        Formatted string for console output.
    """
    lines = [
        "=" * 60,
        "CLUSTER MEMBERSHIP REPORT",
        "=" * 60,
        "",
    ]
    
    for cluster_id in sorted(report['clusters'].keys()):
        cluster = report['clusters'][cluster_id]
        lines.append(f"CLUSTER {cluster_id}: {cluster['label']}")
        lines.append("-" * 40)
        
        chars = cluster.get('characteristics', {})
        if chars:
            lines.append(f"  Characteristics:")
            lines.append(f"    Avg Volatility:    {chars.get('volatility', 0):.1%}")
            lines.append(f"    Avg Annual Return: {chars.get('mean_return', 0):.1%}")
            lines.append(f"    Avg Max Drawdown:  {chars.get('max_drawdown', 0):.1%}")
            lines.append(f"    Avg % Positive:    {chars.get('pos_days_pct', 0):.1%}")
        
        lines.append(f"  Stocks ({cluster['n_stocks']}):")
        # Show tickers in rows of 6
        tickers = cluster['tickers']
        for i in range(0, len(tickers), 6):
            row = tickers[i:i+6]
            lines.append(f"    {', '.join(row)}")
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def get_cluster_performance_by_predictions(
    predictions_df: pd.DataFrame,
    cluster_map: dict[str, int],
    ticker_col: str = "ticker",
    actual_col: str = "actual_return",
    predicted_col: str = "predicted_score",
) -> pd.DataFrame:
    """Analyze model performance by cluster.
    
    Args:
        predictions_df: DataFrame with predictions (ticker, actual_return, predicted_score).
        cluster_map: Dictionary mapping ticker -> cluster_id.
        ticker_col: Column name for ticker.
        actual_col: Column name for actual returns.
        predicted_col: Column name for predicted scores.
    
    Returns:
        DataFrame with per-cluster performance metrics.
    """
    df = predictions_df.copy()
    df['cluster'] = df[ticker_col].map(cluster_map).fillna(-1).astype(int)
    
    # Compute metrics per cluster
    results = []
    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        
        if len(cluster_df) < 5:
            continue
        
        # Get cluster label
        sample_ticker = cluster_df[ticker_col].iloc[0]
        cluster_info = cluster_map.get(sample_ticker, -1)
        
        # Compute IC within cluster
        from scipy.stats import spearmanr, pearsonr
        ic, _ = pearsonr(cluster_df[predicted_col], cluster_df[actual_col])
        rank_ic, _ = spearmanr(cluster_df[predicted_col], cluster_df[actual_col])
        
        # Compute hit rate
        hit_rate = (cluster_df[actual_col] > 0).mean()
        
        # Average return
        avg_return = cluster_df[actual_col].mean()
        
        results.append({
            'cluster': cluster,
            'n_predictions': len(cluster_df),
            'n_unique_tickers': cluster_df[ticker_col].nunique(),
            'pearson_ic': ic,
            'rank_ic': rank_ic,
            'hit_rate': hit_rate,
            'avg_return': avg_return,
            'return_std': cluster_df[actual_col].std(),
        })
    
    return pd.DataFrame(results)


# Cluster interpretation labels (based on NZX clustering results)
# These are dynamically assigned based on cluster characteristics
CLUSTER_INTERPRETATIONS = {
    # (vol_range, return_range) -> label
    "low_vol_positive": "Defensive/Quality",      # vol < 25%, ret > 0
    "low_vol_negative": "Value/Turnaround",       # vol < 25%, ret < 0  
    "mid_vol_positive": "Growth",                 # 25% < vol < 50%, ret > 0
    "mid_vol_negative": "Cyclical/Challenged",    # 25% < vol < 50%, ret < 0
    "high_vol_positive": "Speculative/Momentum",  # vol > 50%, ret > 0
    "high_vol_negative": "Distressed",            # vol > 50%, ret < 0
}


def interpret_cluster(volatility: float, mean_return: float) -> str:
    """Get human-readable cluster interpretation based on characteristics."""
    vol_cat = "low" if volatility < 0.25 else ("mid" if volatility < 0.50 else "high")
    ret_cat = "positive" if mean_return > 0 else "negative"
    key = f"{vol_cat}_vol_{ret_cat}"
    return CLUSTER_INTERPRETATIONS.get(key, "Unknown")
