"""Data caching utilities for expensive data transformations.

The long_to_wide conversion is expensive (~75 seconds). This module provides
caching to parquet for fast subsequent loads (~4 seconds).

Cache invalidation:
- Automatic: cache is invalidated if source file is newer than cache
- Manual: call invalidate_wide_data_cache() or delete cache files

Usage:
    # First call: computes and caches (~75s)
    wide_df = load_cached_wide_data()
    
    # Subsequent calls: loads from cache (~4s)
    wide_df = load_cached_wide_data()
    
    # With features (first call ~85s, subsequent ~5s)
    wide_df = load_cached_wide_data_with_features()
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Callable

import pandas as pd

from config.paths import CACHE_DIR, DATA_LONG_CSV
from config.settings import YEAR_2000_MS
from core.data_loader import load_long_data
from core.long_to_wide import clean_and_classify_tickers, add_macro_prefix, long_to_wide
from core.logging_config import get_logger

logger = get_logger(__name__)

# Cache file paths
WIDE_DATA_CACHE = CACHE_DIR / "wide_data.parquet"
WIDE_DATA_METADATA = CACHE_DIR / "wide_data_meta.json"

# Feature cache paths (includes pre-computed features)
FEATURE_DATA_CACHE = CACHE_DIR / "wide_data_with_features.parquet"
FEATURE_DATA_METADATA = CACHE_DIR / "wide_data_features_meta.json"

# Version for cache invalidation when feature logic changes
FEATURE_CACHE_VERSION = "1.4.0"  


def _get_source_file_hash(path: Path) -> str:
    """Get hash of source file modification time and size for cache invalidation."""
    stat = path.stat()
    content = f"{stat.st_mtime}_{stat.st_size}"
    return hashlib.md5(content.encode()).hexdigest()


def _is_cache_valid() -> bool:
    """Check if cached wide data is still valid.
    
    Cache is valid if:
    1. Cache file exists
    2. Metadata file exists
    3. Source file hash matches (file hasn't changed)
    """
    if not WIDE_DATA_CACHE.exists() or not WIDE_DATA_METADATA.exists():
        return False
    
    try:
        with open(WIDE_DATA_METADATA, "r") as f:
            metadata = json.load(f)
        
        current_hash = _get_source_file_hash(DATA_LONG_CSV)
        return metadata.get("source_hash") == current_hash
    except Exception as e:
        logger.warning(f"Error checking cache validity: {e}")
        return False


def _save_cache_metadata(df: pd.DataFrame) -> None:
    """Save cache metadata for validation."""
    metadata = {
        "source_hash": _get_source_file_hash(DATA_LONG_CSV),
        "created_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": len(df.columns),
        "year_2000_ms": YEAR_2000_MS,
    }
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(WIDE_DATA_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)


def load_cached_wide_data(
    force_refresh: bool = False,
    source_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load wide format data, using cache if available.
    
    This wraps the expensive long_to_wide conversion with parquet caching.
    First call takes ~75s, subsequent calls ~4s.
    
    Args:
        force_refresh: If True, ignore cache and recompute.
        source_path: Path to source CSV. Uses default if None.
    
    Returns:
        Wide format DataFrame with timestamp, ticker, and feature columns.
    """
    # Check cache validity
    if not force_refresh and _is_cache_valid():
        logger.info("Loading wide data from cache...")
        df = pd.read_parquet(WIDE_DATA_CACHE)
        logger.info(f"Loaded cached data: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        return df
    
    # Cache miss - compute from scratch
    logger.info("Cache miss - computing wide data from scratch (this takes ~75 seconds)...")
    
    # Load and transform
    long_df = load_long_data(source_path)
    
    # Filter post-2000
    long_df = long_df[long_df["timestamp"] >= YEAR_2000_MS].copy()
    
    # Clean and transform
    long_df = clean_and_classify_tickers(long_df)
    long_df = add_macro_prefix(long_df)
    wide_df = long_to_wide(long_df)
    
    # Free memory
    del long_df
    
    # Save to cache
    logger.info("Saving to cache...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wide_df.to_parquet(WIDE_DATA_CACHE, index=False, compression="snappy")
    _save_cache_metadata(wide_df)
    
    logger.info(f"Computed and cached: {wide_df.shape[0]:,} rows, {wide_df.shape[1]:,} columns")
    return wide_df


def invalidate_wide_data_cache() -> None:
    """Manually invalidate the wide data cache.
    
    Call this after changing data transformation logic or if cache seems stale.
    """
    if WIDE_DATA_CACHE.exists():
        WIDE_DATA_CACHE.unlink()
        logger.info(f"Deleted cache file: {WIDE_DATA_CACHE}")
    
    if WIDE_DATA_METADATA.exists():
        WIDE_DATA_METADATA.unlink()
        logger.info(f"Deleted metadata file: {WIDE_DATA_METADATA}")


def get_cache_info() -> dict:
    """Get information about the current cache state."""
    if not WIDE_DATA_CACHE.exists():
        return {"status": "no_cache"}
    
    info = {
        "status": "cached",
        "cache_path": str(WIDE_DATA_CACHE),
        "cache_size_mb": WIDE_DATA_CACHE.stat().st_size / 1024 / 1024,
        "is_valid": _is_cache_valid(),
    }
    
    if WIDE_DATA_METADATA.exists():
        with open(WIDE_DATA_METADATA, "r") as f:
            info["metadata"] = json.load(f)
    
    # Add feature cache info
    if FEATURE_DATA_CACHE.exists():
        info["feature_cache"] = {
            "status": "cached",
            "cache_path": str(FEATURE_DATA_CACHE),
            "cache_size_mb": FEATURE_DATA_CACHE.stat().st_size / 1024 / 1024,
            "is_valid": _is_feature_cache_valid(),
        }
        if FEATURE_DATA_METADATA.exists():
            with open(FEATURE_DATA_METADATA, "r") as f:
                info["feature_cache"]["metadata"] = json.load(f)
    else:
        info["feature_cache"] = {"status": "no_cache"}
    
    return info


# =============================================================================
# FEATURE CACHE (wide data + pre-computed features)
# =============================================================================

def _is_feature_cache_valid() -> bool:
    """Check if cached feature data is still valid.
    
    Cache is valid if:
    1. Feature cache file exists
    2. Feature metadata file exists  
    3. Source file hash matches (file hasn't changed)
    4. Feature cache version matches (feature logic hasn't changed)
    """
    if not FEATURE_DATA_CACHE.exists() or not FEATURE_DATA_METADATA.exists():
        return False
    
    try:
        with open(FEATURE_DATA_METADATA, "r") as f:
            metadata = json.load(f)
        
        # Check source file hash
        current_hash = _get_source_file_hash(DATA_LONG_CSV)
        if metadata.get("source_hash") != current_hash:
            logger.debug("Feature cache invalid: source file changed")
            return False
        
        # Check cache version
        if metadata.get("cache_version") != FEATURE_CACHE_VERSION:
            logger.debug("Feature cache invalid: cache version changed")
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Error checking feature cache validity: {e}")
        return False


def _save_feature_cache_metadata(df: pd.DataFrame, feature_sets: List[str]) -> None:
    """Save feature cache metadata for validation."""
    metadata = {
        "source_hash": _get_source_file_hash(DATA_LONG_CSV),
        "cache_version": FEATURE_CACHE_VERSION,
        "created_at": datetime.now().isoformat(),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "feature_sets": feature_sets,
        "year_2000_ms": YEAR_2000_MS,
    }
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_DATA_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)


def load_cached_wide_data_with_features(
    force_refresh: bool = False,
    experimental_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load wide format data with pre-computed features, using cache if available.
    
    This wraps both long_to_wide conversion AND feature engineering with parquet caching.
    First call takes ~85s, subsequent calls ~5s.
    
    Args:
        force_refresh: If True, ignore cache and recompute.
        experimental_features: List of experimental feature sets to apply.
    
    Returns:
        Wide format DataFrame with timestamp, ticker, feature columns.
    """
    feature_sets = experimental_features or []
    
    # Check cache validity
    if not force_refresh and _is_feature_cache_valid():
        logger.info("Loading wide data with features from cache...")
        start = time.perf_counter()
        df = pd.read_parquet(FEATURE_DATA_CACHE)
        elapsed = time.perf_counter() - start
        logger.info(f"Loaded cached feature data: {df.shape[0]:,} rows, {df.shape[1]:,} columns in {elapsed:.2f}s")
        return df
    
    # Cache miss - compute from scratch
    logger.info("Feature cache miss - computing wide data with features (this takes ~85 seconds)...")
    start = time.perf_counter()
    
    # First get wide data (may use its own cache)
    wide_df = load_cached_wide_data(force_refresh=force_refresh)
    
    # Drop sparse columns
    from core.preprocessor import drop_sparse_columns
    wide_df = drop_sparse_columns(wide_df, threshold=0.95)
    
    # Add features
    from features.ratios import add_financial_ratios
    from features.technical import add_technical_features
    from features.alpha_factors import add_alpha_factors
    from features.lag_ma_features import add_lag_ma_features
    from features.attention_features import add_aggregate_attention_features
    
    wide_df = add_financial_ratios(wide_df)
    wide_df = add_technical_features(wide_df)
    wide_df = add_alpha_factors(wide_df)
    wide_df = add_lag_ma_features(wide_df)
    wide_df = add_aggregate_attention_features(wide_df)
    
    # Apply experimental features if requested
    if experimental_features:
        from features.feature_config import apply_experimental_features
        if "alpha_fast" in experimental_features:
            from features.alpha_factors_fast import add_alpha_factors_fast
            wide_df = add_alpha_factors_fast(wide_df)
        else:
            wide_df = apply_experimental_features(wide_df, experimental_features)
    
    # Force float32 to save memory and disk space
    for col in wide_df.columns:
        if wide_df[col].dtype == 'float64':
            wide_df[col] = wide_df[col].astype('float32')
    
    elapsed = time.perf_counter() - start
    logger.info(f"Computed features in {elapsed:.2f}s")
    
    # Save to cache
    logger.info("Saving feature data to cache...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wide_df.to_parquet(FEATURE_DATA_CACHE, index=False, compression="snappy")
    _save_feature_cache_metadata(wide_df, feature_sets)
    
    logger.info(f"Computed and cached: {wide_df.shape[0]:,} rows, {wide_df.shape[1]:,} columns")
    return wide_df


def invalidate_feature_cache() -> None:
    """Manually invalidate the feature data cache.
    
    Call this after changing feature engineering logic.
    """
    if FEATURE_DATA_CACHE.exists():
        FEATURE_DATA_CACHE.unlink()
        logger.info(f"Deleted feature cache file: {FEATURE_DATA_CACHE}")
    
    if FEATURE_DATA_METADATA.exists():
        FEATURE_DATA_METADATA.unlink()
        logger.info(f"Deleted feature metadata file: {FEATURE_DATA_METADATA}")


def invalidate_all_caches() -> None:
    """Invalidate both wide data and feature caches."""
    invalidate_wide_data_cache()
    invalidate_feature_cache()
