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
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    
    return info
