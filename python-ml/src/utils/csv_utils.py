"""CSV utility functions - delegates to io_utils for efficient I/O.

This module provides backward compatibility by wrapping the new io_utils module.
All CSV operations now support automatic Parquet optimization for intermediate files.
"""
from src.utils.io_utils import load_csv, save_csv, clear_cache

# Re-export for backward compatibility
__all__ = ['load_csv', 'save_csv', 'clear_cache']