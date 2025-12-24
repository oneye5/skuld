#!/usr/bin/env python3
"""Main entry point for running the ML pipeline.

Usage:
    uv run scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.rolling_window import run_rolling_windows, print_summary


def main():
    """Run the complete rolling window pipeline."""
    print("=" * 60)
    print("SKULD ML PIPELINE")
    print("=" * 60)
    
    result = run_rolling_windows()
    print_summary(result)
    
    return result


if __name__ == "__main__":
    main()
