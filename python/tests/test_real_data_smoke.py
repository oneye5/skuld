"""Smoke tests against real data_long.csv.

Skipped if the file is absent — these are local-only validation checks.
"""

from pathlib import Path

import pandas as pd
import pytest

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "data_long.csv"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="data/data_long.csv not present"
)


def test_load_real_csv():
    """Real CSV loads without crashing."""
    from skuld_research.data.csv_loader import load_raw_csv

    raw = load_raw_csv(DATA_PATH)
    assert not raw.prices.empty, "No price data loaded"
    assert len(raw.prices.columns) > 100, f"Only {len(raw.prices.columns)} tickers loaded"


def test_pit_loader_on_real_data():
    """PIT snapshot of real data at a recent date returns data."""
    from skuld_research.data.csv_loader import load_raw_csv
    from skuld_research.data.pit_loader import PITLoader

    raw = load_raw_csv(DATA_PATH)
    loader = PITLoader(raw)
    snap = loader.as_of(pd.Timestamp("2025-06-01", tz="UTC"))
    assert not snap.prices.empty
    # PIT invariant: no price after 2025-06-01
    max_date = snap.prices.index.max()
    assert max_date < pd.Timestamp("2025-06-01"), f"PIT violation: {max_date}"


def test_negative_price_detection_real():
    """Run negative-price validation on real data and report."""
    from skuld_research.data.csv_loader import load_raw_csv
    from skuld_research.data.validation import detect_negative_prices

    raw = load_raw_csv(DATA_PATH)
    report = detect_negative_prices(raw.prices)
    # We don't assert zero issues — we just confirm the check runs.
    # Print results for human review.
    if not report.is_clean:
        print(f"\nWARNING: {report.issue_count} negative price observations found:")
        for ticker, dates in report.details.items():
            print(f"  {ticker}: {dates[:5]}{'...' if len(dates) > 5 else ''}")
