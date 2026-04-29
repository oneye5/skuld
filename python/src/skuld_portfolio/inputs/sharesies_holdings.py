"""Parse Sharesies portfolio export CSV to CurrentPortfolio."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from skuld_common.contracts import CurrentPortfolio


class SharesiesSchemaError(ValueError):
    """Raised when Sharesies CSV schema differs from expected format."""
    pass


_EXPECTED_COLUMNS = frozenset([
    "Instrument code",
    "Market code",
    "Name",
    "Shares held",
    "Average cost per share NZD",
    "Latest market price NZD",
    "Total cost NZD",
    "Market value NZD",
])


def parse_sharesies_csv(path: Path, cash_nzd: float = 0.0) -> CurrentPortfolio:
    """Parse a Sharesies portfolio export CSV into a CurrentPortfolio.
    
    Args:
        path: Path to Sharesies CSV export.
        cash_nzd: Cash balance in NZD (not in CSV; supplied separately).
    
    Returns:
        CurrentPortfolio with holdings, prices, and cash.
    
    Raises:
        SharesiesSchemaError: if CSV header differs from expected schema.
        FileNotFoundError: if path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Sharesies CSV not found: {path}")
    
    df = pd.read_csv(path)
    
    # Validate schema
    actual_cols = set(df.columns.tolist())
    if actual_cols != _EXPECTED_COLUMNS:
        missing = _EXPECTED_COLUMNS - actual_cols
        extra = actual_cols - _EXPECTED_COLUMNS
        msg = "Sharesies CSV schema mismatch.\n"
        if missing:
            msg += f"  Missing columns: {sorted(missing)}\n"
        if extra:
            msg += f"  Extra columns: {sorted(extra)}\n"
        msg += "\nRemediation:\n"
        msg += "  1. Ensure you exported the 'Shares portfolio' view from Sharesies.\n"
        msg += "  2. Expected columns: " + ", ".join(sorted(_EXPECTED_COLUMNS))
        raise SharesiesSchemaError(msg)
    
    # Empty file (header only) is an error
    if df.empty:
        raise SharesiesSchemaError(
            "Sharesies CSV is empty (header only, no holdings rows).\n"
            "Remediation: Export a non-empty portfolio or check the file."
        )
    
    # Map columns to ticker, shares, price
    tickers = df["Instrument code"].str.strip().tolist()
    shares = df["Shares held"].astype(float).astype(int).tolist()
    prices = df["Latest market price NZD"].astype(float).tolist()
    
    holdings = pd.Series(shares, index=tickers, dtype=int)
    price_series = pd.Series(prices, index=tickers, dtype=float)
    
    return CurrentPortfolio(
        holdings=holdings,
        prices=price_series,
        cash_nzd=cash_nzd,
    )
