"""60/40 equity/bond benchmark."""
from __future__ import annotations

import pandas as pd

from skuld_common.contracts import PreparedPanel


def sixty_forty(
    panel: PreparedPanel,
    equity_proxy: str = "FNZ.NZ",
    bond_macro_field: str = "long_term_interest_rates",
    duration_years: float = 0.0,
    flat_haircut_bps: float = 50.0,
) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp, tuple[str, ...]]:
    """60/40 equity/bond monthly return path.

    Synthesises a 60/40 monthly return path from:
    - equity leg: panel.returns_monthly[equity_proxy]
    - bond leg: yield-only from panel.macro[bond_macro_field]

    Combined: 0.60 * eq + 0.40 * bd on months where BOTH are finite.
    Applies flat monthly haircut: flat_haircut_bps / 10_000 / 12.

    The default haircut (50bps) reflects ETF management fees and slippage,
    NOT survivorship bias — FNZ.NZ is a live ETF and does not face NZX
    delisting risk.  Pass flat_haircut_bps=400 explicitly if you want
    to apply the same survivorship haircut as the strategy.

    Args:
        panel: PreparedPanel with returns_monthly and macro data.
        equity_proxy: ticker for equity leg (default "FNZ.NZ").
        bond_macro_field: macro field name for bond rates (default
            "long_term_interest_rates").
        flat_haircut_bps: annual flat haircut in basis points (default 50).

    Returns:
        (monthly_returns, coverage_start, coverage_end, notes)
        - monthly_returns: pd.Series of 60/40 net returns (index = month-end dates).
        - coverage_start: first month with both legs finite.
        - coverage_end: last month with both legs finite.
        - notes: tuple of caveats.
    """
    notes_list = []

    # Equity leg
    if equity_proxy not in panel.returns_monthly.columns:
        # Equity proxy missing → return empty
        notes_list.append(f"Equity proxy '{equity_proxy}' not found in panel")
        return (
            pd.Series([], dtype=float),
            pd.Timestamp("1970-01-01"),
            pd.Timestamp("1970-01-01"),
            tuple(notes_list),
        )

    eq_returns = panel.returns_monthly[equity_proxy]

    # Bond leg
    if bond_macro_field not in panel.macro.columns or panel.macro.empty:
        notes_list.append(f"Bond macro field '{bond_macro_field}' not found in panel.macro")
        return (
            pd.Series([], dtype=float),
            pd.Timestamp("1970-01-01"),
            pd.Timestamp("1970-01-01"),
            tuple(notes_list),
        )

    bond_rates_daily = panel.macro[bond_macro_field]

    # Resample to month-end, take last available rate
    bond_rates_monthly = bond_rates_daily.resample("ME").last()

    # Ingested macro series can mix decimal and percentage observations.
    converted_rates = bond_rates_monthly.mask(
        bond_rates_monthly > 1.0, bond_rates_monthly / 100.0
    )
    if not converted_rates.equals(bond_rates_monthly):
        bond_rates_monthly = converted_rates
        notes_list.append("Bond rates auto-converted from percentage to decimal")

    # Monthly bond return: coupon carry only by default; optionally add
    # duration-based price return from month-end yield changes.
    coupon_carry = (1.0 + bond_rates_monthly) ** (1.0 / 12.0) - 1.0
    if duration_years <= 0:
        bond_returns_monthly = coupon_carry
    else:
        bond_returns_monthly = coupon_carry - duration_years * bond_rates_monthly.diff()

    # Align both legs
    df = pd.DataFrame({
        "eq": eq_returns,
        "bd": bond_returns_monthly,
    }).dropna()

    if df.empty:
        notes_list.append("No overlapping months with both equity and bond data")
        return (
            pd.Series([], dtype=float),
            pd.Timestamp("1970-01-01"),
            pd.Timestamp("1970-01-01"),
            tuple(notes_list),
        )

    # 60/40 combination
    combined = 0.60 * df["eq"] + 0.40 * df["bd"]

    # Apply flat haircut
    monthly_haircut = (flat_haircut_bps / 10_000.0) / 12.0
    combined = combined - monthly_haircut

    coverage_start = df.index[0]
    coverage_end = df.index[-1]

    # Add standard caveats
    if duration_years <= 0:
        notes_list.append("yield-only bond proxy (no duration P&L)")
    else:
        notes_list.append(
            f"duration-aware bond proxy ({duration_years:.1f}y configured duration)"
        )
    notes_list.append(f"flat {int(flat_haircut_bps)}bps annual haircut applied uniformly")

    return (combined, coverage_start, coverage_end, tuple(notes_list))
