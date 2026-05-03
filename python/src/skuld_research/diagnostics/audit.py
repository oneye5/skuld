"""Pipeline audit module for data quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from skuld_common.contracts import PreparedPanel


@dataclass(frozen=True)
class PipelineAuditReport:
    # PIT compliance
    pit_max_return_date: pd.Timestamp
    pit_asof: pd.Timestamp
    # True when returns_monthly.index.max() < panel.asof.
    # asof is the date *after* which no data is permitted — the snapshot stands at asof
    # and the last available return is labelled at the month-end *before* asof.
    # Example: asof=2026-01-01, last return date=2025-12-31 → compliant.
    pit_compliant: bool

    # Timestamp alignment
    rebalance_dates_aligned: bool
    n_rebalance_dates: int
    n_misaligned_dates: int

    # Sector coverage
    n_tickers_total: int
    n_tickers_known_sector: int
    frac_known_sector: float

    # Survivorship coverage
    n_tickers_in_panel: int
    n_tickers_in_delisting_csv: int
    # ratio = n_delisting_csv_tickers / n_panel_tickers; may exceed 1.0 if CSV has historical tickers
    delisting_csv_to_panel_ratio: float

    # Data gaps
    mean_nan_frac_returns: float
    max_nan_frac_returns: float
    worst_nan_ticker: str

    # Corporate actions
    n_corporate_action_events: int
    n_corporate_action_tickers: int


def audit_pipeline(
    panel: PreparedPanel,
    *,
    delisting_csv_path: Path | str | None = None,
) -> PipelineAuditReport:
    """Run pipeline audit checks on a PreparedPanel."""
    # PIT compliance
    returns = panel.returns_monthly
    pit_max_return_date = pd.Timestamp(returns.index.max()) if not returns.empty else pd.Timestamp("NaT")
    pit_asof = panel.asof
    pit_compliant = pit_max_return_date < pit_asof

    # Timestamp alignment
    rebalance_index = panel.universe_mask.index
    n_rebalance_dates = len(rebalance_index)
    misaligned = [d for d in rebalance_index if d != d + pd.offsets.MonthEnd(0)]
    n_misaligned_dates = len(misaligned)
    rebalance_dates_aligned = n_misaligned_dates == 0

    # Sector coverage
    sector = panel.sector
    n_tickers_total = len(sector)
    n_tickers_known_sector = int((sector != "Unknown").sum())
    frac_known_sector = n_tickers_known_sector / n_tickers_total if n_tickers_total > 0 else 0.0

    # Survivorship coverage
    mask = panel.universe_mask
    tickers_in_panel = set(mask.columns[mask.any(axis=0)])
    n_tickers_in_panel = len(tickers_in_panel)

    n_tickers_in_delisting_csv = 0
    delisting_csv_to_panel_ratio = 0.0
    if delisting_csv_path is not None:
        p = Path(delisting_csv_path)
        if p.exists():
            df_delist = pd.read_csv(p)
            n_tickers_in_delisting_csv = df_delist["ticker"].nunique()
            if n_tickers_in_panel > 0:
                delisting_csv_to_panel_ratio = n_tickers_in_delisting_csv / n_tickers_in_panel

    # Data gaps
    if returns.empty:
        mean_nan_frac_returns = 0.0
        max_nan_frac_returns = 0.0
        worst_nan_ticker = ""
    else:
        nan_fracs = returns.isna().mean(axis=0)
        mean_nan_frac_returns = float(nan_fracs.mean())
        max_nan_frac_returns = float(nan_fracs.max())
        worst_nan_ticker = str(nan_fracs.idxmax()) if not nan_fracs.empty else ""

    # Corporate actions
    ca = panel.corporate_actions
    n_corporate_action_events = len(ca) if ca is not None else 0
    n_corporate_action_tickers = (
        ca["ticker"].nunique() if (ca is not None and not ca.empty and "ticker" in ca.columns) else 0
    )

    return PipelineAuditReport(
        pit_max_return_date=pit_max_return_date,
        pit_asof=pit_asof,
        pit_compliant=pit_compliant,
        rebalance_dates_aligned=rebalance_dates_aligned,
        n_rebalance_dates=n_rebalance_dates,
        n_misaligned_dates=n_misaligned_dates,
        n_tickers_total=n_tickers_total,
        n_tickers_known_sector=n_tickers_known_sector,
        frac_known_sector=frac_known_sector,
        n_tickers_in_panel=n_tickers_in_panel,
        n_tickers_in_delisting_csv=n_tickers_in_delisting_csv,
        delisting_csv_to_panel_ratio=delisting_csv_to_panel_ratio,
        mean_nan_frac_returns=mean_nan_frac_returns,
        max_nan_frac_returns=max_nan_frac_returns,
        worst_nan_ticker=worst_nan_ticker,
        n_corporate_action_events=n_corporate_action_events,
        n_corporate_action_tickers=n_corporate_action_tickers,
    )
