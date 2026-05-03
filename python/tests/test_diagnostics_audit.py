"""Tests for the pipeline audit module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import PreparedPanel
from skuld_research.diagnostics.audit import PipelineAuditReport, audit_pipeline


def make_panel(
    tickers=("A", "B", "C", "D", "E"),
    n_months=24,
    asof=pd.Timestamp("2026-01-01"),
    sectors=None,
    corporate_actions=None,
    rebalance_dates=None,
) -> PreparedPanel:
    dates_monthly = pd.date_range("2024-01-31", periods=n_months, freq="ME")
    dates_daily = pd.date_range("2024-01-01", periods=n_months * 21, freq="B")
    rng = np.random.default_rng(42)
    tickers = list(tickers)
    returns_m = pd.DataFrame(
        rng.normal(0.01, 0.05, (n_months, len(tickers))),
        index=dates_monthly,
        columns=tickers,
    )
    returns_d = pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(dates_daily), len(tickers))),
        index=dates_daily,
        columns=tickers,
    )
    mc = pd.DataFrame(
        np.ones((n_months, len(tickers))) * 1e8,
        index=dates_monthly,
        columns=tickers,
    )
    sec = pd.Series({t: (sectors or {}).get(t, "Unknown") for t in tickers})
    mask_index = rebalance_dates if rebalance_dates is not None else dates_monthly
    mask = pd.DataFrame(True, index=mask_index, columns=tickers)
    return PreparedPanel(
        returns_daily=returns_d,
        returns_monthly=returns_m,
        market_cap=mc,
        sector=sec,
        universe_mask=mask,
        macro=pd.DataFrame(),
        asof=asof,
        corporate_actions=corporate_actions if corporate_actions is not None else pd.DataFrame(),
    )


# --- PIT compliance ---

def test_pit_compliant():
    panel = make_panel(asof=pd.Timestamp("2026-01-01"))
    report = audit_pipeline(panel)
    assert report.pit_compliant is True
    assert report.pit_max_return_date < report.pit_asof


def test_pit_violation():
    # returns end at 2025-12-31 (24 months from 2024-01-31), asof is 2025-06-01 → violation
    panel = make_panel(asof=pd.Timestamp("2025-06-01"))
    report = audit_pipeline(panel)
    assert report.pit_compliant is False


# --- Timestamp alignment ---

def test_aligned_dates():
    report = audit_pipeline(make_panel())
    assert report.rebalance_dates_aligned is True
    assert report.n_misaligned_dates == 0


def test_misaligned_dates():
    dates_monthly = pd.date_range("2024-01-31", periods=24, freq="ME")
    # Replace one date with a mid-month date
    bad_date = pd.Timestamp("2024-06-15")
    rebalance_dates = dates_monthly.tolist()
    rebalance_dates[5] = bad_date
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    panel = make_panel(rebalance_dates=rebalance_dates)
    report = audit_pipeline(panel)
    assert report.rebalance_dates_aligned is False
    assert report.n_misaligned_dates == 1


# --- Sector coverage ---

def test_all_unknown_sectors():
    report = audit_pipeline(make_panel())
    assert report.frac_known_sector == 0.0
    assert report.n_tickers_known_sector == 0


def test_some_known_sectors():
    sectors = {"A": "Energy", "B": "Finance", "C": "Unknown", "D": "Unknown", "E": "Unknown"}
    panel = make_panel(sectors=sectors)
    report = audit_pipeline(panel)
    assert report.n_tickers_known_sector == 2
    assert report.frac_known_sector == pytest.approx(0.4)


# --- Delisting CSV ---

def test_no_delisting_csv():
    report = audit_pipeline(make_panel(), delisting_csv_path=None)
    assert report.n_tickers_in_delisting_csv == 0
    assert report.delisting_csv_to_panel_ratio == 0.0


def test_delisting_csv_loaded(tmp_path):
    import pandas as pd
    delist_df = pd.DataFrame({"ticker": ["A", "B", "C"]})
    csv_path = tmp_path / "delist.csv"
    delist_df.to_csv(csv_path, index=False)
    panel = make_panel()
    report = audit_pipeline(panel, delisting_csv_path=csv_path)
    assert report.n_tickers_in_delisting_csv == 3
    assert report.delisting_csv_to_panel_ratio > 0.0


# --- Corporate actions ---

def test_corporate_actions_empty():
    report = audit_pipeline(make_panel())
    assert report.n_corporate_action_events == 0
    assert report.n_corporate_action_tickers == 0


def test_corporate_actions_present():
    ca = pd.DataFrame({
        "ticker": ["A", "A", "B"],
        "ex_date": pd.to_datetime(["2024-03-15", "2024-06-15", "2024-09-15"]),
        "type": ["split", "split", "dividend"],
        "factor": [2.0, 0.5, 1.0],
    })
    panel = make_panel(corporate_actions=ca)
    report = audit_pipeline(panel)
    assert report.n_corporate_action_events == 3
    assert report.n_corporate_action_tickers == 2


# --- NaN fractions ---

def test_nan_fraction():
    tickers = ["A", "B", "C", "D", "E"]
    n_months = 24
    dates_monthly = pd.date_range("2024-01-31", periods=n_months, freq="ME")
    dates_daily = pd.date_range("2024-01-01", periods=n_months * 21, freq="B")
    rng = np.random.default_rng(42)
    returns_m = pd.DataFrame(
        rng.normal(0.01, 0.05, (n_months, len(tickers))),
        index=dates_monthly,
        columns=tickers,
    )
    # Inject NaN into ticker "C" (half the rows)
    returns_m.loc[dates_monthly[:12], "C"] = np.nan

    returns_d = pd.DataFrame(
        rng.normal(0.0005, 0.01, (len(dates_daily), len(tickers))),
        index=dates_daily,
        columns=tickers,
    )
    mc = pd.DataFrame(np.ones((n_months, len(tickers))) * 1e8, index=dates_monthly, columns=tickers)
    sec = pd.Series({t: "Unknown" for t in tickers})
    mask = pd.DataFrame(True, index=dates_monthly, columns=tickers)
    panel = PreparedPanel(
        returns_daily=returns_d,
        returns_monthly=returns_m,
        market_cap=mc,
        sector=sec,
        universe_mask=mask,
        macro=pd.DataFrame(),
        asof=pd.Timestamp("2026-01-01"),
        corporate_actions=pd.DataFrame(),
    )
    report = audit_pipeline(panel)
    assert report.max_nan_frac_returns > 0
    assert report.worst_nan_ticker == "C"
