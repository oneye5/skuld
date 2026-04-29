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


def _load_real():
    from skuld_research.data.csv_loader import load_raw_csv

    return load_raw_csv(DATA_PATH)


def test_load_real_csv():
    """Real CSV loads without crashing."""
    raw = _load_real()
    assert not raw.prices.empty, "No price data loaded"
    assert len(raw.prices.columns) > 100, f"Only {len(raw.prices.columns)} tickers loaded"


def test_pit_loader_on_real_data():
    """PIT snapshot of real data at a recent date returns data."""
    from skuld_research.data.pit_loader import PITLoader

    raw = _load_real()
    loader = PITLoader(raw)
    snap = loader.as_of(pd.Timestamp("2025-06-01", tz="UTC"))
    assert not snap.prices.empty
    max_date = snap.prices.index.max()
    assert max_date < pd.Timestamp("2025-06-01"), f"PIT violation: {max_date}"


def test_negative_price_detection_real():
    """Run negative-price validation on real data and report."""
    from skuld_common.validation import detect_negative_prices

    raw = _load_real()
    report = detect_negative_prices(raw.prices)
    if not report.is_clean:
        print(f"\nWARNING: {report.issue_count} negative price observations found:")
        for ticker, dates in report.details.items():
            print(f"  {ticker}: {dates[:5]}{'...' if len(dates) > 5 else ''}")


def test_aia_2017_no_unhandled_corporate_action():
    """Kept as a named test for traceability; the same window is also covered
    by `test_known_corporate_actions_handled_by_adj_close[AIA 2017 ...]`.
    """
    from skuld_research.data import PITLoader, build_prepared_panel

    raw = _load_real()
    snap = PITLoader(raw).as_of(pd.Timestamp("2018-06-01", tz="UTC"))
    panel = build_prepared_panel(snap)
    if "AIA.NZ" not in panel.returns_daily.columns:
        pytest.skip("AIA.NZ not in real data")
    aia_2017 = panel.returns_daily["AIA.NZ"].loc["2017-01-01":"2017-12-31"].dropna()
    assert (aia_2017.abs() <= 0.30).all()


# (ticker, event_label, window_start, window_end, snap_asof)
# Each tuple is a known NZX corporate-action window where Yahoo's `adj_close`
# is expected to fully back-adjust the event. A daily return outside ±30%
# inside the window is the canary for an unhandled split / capital return /
# rights issue.
_KNOWN_CORP_ACTION_EVENTS = [
    # AIA: ~$0.50/share capital return, ex-date late Nov 2017.
    ("AIA.NZ", "AIA 2017 capital return", "2017-01-01", "2017-12-31", "2018-06-01"),
    # Spark: $0.25/share special dividend (capital return) Oct/Nov 2014,
    # part of the post-Telecom-demerger capital management programme.
    ("SPK.NZ", "SPK 2014 special dividend", "2014-01-01", "2014-12-31", "2015-06-01"),
    # Fletcher Building: 1-for-4.46 renounceable rights issue at $4.80,
    # ex-rights date late April / early May 2018.
    ("FBU.NZ", "FBU 2018 rights issue", "2018-03-01", "2018-06-30", "2018-12-01"),
    # Ryman Healthcare: 1-for-2.81 renounceable rights issue at $5.00,
    # announced Feb 2023, retail leg through April 2023.
    ("RYM.NZ", "RYM 2023 rights issue", "2023-01-01", "2023-06-30", "2023-12-01"),
]


@pytest.mark.parametrize(
    ("ticker", "label", "window_start", "window_end", "snap_asof"),
    _KNOWN_CORP_ACTION_EVENTS,
    ids=[ev[1] for ev in _KNOWN_CORP_ACTION_EVENTS],
)
def test_known_corporate_actions_handled_by_adj_close(
    ticker: str,
    label: str,
    window_start: str,
    window_end: str,
    snap_asof: str,
):
    """For each known event, the resulting daily return series must stay
    inside ±30% across the event window. A breach indicates Yahoo's
    `adj_close` did not back-adjust the event and we'd need explicit
    handling from `corporate_actions`.
    """
    from skuld_research.data import PITLoader, build_prepared_panel

    raw = _load_real()
    snap = PITLoader(raw).as_of(pd.Timestamp(snap_asof, tz="UTC"))
    panel = build_prepared_panel(snap)
    if ticker not in panel.returns_daily.columns:
        pytest.skip(f"{ticker} not in real data")
    series = panel.returns_daily[ticker].loc[window_start:window_end].dropna()
    if series.empty:
        pytest.skip(f"{ticker} has no observations in {window_start}..{window_end}")
    max_abs = float(series.abs().max())
    assert max_abs <= 0.30, (
        f"{label}: suspicious daily return in {ticker} "
        f"({window_start}..{window_end}); max |return| = {max_abs:.4f}. "
        f"Likely unadjusted corporate action."
    )


def test_aia_2017_total_return_matches_reference():
    """Pipeline total return for AIA 2017 is within 10 bp of a hard-coded reference.

    Reference derivation:
      adj_close 2017-01-04 = 5.646134  (first trading day)
      adj_close 2017-12-29 = 5.831259  (last trading day)
      total return = 5.831259 / 5.646134 - 1 = 0.032788  (3.2788%)

    AIA paid two dividends in 2017: NZD 0.100 (ex 2017-03-20) and
    NZD 0.105 (ex 2017-10-05), totalling NZD 0.205/share.

    Independent cross-check — October ex-date (div = NZD 0.105):
      observed adj_return on 2017-10-05 = -0.9479%
      naive return if NOT adjusted    = -0.105 / 5.696277 = -1.8433%
      difference = 0.8954 pp confirms the dividend IS embedded in adj_close.
    If adj_close were raw (unadjusted for dividends) the gap would be ≈0.
    """
    from skuld_research.data import PITLoader

    raw = _load_real()
    snap = PITLoader(raw).as_of(pd.Timestamp("2018-06-01", tz="UTC"))

    if "AIA.NZ" not in snap.prices.columns:
        pytest.skip("AIA.NZ not in real data")

    aia = snap.prices["AIA.NZ"].dropna().sort_index()

    # ── regression guard: total return within 10 bp of reference ──────────
    aia_2017 = aia.loc["2017-01-01":"2017-12-31"]
    total_return = (1 + aia_2017.pct_change(fill_method=None).dropna()).prod() - 1
    REFERENCE = 0.032788  # 5.831259 / 5.646134 - 1
    assert abs(total_return - REFERENCE) < 0.001, (
        f"AIA 2017 total return {total_return:.6f} deviates from reference "
        f"{REFERENCE:.6f} by {abs(total_return - REFERENCE) * 100:.4f} pp "
        f"(tolerance 10 bp)"
    )

    # ── cross-check: October 2017 dividend (NZD 0.105) is embedded ────────
    # If the October dividend were NOT in adj_close, the ex-date return would
    # be ≈ -0.105 / prev_close ≈ -1.84%.  We observe ≈ -0.95%, a gap of
    # ~0.90 pp, which can only arise if back-adjustment absorbed the dividend.
    ex_date_idx = aia.index[aia.index >= pd.Timestamp("2017-10-05")][0]
    prev_idx = aia.index[aia.index < ex_date_idx][-1]
    adj_prev = float(aia[prev_idx])
    oct_return = float(aia[ex_date_idx]) / adj_prev - 1
    naive_if_unadjusted = -0.105 / adj_prev  # ≈ -1.84%
    assert abs(oct_return - naive_if_unadjusted) > 0.005, (
        f"October ex-date adj_return ({oct_return:.4%}) is consistent with an "
        f"unadjusted series (naive = {naive_if_unadjusted:.4%}); "
        f"dividend may not be embedded in adj_close"
    )


def test_prepared_panel_on_real_data():
    """Builder runs end-to-end on real data and produces a sane universe."""
    from skuld_research.data import PITLoader, build_prepared_panel

    raw = _load_real()
    snap = PITLoader(raw).as_of(pd.Timestamp("2025-06-01", tz="UTC"))
    panel = build_prepared_panel(snap)
    assert not panel.returns_daily.empty
    recent = panel.universe_mask.loc["2024-01-01":"2024-12-31"]
    assert (recent.sum(axis=1) >= 30).any(), (
        f"Universe too narrow on every 2024 rebalance date: max={recent.sum(axis=1).max()}"
    )


@pytest.mark.slow
def test_momentum_walk_forward_on_real_data():
    """Full walk-forward of momentum-only strategy on real NZX data.

    This is the Milestone 4 completion test.
    Done when: WalkForwardResult contains returns, costs, turnover,
    observed + augmented drawdown, and raw + delisting-adjusted Sharpe.
    """
    from pathlib import Path
    from skuld_research.backtest.walk_forward import FoldSpec, WalkForwardEngine
    from skuld_research.backtest.engine import BacktestConfig
    from skuld_research.data import PITLoader, build_prepared_panel
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_common.contracts import WalkForwardResult

    raw = _load_real()
    snap = PITLoader(raw).as_of(pd.Timestamp("2026-01-01", tz="UTC"))
    panel = build_prepared_panel(snap)

    rebalance_dates = panel.universe_mask.index
    assert len(rebalance_dates) >= 6, (
        f"Too few rebalance dates: {len(rebalance_dates)}"
    )

    n = len(rebalance_dates)
    mid = n // 2

    folds = [
        FoldSpec(0, rebalance_dates[1], rebalance_dates[mid]),
        FoldSpec(1, rebalance_dates[mid + 1], rebalance_dates[-1]),
    ]

    csv_path = Path(__file__).parent.parent / "src" / "survivorship" / "nzx_delistings.csv"
    # M5 introduced degenerate-fold rejection; for this integration smoke test
    # we relax the threshold so that historically-sparse early folds still
    # contribute to the OOS aggregate (the canonical M6 report applies the
    # production default of 0.5).
    smoke_config = BacktestConfig(degenerate_fold_max_empty_frac=1.0)
    wf = WalkForwardEngine(
        factors=[MomentumFactor()],
        panel=panel,
        folds=folds,
        delisting_csv_path=csv_path if csv_path.exists() else None,
        backtest_config=smoke_config,
        monte_carlo_seeds=200,  # fewer for speed
    )
    result = wf.run()

    # --- Milestone 4 done criteria ---
    assert isinstance(result, WalkForwardResult)
    assert len(result.oos_returns) > 0
    assert result.oos_max_drawdown_observed <= 0
    assert result.oos_max_drawdown_augmented_median <= result.oos_max_drawdown_observed + 1e-9
    assert result.oos_sharpe_flat_haircut <= result.oos_sharpe_raw + 1e-9
    assert result.oos_sharpe_delisting_adjusted <= result.oos_sharpe_flat_haircut + 1e-9

    # Total costs must be non-negative
    total_cost = sum(float(f.result.costs_nzd.sum()) for f in result.folds)
    assert total_cost >= 0

    # Turnover must be non-negative
    for fold in result.folds:
        assert (fold.result.turnover >= 0).all()

    print("\n=== Milestone 4: Momentum Walk-Forward (Real NZX Data) ===")
    print(f"OOS periods (months):                      {len(result.oos_returns)}")
    print(f"Avg monthly net return:                    {result.oos_returns.mean():.4%}")
    print(f"Annualised return:                         {result.oos_returns.mean() * 12:.2%}")
    print(f"Sharpe (raw):                              {result.oos_sharpe_raw:.3f}")
    print(f"Sharpe (400 bps flat haircut):             {result.oos_sharpe_flat_haircut:.3f}")
    print(f"Sharpe (delisting-adjusted):               {result.oos_sharpe_delisting_adjusted:.3f}")
    print(f"Max drawdown (observed):                   {result.oos_max_drawdown_observed:.2%}")
    print(f"Max drawdown (augmented median MC):        {result.oos_max_drawdown_augmented_median:.2%}")
    print(f"Max drawdown (augmented 90th pct MC):      {result.oos_max_drawdown_augmented_p90:.2%}")
    print(f"Avg monthly one-sided turnover:            {result.oos_avg_turnover:.2%}")
    print(f"Total cost NZD (at $10K initial NAV):      ${total_cost:,.2f}")
    for fold in result.folds:
        r = fold.result
        print(f"\n  Fold {fold.fold_id}: {fold.test_start.date()} → {fold.test_end.date()}")
        print(f"    Periods: {r.n_periods}, Avg positions: {r.avg_positions:.1f}")
        print(f"    Sharpe (raw): {r.sharpe_raw:.3f}, (haircut): {r.sharpe_flat_haircut:.3f}")
        print(f"    Max drawdown: {r.drawdown.min():.2%}")


@pytest.mark.slow
def test_momentum_diagnostics_on_real_data():
    """Full diagnostics pipeline (IC, decay, decomposition) on real NZX data.

    This is the Milestone 4.5 completion test.
    Done when: diagnostics reports contain finite numbers and the markdown
    renders without error.
    """
    import numpy as np
    from skuld_research.data import PITLoader, build_prepared_panel
    from skuld_research.factors.momentum import MomentumFactor
    from skuld_research.diagnostics.panels import (
        score_panel,
        quintile_spread_returns,
        market_proxy_returns,
    )
    from skuld_research.diagnostics.ic import ranking_ic
    from skuld_research.diagnostics.decay import alpha_decay
    from skuld_research.diagnostics.decomposition import factor_decomposition

    raw = _load_real()
    snap = PITLoader(raw).as_of(pd.Timestamp("2026-01-01", tz="UTC"))
    panel = build_prepared_panel(snap)

    # Build momentum score panel
    momentum = MomentumFactor()
    scores = score_panel(momentum, panel)

    # IC at horizon=1
    ic = ranking_ic(
        scores,
        panel.returns_monthly,
        horizon_months=1,
        factor_name="momentum",
        min_cross_section=10,
    )

    # Alpha decay
    decay = alpha_decay(
        scores,
        panel.returns_monthly,
        horizons=(1, 2, 3, 6, 12),
        factor_name="momentum",
        min_cross_section=10,
    )

    # Factor decomposition
    momentum_spread = quintile_spread_returns(scores, panel.returns_monthly)
    market_ret = market_proxy_returns(panel)

    decomp = factor_decomposition(
        strategy_returns=momentum_spread,
        market_returns=market_ret,
        factor_returns_dict={"momentum": momentum_spread},
    )

    # --- Milestone 4.5 done criteria ---
    # All stats should be finite
    assert not np.isnan(ic.ic_mean)
    assert not np.isnan(ic.ic_std)
    assert ic.n_obs > 0

    # Decay should have results for all horizons
    for h in decay.horizons:
        assert not np.isnan(decay.ic_by_horizon[h].ic_mean)

    # Decomposition should have finite coefficients
    assert not np.isnan(decomp.residual_alpha_annualised)
    assert not np.isnan(decomp.r_squared)
    assert decomp.n_obs > 0

    print("\n=== Milestone 4.5: Momentum Diagnostics (Real NZX Data) ===")
    print(f"IC (horizon=1):                            {ic.ic_mean:.4f} (t={ic.t_stat_newey_west:.2f})")
    print(f"IC Std:                                    {ic.ic_std:.4f}")
    print(f"IC IR (annualized):                        {ic.ic_ir:.4f}")
    print(f"Observations:                              {ic.n_obs}")
    print(f"\nAlpha Decay:")
    for h in sorted(decay.horizons):
        ic_h = decay.ic_by_horizon[h]
        print(f"  {h:2d} months: IC={ic_h.ic_mean:.4f}, IR={ic_h.ic_ir:.4f}, t={ic_h.t_stat_newey_west:.2f}")
    print(f"Peak horizon:                              {decay.peak_horizon} months")
    print(f"\nFactor Decomposition:")
    for reg in decomp.regressors:
        beta = decomp.coefficients[reg]
        t = decomp.t_stats[reg]
        print(f"  {reg:12s}: β={beta:6.4f}, t={t:6.2f}")
    print(f"Residual alpha (annualized):               {decomp.residual_alpha_annualised:.4f} (t={decomp.residual_alpha_t_stat:.2f})")
    print(f"R²:                                        {decomp.r_squared:.4f}")
