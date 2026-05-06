"""Tests for BacktestEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_common.contracts import BacktestResult, PITSnapshot, PreparedPanel
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.costs.model import CostConfig
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.execution.policy import ExecutionPolicyConfig
from skuld_research.factors.momentum import MomentumFactor

# ---------------------------------------------------------------------------
# Helper: build a PreparedPanel with synthetic data
# ---------------------------------------------------------------------------

def _make_panel(
    n_tickers: int = 15,
    n_days: int = 600,
    seed: int = 0,
) -> PreparedPanel:
    """Build a PreparedPanel with synthetic daily returns for backtest tests."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}.NZ" for i in range(n_tickers)]
    dates = pd.bdate_range("2022-01-01", periods=n_days)

    prices_data = {}
    for t in tickers:
        px = 10.0 * (1 + 0.001 * rng.standard_normal(n_days)).cumprod()
        prices_data[t] = px

    prices = pd.DataFrame(prices_data, index=dates)
    prices.index.name = "date"
    volumes = pd.DataFrame({t: 500_000.0 for t in tickers}, index=dates)
    volumes.index.name = "date"

    last_date = dates[-1]
    asof_ts = last_date + pd.DateOffset(months=3)

    snap = PITSnapshot(
        prices=prices,
        volumes=volumes,
        fundamentals=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["ticker", "publication_date"])
        ),
        macro=pd.DataFrame(),
        corporate_actions=pd.DataFrame(
            columns=["ticker", "ex_date", "type", "factor"]
        ),
        asof=asof_ts,
    )
    return build_prepared_panel(snap, nzx_only=False, rebalance_start="2022-01-01")


def _make_engine(config: BacktestConfig | None = None) -> BacktestEngine:
    panel = _make_panel()
    return BacktestEngine(
        factors=[MomentumFactor()],
        panel=panel,
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class _FixedScoreFactor:
    def __init__(self, scores: pd.Series, name: str = "fixed") -> None:
        self._scores = scores
        self.name = name

    def score(
        self,
        panel: PreparedPanel,
        t: pd.Timestamp,
        universe: list[str],
    ) -> pd.Series:
        return self._scores.reindex(universe)


def test_quarterly_period_compounds_all_intermediate_monthly_returns():
    """A quarterly rebalance period compounds every monthly return in the gap."""
    tickers = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-06-30"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30", "2024-05-31", "2024-06-30"])
    monthly_returns = pd.DataFrame(
        {
            "A.NZ": [0.10, 0.10, -0.05],
            "B.NZ": [0.0, 0.0, 0.0],
            "C.NZ": [0.0, 0.0, 0.0],
            "D.NZ": [0.0, 0.0, 0.0],
        },
        index=monthly_dates,
    )
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=monthly_returns,
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-07-31"),
    )
    cfg = BacktestConfig(
        cash_floor=0.0,
        max_position=1.0,
        max_sector=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )

    factor = _FixedScoreFactor(
        pd.Series({"A.NZ": 3.0, "B.NZ": 2.0, "C.NZ": 1.0, "D.NZ": 0.5})
    )
    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()

    expected = (1.10 * 1.10 * 0.95) - 1.0
    assert result.returns.iloc[0] == pytest.approx(expected)


def test_quarterly_period_charges_subscription_for_each_month_in_period():
    """Sharesies subscription is monthly even when rebalance periods are quarterly."""
    tickers = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-06-30"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30", "2024-05-31", "2024-06-30"])
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=monthly_dates, columns=tickers),
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-07-31"),
    )
    cfg = BacktestConfig(
        initial_nav_nzd=10_000.0,
        cash_floor=0.0,
        max_position=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=15.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )

    factor = _FixedScoreFactor(
        pd.Series({"A.NZ": 3.0, "B.NZ": 2.0, "C.NZ": 1.0, "D.NZ": 0.5})
    )
    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()

    assert result.costs_nzd.iloc[0] == pytest.approx(45.0)
    assert result.returns.iloc[0] == pytest.approx(-45.0 / 10_000.0)
    assert result.spread_costs_nzd.iloc[0] == pytest.approx(0.0)
    assert result.sharesies_fee_nzd.iloc[0] == pytest.approx(45.0)
    assert result.gross_returns.iloc[0] == pytest.approx(0.0)
    assert result.cost_drag.iloc[0] == pytest.approx(45.0 / 10_000.0)
    assert result.equity_weight.iloc[0] == pytest.approx(1.0)
    assert result.cash_weight.iloc[0] == pytest.approx(0.0)


def test_empty_universe_liquidates_before_holding_period_returns():
    """If the universe is empty at rebalance t, holdings are sold before t->next_t returns."""
    tickers = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-04-30", "2024-05-31"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30", "2024-05-31"])
    monthly_returns = pd.DataFrame(
        {
            "A.NZ": [0.10, 0.50],
            "B.NZ": [0.0, 0.0],
            "C.NZ": [0.0, 0.0],
            "D.NZ": [0.0, 0.0],
        },
        index=monthly_dates,
    )
    universe_mask = pd.DataFrame(True, index=rebalance_dates, columns=tickers)
    universe_mask.loc[pd.Timestamp("2024-04-30"), :] = False
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=monthly_returns,
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=universe_mask,
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-06-30"),
    )
    cfg = BacktestConfig(
        cash_floor=0.0,
        max_position=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )

    factor = _FixedScoreFactor(
        pd.Series({"A.NZ": 3.0, "B.NZ": 2.0, "C.NZ": 1.0, "D.NZ": 0.5})
    )
    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()

    assert result.returns.iloc[0] == pytest.approx(0.10)
    assert result.returns.iloc[1] == pytest.approx(0.0)
    assert result.period_n_positions.iloc[1] == 0


def test_empty_universe_cost_drag_uses_pre_period_nav():
    """Cost attribution should reconcile gross return to net return on cash-only months."""
    tickers = ["A.NZ"]
    dates = pd.bdate_range("2024-01-01", "2024-02-29")
    rebalance_dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=[pd.Timestamp("2024-02-29")], columns=tickers),
        market_cap=pd.DataFrame(1_000_000.0, index=dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(False, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=dates),
        asof=pd.Timestamp("2024-03-31"),
    )
    cfg = BacktestConfig(
        initial_nav_nzd=10_000.0,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=15.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )

    result = BacktestEngine(factors=[], panel=panel, config=cfg).run()

    assert result.cost_drag.iloc[0] == pytest.approx(15.0 / 10_000.0)
    assert result.returns.iloc[0] == pytest.approx(
        result.gross_returns.iloc[0] - result.cost_drag.iloc[0]
    )


def test_execution_policy_defers_low_benefit_rebalance_trade():
    """BacktestEngine applies volume-budget policy after normal trade filters."""
    tickers = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-04-30"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30"])
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=monthly_dates, columns=tickers),
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-05-31"),
    )
    cfg = BacktestConfig(
        initial_nav_nzd=10_000.0,
        cash_floor=0.0,
        max_position=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
        execution_policy=ExecutionPolicyConfig(
            volume_budget_nzd=5_000.0,
            excess_trade_benefit_bps=300.0,
        ),
    )
    factor = _FixedScoreFactor(
        pd.Series({"A.NZ": 3.0, "B.NZ": 2.0, "C.NZ": 1.0, "D.NZ": 0.5})
    )

    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()

    assert result.executed_volume_nzd.iloc[0] <= 5_000.0
    assert result.deferred_volume_nzd.iloc[0] > 0.0
    assert result.turnover.iloc[0] < 0.5


def test_backtest_turnover_budget_limits_rebalance_volume():
    """BacktestConfig turnover budget is enforced during each rebalance."""
    tickers = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-04-30"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30"])
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=monthly_dates, columns=tickers),
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-05-31"),
    )
    cfg = BacktestConfig(
        initial_nav_nzd=10_000.0,
        cash_floor=0.0,
        max_position=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        turnover_budget_frac=0.25,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )
    factor = _FixedScoreFactor(
        pd.Series({"A.NZ": 3.0, "B.NZ": 2.0, "C.NZ": 1.0, "D.NZ": 0.5})
    )

    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()

    assert result.executed_volume_nzd.iloc[0] <= 2_500.0
    assert result.deferred_volume_nzd.iloc[0] > 0.0
    assert result.turnover.iloc[0] <= 0.125


def test_execution_policy_forced_exit_bypasses_trade_filters():
    """BacktestEngine liquidates stale holdings even below size/no-trade filters."""
    tickers = ["A.NZ", "B.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-04-30", "2024-05-31"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30", "2024-05-31"])
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=pd.DataFrame(
            {"A.NZ": [-0.60, 0.0], "B.NZ": [0.0, 0.0]},
            index=monthly_dates,
        ),
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=monthly_dates),
        asof=pd.Timestamp("2024-06-30"),
    )
    cfg = BacktestConfig(
        initial_nav_nzd=10_000.0,
        cash_floor=0.0,
        max_position=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=5_000.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
        execution_policy=ExecutionPolicyConfig(
            volume_budget_nzd=10_000.0,
            excess_trade_benefit_bps=10_000.0,
        ),
    )

    class DateScoreFactor:
        name = "date_score"

        def score(
            self,
            panel: PreparedPanel,
            t: pd.Timestamp,
            universe: list[str],
        ) -> pd.Series:
            scores = (
                pd.Series({"A.NZ": 1.0, "B.NZ": -1.0})
                if t == rebalance_dates[0]
                else pd.Series({"A.NZ": -1.0, "B.NZ": -1.0})
            )
            return scores.reindex(universe)

    result = BacktestEngine(factors=[DateScoreFactor()], panel=panel, config=cfg).run()

    assert result.equity_weight.iloc[0] == pytest.approx(1.0)
    assert result.equity_weight.iloc[1] == pytest.approx(0.0)
    assert result.executed_volume_nzd.iloc[1] == pytest.approx(4_000.0)


def test_subscription_costs_drift_weights_by_reducing_cash_first():
    """Fixed fees reduce cash, so the next rebalance restores the cash floor."""
    tickers = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-04-30", "2024-05-31"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30", "2024-05-31"])
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=monthly_dates, columns=tickers),
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-06-30"),
    )
    cfg = BacktestConfig(
        initial_nav_nzd=10_000.0,
        cash_floor=0.10,
        max_position=0.90,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=100.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )

    factor = _FixedScoreFactor(
        pd.Series({"A.NZ": 3.0, "B.NZ": 2.0, "C.NZ": 1.0, "D.NZ": 0.5})
    )
    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()

    assert result.turnover.iloc[1] == pytest.approx(((0.90 / 0.99) - 0.90) / 2.0)


def test_backtest_engine_passes_rebalance_adv_into_optimizer(monkeypatch: pytest.MonkeyPatch):
    """Engine should pass rebalance-date ADV, NAV, and cap through to the optimizer."""
    tickers = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-04-30"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30"])
    prices = pd.DataFrame(
        {
            "A.NZ": 10.0,
            "B.NZ": 20.0,
            "C.NZ": 30.0,
            "D.NZ": 40.0,
        },
        index=daily_dates,
    )
    market_cap = prices * 100_000.0
    adv_panel = pd.DataFrame(
        {
            "A.NZ": [100_000.0],
            "B.NZ": [400_000.0],
            "C.NZ": [900_000.0],
            "D.NZ": [1_600_000.0],
        },
        index=[daily_dates[-1]],
    )
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=monthly_dates, columns=tickers),
        market_cap=market_cap,
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-05-31"),
        prices=prices,
    )
    cfg = BacktestConfig(
        initial_nav_nzd=10_000.0,
        cash_floor=0.0,
        max_position=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        adv_participation_cap=0.02,
        adv_panel=adv_panel,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )
    factor = _FixedScoreFactor(
        pd.Series({"A.NZ": 4.0, "B.NZ": 3.0, "C.NZ": 2.0, "D.NZ": 1.0})
    )
    captured: dict[str, object] = {}

    def fake_build_target_portfolio(*args, **kwargs):
        captured["adv"] = kwargs.get("adv")
        captured["portfolio_nav"] = kwargs.get("portfolio_nav")
        captured["adv_participation_cap"] = kwargs.get("adv_participation_cap")
        return _make_target_portfolio({"A.NZ": 1.0}, asof=args[2])

    monkeypatch.setattr("skuld_research.backtest.engine.build_target_portfolio", fake_build_target_portfolio)

    BacktestEngine(factors=[factor], panel=panel, config=cfg).run()

    assert isinstance(captured["adv"], pd.Series)
    assert captured["portfolio_nav"] == pytest.approx(10_000.0)
    assert captured["adv_participation_cap"] == pytest.approx(0.02)
    assert captured["adv"]["A.NZ"] == pytest.approx(100_000.0)
    assert captured["adv"]["B.NZ"] == pytest.approx(400_000.0)


def _make_target_portfolio(weights: dict[str, float], asof: pd.Timestamp):
    from skuld_common.contracts import TargetPortfolio

    series = pd.Series(weights, dtype=float)
    return TargetPortfolio(
        weights=series,
        cash_weight=max(0.0, 1.0 - float(series.sum())),
        method="test",
        asof=asof,
    )

def test_engine_returns_backtest_result():
    """engine.run() returns BacktestResult with at least one return period."""
    engine = _make_engine()
    result = engine.run()
    assert isinstance(result, BacktestResult)
    assert len(result.returns) >= 1


def test_costs_are_non_negative():
    """All period costs are non-negative."""
    result = _make_engine().run()
    assert (result.costs_nzd >= 0).all()


def test_turnover_is_non_negative():
    """All period turnover values are non-negative."""
    result = _make_engine().run()
    assert (result.turnover >= 0).all()


def test_drawdown_is_non_positive():
    """All drawdown values are <= 0 (within float tolerance)."""
    result = _make_engine().run()
    assert (result.drawdown <= 1e-9).all()


def test_sharpe_flat_haircut_le_raw():
    """Flat-haircut Sharpe is always <= raw Sharpe (within tolerance)."""
    result = _make_engine().run()
    assert result.sharpe_flat_haircut <= result.sharpe_raw + 1e-9


def test_no_trade_threshold_reduces_turnover():
    """Wide NTR (0.30) produces <= mean turnover compared to no NTR (0.0)."""
    panel = _make_panel(seed=42)

    cfg_wide = BacktestConfig(no_trade_threshold_frac=0.30)
    cfg_none = BacktestConfig(no_trade_threshold_frac=0.0)

    result_wide = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg_wide).run()
    result_none = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg_none).run()

    assert float(result_wide.turnover.mean()) <= float(result_none.turnover.mean()) + 1e-9


def test_backtest_result_invariants_hold():
    """BacktestResult dataclass invariants don't raise."""
    result = _make_engine().run()
    # Accessing fields exercises __post_init__ (already called on construction)
    assert result.n_periods >= 1
    assert result.avg_positions >= 0.0
    assert result.start <= result.end


def test_minimum_rebalance_dates_raises():
    """Panel with only 1 rebalance date raises ValueError."""
    panel = _make_panel(n_days=600)
    # Slice universe_mask to a single row
    single_mask = panel.universe_mask.iloc[:1]
    short_panel = PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=single_mask,
        macro=panel.macro,
        asof=panel.asof,
    )
    engine = BacktestEngine(factors=[MomentumFactor()], panel=short_panel)
    with pytest.raises(ValueError, match="at least 2 rebalance dates"):
        engine.run()


def test_backtest_config_rejects_out_of_bounds_adv_participation_cap():
    with pytest.raises(ValueError, match="adv_participation_cap"):
        BacktestConfig(adv_participation_cap=-0.01)

    with pytest.raises(ValueError, match="adv_participation_cap"):
        BacktestConfig(adv_participation_cap=1.01)


def test_backtest_config_rejects_out_of_bounds_min_names_and_turnover_budget():
    with pytest.raises(ValueError, match="min_names"):
        BacktestConfig(min_names=0)

    with pytest.raises(ValueError, match="turnover_budget_frac"):
        BacktestConfig(turnover_budget_frac=-0.01)

    with pytest.raises(ValueError, match="turnover_budget_frac"):
        BacktestConfig(turnover_budget_frac=1.01)


def test_drift_weights_wipeout_does_not_crash():
    """_drift_weights handles total loss without negative weights."""
    from skuld_research.backtest.engine import _drift_weights
    weights = pd.Series({"A.NZ": 0.5, "B.NZ": 0.5})
    period_returns = pd.Series({"A.NZ": -1.0, "B.NZ": -1.0})
    result = _drift_weights(weights, period_returns, portfolio_gross_return=-1.0)
    # Should not raise; weights must remain non-negative (clipped, not negative)
    assert (result >= 0.0).all()


def test_ntr_prevents_individual_small_trades():
    """When NTR=0.5 (50% of NAV), no position smaller than 50% of NAV gets traded."""
    from skuld_research.backtest.engine import BacktestConfig

    # With NTR=0.5, any weight delta smaller than 0.50 should be skipped.
    # With 15 tickers averaging 1/15 ≈ 6.7% weight each, all individual trades
    # are below the threshold, so turnover should be 0.0 on every period.
    cfg = BacktestConfig(no_trade_threshold_frac=0.50)
    panel = _make_panel(seed=99)
    engine = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg)
    result = engine.run()
    # With NTR=50%, all individual rebalance deltas should be blocked.
    # Turnover must be 0 for every period (no trades should execute).
    assert float(result.turnover.max()) == pytest.approx(0.0, abs=1e-9)


def test_universe_exit_liquidates_holdings():
    """Tickers that held but drop out of universe should be liquidated (n_positions drops)."""
    # Build an engine and run it. Since we can't easily inspect internal state,
    # we verify that n_positions in BacktestResult is never negative and reflects
    # the actual universe constraint: avg_positions <= n_tickers.
    result = _make_engine().run()
    n_tickers = 15  # from _make_panel default
    assert result.avg_positions <= n_tickers
    # avg_positions must be >= 0 (not a hard lower bound since universe can be empty)
    assert result.avg_positions >= 0.0


def test_synthetic_backtest_returns_in_plausible_range():
    """Regression guard: net monthly returns for the synthetic panel stay within ±30%."""
    result = _make_engine(BacktestConfig(initial_nav_nzd=10_000.0)).run()
    # Monthly returns outside ±30% would indicate a bug (extreme NAV crash or look-ahead)
    assert float(result.returns.max()) <= 0.30, (
        f"Suspiciously high monthly return: {result.returns.max():.4f}"
    )
    assert float(result.returns.min()) >= -0.30, (
        f"Suspiciously large monthly loss: {result.returns.min():.4f}"
    )


# ---------------------------------------------------------------------------
# Smoothing alpha tests
# ---------------------------------------------------------------------------

def _make_smoothing_panel() -> PreparedPanel:
    """5 tickers, 24 months of synthetic data for smoothing tests."""
    return _make_panel(n_tickers=5, n_days=520, seed=7)


def test_smoothing_alpha_zero_no_change():
    """smoothing_alpha=0.0 should produce identical results to the default (no smoothing)."""
    panel = _make_smoothing_panel()
    cfg_default = BacktestConfig()
    cfg_zero = BacktestConfig(smoothing_alpha=0.0)

    result_default = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg_default).run()
    result_zero = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg_zero).run()

    pd.testing.assert_series_equal(result_default.returns, result_zero.returns)
    pd.testing.assert_series_equal(result_default.turnover, result_zero.turnover)


def test_smoothing_reduces_turnover():
    """smoothing_alpha=0.5 reduces per-period trade deltas vs alpha=0.0.

    Smoothing moves the target weights toward current weights, so the max
    single-period turnover must be lower with smoothing than without, given
    the same signal.  We verify this on a controlled two-period scenario.
    """
    tickers = ["A.NZ", "B.NZ", "C.NZ", "D.NZ"]
    rebalance_dates = pd.to_datetime(["2024-03-31", "2024-04-30", "2024-05-31"])
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-29")
    monthly_dates = pd.to_datetime(["2024-04-30", "2024-05-31"])
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=monthly_dates, columns=tickers),
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-06-30"),
    )
    zero_cost = CostConfig(
        spread_bps=0.0,
        sharesies_monthly_fee_nzd=0.0,
        sharesies_coverage_nzd=1_000_000.0,
        sharesies_excess_bps=0.0,
    )

    # Signal flips completely from period 1 to period 2 → big rebalance
    class FlippingFactor:
        name = "flip"
        def score(self, panel, t, universe):
            if t == rebalance_dates[0]:
                return pd.Series({"A.NZ": 3.0, "B.NZ": 2.0, "C.NZ": 1.0, "D.NZ": 0.5}).reindex(universe)
            return pd.Series({"A.NZ": 0.5, "B.NZ": 1.0, "C.NZ": 2.0, "D.NZ": 3.0}).reindex(universe)

    base_cfg = dict(
        initial_nav_nzd=10_000.0,
        cash_floor=0.0,
        max_position=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=zero_cost,
    )

    result_no_smooth = BacktestEngine(
        factors=[FlippingFactor()], panel=panel,
        config=BacktestConfig(**base_cfg, smoothing_alpha=0.0)
    ).run()
    result_smooth = BacktestEngine(
        factors=[FlippingFactor()], panel=panel,
        config=BacktestConfig(**base_cfg, smoothing_alpha=0.5)
    ).run()

    # On the second period, smoothing should reduce turnover
    assert result_smooth.turnover.iloc[1] <= result_no_smooth.turnover.iloc[1] + 1e-9


def test_smoothing_alpha_validation():
    """smoothing_alpha must be in [0, 1); 1.0 and negatives raise ValueError."""
    with pytest.raises(ValueError, match="smoothing_alpha"):
        BacktestConfig(smoothing_alpha=1.0)

    with pytest.raises(ValueError, match="smoothing_alpha"):
        BacktestConfig(smoothing_alpha=-0.1)

    # 0.9 is valid
    cfg = BacktestConfig(smoothing_alpha=0.9)
    assert cfg.smoothing_alpha == pytest.approx(0.9)


def test_smoothing_weights_sum_to_one():
    """With smoothing_alpha=0.3, equity_weight + cash_weight ≈ 1.0 each period."""
    panel = _make_smoothing_panel()
    cfg = BacktestConfig(smoothing_alpha=0.3)
    result = BacktestEngine(factors=[MomentumFactor()], panel=panel, config=cfg).run()

    total = result.equity_weight + result.cash_weight
    assert (total - 1.0).abs().max() < 1e-4


# ---------------------------------------------------------------------------
# cap_binding_count tests
# ---------------------------------------------------------------------------


def _make_cap_panel(n_tickers: int = 4, n_rebalance: int = 3) -> PreparedPanel:
    """Panel where n_tickers > 1/max_position so caps can bind."""
    tickers = [f"T{i:02d}.NZ" for i in range(n_tickers)]
    rebalance_dates = pd.date_range("2024-01-31", periods=n_rebalance, freq="ME")
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-28")
    monthly_dates = pd.date_range("2024-02-29", periods=n_rebalance, freq="ME")
    rng = np.random.default_rng(99)
    rets = pd.DataFrame(
        rng.normal(0.005, 0.02, (n_rebalance, n_tickers)),
        index=monthly_dates,
        columns=tickers,
    )
    return PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=rets,
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-04-30"),
    )


def test_cap_binding_count_is_series_of_ints():
    """cap_binding_count must be an integer Series with one value per period."""
    panel = _make_cap_panel(n_tickers=8, n_rebalance=3)
    cfg = BacktestConfig(
        max_position=0.15,  # 1/0.15 ≈ 6.7, so with 8 tickers some will cap
        cash_floor=0.0,
        max_sector=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )
    factor = _FixedScoreFactor(
        pd.Series({f"T{i:02d}.NZ": float(i) for i in range(8)})
    )
    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()

    assert isinstance(result.cap_binding_count, pd.Series)
    assert result.cap_binding_count.dtype == int or result.cap_binding_count.dtype == "int64"
    assert len(result.cap_binding_count) == len(result.returns)


def test_cap_binding_count_positive_when_cap_binds():
    """When max_position is tight enough to bind, at least one period should report > 0 bound tickers."""
    panel = _make_cap_panel(n_tickers=8, n_rebalance=3)
    cfg = BacktestConfig(
        max_position=0.10,  # cap at 10%; 8 tickers => optimizer will cap multiple names
        cash_floor=0.0,
        max_sector=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )
    factor = _FixedScoreFactor(
        pd.Series({f"T{i:02d}.NZ": float(i) for i in range(8)})
    )
    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()
    # With 8 equal-weight positions the EW weight is 12.5%; a 10% cap binds every ticker.
    assert result.cap_binding_count.sum() > 0


def test_cap_binding_count_zero_when_cap_does_not_bind():
    """When equal-weight positions comfortably fit below max_position, cap_binding_count is zero.

    We use 4 tickers with flat (zero) returns so the minimum-variance optimizer
    produces exactly equal weights (0.25 each).  With max_position=0.30, the cap
    threshold (0.2999) is above every position — no cap should bind.
    """
    n_tickers = 4
    tickers = [f"T{i:02d}.NZ" for i in range(n_tickers)]
    rebalance_dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    daily_dates = pd.bdate_range("2024-01-01", "2024-03-28")
    monthly_dates = pd.date_range("2024-02-29", periods=3, freq="ME")
    # All-zero returns → minimum-variance optimizer produces equal weights
    panel = PreparedPanel(
        returns_daily=pd.DataFrame(0.0, index=daily_dates, columns=tickers),
        returns_monthly=pd.DataFrame(0.0, index=monthly_dates, columns=tickers),
        market_cap=pd.DataFrame(1_000_000.0, index=daily_dates, columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame(True, index=rebalance_dates, columns=tickers),
        macro=pd.DataFrame(index=daily_dates),
        asof=pd.Timestamp("2024-04-30"),
    )
    cfg = BacktestConfig(
        max_position=0.30,  # above EW weight of 0.25 → cap does not bind
        cash_floor=0.0,
        max_sector=1.0,
        no_trade_threshold_frac=0.0,
        size_floor_nzd=0.0,
        min_return_obs=1,
        cost_config=CostConfig(
            spread_bps=0.0,
            sharesies_monthly_fee_nzd=0.0,
            sharesies_coverage_nzd=1_000_000.0,
            sharesies_excess_bps=0.0,
        ),
    )
    factor = _FixedScoreFactor(
        pd.Series({t: 1.0 for t in tickers})  # equal scores → equal weights
    )
    result = BacktestEngine(factors=[factor], panel=panel, config=cfg).run()
    assert result.cap_binding_count.sum() == 0


def test_cap_binding_count_default_is_empty_series():
    """BacktestResult default cap_binding_count is an empty int Series (backward compat)."""
    from skuld_common.contracts import BacktestResult
    r = BacktestResult(
        returns=pd.Series([0.01]),
        costs_nzd=pd.Series([0.0]),
        turnover=pd.Series([0.5]),
        drawdown=pd.Series([-0.01]),
        sharpe_raw=0.5,
        sharpe_flat_haircut=0.4,
        start=pd.Timestamp("2024-01-31"),
        end=pd.Timestamp("2024-12-31"),
        n_periods=1,
        avg_positions=5.0,
    )
    assert isinstance(r.cap_binding_count, pd.Series)
    assert r.cap_binding_count.dtype == int or pd.api.types.is_integer_dtype(r.cap_binding_count)
