"""Tests for the production recommendation pipeline."""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from skuld_common.contracts import (
    CombinedScores,
    CurrentPortfolio,
    PreparedPanel,
    TargetPortfolio,
    TradeList,
)
from skuld_research.config.spec import BacktestEngineSpec, BacktestSpec, CostSpec, ExecutionPolicySpec


def _empty_trade_list(asof: pd.Timestamp) -> TradeList:
    return TradeList(
        trades=pd.DataFrame(
            columns=[
                "ticker",
                "action",
                "current_shares",
                "target_shares",
                "delta_shares",
                "current_value_nzd",
                "target_value_nzd",
                "delta_value_nzd",
                "est_round_trip_cost_nzd",
                "in_no_trade_region",
                "below_size_floor",
                "deferred_to_next_month",
                "sharesies_fee_band",
            ]
        ),
        total_volume_nzd=0.0,
        total_estimated_cost_nzd=0.0,
        asof=asof,
        config_hash="hash",
    )


def test_recommend_passes_execution_policy_spec_to_plan_trades(monkeypatch, tmp_path: Path):
    """Live recommendations use the same execution-policy spec as research."""
    import skuld_portfolio.pipeline.recommend as recommend_module

    asof = cast(pd.Timestamp, pd.Timestamp("2026-01-30"))
    tickers = ["AIR.NZ", "FBU.NZ"]
    spec = BacktestSpec(
        name="policy_spec",
        asof=datetime.date(2026, 1, 30),
        backtest=BacktestEngineSpec(turnover_budget_frac=0.30),
        execution_policy=ExecutionPolicySpec(
            kind="volume_budget",
            monthly_volume_budget_nzd=1_234.0,
            min_trade_benefit_bps=55.0,
            excess_trade_benefit_bps=222.0,
        ),
    )
    panel = PreparedPanel(
        returns_daily=pd.DataFrame([[0.0, 0.0]], index=[asof], columns=tickers),
        returns_monthly=pd.DataFrame([[0.0, 0.0]], index=[asof], columns=tickers),
        market_cap=pd.DataFrame([[1.0, 1.0]], index=[asof], columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame([[True, True]], index=[asof], columns=tickers),
        macro=pd.DataFrame(index=[asof]),
        asof=asof,
    )

    class FakePITLoader:
        def __init__(self, raw):
            self.raw = raw

        def as_of(self, requested_asof):
            return type(
                "FakeSnapshot",
                (),
                {"prices": pd.DataFrame([[1.0, 1.0]], index=[asof], columns=tickers)},
            )()

    captured = {}

    def fake_plan_trades(**kwargs):
        captured["execution_policy"] = kwargs["execution_policy"]
        trades = pd.DataFrame([
            {
                "ticker": "AIR.NZ",
                "action": "HOLD",
                "current_shares": 0,
                "target_shares": 0,
                "delta_shares": 0,
                "current_value_nzd": 0.0,
                "target_value_nzd": 0.0,
                "delta_value_nzd": 0.0,
                "est_round_trip_cost_nzd": 0.0,
                "in_no_trade_region": True,
                "below_size_floor": False,
                "deferred_to_next_month": False,
                "sharesies_fee_band": "subscription_only",
            }
        ])
        return TradeList(
            trades=trades,
            total_volume_nzd=0.0,
            total_estimated_cost_nzd=0.0,
            asof=asof,
            config_hash="hash",
        )

    monkeypatch.setattr(recommend_module, "load_spec", lambda path: spec)
    monkeypatch.setattr(recommend_module, "spec_hash", lambda loaded_spec: "hash")
    monkeypatch.setattr(recommend_module, "load_raw_csv", lambda path, **kwargs: object())
    monkeypatch.setattr(recommend_module, "PITLoader", FakePITLoader)
    monkeypatch.setattr(recommend_module, "build_prepared_panel", lambda snap, **kwargs: panel)
    monkeypatch.setattr(recommend_module, "build_factors_from_specs", lambda factors: [])
    monkeypatch.setattr(
        recommend_module,
        "combine_signals",
        lambda signals, universe, sector, date: CombinedScores(
            scores=pd.Series([3.0, 1.0], index=tickers),
            component_scores=pd.DataFrame(index=tickers),
            asof=date,
        ),
    )
    monkeypatch.setattr(
        recommend_module,
        "build_target_portfolio",
        lambda *args, **kwargs: TargetPortfolio(
            weights=pd.Series([0.3, 0.3], index=tickers),
            cash_weight=0.4,
            method="test",
            asof=asof,
        ),
    )
    monkeypatch.setattr(recommend_module, "read_cash_yaml", lambda path: 10_000.0)
    monkeypatch.setattr(
        recommend_module,
        "parse_sharesies_csv",
        lambda path, cash_nzd: CurrentPortfolio(
            holdings=pd.Series([0, 0], index=tickers, dtype=int),
            prices=pd.Series([1.0, 1.0], index=tickers, dtype=float),
            cash_nzd=cash_nzd,
        ),
    )
    monkeypatch.setattr(recommend_module, "plan_trades", fake_plan_trades)

    recommend_module.recommend(
        tmp_path / "spec.yaml",
        tmp_path / "holdings.csv",
        tmp_path / "cash.yaml",
        asof,
        tmp_path / "data.csv",
    )

    policy = captured["execution_policy"]
    assert policy.volume_budget_nzd == 1_234.0
    assert policy.turnover_budget_frac == 0.30
    assert policy.min_trade_benefit_bps == 55.0
    assert policy.excess_trade_benefit_bps == 222.0


def test_recommend_threads_data_and_target_realism_settings(monkeypatch, tmp_path: Path):
    """Live recommendations should use the same cleaned data and target controls as research."""
    import skuld_portfolio.pipeline.recommend as recommend_module
    from skuld_research.config.spec import AnomalyFilterSpec, ScrubbingSpec, UniverseSpec

    asof = cast(pd.Timestamp, pd.Timestamp("2026-01-30"))
    tickers = ["AIR.NZ", "FBU.NZ"]
    spec = BacktestSpec(
        name="realism_spec",
        asof=datetime.date(2026, 1, 30),
        universe=UniverseSpec(rebalance_freq="BQE"),
        backtest=BacktestEngineSpec(min_names=10, adv_participation_cap=0.02),
        scrubbing=ScrubbingSpec(kind="round_trip"),
        anomaly_filter=AnomalyFilterSpec(kind="mask_extremes"),
    )
    panel = PreparedPanel(
        returns_daily=pd.DataFrame([[0.0, 0.0]], index=[asof], columns=tickers),
        returns_monthly=pd.DataFrame([[0.0, 0.0]], index=[asof], columns=tickers),
        market_cap=pd.DataFrame([[1.0, 1.0]], index=[asof], columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame([[True, True]], index=[asof], columns=tickers),
        macro=pd.DataFrame(index=[asof]),
        asof=asof,
        prices=pd.DataFrame([[10.0, 20.0]], index=[asof], columns=tickers),
    )
    snap = type(
        "FakeSnapshot",
        (),
        {
            "prices": panel.prices,
            "volumes": pd.DataFrame([[1_000.0, 2_000.0]], index=[asof], columns=tickers),
        },
    )()
    captured: dict[str, object] = {}

    class FakePITLoader:
        def __init__(self, raw):
            self.raw = raw

        def as_of(self, requested_asof):
            return snap

    def fake_load_raw_csv(path, **kwargs):
        captured["scrub"] = kwargs.get("scrub")
        captured["adjustments"] = kwargs.get("adjustments")
        return object()

    def fake_build_prepared_panel(snap_arg, **kwargs):
        captured["panel_kwargs"] = kwargs
        return panel

    def fake_build_target_portfolio(*args, **kwargs):
        captured["target_kwargs"] = kwargs
        return TargetPortfolio(
            weights=pd.Series([0.3, 0.3], index=tickers),
            cash_weight=0.4,
            method="test",
            asof=asof,
        )

    monkeypatch.setattr(recommend_module, "load_spec", lambda path: spec)
    monkeypatch.setattr(recommend_module, "spec_hash", lambda loaded_spec: "hash")
    monkeypatch.setattr(recommend_module, "load_raw_csv", fake_load_raw_csv)
    monkeypatch.setattr(recommend_module, "PITLoader", FakePITLoader)
    monkeypatch.setattr(recommend_module, "build_prepared_panel", fake_build_prepared_panel)
    monkeypatch.setattr(recommend_module, "build_factors_from_specs", lambda factors: [])
    monkeypatch.setattr(
        recommend_module,
        "combine_signals",
        lambda signals, universe, sector, date: CombinedScores(
            scores=pd.Series([3.0, 1.0], index=tickers),
            component_scores=pd.DataFrame(index=tickers),
            asof=date,
        ),
    )
    monkeypatch.setattr(recommend_module, "build_target_portfolio", fake_build_target_portfolio)
    monkeypatch.setattr(recommend_module, "read_cash_yaml", lambda path: 10_000.0)
    monkeypatch.setattr(
        recommend_module,
        "parse_sharesies_csv",
        lambda path, cash_nzd: CurrentPortfolio(
            holdings=pd.Series([0, 0], index=tickers, dtype=int),
            prices=pd.Series([10.0, 20.0], index=tickers, dtype=float),
            cash_nzd=cash_nzd,
        ),
    )
    monkeypatch.setattr(
        recommend_module,
        "plan_trades",
        lambda **kwargs: _empty_trade_list(asof),
    )

    recommend_module.recommend(
        tmp_path / "spec.yaml",
        tmp_path / "holdings.csv",
        tmp_path / "cash.yaml",
        asof,
        tmp_path / "data.csv",
    )

    assert captured["scrub"] == spec.scrubbing
    assert captured["adjustments"] == spec.adjustments
    assert captured["panel_kwargs"]["anomaly_filter"] == spec.anomaly_filter
    assert captured["panel_kwargs"]["rebalance_freq"] == "BQE"
    assert captured["target_kwargs"]["min_names"] == 10
    assert captured["target_kwargs"]["adv_participation_cap"] == 0.02
    assert isinstance(captured["target_kwargs"]["adv"], pd.Series)


def test_recommend_passes_ar_spread_to_trade_planner(monkeypatch, tmp_path: Path):
    """Live recommendations should price planner size floors/costs with AR spreads."""
    import skuld_portfolio.pipeline.recommend as recommend_module

    asof = cast(pd.Timestamp, pd.Timestamp("2026-01-30"))
    tickers = ["AIR.NZ"]
    spec = BacktestSpec(
        name="ar_spread_spec",
        asof=datetime.date(2026, 1, 30),
        cost=CostSpec(spread_model="abdi_ranaldo"),
    )
    panel = PreparedPanel(
        returns_daily=pd.DataFrame([[0.0]], index=[asof], columns=tickers),
        returns_monthly=pd.DataFrame([[0.0]], index=[asof], columns=tickers),
        market_cap=pd.DataFrame([[1.0]], index=[asof], columns=tickers),
        sector=pd.Series("Unknown", index=tickers),
        universe_mask=pd.DataFrame([[True]], index=[asof], columns=tickers),
        macro=pd.DataFrame(index=[asof]),
        asof=asof,
        prices=pd.DataFrame([[10.0]], index=[asof], columns=tickers),
    )
    snap = type(
        "FakeSnapshot",
        (),
        {"prices": panel.prices, "volumes": pd.DataFrame([[1_000.0]], index=[asof], columns=tickers)},
    )()
    captured: dict[str, object] = {}

    class FakePITLoader:
        def __init__(self, raw):
            self.raw = raw

        def as_of(self, requested_asof):
            return snap

    monkeypatch.setattr(recommend_module, "load_spec", lambda path: spec)
    monkeypatch.setattr(recommend_module, "spec_hash", lambda loaded_spec: "hash")
    monkeypatch.setattr(recommend_module, "load_raw_csv", lambda path, **kwargs: object())
    monkeypatch.setattr(recommend_module, "PITLoader", FakePITLoader)
    monkeypatch.setattr(recommend_module, "build_prepared_panel", lambda snap_arg, **kwargs: panel)
    monkeypatch.setattr(recommend_module, "build_factors_from_specs", lambda factors: [])
    monkeypatch.setattr(
        recommend_module,
        "combine_signals",
        lambda signals, universe, sector, date: CombinedScores(
            scores=pd.Series([3.0], index=tickers),
            component_scores=pd.DataFrame(index=tickers),
            asof=date,
        ),
    )
    monkeypatch.setattr(
        recommend_module,
        "build_target_portfolio",
        lambda *args, **kwargs: TargetPortfolio(
            weights=pd.Series([0.3], index=tickers),
            cash_weight=0.7,
            method="test",
            asof=asof,
        ),
    )
    monkeypatch.setattr(recommend_module, "read_cash_yaml", lambda path: 10_000.0)
    monkeypatch.setattr(
        recommend_module,
        "parse_sharesies_csv",
        lambda path, cash_nzd: CurrentPortfolio(
            holdings=pd.Series([0], index=tickers, dtype=int),
            prices=pd.Series([10.0], index=tickers, dtype=float),
            cash_nzd=cash_nzd,
        ),
    )
    monkeypatch.setattr(
        recommend_module,
        "load_raw_ohlc",
        lambda *args, **kwargs: (
            pd.DataFrame([[1.0]], index=[asof], columns=tickers),
            pd.DataFrame([[1.0]], index=[asof], columns=tickers),
            pd.DataFrame([[1.0]], index=[asof], columns=tickers),
        ),
    )
    monkeypatch.setattr(
        recommend_module,
        "compute_abdi_ranaldo_spread_panel",
        lambda *args, **kwargs: pd.DataFrame(
            [[37.0], [99.0]],
            index=[asof - pd.offsets.BDay(3), asof - pd.offsets.BDay(1)],
            columns=tickers,
        ),
    )
    def fake_plan_trades(**kwargs):
        captured["spread"] = kwargs["per_ticker_spread_bps"]
        return _empty_trade_list(asof)

    monkeypatch.setattr(recommend_module, "plan_trades", fake_plan_trades)

    recommend_module.recommend(
        tmp_path / "spec.yaml",
        tmp_path / "holdings.csv",
        tmp_path / "cash.yaml",
        asof,
        tmp_path / "data.csv",
    )

    assert isinstance(captured["spread"], pd.Series)
    assert captured["spread"]["AIR.NZ"] == 37.0
