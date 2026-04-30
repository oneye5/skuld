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
from skuld_research.config.spec import BacktestSpec, ExecutionPolicySpec


def test_recommend_passes_execution_policy_spec_to_plan_trades(monkeypatch, tmp_path: Path):
    """Live recommendations use the same execution-policy spec as research."""
    import skuld_portfolio.pipeline.recommend as recommend_module

    asof = cast(pd.Timestamp, pd.Timestamp("2026-01-30"))
    tickers = ["AIR.NZ", "FBU.NZ"]
    spec = BacktestSpec(
        name="policy_spec",
        asof=datetime.date(2026, 1, 30),
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
    monkeypatch.setattr(recommend_module, "load_raw_csv", lambda path: object())
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
    assert policy.min_trade_benefit_bps == 55.0
    assert policy.excess_trade_benefit_bps == 222.0
