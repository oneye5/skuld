"""Tests for cost-aware execution policy."""
from __future__ import annotations

import pandas as pd

from skuld_research.execution.policy import ExecutionPolicyConfig, apply_execution_policy


def test_policy_defers_low_benefit_trade_past_volume_budget():
    """A trade that breaches the covered-volume budget is deferred if benefit is low."""
    delta = pd.Series({"A.NZ": 0.20, "B.NZ": 0.10})
    alpha = pd.Series({"A.NZ": 300.0, "B.NZ": 50.0})

    result = apply_execution_policy(
        delta,
        nav_nzd=10_000.0,
        expected_alpha_bps=alpha,
        config=ExecutionPolicyConfig(
            volume_budget_nzd=2_500.0,
            excess_trade_benefit_bps=190.0,
        ),
    )

    assert result.executable_delta_weights["A.NZ"] == 0.20
    assert result.executable_delta_weights["B.NZ"] == 0.0
    assert result.deferred["B.NZ"] is True
    assert result.executed_volume_nzd == 2_000.0
    assert result.deferred_volume_nzd == 1_000.0


def test_policy_allows_high_benefit_trade_past_volume_budget():
    """A high-benefit trade may exceed the fee cliff when expected edge clears cost."""
    delta = pd.Series({"A.NZ": 0.20, "B.NZ": 0.10})
    alpha = pd.Series({"A.NZ": 300.0, "B.NZ": 250.0})

    result = apply_execution_policy(
        delta,
        nav_nzd=10_000.0,
        expected_alpha_bps=alpha,
        config=ExecutionPolicyConfig(
            volume_budget_nzd=2_500.0,
            excess_trade_benefit_bps=190.0,
        ),
    )

    assert result.executable_delta_weights["A.NZ"] == 0.20
    assert result.executable_delta_weights["B.NZ"] == 0.10
    assert result.deferred["B.NZ"] is False
    assert result.executed_volume_nzd == 3_000.0
    assert result.excess_volume_nzd == 500.0


def test_policy_uses_directional_benefit_for_sells():
    """Selling high-alpha names is low benefit; selling low-alpha names is high benefit."""
    delta = pd.Series({"HIGH.NZ": -0.10, "LOW.NZ": -0.10})
    alpha = pd.Series({"HIGH.NZ": 300.0, "LOW.NZ": -300.0})

    result = apply_execution_policy(
        delta,
        nav_nzd=10_000.0,
        expected_alpha_bps=alpha,
        config=ExecutionPolicyConfig(
            volume_budget_nzd=1_000.0,
            min_trade_benefit_bps=50.0,
            excess_trade_benefit_bps=190.0,
        ),
    )

    assert result.executable_delta_weights["LOW.NZ"] == -0.10
    assert result.executable_delta_weights["HIGH.NZ"] == 0.0
    assert result.deferred["HIGH.NZ"] is True


def test_policy_minimum_benefit_adds_hysteresis_even_under_budget():
    """Low-benefit churn can be deferred even when there is unused fee budget."""
    delta = pd.Series({"A.NZ": 0.05, "B.NZ": -0.05})
    alpha = pd.Series({"A.NZ": 20.0, "B.NZ": -150.0})

    result = apply_execution_policy(
        delta,
        nav_nzd=10_000.0,
        expected_alpha_bps=alpha,
        config=ExecutionPolicyConfig(
            volume_budget_nzd=5_000.0,
            min_trade_benefit_bps=50.0,
        ),
    )

    assert result.executable_delta_weights["A.NZ"] == 0.0
    assert result.executable_delta_weights["B.NZ"] == -0.05
    assert result.deferred["A.NZ"] is True
    assert result.deferred["B.NZ"] is False


def test_policy_forced_trade_executes_even_without_budget():
    """Forced trades, such as liquidating unscored holdings, bypass the fee budget."""
    delta = pd.Series({"A.NZ": -0.40})
    alpha = pd.Series({"A.NZ": 0.0})
    forced = pd.Series({"A.NZ": True})

    result = apply_execution_policy(
        delta,
        nav_nzd=10_000.0,
        expected_alpha_bps=alpha,
        config=ExecutionPolicyConfig(
            volume_budget_nzd=1_000.0,
            excess_trade_benefit_bps=10_000.0,
        ),
        forced=forced,
    )

    assert result.executable_delta_weights["A.NZ"] == -0.40
    assert result.deferred["A.NZ"] is False
    assert result.executed_volume_nzd == 4_000.0


def test_disabled_policy_executes_all_filtered_trades():
    """Default config is inert unless an execution policy is explicitly enabled."""
    delta = pd.Series({"A.NZ": 0.30, "B.NZ": 0.30})

    result = apply_execution_policy(
        delta,
        nav_nzd=10_000.0,
        expected_alpha_bps=None,
        config=ExecutionPolicyConfig(),
    )

    assert result.executable_delta_weights.equals(delta.astype(float))
    assert result.deferred.eq(False).all()
    assert result.executed_volume_nzd == 6_000.0
