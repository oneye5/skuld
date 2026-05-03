"""Cost-aware execution policy for target-portfolio deltas."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExecutionPolicyConfig:
    """Parameters for monthly trade deferral."""

    volume_budget_nzd: float | None = None
    turnover_budget_frac: float | None = None
    min_trade_benefit_bps: float = 0.0
    excess_trade_benefit_bps: float = 190.0

    @property
    def enabled(self) -> bool:
        """Whether this policy can alter trades."""
        return (
            self.volume_budget_nzd is not None
            or self.turnover_budget_frac is not None
            or self.min_trade_benefit_bps > 0.0
        )


@dataclass(frozen=True)
class ExecutionPolicyResult:
    """Execution policy output for one rebalance."""

    executable_delta_weights: pd.Series
    deferred: pd.Series
    executed_volume_nzd: float
    deferred_volume_nzd: float
    excess_volume_nzd: float


def apply_execution_policy(
    delta_weights: pd.Series,
    *,
    nav_nzd: float,
    expected_alpha_bps: pd.Series | None,
    config: ExecutionPolicyConfig,
    forced: pd.Series | None = None,
) -> ExecutionPolicyResult:
    """Apply cost-aware monthly trade deferral to weight deltas.

    The policy ranks candidate trades by directional expected benefit, executes
    high-benefit trades first, and defers low-benefit trades that either fail a
    minimum-benefit hurdle or would exceed the monthly covered-volume budget.
    ``delta_weights`` are signed fractions of NAV; trade volume is
    ``abs(delta_weight) * nav_nzd``.
    """
    deltas = delta_weights.astype(float).copy()
    trade_values = deltas.abs() * nav_nzd
    if deltas.empty:
        return ExecutionPolicyResult(
            executable_delta_weights=deltas,
            deferred=pd.Series(dtype=object),
            executed_volume_nzd=0.0,
            deferred_volume_nzd=0.0,
            excess_volume_nzd=0.0,
        )

    if not config.enabled:
        executed_volume = float(trade_values.sum())
        return ExecutionPolicyResult(
            executable_delta_weights=deltas,
            deferred=pd.Series(False, index=deltas.index, dtype=object),
            executed_volume_nzd=executed_volume,
            deferred_volume_nzd=0.0,
            excess_volume_nzd=0.0,
        )

    alpha = (
        expected_alpha_bps.reindex(deltas.index, fill_value=0.0).astype(float)
        if expected_alpha_bps is not None
        else pd.Series(0.0, index=deltas.index)
    )
    forced_mask = (
        forced.reindex(deltas.index, fill_value=False).astype(bool)
        if forced is not None
        else pd.Series(False, index=deltas.index)
    )

    executable = pd.Series(0.0, index=deltas.index)
    deferred = pd.Series(False, index=deltas.index, dtype=object)
    executed_volume = 0.0
    deferred_volume = 0.0
    volume_budget = config.volume_budget_nzd
    turnover_budget = (
        2.0 * config.turnover_budget_frac * nav_nzd
        if config.turnover_budget_frac is not None
        else None
    )

    directional_benefit = alpha * deltas.apply(lambda value: 1.0 if value >= 0 else -1.0)
    ranked = sorted(deltas.index, key=lambda tk: float(directional_benefit[tk]), reverse=True)
    for ticker in ranked:
        trade_value = float(trade_values[ticker])
        if trade_value <= 1e-12:
            continue

        benefit_bps = float(directional_benefit[ticker])
        is_forced = bool(forced_mask[ticker])

        if not is_forced and benefit_bps < config.min_trade_benefit_bps:
            deferred[ticker] = True
            deferred_volume += trade_value
            continue

        crosses_volume_budget = (
            volume_budget is not None and executed_volume + trade_value > volume_budget
        )
        crosses_turnover_budget = (
            turnover_budget is not None and executed_volume + trade_value > turnover_budget
        )
        if not is_forced and crosses_turnover_budget:
            deferred[ticker] = True
            deferred_volume += trade_value
            continue
        if (
            not is_forced
            and crosses_volume_budget
            and benefit_bps < config.excess_trade_benefit_bps
        ):
            deferred[ticker] = True
            deferred_volume += trade_value
            continue

        executable[ticker] = float(deltas[ticker])
        executed_volume += trade_value

    excess_volume = (
        max(0.0, executed_volume - volume_budget)
        if volume_budget is not None
        else 0.0
    )
    return ExecutionPolicyResult(
        executable_delta_weights=executable,
        deferred=deferred,
        executed_volume_nzd=executed_volume,
        deferred_volume_nzd=deferred_volume,
        excess_volume_nzd=excess_volume,
    )
