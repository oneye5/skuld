"""Apply cash overlay to a target portfolio."""
from __future__ import annotations

import pandas as pd

from skuld_common.contracts import PreparedPanel, TargetPortfolio
from skuld_research.overlay.rules import OverlayRule


def apply_cash_overlay(
    target: TargetPortfolio,
    panel: PreparedPanel,
    rule: OverlayRule,
    asof: pd.Timestamp,
) -> TargetPortfolio:
    """Apply cash overlay rule to a target portfolio.
    
    Evaluates the overlay rule and raises cash to the maximum of the target's
    existing cash weight and the rule's desired cash fraction. Equity weights
    are re-normalised to preserve relative proportions while summing to
    (1 - new_cash_weight).
    
    Args:
        target: Target portfolio from Stage 5 (portfolio constructor).
        panel: PreparedPanel for rule evaluation.
        rule: OverlayRule instance.
        asof: Rebalance date.
    
    Returns:
        New TargetPortfolio with cash potentially raised and equity weights
        re-normalised. If the rule returns a cash fraction <= target's existing
        cash_weight, the target is returned unchanged (same object).
    """
    desired_cash = rule.evaluate(panel, asof)

    # Clamp to [0, 1] as a safety (rules should enforce this, but belt-and-suspenders)
    desired_cash = max(0.0, min(1.0, desired_cash))

    # Take the maximum of existing cash floor and rule's desired cash
    new_cash = max(target.cash_weight, desired_cash)

    # If no change, return the original target
    if abs(new_cash - target.cash_weight) < 1e-9:
        return target

    # Re-normalise equity weights to sum to (1 - new_cash)
    old_equity_total = target.weights.sum()
    if old_equity_total < 1e-9:
        # Edge case: target had no equity → keep it that way
        new_weights = target.weights.copy()
    else:
        new_equity_total = 1.0 - new_cash
        scale = new_equity_total / old_equity_total
        new_weights = target.weights * scale

    # Validate invariants (same as TargetPortfolio's implicit contract)
    equity_sum = new_weights.sum()
    total = equity_sum + new_cash
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"apply_cash_overlay produced invalid portfolio: "
            f"equity_sum={equity_sum:.6f}, cash={new_cash:.6f}, total={total:.6f}"
        )

    if (new_weights < -1e-9).any():
        raise ValueError(
            f"apply_cash_overlay produced negative weights: "
            f"min={new_weights.min():.6f}"
        )

    return TargetPortfolio(
        weights=new_weights,
        cash_weight=new_cash,
        method=target.method,
        asof=target.asof,
    )
