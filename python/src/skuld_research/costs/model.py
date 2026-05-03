"""Transaction cost model: bid/ask spread + Sharesies monthly subscription.

Sharesies $15/month plan (as of Nov 2025 — see docs/sharesies-pricing.md):
- $15/month subscription is charged every month regardless of trading activity.
- The subscription covers the first $5,000 of buy/sell volume (no extra fee on coverage).
- Volume above $5,000 in a month: 1.9% (190 bps) on the EXCESS only.

Spread cost semantics:
- `spread_bps` is the cost charged on EACH SIDE of a trade (i.e. per-side
  rate). A buy of $1k followed by a sell of $1k generates total_volume = $2k
  and a spread cost of `$2k * spread_bps / 1e4`.
- Therefore `spread_bps` corresponds to half of a quoted bid-ask spread:
  a market with a 100bps quoted spread implies `spread_bps = 50` (you cross
  half the spread per side). Setting `spread_bps = 200` models a 400bps
  quoted spread — a deliberately conservative all-tickers-equal assumption
  for NZX, retained for backward compatibility with pre-AR specs.
- For a per-ticker, time-varying spread estimate, pass `per_ticker_spread_bps`
  to `compute_period_costs` (see `spread_estimator.py`).

Simplification: per-order $25 NZD cap is not modelled; conservative for large single orders.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    """Transaction cost parameters (all configurable)."""
    spread_bps: float = 200.0                      # per-side spread rate in bps (see module docstring)
    sharesies_monthly_fee_nzd: float = 15.0        # subscription fee charged every month
    sharesies_coverage_nzd: float = 5_000.0        # volume covered by subscription (no extra fee)
    sharesies_excess_bps: float = 190.0            # 1.9% on volume above coverage
    sharesies_per_order_cap_nzd: float = 25.0      # max fee per single trade


@dataclass(frozen=True)
class CostBreakdown:
    """Full cost breakdown for one rebalance period."""
    spread_cost_nzd: float
    sharesies_fee_nzd: float
    total_cost_nzd: float
    sharesies_fee_band: str   # 'subscription_only' | 'subscription_plus_excess'


class CostModel:
    """Computes transaction costs for a set of trades in one rebalance period.

    The Sharesies fee is monthly (not per-trade): the fee depends on total
    volume across all trades in the month, not on individual trade sizes.
    The spread cost is proportional to total absolute trade volume.
    """

    def __init__(self, config: CostConfig | None = None) -> None:
        self.config = config or CostConfig()

    def compute_period_costs(
        self,
        trade_values_nzd: pd.Series,
        *,
        per_ticker_spread_bps: pd.Series | None = None,
    ) -> CostBreakdown:
        """Compute full cost breakdown for all trades in one rebalance period.

        Args:
            trade_values_nzd: Series of trade values in NZD (can be signed or
                unsigned; absolute values are used). Each entry is one trade.
                The index identifies the ticker. An empty Series means no
                trades this period.
            per_ticker_spread_bps: Optional Series of per-side spread bps,
                indexed by ticker. When provided, each trade is charged its
                own ticker-specific spread; when absent, the flat
                `config.spread_bps` is applied to all trades. Tickers in
                `trade_values_nzd` that are missing from this series fall back
                to the flat config rate.

        Returns:
            CostBreakdown with spread cost, Sharesies fee, and total cost.
        """
        abs_values = trade_values_nzd.abs()
        total_volume = float(abs_values.sum())

        if per_ticker_spread_bps is None or abs_values.empty:
            spread_cost = total_volume * self.config.spread_bps / 10_000
        else:
            # Per-trade per-side bps; missing tickers fall back to flat rate.
            bps = per_ticker_spread_bps.reindex(abs_values.index).fillna(
                self.config.spread_bps
            )
            spread_cost = float((abs_values * bps / 10_000).sum())

        # Subscription is always charged regardless of trading activity.
        fee = self.config.sharesies_monthly_fee_nzd
        excess = total_volume - self.config.sharesies_coverage_nzd
        if excess > 0:
            # We apply the per-order cap. Since we don't know the exact chronological
            # order to apply the $5000 coverage, we compute the total fee as if there
            # were no coverage (but with caps), then scale it down by the proportion
            # of volume that is excess.
            uncapped_rates = abs_values * self.config.sharesies_excess_bps / 10_000
            capped_fees = uncapped_rates.clip(upper=self.config.sharesies_per_order_cap_nzd)
            total_capped_fee = float(capped_fees.sum())
            excess_ratio = excess / total_volume
            fee += total_capped_fee * excess_ratio
            band = "subscription_plus_excess"
        else:
            band = "subscription_only"

        total = spread_cost + fee
        return CostBreakdown(
            spread_cost_nzd=spread_cost,
            sharesies_fee_nzd=fee,
            total_cost_nzd=total,
            sharesies_fee_band=band,
        )
