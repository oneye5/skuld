"""Survivorship-bias adjustment for NZX backtest results.

Two adjustment modes:
1. Flat haircut: subtract FLAT_HAIRCUT_DEFAULT (400 bps/yr) from annualised return.
2. Probabilistic (when delisting CSV is loaded):
   a. Factor-conditional annual drag = p × μ_d (where p = loss-type delisting rate,
      μ_d = mean terminal return of loss delistings).
   b. Monte Carlo augmented drawdown: inject random delistings into the return series.

Reference: §1.1 of the Skuld implementation plan (docs/2026-04-20-skuld-implementation-plan.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from skuld_research.backtest.metrics import compute_drawdown_series, compute_max_drawdown

FLAT_HAIRCUT_BPS_DEFAULT: float = 400.0
UNIVERSE_SIZE_NZX: int = 146   # full NZX universe; used as denominator for per-name loss rate.
# Mathematically this is correct for equal-weighted portfolios: portfolio drag
# = n_names × (rate_per_name × weight_per_name × |terminal_return|)
# = n_names × (rate_per_name × 1/n_names × |terminal_return|)
# = rate_per_name × |terminal_return|   (n cancels)
# So unconditional_annual_drag is independent of portfolio size.  The 146-name
# denominator gives the unconditional rate for a randomly-selected NZX name.
# Held names are liquidity-screened, so the true per-name rate may be lower —
# meaning the 146-based drag is a conservative upper bound.


@dataclass(frozen=True)
class DelistingStats:
    """Statistics derived from the NZX delisting research sample.

    Attributes:
        annual_loss_rate: annual probability of a loss-type delisting per
            name in the universe (p = n_loss_delistings / years / universe_size).
        mean_terminal_return: mean of terminal_return_12m for loss delistings (< 0).
        std_terminal_return: std of terminal_return_12m for loss delistings (> 0).
        n_loss_delistings: number of loss-type delistings in the sample.
        years_observed: length of the observation window in years.
    """
    annual_loss_rate: float
    mean_terminal_return: float
    std_terminal_return: float
    n_loss_delistings: int
    years_observed: float

    @property
    def unconditional_annual_drag(self) -> float:
        """Expected annual drag = p × μ_d (will be negative since μ_d < 0)."""
        return self.annual_loss_rate * self.mean_terminal_return


class SurvivorshipAdjuster:
    """Applies survivorship-bias adjustments to backtest Sharpe and drawdown.

    Usage with delisting data:
        adjuster = SurvivorshipAdjuster(delisting_csv_path="path/to/nzx_delistings.csv")
        adj_sharpe = adjuster.delisting_adjusted_sharpe(sharpe_raw, ann_ret, ann_vol)
        med_mdd, p90_mdd = adjuster.augmented_max_drawdown(monthly_returns, n_names_avg=20)

    Usage without delisting data (flat haircut only):
        adjuster = SurvivorshipAdjuster()
        adj_sharpe = adjuster.flat_haircut_sharpe(sharpe_raw, ann_ret, ann_vol)
    """

    def __init__(
        self,
        delisting_csv_path: Path | str | None = None,
        flat_haircut_bps: float = FLAT_HAIRCUT_BPS_DEFAULT,
        monte_carlo_seeds: int = 1_000,
        rng_seed: int = 42,
    ) -> None:
        self.flat_haircut_annual = flat_haircut_bps / 10_000
        self.monte_carlo_seeds = monte_carlo_seeds
        self._rng = np.random.default_rng(rng_seed)
        self._stats: DelistingStats | None = None

        if delisting_csv_path is not None:
            self._stats = self._load_delisting_stats(Path(delisting_csv_path))

    @staticmethod
    def _load_delisting_stats(csv_path: Path) -> DelistingStats:
        """Load the NZX delisting CSV and compute summary statistics.

        Only 'involuntary' and 'voluntary' rows with terminal_return_12m < 0
        are counted as loss delistings. 'merger' rows are excluded.
        """
        df = pd.read_csv(csv_path)

        loss = df[
            df["reason"].isin(["involuntary", "voluntary"]) &
            (df["terminal_return_12m"].astype(float) < 0)
        ].copy()
        n = len(loss)
        if n == 0:
            raise ValueError(f"No loss-type delistings found in {csv_path}")

        all_dates = pd.to_datetime(df["delisted_date"], errors="coerce").dropna()
        years = max((all_dates.max() - all_dates.min()).days / 365.25, 1.0)
        annual_loss_rate = (n / years) / UNIVERSE_SIZE_NZX

        rets = loss["terminal_return_12m"].astype(float)
        return DelistingStats(
            annual_loss_rate=float(annual_loss_rate),
            mean_terminal_return=float(rets.mean()),
            std_terminal_return=float(rets.std(ddof=1)) if n > 1 else 0.0,
            n_loss_delistings=n,
            years_observed=float(years),
        )

    def flat_haircut_sharpe(
        self,
        sharpe_raw: float,
        annualised_return: float,
        annualised_vol: float,
    ) -> float:
        """Apply flat 400 bps/yr annual return haircut and recompute Sharpe."""
        if annualised_vol < 1e-12:
            return 0.0
        return (annualised_return - self.flat_haircut_annual) / annualised_vol

    def delisting_adjusted_sharpe(
        self,
        sharpe_raw: float,
        annualised_return: float,
        annualised_vol: float,
    ) -> float:
        """Probabilistic delisting-adjusted Sharpe ratio.

        Uses the MORE CONSERVATIVE of:
        - The flat 400 bps/yr haircut.
        - The probabilistic unconditional annual drag (p × |μ_d|).

        If no delisting stats are loaded, falls back to flat haircut.
        """
        if self._stats is None:
            return self.flat_haircut_sharpe(sharpe_raw, annualised_return, annualised_vol)
        if annualised_vol < 1e-12:
            return 0.0
        prob_drag = abs(self._stats.unconditional_annual_drag)  # positive drag
        conservative_drag = max(prob_drag, self.flat_haircut_annual)
        return (annualised_return - conservative_drag) / annualised_vol

    def augmented_max_drawdown(
        self,
        monthly_returns: pd.Series,
        n_names_avg: float,
        n_simulations: int | None = None,
    ) -> tuple[float, float]:
        """Monte Carlo augmented max drawdown via random delisting injection.

        For each simulation:
        1. Copy the monthly return series.
        2. Each month, randomly inject delistings: each of n_names_avg positions
           faces an annual delisting probability of annual_loss_rate. The monthly
           probability per name is annual_loss_rate / 12. Number of delistings
           per month is drawn from Binomial(int(n_names), p_monthly).
        3. Each delisting causes a portfolio loss of terminal_return / n_names
           (equal-weight approximation). Terminal return drawn from N(μ_d, σ_d),
           clipped to [−1, 0].
        4. Compute cumulative NAV and max drawdown of the augmented series.
        5. Return (median, 10th-pct) of max drawdown across simulations.
           (10th-pct because drawdown is negative, so 10th-pct is the more extreme tail.)

        Falls back to (observed_mdd, observed_mdd) if no stats are loaded.

        Returns:
            (median_augmented_mdd, p90_augmented_mdd) — both ≤ 0.
            The p90 here refers to the 10th percentile of the drawdown distribution
            (i.e., the more negative / worse 10% of outcomes).
        """
        observed_mdd = compute_max_drawdown(monthly_returns)

        if self._stats is None or monthly_returns.empty:
            return observed_mdd, observed_mdd

        n_sim = n_simulations if n_simulations is not None else self.monte_carlo_seeds
        rets = monthly_returns.values.astype(float)
        T = len(rets)
        n_names = max(1, int(round(n_names_avg)))
        p_monthly = self._stats.annual_loss_rate / 12
        mu_d = self._stats.mean_terminal_return
        sigma_d = max(self._stats.std_terminal_return, 1e-6)

        mdd_samples = np.empty(n_sim)
        for s in range(n_sim):
            sim_rets = rets.copy()
            for t_idx in range(T):
                n_delistings = int(self._rng.binomial(n_names, p_monthly))
                if n_delistings > 0:
                    losses = self._rng.normal(mu_d, sigma_d, n_delistings)
                    losses = np.clip(losses, -1.0, 0.0)
                    portfolio_impact = float(losses.sum()) / n_names
                    sim_rets[t_idx] += portfolio_impact
            nav = np.cumprod(1.0 + sim_rets)
            running_max = np.maximum.accumulate(nav)
            drawdowns = nav / running_max - 1.0
            mdd_samples[s] = float(np.min(drawdowns))

        return float(np.median(mdd_samples)), float(np.percentile(mdd_samples, 10))
