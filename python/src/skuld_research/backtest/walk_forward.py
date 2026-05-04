"""Walk-forward evaluation for the Skuld backtest engine.

For factor strategies with no tunable parameters (e.g., momentum),
there is no training phase: each fold is purely OOS evaluation.
The engine restricts `panel.universe_mask` to each fold's date range
while passing the full panel (PIT-safe) to BacktestEngine for signal computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from skuld_common.contracts import BacktestResult, FoldResult, PreparedPanel, WalkForwardResult
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.backtest.metrics import compute_drawdown_series
from skuld_research.factors.protocols import SignalGenerator
from skuld_research.survivorship.bias import SurvivorshipAdjuster


@dataclass(frozen=True)
class FoldSpec:
    """Definition of one OOS fold in the walk-forward.

    Attributes:
        fold_id: 0-based fold index.
        test_start: first rebalance date included in this fold.
        test_end: last rebalance date included in this fold (inclusive).
    """
    fold_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.fold_id < 0:
            raise ValueError(f"FoldSpec.fold_id must be >= 0, got {self.fold_id}")
        if self.test_end < self.test_start:
            raise ValueError(
                f"FoldSpec.test_end ({self.test_end}) must be >= test_start ({self.test_start})"
            )


class WalkForwardEngine:
    """Runs a walk-forward evaluation of a factor strategy.

    For strategies with no free parameters (like momentum), there is no
    training step — each fold is purely OOS. The full panel is passed to
    BacktestEngine; only universe_mask is restricted per fold so that the
    engine iterates over the correct rebalance dates while still having
    access to the full history for signal computation (PIT-safe).

    Args:
        factors: list of SignalGenerator instances (e.g., [MomentumFactor()]).
        panel: PreparedPanel covering the full available history.
        folds: list of FoldSpec defining each OOS window. Must be non-overlapping.
        delisting_csv_path: optional path to nzx_delistings.csv for probabilistic
            survivorship adjustment. If None, falls back to flat 400 bps haircut only.
        backtest_config: BacktestConfig shared across all folds.
        monte_carlo_seeds: number of MC simulations for augmented drawdown.
        mc_rng_seed: seed for MC reproducibility.
        precomputed_returns: optional pd.Series of monthly returns. When provided,
            the engine skips portfolio construction and uses these returns directly
            (aligned to fold windows), with zero costs/turnover. Used for benchmarks.
        overlay_rule: optional OverlayRule for cash overlay. Defaults to NoOverlay().
    """

    def __init__(
        self,
        factors: list[SignalGenerator],
        panel: PreparedPanel,
        folds: list[FoldSpec],
        delisting_csv_path: Path | str | None = None,
        backtest_config: BacktestConfig | None = None,
        monte_carlo_seeds: int = 1_000,
        mc_rng_seed: int = 42,
        precomputed_returns: pd.Series | None = None,
        overlay_rule = None,
        spread_panel: pd.DataFrame | None = None,
    ) -> None:
        self.factors = factors
        self.panel = panel
        self.folds = sorted(folds, key=lambda f: f.test_start)
        # Validate that folds do not overlap
        for i in range(len(self.folds) - 1):
            a, b = self.folds[i], self.folds[i + 1]
            if b.test_start <= a.test_end:
                raise ValueError(
                    f"Folds {a.fold_id} and {b.fold_id} overlap: "
                    f"fold {a.fold_id} ends {a.test_end}, "
                    f"fold {b.fold_id} starts {b.test_start}"
                )
        self.delisting_csv = Path(delisting_csv_path) if delisting_csv_path else None
        self.bc = backtest_config or BacktestConfig()
        self.mc_seeds = monte_carlo_seeds
        self.mc_rng_seed = mc_rng_seed
        self.precomputed_returns = precomputed_returns
        self.overlay_rule = overlay_rule
        self.spread_panel = spread_panel

    def run(self) -> WalkForwardResult:
        """Run all OOS folds and aggregate into a WalkForwardResult."""
        # Short-circuit for precomputed returns (benchmarks)
        if self.precomputed_returns is not None:
            return self._run_precomputed()

        adjuster = SurvivorshipAdjuster(
            delisting_csv_path=self.delisting_csv,
            flat_haircut_bps=self.bc.flat_haircut_bps,
            monte_carlo_seeds=self.mc_seeds,
            rng_seed=self.mc_rng_seed,
        )

        fold_results: list[FoldResult] = []
        oos_returns_parts: list[pd.Series] = []
        rejection_reasons: list[str] = []
        rejected_fold_ids: set[int] = set()
        n_kept = 0
        n_rejected = 0

        for spec in self.folds:
            panel_fold = _restrict_panel_to_fold(self.panel, spec)
            engine = BacktestEngine(
                factors=self.factors,
                panel=panel_fold,
                config=self.bc,
                overlay_rule=self.overlay_rule,
                spread_panel=self.spread_panel,
            )
            fold_bt: BacktestResult = engine.run()

            # Degenerate-fold rejection
            min_pos = self.bc.min_positions_per_month
            max_empty_frac = self.bc.degenerate_fold_max_empty_frac

            # Only apply rejection if period_n_positions is populated
            if not fold_bt.period_n_positions.empty and len(fold_bt.period_n_positions) > 0:
                # Count empty months (n_positions < min_pos)
                empty_months = (fold_bt.period_n_positions < min_pos).sum()
                total_months = len(fold_bt.period_n_positions)
                empty_frac = empty_months / max(1, total_months)
                is_degenerate = empty_frac > max_empty_frac
            else:
                # No position data → consider valid (backward compatibility)
                empty_frac = 0.0
                is_degenerate = False

            fold_results.append(FoldResult(
                fold_id=spec.fold_id,
                test_start=spec.test_start,
                test_end=spec.test_end,
                result=fold_bt,
            ))

            if is_degenerate:
                n_rejected += 1
                rejected_fold_ids.add(spec.fold_id)
                pct = int(empty_frac * 100)
                rejection_reasons.append(f"fold {spec.fold_id}: {pct}% empty months")
            else:
                n_kept += 1
                oos_returns_parts.append(fold_bt.returns)

        # Concatenate OOS returns from kept folds only
        if oos_returns_parts:
            oos_returns = pd.concat(oos_returns_parts).sort_index()
        else:
            # All folds rejected → empty returns
            oos_returns = pd.Series([], dtype=float)

        # Aggregate stats
        if len(oos_returns) > 0:
            n = len(oos_returns)
            mu = float(oos_returns.mean()) * 12.0
            vol = float(oos_returns.std(ddof=1)) * (12.0 ** 0.5) if n > 1 else 0.0

            rf = self.bc.risk_free_annual
            oos_sharpe_raw = ((mu - rf) / vol) if vol > 1e-12 else 0.0
            haircut = self.bc.flat_haircut_bps / 10_000.0
            oos_sharpe_hc = ((mu - rf - haircut) / vol) if vol > 1e-12 else 0.0
            oos_sharpe_dl = adjuster.delisting_adjusted_sharpe(
                sharpe_raw=oos_sharpe_raw,
                annualised_return=mu,
                annualised_vol=vol,
            )

            oos_drawdown = compute_drawdown_series(oos_returns)
            oos_mdd_obs = float(oos_drawdown.min()) if not oos_drawdown.empty else 0.0
            oos_mdd_obs = min(oos_mdd_obs, 0.0)

            # Average positions for MC (kept folds only)
            kept_folds = [fr for fr in fold_results if fr.fold_id not in rejected_fold_ids]
            avg_pos = (
                sum(fr.result.avg_positions for fr in kept_folds) / max(1, len(kept_folds))
            )
            mc_med, mc_p90 = adjuster.augmented_max_drawdown(
                oos_returns, n_names_avg=avg_pos
            )

            total_cost = sum(float(fr.result.costs_nzd.sum()) for fr in kept_folds)
            avg_turnover = (
                sum(float(fr.result.turnover.mean()) for fr in kept_folds) / max(1, len(kept_folds))
            )

            oos_hit_rate = float((oos_returns > 0).mean())

            try:
                from scipy import stats as scipy_stats
                oos_skewness = (
                    float(scipy_stats.skew(oos_returns.values, bias=False))
                    if n >= 3
                    else 0.0
                )
            except Exception:
                oos_skewness = 0.0

            oos_mdd_for_calmar = float(oos_drawdown.min()) if not oos_drawdown.empty else 0.0
            oos_calmar_ratio = (
                ((mu - rf) / abs(oos_mdd_for_calmar))
                if abs(oos_mdd_for_calmar) > 1e-12
                else 0.0
            )
        else:
            # No kept folds → zero everything
            oos_sharpe_raw = 0.0
            oos_sharpe_hc = 0.0
            oos_sharpe_dl = 0.0
            oos_drawdown = pd.Series([], dtype=float)
            oos_mdd_obs = 0.0
            mc_med = 0.0
            mc_p90 = 0.0
            total_cost = 0.0
            avg_turnover = 0.0
            oos_hit_rate = 0.0
            oos_skewness = 0.0
            oos_calmar_ratio = 0.0

        # Per-regime Sharpe (using kept folds only)
        oos_sharpe_by_regime = _compute_per_regime_sharpe(
            self.panel, oos_returns, self.bc.risk_free_annual
        )

        # Stationary bootstrap CI
        n_boot = getattr(self.bc, "bootstrap_n_resamples", 2000)
        oos_sharpe_stationary_bootstrap_ci = _compute_stationary_bootstrap_ci(oos_returns, n_boot, self.bc.risk_free_annual)

        return WalkForwardResult(
            folds=tuple(fold_results),
            oos_returns=oos_returns,
            oos_sharpe_raw=oos_sharpe_raw,
            oos_sharpe_flat_haircut=oos_sharpe_hc,
            oos_sharpe_delisting_adjusted=oos_sharpe_dl,
            oos_drawdown_observed=oos_drawdown,
            oos_max_drawdown_observed=oos_mdd_obs,
            oos_max_drawdown_augmented_median=mc_med,
            oos_max_drawdown_augmented_p90=mc_p90,
            oos_avg_turnover=avg_turnover,
            oos_total_cost_nzd=total_cost,
            oos_hit_rate=oos_hit_rate,
            oos_skewness=oos_skewness,
            oos_calmar_ratio=oos_calmar_ratio,
            n_kept_folds=n_kept,
            n_rejected_folds=n_rejected,
            rejection_reasons=tuple(rejection_reasons),
            oos_sharpe_by_regime=oos_sharpe_by_regime,
            oos_sharpe_stationary_bootstrap_ci=oos_sharpe_stationary_bootstrap_ci,
        )

    def _run_precomputed(self) -> WalkForwardResult:
        """Short-circuit: use precomputed returns directly, zero costs/turnover."""
        precomp = self.precomputed_returns
        assert precomp is not None

        # Empty benchmark series (e.g. macro field missing on a tiny synthetic panel)
        # may carry the default RangeIndex; coerce to DatetimeIndex so per-fold slicing
        # is well-typed regardless of upstream content.
        if not isinstance(precomp.index, pd.DatetimeIndex):
            precomp = precomp.copy()
            precomp.index = pd.DatetimeIndex(precomp.index, name=precomp.index.name)

        # Extract returns for each fold window
        fold_results: list[FoldResult] = []
        oos_returns_parts: list[pd.Series] = []

        for spec in self.folds:
            fold_returns = precomp.loc[
                (precomp.index >= spec.test_start) & (precomp.index <= spec.test_end)
            ]

            # Build a minimal BacktestResult with zero costs/turnover
            fold_bt = BacktestResult(
                returns=fold_returns,
                costs_nzd=pd.Series(0.0, index=fold_returns.index),
                turnover=pd.Series(0.0, index=fold_returns.index),
                drawdown=compute_drawdown_series(fold_returns),
                sharpe_raw=0.0,  # Will be recomputed in aggregate
                sharpe_flat_haircut=0.0,
                start=fold_returns.index[0] if not fold_returns.empty else spec.test_start,
                end=fold_returns.index[-1] if not fold_returns.empty else spec.test_end,
                n_periods=len(fold_returns),
                avg_positions=0.0,
                hit_rate=0.0,
                skewness=0.0,
                calmar_ratio=0.0,
                period_n_positions=pd.Series(0, index=fold_returns.index, dtype=int),
            )

            fold_results.append(FoldResult(
                fold_id=spec.fold_id,
                test_start=spec.test_start,
                test_end=spec.test_end,
                result=fold_bt,
            ))

            oos_returns_parts.append(fold_returns)

        # Concatenate OOS returns
        if oos_returns_parts:
            oos_returns = pd.concat(oos_returns_parts).sort_index()
        else:
            oos_returns = pd.Series([], dtype=float)

        # Compute aggregate stats
        if len(oos_returns) > 0:
            n = len(oos_returns)
            mu = float(oos_returns.mean()) * 12.0
            vol = float(oos_returns.std(ddof=1)) * (12.0 ** 0.5) if n > 1 else 0.0

            rf = self.bc.risk_free_annual
            oos_sharpe_raw = ((mu - rf) / vol) if vol > 1e-12 else 0.0
            # Precomputed benchmark returns are already net of their own costs or
            # haircuts; do not apply the strategy's survivorship/cost stress again.
            oos_sharpe_hc = oos_sharpe_raw
            oos_sharpe_dl = oos_sharpe_raw

            oos_drawdown = compute_drawdown_series(oos_returns)
            oos_mdd_obs = float(oos_drawdown.min()) if not oos_drawdown.empty else 0.0
            oos_mdd_obs = min(oos_mdd_obs, 0.0)

            # No MC augmentation for precomputed returns
            mc_med = oos_mdd_obs
            mc_p90 = oos_mdd_obs

            oos_hit_rate = float((oos_returns > 0).mean())

            try:
                from scipy import stats as scipy_stats
                oos_skewness = (
                    float(scipy_stats.skew(oos_returns.values, bias=False))
                    if n >= 3
                    else 0.0
                )
            except Exception:
                oos_skewness = 0.0

            oos_calmar_ratio = (
                ((mu - rf) / abs(oos_mdd_obs))
                if abs(oos_mdd_obs) > 1e-12
                else 0.0
            )
        else:
            oos_sharpe_raw = 0.0
            oos_sharpe_hc = 0.0
            oos_sharpe_dl = 0.0
            oos_drawdown = pd.Series([], dtype=float)
            oos_mdd_obs = 0.0
            mc_med = 0.0
            mc_p90 = 0.0
            oos_hit_rate = 0.0
            oos_skewness = 0.0
            oos_calmar_ratio = 0.0

        # Per-regime Sharpe
        oos_sharpe_by_regime = _compute_per_regime_sharpe(
            self.panel, oos_returns, self.bc.risk_free_annual
        )

        return WalkForwardResult(
            folds=tuple(fold_results),
            oos_returns=oos_returns,
            oos_sharpe_raw=oos_sharpe_raw,
            oos_sharpe_flat_haircut=oos_sharpe_hc,
            oos_sharpe_delisting_adjusted=oos_sharpe_dl,
            oos_drawdown_observed=oos_drawdown,
            oos_max_drawdown_observed=oos_mdd_obs,
            oos_max_drawdown_augmented_median=mc_med,
            oos_max_drawdown_augmented_p90=mc_p90,
            oos_avg_turnover=0.0,  # Zero for precomputed
            oos_total_cost_nzd=0.0,  # Zero for precomputed
            oos_hit_rate=oos_hit_rate,
            oos_skewness=oos_skewness,
            oos_calmar_ratio=oos_calmar_ratio,
            n_kept_folds=len(self.folds),
            n_rejected_folds=0,
            rejection_reasons=(),
            oos_sharpe_by_regime=oos_sharpe_by_regime,
        )


def _restrict_panel_to_fold(panel, spec: FoldSpec):
    """Return a PreparedPanel with universe_mask restricted to the fold's date range.

    The full returns_daily, returns_monthly, market_cap, sector are kept intact
    so BacktestEngine can compute signals using the full history (PIT-safe).
    Only the rebalance schedule (universe_mask) is restricted to the fold window.
    """
    from skuld_common.contracts import PreparedPanel

    mask = panel.universe_mask
    fold_dates = mask.index[
        (mask.index >= spec.test_start) & (mask.index <= spec.test_end)
    ]
    if fold_dates.empty:
        raise ValueError(
            f"Fold {spec.fold_id} [{spec.test_start}, {spec.test_end}] "
            f"has no rebalance dates in panel.universe_mask.index"
        )
    fold_mask = mask.loc[fold_dates]
    return PreparedPanel(
        returns_daily=panel.returns_daily,
        returns_monthly=panel.returns_monthly,
        market_cap=panel.market_cap,
        sector=panel.sector,
        universe_mask=fold_mask,
        macro=panel.macro,
        asof=panel.asof,
        prices=panel.prices,
        corporate_actions=panel.corporate_actions,
    )


def _compute_stationary_bootstrap_ci(
    oos_returns: pd.Series,
    n_resamples: int,
    rf_annual: float,
) -> tuple[float, float]:
    """Return (ci_low_95, ci_high_95) from stationary block bootstrap, or (nan, nan)."""
    if len(oos_returns) < 4:
        return (float("nan"), float("nan"))
    try:
        from skuld_research.stats.bootstrap import stationary_bootstrap_sharpe
        result = stationary_bootstrap_sharpe(
            oos_returns.dropna(),
            n_resamples=n_resamples,
            rf_annual=rf_annual,
        )
        return (result.ci_low_95, result.ci_high_95)
    except Exception:
        return (float("nan"), float("nan"))


def _compute_per_regime_sharpe(
    panel: PreparedPanel,
    oos_returns: pd.Series,
    rf_annual: float,
) -> dict[str, float]:
    """Compute per-regime annualised Sharpe from OOS returns."""
    if oos_returns.empty:
        return {}

    # Import here to avoid circular dependency
    from skuld_research.stats.regimes import label_regimes

    regime_labels = label_regimes(panel)

    # Align returns with regime labels
    aligned = oos_returns.to_frame("ret").join(regime_labels.to_frame("regime"), how="inner")

    sharpes = {}
    for regime in ["bull", "bear", "chop"]:
        regime_rets = aligned[aligned["regime"] == regime]["ret"]
        if len(regime_rets) > 1:
            mu = regime_rets.mean() * 12.0
            vol = regime_rets.std(ddof=1) * (12.0 ** 0.5)
            sharpe = ((mu - rf_annual) / vol) if vol > 1e-12 else 0.0
            sharpes[regime] = sharpe

    return sharpes
