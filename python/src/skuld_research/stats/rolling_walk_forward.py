"""Rolling walk-forward driver (5y train / 1y OOS / 1y step)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from skuld_common.contracts import PreparedPanel, WalkForwardResult
from skuld_research.backtest.engine import BacktestConfig
from skuld_research.backtest.walk_forward import FoldSpec, WalkForwardEngine
from skuld_research.factors.protocols import SignalGenerator


class RollingWalkForwardEngine:
    """Rolling walk-forward driver with fixed train/OOS/step windows.
    
    For parameter-free strategies (e.g., momentum), the "training" window
    simply ensures the panel has sufficient history before the first OOS period.
    
    Args:
        panel: PreparedPanel covering full history.
        factors: list of SignalGenerator instances.
        train_years: warm-up years before first OOS (default 5).
        oos_years: out-of-sample years per fold (default 1).
        step_years: step size between folds (default 1).
        delisting_csv_path: optional path to delisting CSV.
        backtest_config: BacktestConfig for the engine.
        monte_carlo_seeds: MC simulations for augmented drawdown.
        mc_rng_seed: RNG seed for MC.
        precomputed_returns: optional pre-built monthly returns Series. When
            provided, portfolio construction is skipped and these returns are
            sliced to each fold window directly (zero costs/turnover). Used
            for benchmark pass-through.
        overlay_rule: optional OverlayRule for cash overlay. Defaults to NoOverlay().
    """
    
    def __init__(
        self,
        panel: PreparedPanel,
        factors: list[SignalGenerator],
        train_years: int = 5,
        oos_years: int = 1,
        step_years: int = 1,
        delisting_csv_path: Path | str | None = None,
        backtest_config: BacktestConfig | None = None,
        monte_carlo_seeds: int = 1_000,
        mc_rng_seed: int = 42,
        precomputed_returns: pd.Series | None = None,
        overlay_rule = None,
        spread_panel: pd.DataFrame | None = None,
    ) -> None:
        self.panel = panel
        self.factors = factors
        self.train_years = train_years
        self.oos_years = oos_years
        self.step_years = step_years
        self.delisting_csv = delisting_csv_path
        self.bc = backtest_config or BacktestConfig()
        self.mc_seeds = monte_carlo_seeds
        self.mc_rng_seed = mc_rng_seed
        self.precomputed_returns = precomputed_returns
        self.overlay_rule = overlay_rule
        self.spread_panel = spread_panel
    
    def _build_folds(self) -> list[FoldSpec]:
        """Build rolling folds for the panel.
        
        Returns:
            List of FoldSpec instances.
        """
        rebalance_dates = self.panel.universe_mask.index.tolist()
        
        if len(rebalance_dates) < 2:
            raise ValueError("Need at least 2 rebalance dates")
        
        start_date = rebalance_dates[0]
        end_date = rebalance_dates[-1]
        
        # First OOS start: start_date + train_years
        first_oos_start = start_date + pd.DateOffset(years=self.train_years)
        
        # Generate folds
        folds = []
        fold_id = 0
        current_oos_start = first_oos_start
        
        while current_oos_start <= end_date:
            oos_end = current_oos_start + pd.DateOffset(years=self.oos_years)
            
            # Find rebalance dates in [current_oos_start, oos_end)
            fold_dates = [
                d for d in rebalance_dates
                if current_oos_start <= d < oos_end
            ]
            
            if not fold_dates:
                # No rebalance dates in this window → skip
                current_oos_start = current_oos_start + pd.DateOffset(years=self.step_years)
                continue
            
            test_start = fold_dates[0]
            test_end = fold_dates[-1]

            # Skip folds where the universe mask is empty at every OOS rebalance
            # date. These folds cannot produce a portfolio by definition (no
            # tickers pass the mcap/ADV/history filter) and should not be
            # counted as "rejected" — they were never candidate folds. Counting
            # them inflates n_rejected and the multiple-testing denominator.
            mask_in_window = self.panel.universe_mask.loc[fold_dates]
            if mask_in_window.to_numpy().sum() == 0:
                current_oos_start = current_oos_start + pd.DateOffset(years=self.step_years)
                continue

            folds.append(FoldSpec(
                fold_id=fold_id,
                test_start=test_start,
                test_end=test_end,
            ))

            fold_id += 1
            current_oos_start = current_oos_start + pd.DateOffset(years=self.step_years)
        
        if not folds:
            raise ValueError(
                f"No OOS folds generated. Panel spans {start_date} to {end_date}; "
                f"train_years={self.train_years}, oos_years={self.oos_years}"
            )
        
        return folds
    
    def run(self) -> WalkForwardResult:
        """Run rolling walk-forward and return aggregated result."""
        folds = self._build_folds()
        
        # Delegate to WalkForwardEngine
        wf = WalkForwardEngine(
            factors=self.factors,
            panel=self.panel,
            folds=folds,
            delisting_csv_path=self.delisting_csv,
            backtest_config=self.bc,
            monte_carlo_seeds=self.mc_seeds,
            mc_rng_seed=self.mc_rng_seed,
            precomputed_returns=self.precomputed_returns,
            overlay_rule=self.overlay_rule,
            spread_panel=self.spread_panel,
        )
        
        return wf.run()
