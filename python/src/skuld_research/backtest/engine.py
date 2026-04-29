"""Monthly rebalance backtest engine.

Iterates over panel.universe_mask rebalance dates, scores factor signals,
constructs portfolios, applies transaction costs, tracks NAV and weight drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from skuld_common.contracts import BacktestResult, CombinedScores, PreparedPanel, TargetPortfolio
from skuld_research.costs.model import CostBreakdown, CostConfig, CostModel
from skuld_research.factors.combiner import combine_signals
from skuld_research.factors.protocols import SignalGenerator
from skuld_research.portfolio.optimizer import build_target_portfolio
from skuld_research.backtest.metrics import compute_drawdown_series
from skuld_research.overlay.rules import NoOverlay, OverlayRule
from skuld_research.overlay.apply import apply_cash_overlay


@dataclass
class BacktestConfig:
    """All parameters governing a single backtest run.

    NZX structural note: ``max_position`` defaults to 0.25 (25%) because the NZX
    investable universe at any rebalance date is typically 4–10 names.  At 5% the
    optimizer would be over-constrained and the resulting portfolio would diverge
    far from its risk-parity target.  Change this value if applying the engine to
    a deeper universe (e.g., use 0.05 for S&P 500).  ``risk_free_annual`` defaults
    to 0.0; set to e.g. 0.035 for a NZ OCR assumption.
    """
    initial_nav_nzd: float = 10_000.0
    cash_floor: float = 0.05
    max_position: float = 0.25   # raised from 0.05: NZX typically has 4-10 positions
    max_sector: float = 0.25
    score_lambda: float = 0.0
    no_trade_threshold_frac: float = 0.005   # 0.5% of NAV min drift to rebalance
    size_floor_nzd: float = 50.0              # minimum trade in NZD
    size_floor_cost_multiple: float = 5.0     # skip if trade < N × round-trip cost
    return_window_days: int = 252
    min_return_obs: int = 63
    cost_config: CostConfig = field(default_factory=CostConfig)
    flat_haircut_bps: float = 400.0
    risk_free_annual: float = 0.0
    min_positions_per_month: int = 1  # NZX routinely has 4-name months; only reject if truly empty
    degenerate_fold_max_empty_frac: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 < self.max_position <= 1.0):
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")


class BacktestEngine:
    """Monthly rebalance backtest engine.

    Args:
        factors: list of SignalGenerator instances (e.g., [MomentumFactor()]).
        panel: PreparedPanel (already PIT-safe).
        config: BacktestConfig.
        cost_model: optional pre-built CostModel; defaults to CostModel(config.cost_config).
        overlay_rule: optional OverlayRule for cash overlay. Defaults to NoOverlay().
        spread_panel: optional date x ticker DataFrame of per-side spread bps.
            When provided, the cost model uses the per-ticker per-period bps
            looked up from this panel (ffilled to the rebalance date) instead
            of the flat ``cost_config.spread_bps``. Use this with the
            Abdi-Ranaldo OHLC spread estimator (see
            ``skuld_research.costs.spread_estimator``).
    """

    def __init__(
        self,
        factors: list[SignalGenerator],
        panel: PreparedPanel,
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        overlay_rule: OverlayRule | None = None,
        spread_panel: pd.DataFrame | None = None,
    ) -> None:
        self.factors = factors
        self.panel = panel
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or CostModel(self.config.cost_config)
        self.overlay_rule = overlay_rule or NoOverlay()
        self.spread_panel = spread_panel

    def _spread_lookup(self, t: pd.Timestamp) -> pd.Series | None:
        """Return per-ticker per-side bps for rebalance date `t` (or None).

        Uses the most recent row at or before `t`. If `t` precedes the panel
        or no panel was supplied, returns None and the engine falls back to
        the flat `cost_config.spread_bps`.
        """
        if self.spread_panel is None or self.spread_panel.empty:
            return None
        idx = self.spread_panel.index
        if t in idx:
            return self.spread_panel.loc[t]
        avail = idx[idx <= t]
        if avail.empty:
            return None
        return self.spread_panel.loc[avail[-1]]

    def run(self) -> BacktestResult:
        """Run the backtest over all rebalance dates in panel.universe_mask.

        Returns a BacktestResult with net returns, costs, turnover,
        drawdown, and Sharpe ratios.
        """
        panel = self.panel
        cfg = self.config
        rebalance_dates = panel.universe_mask.index.tolist()

        if len(rebalance_dates) < 2:
            raise ValueError("Need at least 2 rebalance dates to run a backtest")

        nav = cfg.initial_nav_nzd
        current_weights: pd.Series = pd.Series(dtype=float)

        period_records: list[dict] = []

        for i, t in enumerate(rebalance_dates[:-1]):
            next_t = rebalance_dates[i + 1]
            t_naive = t.tz_localize(None) if getattr(t, "tzinfo", None) else t

            # Per-ticker spread bps for this rebalance date (or None for flat).
            spread_at_t = self._spread_lookup(t_naive)

            universe = [
                tk for tk in panel.universe_mask.columns
                if bool(panel.universe_mask.loc[t, tk])
            ]

            if not universe:
                # Force liquidation if we hold anything; then charge the subscription fee.
                if not current_weights.empty and (current_weights > 1e-6).any():
                    # Realise returns for the closing period before liquidating
                    monthly_rets = panel.returns_monthly
                    if next_t in monthly_rets.index:
                        next_t_lookup = next_t
                    else:
                        avail = monthly_rets.index[monthly_rets.index <= next_t]
                        next_t_lookup = avail[-1] if not avail.empty else None

                    if next_t_lookup is not None:
                        period_rets_all = monthly_rets.loc[next_t_lookup]
                        period_rets = period_rets_all.reindex(current_weights.index, fill_value=0.0)
                        gross_return = float((current_weights * period_rets).sum())
                    else:
                        gross_return = 0.0

                    # Spread cost on the full liquidation value
                    liquidation_values = current_weights * nav
                    liq_cost = self.cost_model.compute_period_costs(
                        liquidation_values, per_ticker_spread_bps=spread_at_t,
                    )
                    cost_drag = liq_cost.total_cost_nzd / nav
                    net_return = gross_return - cost_drag
                    turnover = float(current_weights.sum())
                    current_weights = pd.Series(dtype=float)
                else:
                    # Holding cash; only subscription fee applies
                    liq_cost = self.cost_model.compute_period_costs(pd.Series(dtype=float))
                    net_return = -liq_cost.total_cost_nzd / nav
                    turnover = 0.0

                nav = nav * (1.0 + net_return)
                period_records.append({
                    "date": next_t,
                    "net_return": net_return,
                    "cost_nzd": liq_cost.total_cost_nzd,
                    "turnover": turnover,
                    "n_positions": 0,
                })
                continue

            # Score signals
            signals: dict[str, pd.Series] = {}
            for factor in self.factors:
                signals[factor.name] = factor.score(panel, t, universe)

            combined: CombinedScores = combine_signals(
                signals, universe, panel.sector, t
            )

            target: TargetPortfolio = build_target_portfolio(
                combined, panel, t,
                cash_floor=cfg.cash_floor,
                max_position=cfg.max_position,
                max_sector=cfg.max_sector,
                score_lambda=cfg.score_lambda,
                return_window_days=cfg.return_window_days,
                min_return_obs=cfg.min_return_obs,
            )

            # Apply cash overlay (may raise cash beyond the floor)
            target = apply_cash_overlay(target, panel, self.overlay_rule, t)

            # All tickers we need to consider (target + current holdings)
            all_tickers = list(
                set(list(target.weights.index)) | set(list(current_weights.index))
            )
            target_full = target.weights.reindex(all_tickers, fill_value=0.0)
            current_full = current_weights.reindex(all_tickers, fill_value=0.0)
            delta_w = target_full - current_full

            # No-trade region: skip if delta_weight is too small
            in_ntr = delta_w.abs() < cfg.no_trade_threshold_frac

            # Size floor: skip if trade value < max(size_floor_nzd, N × spread_cost).
            # Use per-ticker bps when a spread panel is provided so the floor
            # scales with realistic per-name costs.
            delta_values = delta_w.abs() * nav
            if spread_at_t is None:
                bps_per_trade = float(cfg.cost_config.spread_bps)
                spread_cost_per_trade = delta_values * bps_per_trade / 10_000
            else:
                bps_series = spread_at_t.reindex(delta_values.index).fillna(
                    cfg.cost_config.spread_bps
                )
                spread_cost_per_trade = delta_values * bps_series / 10_000
            size_floor_per_trade = np.maximum(
                cfg.size_floor_nzd,
                cfg.size_floor_cost_multiple * spread_cost_per_trade,
            )
            below_floor = delta_values < size_floor_per_trade

            skip = in_ntr | below_floor
            executable_delta = delta_w.copy()
            executable_delta[skip] = 0.0

            # Transaction costs
            exec_abs_values = (executable_delta.abs() * nav)
            cost_bd: CostBreakdown = self.cost_model.compute_period_costs(
                exec_abs_values, per_ticker_spread_bps=spread_at_t,
            )

            # New actual weights
            new_weights = current_full + executable_delta
            new_weights = new_weights.clip(lower=0.0)
            # Prevent leverage: if skipping sells caused total equity > 100%, normalise down.
            total_w = new_weights.sum()
            if total_w > 1.0:
                new_weights = new_weights / total_w

            # Monthly return for the period [t → next_t]
            monthly_rets = panel.returns_monthly
            if next_t in monthly_rets.index:
                next_t_lookup = next_t
            else:
                avail = monthly_rets.index[monthly_rets.index <= next_t]
                if avail.empty:
                    period_records.append({
                        "date": next_t,
                        "net_return": 0.0,
                        "cost_nzd": 0.0,
                        "turnover": 0.0,
                        "n_positions": int((new_weights > 1e-6).sum()),
                    })
                    current_weights = new_weights
                    continue
                next_t_lookup = avail[-1]

            period_rets_all = monthly_rets.loc[next_t_lookup]
            period_rets = period_rets_all.reindex(new_weights.index, fill_value=0.0)
            gross_return = float((new_weights * period_rets).sum())
            cost_drag = cost_bd.total_cost_nzd / nav
            net_return = gross_return - cost_drag

            # One-sided turnover (fraction of NAV traded)
            turnover = float(executable_delta.abs().sum()) / 2.0

            n_pos = int((new_weights > 1e-6).sum())
            period_records.append({
                "date": next_t,
                "net_return": net_return,
                "cost_nzd": cost_bd.total_cost_nzd,
                "turnover": turnover,
                "n_positions": n_pos,
            })

            # Update NAV, drift weights
            nav = nav * (1.0 + net_return)
            drifted = _drift_weights(new_weights, period_rets, gross_return)
            current_weights = drifted

        if not period_records:
            raise ValueError("No periods were evaluated in the backtest")

        return _build_result(period_records, cfg.flat_haircut_bps, cfg.risk_free_annual)


def _drift_weights(
    weights: pd.Series,
    period_returns: pd.Series,
    portfolio_gross_return: float,
) -> pd.Series:
    """Drift weights forward by period returns."""
    denom = 1.0 + portfolio_gross_return
    if abs(denom) < 1e-12:
        # Portfolio completely wiped out; all position values go to zero.
        return pd.Series(0.0, index=weights.index)
    rets_aligned = period_returns.reindex(weights.index, fill_value=0.0)
    drifted = weights * (1.0 + rets_aligned) / denom
    return drifted.clip(lower=0.0)


def _build_result(records: list[dict], flat_haircut_bps: float, rf_annual: float = 0.0) -> BacktestResult:
    """Assemble BacktestResult from period records."""
    df = pd.DataFrame(records).set_index("date")
    df.index = pd.DatetimeIndex(df.index)

    returns = df["net_return"].astype(float)
    costs_nzd = df["cost_nzd"].astype(float)
    turnover = df["turnover"].astype(float)
    drawdown = compute_drawdown_series(returns)

    n = len(returns)
    mu = float(returns.mean()) * 12.0
    vol = float(returns.std(ddof=1)) * np.sqrt(12.0) if n > 1 else 0.0
    sharpe_raw = ((mu - rf_annual) / vol) if vol > 1e-12 else 0.0
    haircut = flat_haircut_bps / 10_000.0
    sharpe_hc = ((mu - rf_annual - haircut) / vol) if vol > 1e-12 else 0.0

    # Additional statistics
    hit_rate = float((returns > 0).mean()) if n > 0 else 0.0

    try:
        from scipy import stats as scipy_stats
        skewness = float(scipy_stats.skew(returns.values, bias=False)) if n >= 3 else 0.0
    except Exception:
        skewness = 0.0

    mdd = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar_ratio = ((mu - rf_annual) / abs(mdd)) if abs(mdd) > 1e-12 else 0.0
    
    period_n_positions = df["n_positions"].astype(int)

    return BacktestResult(
        returns=returns,
        costs_nzd=costs_nzd,
        turnover=turnover,
        drawdown=drawdown,
        sharpe_raw=sharpe_raw,
        sharpe_flat_haircut=sharpe_hc,
        start=pd.Timestamp(returns.index[0]),
        end=pd.Timestamp(returns.index[-1]),
        n_periods=n,
        avg_positions=float(df["n_positions"].mean()),
        hit_rate=hit_rate,
        skewness=skewness,
        calmar_ratio=calmar_ratio,
        period_n_positions=period_n_positions,
    )
