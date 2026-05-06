"""Monthly rebalance backtest engine.

Iterates over panel.universe_mask rebalance dates, scores factor signals,
constructs portfolios, applies transaction costs, tracks NAV and weight drift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from skuld_common.contracts import BacktestResult, CombinedScores, PreparedPanel, TargetPortfolio
from skuld_research.backtest.metrics import compute_drawdown_series
from skuld_research.costs.model import CostBreakdown, CostConfig, CostModel
from skuld_research.execution.policy import ExecutionPolicyConfig, apply_execution_policy
from skuld_research.factors.combiner import combine_signals
from skuld_research.factors.protocols import SignalGenerator
from skuld_research.overlay.apply import apply_cash_overlay
from skuld_research.overlay.rules import NoOverlay, OverlayRule
from skuld_research.portfolio.optimizer import build_target_portfolio


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
    min_names: int | None = None
    score_lambda: float = 0.0
    no_trade_threshold_frac: float = 0.005   # 0.5% of NAV min drift to rebalance
    size_floor_nzd: float = 50.0              # minimum trade in NZD
    size_floor_cost_multiple: float = 5.0     # skip if trade < N × round-trip cost
    return_window_days: int = 252
    min_return_obs: int = 63
    adv_participation_cap: float | None = 0.01
    cost_config: CostConfig = field(default_factory=CostConfig)
    flat_haircut_bps: float = 400.0
    risk_free_annual: float = 0.0
    min_positions_per_month: int = 1  # NZX routinely has 4-name months; only reject if truly empty
    degenerate_fold_max_empty_frac: float = 0.5
    turnover_budget_frac: float | None = None
    smoothing_alpha: float = 0.0   # weight toward prior portfolio (0=no smoothing, 1=never trade)
    execution_policy: ExecutionPolicyConfig = field(default_factory=ExecutionPolicyConfig)
    adv_panel: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.max_position <= 1.0):
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")
        if self.adv_participation_cap is not None and not (0.0 <= self.adv_participation_cap <= 1.0):
            raise ValueError(
                "adv_participation_cap must be in [0, 1] or None, "
                f"got {self.adv_participation_cap}"
            )
        if self.min_names is not None and self.min_names < 1:
            raise ValueError(f"min_names must be >= 1 or None, got {self.min_names}")
        if self.turnover_budget_frac is not None and not (0.0 <= self.turnover_budget_frac <= 1.0):
            raise ValueError(
                "turnover_budget_frac must be in [0, 1] or None, "
                f"got {self.turnover_budget_frac}"
            )
        if self.turnover_budget_frac is not None and self.execution_policy.turnover_budget_frac is None:
            self.execution_policy = replace(
                self.execution_policy,
                turnover_budget_frac=self.turnover_budget_frac,
            )
        if not (0.0 <= self.smoothing_alpha < 1.0):
            raise ValueError(f"smoothing_alpha must be in [0, 1), got {self.smoothing_alpha}")


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

    def _adv_lookup(self, t: pd.Timestamp) -> pd.Series | None:
        """Return rebalance-date ADV in NZD for `t` (or None if unavailable)."""
        if self.config.adv_panel is None or self.config.adv_panel.empty:
            return None
        idx = self.config.adv_panel.index
        if t in idx:
            return self.config.adv_panel.loc[t]
        avail = idx[idx <= t]
        if avail.empty:
            return None
        return self.config.adv_panel.loc[avail[-1]]

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
                logger.warning(
                    "rebalance %s: universe collapsed to 0 tickers — holding cash", t.date()
                )
                nav_before_cost = nav
                # Force liquidation if we hold anything; then charge the subscription fee.
                if not current_weights.empty and (current_weights > 1e-6).any():
                    gross_return = 0.0

                    # Spread cost on the full liquidation value
                    liquidation_values = current_weights * nav
                    liq_cost = self.cost_model.compute_period_costs(
                        liquidation_values, per_ticker_spread_bps=spread_at_t,
                    )
                    liq_cost = _add_holding_period_subscription_fees(
                        liq_cost,
                        panel.returns_monthly,
                        t_naive,
                        next_t,
                        cfg.cost_config.sharesies_monthly_fee_nzd,
                    )
                    cost_drag = liq_cost.total_cost_nzd / nav_before_cost
                    net_return = gross_return - cost_drag
                    turnover = float(current_weights.sum())
                    current_weights = pd.Series(dtype=float)
                else:
                    # Holding cash; only subscription fee applies
                    gross_return = 0.0
                    liq_cost = self.cost_model.compute_period_costs(pd.Series(dtype=float))
                    liq_cost = _add_holding_period_subscription_fees(
                        liq_cost,
                        panel.returns_monthly,
                        t_naive,
                        next_t,
                        cfg.cost_config.sharesies_monthly_fee_nzd,
                    )
                    cost_drag = liq_cost.total_cost_nzd / nav_before_cost
                    net_return = -cost_drag
                    turnover = 0.0

                nav = nav * (1.0 + net_return)
                period_records.append({
                    "date": next_t,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "cost_nzd": liq_cost.total_cost_nzd,
                    "spread_cost_nzd": liq_cost.spread_cost_nzd,
                    "sharesies_fee_nzd": liq_cost.sharesies_fee_nzd,
                    "cost_drag": cost_drag,
                    "turnover": turnover,
                    "n_positions": 0,
                    "equity_weight": 0.0,
                    "cash_weight": 1.0,
                    "executed_volume_nzd": turnover * nav_before_cost,
                    "deferred_volume_nzd": 0.0,
                    "excess_volume_nzd": 0.0,
                    "cap_binding_count": 0,
                })
                continue

            if len(universe) < 5:
                logger.warning(
                    "rebalance %s: thin universe (%d tickers) — factor scores unreliable",
                    t.date(), len(universe),
                )

            # Score signals
            signals: dict[str, pd.Series] = {}
            for factor in self.factors:
                signals[factor.name] = factor.score(panel, t, universe)

            combined: CombinedScores = combine_signals(
                signals, universe, panel.sector, t
            )
            adv_at_t = self._adv_lookup(t_naive)

            target: TargetPortfolio = build_target_portfolio(
                combined, panel, t,
                cash_floor=cfg.cash_floor,
                max_position=cfg.max_position,
                max_sector=cfg.max_sector,
                min_names=cfg.min_names,
                score_lambda=cfg.score_lambda,
                adv=adv_at_t,
                portfolio_nav=nav,
                adv_participation_cap=cfg.adv_participation_cap,
                return_window_days=cfg.return_window_days,
                min_return_obs=cfg.min_return_obs,
            )

            # Count tickers whose weight is at or above the per-name cap.
            # A tolerance of 1e-4 absorbs floating-point redistribution artefacts.
            n_capped = int((target.weights >= cfg.max_position - 1e-4).sum())

            # Apply cash overlay (may raise cash beyond the floor)
            target = apply_cash_overlay(target, panel, self.overlay_rule, t)

            # Portfolio weight smoothing: blend target with current weights
            if cfg.smoothing_alpha > 0.0 and not current_weights.empty:
                blend_tickers = list(
                    set(list(target.weights.index)) | set(list(current_weights.index))
                )
                t_full = target.weights.reindex(blend_tickers, fill_value=0.0)
                c_full = current_weights.reindex(blend_tickers, fill_value=0.0)
                blended = (1.0 - cfg.smoothing_alpha) * t_full + cfg.smoothing_alpha * c_full
                # Drop near-zero positions
                blended = blended[blended > 1e-4]
                total_blended = blended.sum()
                if total_blended > 1e-12:
                    # Rescale to equity sleeve (1 - cash_floor)
                    blended = blended * (1.0 - cfg.cash_floor) / total_blended
                cash_weight = max(cfg.cash_floor, 1.0 - float(blended.sum()))
                target = replace(target, weights=blended, cash_weight=cash_weight)

            # All tickers we need to consider (target + current holdings)
            all_tickers = list(
                set(list(target.weights.index)) | set(list(current_weights.index))
            )
            target_full = target.weights.reindex(all_tickers, fill_value=0.0)
            current_full = current_weights.reindex(all_tickers, fill_value=0.0)
            delta_w = target_full - current_full
            # Existing holdings absent from the new target are forced exits;
            # otherwise stale positions can remain stranded behind trade filters.
            forced = (current_full > 1e-9) & (target_full <= 1e-9)

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

            skip = (in_ntr | below_floor) & ~forced
            executable_delta = delta_w.copy()
            executable_delta[skip] = 0.0

            executed_volume_nzd = float(executable_delta.abs().sum() * nav)
            deferred_volume_nzd = 0.0
            excess_volume_nzd = 0.0

            if cfg.execution_policy.enabled:
                policy_result = apply_execution_policy(
                    executable_delta,
                    nav_nzd=nav,
                    expected_alpha_bps=combined.scores * 100.0,
                    config=cfg.execution_policy,
                    forced=forced,
                )
                executable_delta = policy_result.executable_delta_weights
                executed_volume_nzd = policy_result.executed_volume_nzd
                deferred_volume_nzd = policy_result.deferred_volume_nzd
                excess_volume_nzd = policy_result.excess_volume_nzd

            # Transaction costs
            exec_abs_values = (executable_delta.abs() * nav)
            cost_bd: CostBreakdown = self.cost_model.compute_period_costs(
                exec_abs_values, per_ticker_spread_bps=spread_at_t,
            )
            cost_bd = _add_holding_period_subscription_fees(
                cost_bd,
                panel.returns_monthly,
                t_naive,
                next_t,
                cfg.cost_config.sharesies_monthly_fee_nzd,
            )

            # New actual weights
            new_weights = current_full + executable_delta
            new_weights = new_weights.clip(lower=0.0)
            # Prevent leverage: if skipping sells caused total equity > 100%, normalise down.
            total_w = new_weights.sum()
            if total_w > 1.0:
                new_weights = new_weights / total_w

            # Compound every monthly return in the holding period [t -> next_t].
            gross_return, drifted = _compound_period_returns(
                new_weights, panel.returns_monthly, t_naive, next_t
            )
            cost_drag = cost_bd.total_cost_nzd / nav
            net_return = gross_return - cost_drag

            # One-sided turnover (fraction of NAV traded)
            turnover = float(executable_delta.abs().sum()) / 2.0

            n_pos = int((new_weights > 1e-6).sum())
            period_records.append({
                "date": next_t,
                "gross_return": gross_return,
                "net_return": net_return,
                "cost_nzd": cost_bd.total_cost_nzd,
                "spread_cost_nzd": cost_bd.spread_cost_nzd,
                "sharesies_fee_nzd": cost_bd.sharesies_fee_nzd,
                "cost_drag": cost_drag,
                "turnover": turnover,
                "n_positions": n_pos,
                "equity_weight": float(new_weights.sum()),
                "cash_weight": max(0.0, 1.0 - float(new_weights.sum())),
                "executed_volume_nzd": executed_volume_nzd,
                "deferred_volume_nzd": deferred_volume_nzd,
                "excess_volume_nzd": excess_volume_nzd,
                "cap_binding_count": n_capped,
            })

            # Update NAV, drift weights
            nav = nav * (1.0 + net_return)
            drifted = _apply_cost_drag_to_weights(drifted, cost_drag)
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


def _compound_period_returns(
    weights: pd.Series,
    monthly_returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, pd.Series]:
    """Compound all monthly returns in ``(start, end]`` and drift weights."""
    if weights.empty:
        return 0.0, weights

    start_naive = start.tz_localize(None) if getattr(start, "tzinfo", None) else start
    end_naive = end.tz_localize(None) if getattr(end, "tzinfo", None) else end
    idx = monthly_returns.index
    period_idx = idx[(idx > start_naive) & (idx <= end_naive)]
    if period_idx.empty:
        return 0.0, weights

    drifted = weights.copy()
    compound = 1.0
    for month in period_idx:
        period_rets = monthly_returns.loc[month].reindex(drifted.index, fill_value=0.0)
        month_return = float((drifted * period_rets).sum())
        compound *= 1.0 + month_return
        drifted = _drift_weights(drifted, period_rets, month_return)

    return compound - 1.0, drifted


def _apply_cost_drag_to_weights(weights: pd.Series, cost_drag: float) -> pd.Series:
    """Re-express equity weights after fees reduce NAV, assuming fees come from cash."""
    if weights.empty or cost_drag <= 0.0:
        return weights
    denom = 1.0 - cost_drag
    if denom <= 1e-12:
        return pd.Series(0.0, index=weights.index)
    return (weights / denom).clip(lower=0.0)


def _holding_period_month_count(
    monthly_returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    """Count monthly return rows in ``(start, end]`` with a minimum of one fee month."""
    start_naive = start.tz_localize(None) if getattr(start, "tzinfo", None) else start
    end_naive = end.tz_localize(None) if getattr(end, "tzinfo", None) else end
    idx = monthly_returns.index
    return max(1, int(((idx > start_naive) & (idx <= end_naive)).sum()))


def _add_holding_period_subscription_fees(
    cost: CostBreakdown,
    monthly_returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    monthly_fee_nzd: float,
) -> CostBreakdown:
    """Add fixed monthly subscription fees beyond the one charged by CostModel."""
    extra_months = _holding_period_month_count(monthly_returns, start, end) - 1
    if extra_months <= 0 or monthly_fee_nzd == 0.0:
        return cost
    extra_fee = extra_months * monthly_fee_nzd
    return replace(
        cost,
        sharesies_fee_nzd=cost.sharesies_fee_nzd + extra_fee,
        total_cost_nzd=cost.total_cost_nzd + extra_fee,
    )


def _build_result(
    records: list[dict], flat_haircut_bps: float, rf_annual: float = 0.0
) -> BacktestResult:
    """Assemble BacktestResult from period records."""
    df = pd.DataFrame(records).set_index("date")
    df.index = pd.DatetimeIndex(df.index)

    returns = df["net_return"].astype(float)
    gross_returns = df["gross_return"].astype(float)
    costs_nzd = df["cost_nzd"].astype(float)
    spread_costs_nzd = df["spread_cost_nzd"].astype(float)
    sharesies_fee_nzd = df["sharesies_fee_nzd"].astype(float)
    cost_drag = df["cost_drag"].astype(float)
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
    equity_weight = df["equity_weight"].astype(float)
    cash_weight = df["cash_weight"].astype(float)
    executed_volume_nzd = df["executed_volume_nzd"].astype(float)
    deferred_volume_nzd = df["deferred_volume_nzd"].astype(float)
    excess_volume_nzd = df["excess_volume_nzd"].astype(float)
    cap_binding_count = df["cap_binding_count"].astype(int) if "cap_binding_count" in df.columns else pd.Series(0, index=df.index, dtype=int)

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
        gross_returns=gross_returns,
        spread_costs_nzd=spread_costs_nzd,
        sharesies_fee_nzd=sharesies_fee_nzd,
        cost_drag=cost_drag,
        equity_weight=equity_weight,
        cash_weight=cash_weight,
        executed_volume_nzd=executed_volume_nzd,
        deferred_volume_nzd=deferred_volume_nzd,
        excess_volume_nzd=excess_volume_nzd,
        cap_binding_count=cap_binding_count,
    )
