"""Run a backtest from a BacktestSpec."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from skuld_common.contracts import (
    BenchmarkResult,
    DominanceResult,
    GatingDecision,
    PITSnapshot,
    PreparedPanel,
    WalkForwardResult,
)
from skuld_research.backtest.engine import BacktestConfig
from skuld_research.backtest.walk_forward import FoldSpec, WalkForwardEngine
from skuld_research.benchmarks.nz_td_floor import nz_td_floor
from skuld_research.benchmarks.nzx_equal_weighted_fixed_universe import (
    nzx_equal_weighted_fixed_universe,
)
from skuld_research.benchmarks.sixty_forty import sixty_forty
from skuld_research.config.factors import build_factors_from_specs
from skuld_research.config.hashing import spec_hash
from skuld_research.config.loader import find_python_root
from skuld_research.config.spec import BacktestSpec
from skuld_research.costs.model import CostConfig
from skuld_research.costs.spread_estimator import compute_abdi_ranaldo_spread_panel
from skuld_research.data.csv_loader import load_raw_csv, load_raw_ohlc
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.data.scrubber import ScrubReport
from skuld_research.execution.policy import ExecutionPolicyConfig
from skuld_research.overlay import NoOverlay, NzxMA200AndAggMomentumRule, OverlayRule
from skuld_research.reporting._seeds import derive_child_seeds
from skuld_research.stats.gating import evaluate as evaluate_gating
from skuld_research.stats.ledger import TrialLedger
from skuld_research.stats.rolling_walk_forward import RollingWalkForwardEngine


def _compute_adv_panel(snap: PITSnapshot, panel: PreparedPanel, adv_window: int) -> pd.DataFrame:
    """Compute rolling dollar ADV from the PIT snapshot, aligned to panel dates."""
    if panel.prices.empty:
        return pd.DataFrame(index=panel.returns_daily.index, columns=panel.returns_daily.columns)

    prices_raw = panel.prices.sort_index()
    if not isinstance(prices_raw.index, pd.DatetimeIndex):
        return pd.DataFrame(index=panel.returns_daily.index, columns=panel.returns_daily.columns)
    prices_daily = prices_raw.resample("D").last()

    volumes_raw = snap.volumes.reindex(columns=prices_daily.columns).sort_index()
    if volumes_raw.empty or not isinstance(volumes_raw.index, pd.DatetimeIndex):
        volumes_daily = pd.DataFrame(index=prices_daily.index, columns=prices_daily.columns)
    else:
        volumes_daily = volumes_raw.resample("D").sum(min_count=1).reindex(prices_daily.index)

    dollar_volume = (prices_daily * volumes_daily).where(volumes_daily > 0)
    return dollar_volume.rolling(adv_window, min_periods=1).mean()


def _compute_share_adv_panel(snap: PITSnapshot, panel: PreparedPanel, adv_window: int) -> pd.DataFrame:
    """Compute rolling share ADV from the PIT snapshot, aligned to panel dates."""
    volumes_raw = snap.volumes.reindex(columns=panel.returns_daily.columns).sort_index()
    if volumes_raw.empty or not isinstance(volumes_raw.index, pd.DatetimeIndex):
        return pd.DataFrame(index=panel.returns_daily.index, columns=panel.returns_daily.columns)
    volumes_daily = volumes_raw.resample("D").sum(min_count=1).reindex(panel.returns_daily.index)
    return volumes_daily.where(volumes_daily > 0).rolling(adv_window, min_periods=1).mean()


def _label_spread_by_next_observation(spread_panel: pd.DataFrame) -> pd.DataFrame:
    if spread_panel.empty:
        return spread_panel
    relabelled = spread_panel.iloc[:-1].copy()
    relabelled.index = spread_panel.index[1:]
    return relabelled


@dataclass(frozen=True)
class RunResult:
    """Result of run_from_spec."""

    spec: BacktestSpec
    spec_hash: str
    panel: PreparedPanel
    strategy_two_fold: WalkForwardResult | None
    strategy_rolling: WalkForwardResult
    benchmarks: tuple[BenchmarkResult, ...]
    gating: GatingDecision
    dominance: DominanceResult
    master_seed: int
    panel_coverage_start: pd.Timestamp
    panel_coverage_end: pd.Timestamp
    scrub_report: ScrubReport


def run_from_spec(
    spec: BacktestSpec,
    *,
    raw_csv_path: Path | str,
    write_ledger: bool = True,
    ledger_root: Path | None = None,
) -> RunResult:
    """Run a complete backtest from a BacktestSpec.

    Args:
        spec: BacktestSpec instance.
        raw_csv_path: Path to data_long.csv.
        write_ledger: Whether to append to trial ledger (default True).
        ledger_root: Optional override for ledger root (defaults to python/trial_ledger).

    Returns:
        RunResult with all computed results.
    """
    # Compute spec hash
    h = spec_hash(spec)

    # Derive child seeds
    child_seeds = derive_child_seeds(spec.master_seed)

    # Load data
    raw = load_raw_csv(Path(raw_csv_path), scrub=spec.scrubbing, adjustments=spec.adjustments)
    snap = PITLoader(raw).as_of(pd.Timestamp(spec.asof, tz="UTC"))

    # Build panel
    panel = build_prepared_panel(
        snap,
        min_adv_dollars=spec.universe.min_adv_dollars,
        min_market_cap_nzd=spec.universe.min_market_cap_nzd,
        min_history_days=spec.universe.min_history_days,
        adv_window=spec.universe.adv_window,
        mc_ffill_days=spec.universe.mc_ffill_days,
        nzx_only=spec.universe.nzx_only,
        rebalance_freq=spec.universe.rebalance_freq,
        anomaly_filter=spec.anomaly_filter,
    )

    # Build BacktestConfig
    cost_config = CostConfig(
        spread_bps=spec.cost.spread_bps,
        sharesies_monthly_fee_nzd=spec.cost.sharesies_monthly_fee_nzd,
        sharesies_coverage_nzd=spec.cost.sharesies_coverage_nzd,
        sharesies_excess_bps=spec.cost.sharesies_excess_bps,
    )
    execution_policy = ExecutionPolicyConfig(
        volume_budget_nzd=(
            spec.execution_policy.monthly_volume_budget_nzd
            if spec.execution_policy.kind == "volume_budget"
            else None
        ),
        turnover_budget_frac=spec.backtest.turnover_budget_frac,
        min_trade_benefit_bps=(
            spec.execution_policy.min_trade_benefit_bps
            if spec.execution_policy.kind == "volume_budget"
            else 0.0
        ),
        excess_trade_benefit_bps=(
            spec.execution_policy.excess_trade_benefit_bps
            if spec.execution_policy.kind == "volume_budget"
            else 190.0
        ),
    )

    backtest_config = BacktestConfig(
        initial_nav_nzd=spec.backtest.initial_nav_nzd,
        cash_floor=spec.backtest.cash_floor,
        max_position=spec.backtest.max_position,
        max_sector=spec.backtest.max_sector,
        min_names=spec.backtest.min_names,
        score_lambda=spec.backtest.score_lambda,
        no_trade_threshold_frac=spec.backtest.no_trade_threshold_frac,
        size_floor_nzd=spec.backtest.size_floor_nzd,
        size_floor_cost_multiple=spec.backtest.size_floor_cost_multiple,
        return_window_days=spec.backtest.return_window_days,
        min_return_obs=spec.backtest.min_return_obs,
        adv_participation_cap=spec.backtest.adv_participation_cap,
        cost_config=cost_config,
        flat_haircut_bps=spec.backtest.flat_haircut_bps,
        risk_free_annual=spec.backtest.risk_free_annual,
        min_positions_per_month=spec.backtest.min_positions_per_month,
        degenerate_fold_max_empty_frac=spec.backtest.degenerate_fold_max_empty_frac,
        turnover_budget_frac=spec.backtest.turnover_budget_frac,
        smoothing_alpha=spec.backtest.smoothing_alpha,
        execution_policy=execution_policy,
        adv_panel=_compute_adv_panel(snap, panel, spec.universe.adv_window),
    )

    # Build spread panel if requested
    spread_panel: pd.DataFrame | None = None
    if spec.cost.spread_model == "abdi_ranaldo":
        high, low, close = load_raw_ohlc(
            Path(raw_csv_path), scrub=spec.scrubbing, adjustments=spec.adjustments
        )
        spread_panel = compute_abdi_ranaldo_spread_panel(
            high, low, close,
            window=spec.cost.spread_estimator_window,
            min_obs=spec.cost.spread_estimator_min_obs,
            scale=spec.cost.spread_estimator_scale,
            min_bps_per_side=spec.cost.spread_estimator_min_bps_per_side,
        )
        # The AR estimator at date t uses eta[t+1], so the row labelled t is
        # only safe once the next OHLC observation exists. Relabel each row to
        # that next observed source row, not merely the next business day.
        if isinstance(spread_panel.index, pd.DatetimeIndex):
            spread_panel = _label_spread_by_next_observation(spread_panel)

    # Build factors
    factors = build_factors_from_specs(spec.factors)

    # Build overlay rule
    overlay_rule: OverlayRule
    if spec.overlay is None or spec.overlay.kind == "none":
        overlay_rule = NoOverlay()
    elif spec.overlay.kind == "nzx_ma200_agg_momentum":
        overlay_rule = NzxMA200AndAggMomentumRule(
            defensive_cash_fraction=spec.overlay.defensive_cash_fraction,
            momentum_aggregate_lookback_months=spec.overlay.momentum_aggregate_lookback_months,
        )
    else:
        raise ValueError(f"Unknown overlay kind: {spec.overlay.kind}")

    # Resolve delisting CSV path
    python_root = find_python_root()
    delisting_csv_path = python_root / spec.survivorship.delisting_csv_relpath

    # Build 2-fold driver folds if enabled
    strategy_two_fold = None
    if spec.walk_forward.two_fold_enabled:
        rebalance_dates = panel.universe_mask.index.tolist()
        n = len(rebalance_dates)
        if n < 4:
            raise ValueError(f"Need at least 4 rebalance dates for 2-fold driver, got {n}")

        mid = n // 2
        folds_2fold = [
            FoldSpec(fold_id=0, test_start=rebalance_dates[1], test_end=rebalance_dates[mid - 1]),
            FoldSpec(fold_id=1, test_start=rebalance_dates[mid], test_end=rebalance_dates[-1]),
        ]

        strategy_two_fold = WalkForwardEngine(
            factors=factors,
            panel=panel,
            folds=folds_2fold,
            delisting_csv_path=delisting_csv_path,
            backtest_config=backtest_config,
            monte_carlo_seeds=spec.survivorship.monte_carlo_seeds,
            mc_rng_seed=child_seeds["mc_delisting"],
            overlay_rule=overlay_rule,
            spread_panel=spread_panel,
        ).run()

    # Run rolling walk-forward
    strategy_rolling = RollingWalkForwardEngine(
        panel=panel,
        factors=factors,
        train_years=spec.walk_forward.rolling.train_years,
        oos_years=spec.walk_forward.rolling.oos_years,
        step_years=spec.walk_forward.rolling.step_years,
        delisting_csv_path=delisting_csv_path,
        backtest_config=backtest_config,
        monte_carlo_seeds=spec.survivorship.monte_carlo_seeds,
        mc_rng_seed=child_seeds["mc_delisting"],
        overlay_rule=overlay_rule,
        spread_panel=spread_panel,
    ).run()

    # Build folds for benchmarks (only if 2-fold enabled)
    folds_2fold = None
    if spec.walk_forward.two_fold_enabled:
        rebalance_dates = panel.universe_mask.index.tolist()
        n = len(rebalance_dates)
        mid = n // 2
        folds_2fold = [
            FoldSpec(fold_id=0, test_start=rebalance_dates[1], test_end=rebalance_dates[mid - 1]),
            FoldSpec(fold_id=1, test_start=rebalance_dates[mid], test_end=rebalance_dates[-1]),
        ]

    # Run benchmarks
    _rolling_kwargs = dict(
        panel=panel,
        factors=[],
        train_years=spec.walk_forward.rolling.train_years,
        oos_years=spec.walk_forward.rolling.oos_years,
        step_years=spec.walk_forward.rolling.step_years,
        backtest_config=backtest_config,
        monte_carlo_seeds=spec.survivorship.monte_carlo_seeds,
        mc_rng_seed=child_seeds["mc_delisting"],
    )
    _2fold_kwargs = dict(
        factors=[],
        panel=panel,
        folds=folds_2fold,
        backtest_config=backtest_config,
        monte_carlo_seeds=spec.survivorship.monte_carlo_seeds,
        mc_rng_seed=child_seeds["mc_delisting"],
    ) if folds_2fold else None

    # 1. NZ TD floor
    td_floor_bt = nz_td_floor(panel, snap.asof, default_floor=spec.benchmarks.td_floor_default)
    td_floor_2fold = (
        WalkForwardEngine(**_2fold_kwargs, precomputed_returns=td_floor_bt.returns).run()
        if _2fold_kwargs
        else None
    )
    td_floor_rolling_wf = RollingWalkForwardEngine(
        **_rolling_kwargs, precomputed_returns=td_floor_bt.returns
    ).run()
    td_bench = BenchmarkResult(
        name="NZ TD floor",
        wf_two_fold=td_floor_2fold or td_floor_rolling_wf,  # fallback if no 2-fold
        wf_rolling=td_floor_rolling_wf,
        coverage_start=td_floor_bt.start,
        coverage_end=td_floor_bt.end,
        notes=(
            f"Notional term deposit, {spec.benchmarks.td_floor_default*100:.1f}% "
            "default floor",
        ),
    )

    # 2. NZX equal-weighted
    nzx_bt = nzx_equal_weighted_fixed_universe(
        panel,
        mcap_floor_nzd=spec.benchmarks.nzx_eq_mcap_floor_nzd,
        adv_floor_shares=spec.benchmarks.nzx_eq_adv_floor_shares,
        share_adv=_compute_share_adv_panel(snap, panel, spec.universe.adv_window),
        backtest_config=backtest_config,
    )
    nzx_2fold = (
        WalkForwardEngine(**_2fold_kwargs, precomputed_returns=nzx_bt.returns).run()
        if _2fold_kwargs
        else None
    )
    nzx_rolling_wf = RollingWalkForwardEngine(
        **_rolling_kwargs, precomputed_returns=nzx_bt.returns
    ).run()
    nzx_bench = BenchmarkResult(
        name="NZX equal-weighted",
        wf_two_fold=nzx_2fold or nzx_rolling_wf,
        wf_rolling=nzx_rolling_wf,
        coverage_start=nzx_bt.start,
        coverage_end=nzx_bt.end,
        notes=(
            f"{spec.benchmarks.nzx_eq_mcap_floor_nzd/1e6:.0f}M NZD mcap floor",
            f"{spec.benchmarks.nzx_eq_adv_floor_shares:,} share ADV floor",
        ),
    )

    # 3. 60/40
    sixty_forty_returns, sf_start, sf_end, sf_notes = sixty_forty(
        panel,
        equity_proxy=spec.benchmarks.sixty_forty_equity_proxy,
        bond_macro_field=spec.benchmarks.sixty_forty_bond_macro_field,
        duration_years=spec.benchmarks.sixty_forty_bond_duration_years,
        flat_haircut_bps=spec.benchmarks.sixty_forty_flat_haircut_bps,
    )
    sf_2fold = (
        WalkForwardEngine(**_2fold_kwargs, precomputed_returns=sixty_forty_returns).run()
        if _2fold_kwargs
        else None
    )
    sf_rolling_wf = RollingWalkForwardEngine(
        **_rolling_kwargs, precomputed_returns=sixty_forty_returns
    ).run()
    sf_bench = BenchmarkResult(
        name="60/40",
        wf_two_fold=sf_2fold or sf_rolling_wf,
        wf_rolling=sf_rolling_wf,
        coverage_start=sf_start,
        coverage_end=sf_end,
        notes=sf_notes,
    )

    benchmarks = (td_bench, nzx_bench, sf_bench)

    bench_oos_returns = {
        bench.name: bench.wf_rolling.oos_returns
        for bench in benchmarks
        if not bench.wf_rolling.oos_returns.empty
    }

    # Gating
    if ledger_root is None:
        ledger_root = python_root / "trial_ledger"

    ledger = TrialLedger(ledger_root, spec.output.ledger_scope)

    gating = evaluate_gating(
        strategy_rolling,
        ledger,
        benchmarks=bench_oos_returns,
        td_benchmark_name=td_bench.name,
        sanity_floor=spec.gating.sanity_floor,
        alpha=spec.gating.alpha,
        n_resamples=spec.gating.bootstrap_n_resamples,
        dominance_n_resamples=spec.gating.dominance_n_resamples,
        rng_seed=child_seeds["bootstrap"],
        n_trials_prior_override=spec.n_trials_prior,
        rf_annual=spec.backtest.risk_free_annual,
    )

    if gating.dominance is None:
        raise RuntimeError("Expected dominance result when benchmark returns are provided")
    dominance = gating.dominance

    # Write ledger entry if requested and not already present
    if write_ledger and not ledger.contains(h):
        entry = {
            "spec_hash": h,
            "spec_summary": f"{spec.name} @ {spec.asof}",
            "wf_sharpe": strategy_rolling.oos_sharpe_delisting_adjusted,
            "wf_n_obs": len(strategy_rolling.oos_returns),
            "kept_folds": strategy_rolling.n_kept_folds,
            "rejected_folds": strategy_rolling.n_rejected_folds,
            "git_sha": None,  # filled by caller if available
            "entered_at": datetime.utcnow().isoformat() + "Z",
        }
        ledger.append(entry)

    return RunResult(
        spec=spec,
        spec_hash=h,
        panel=panel,
        strategy_two_fold=strategy_two_fold,
        strategy_rolling=strategy_rolling,
        benchmarks=benchmarks,
        gating=gating,
        dominance=dominance,
        master_seed=spec.master_seed,
        panel_coverage_start=panel.returns_daily.index[0],
        panel_coverage_end=panel.returns_daily.index[-1],
        scrub_report=raw.scrub_report,
    )
