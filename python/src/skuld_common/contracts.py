"""Core data contract types for Skuld pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class PITSnapshot:
    """All values knowable strictly before `asof`. Enforced, not asked nicely.

    Attributes:
        prices: index=date, columns=ticker, values=adj_close
        volumes: index=date, columns=ticker, values=volume
        fundamentals: MultiIndex (ticker, publication_date), columns=feature
        macro: index=date, columns=macro_feature
        corporate_actions: columns: ticker, ex_date, type, factor
        asof: the timestamp this snapshot was built for
        sector_labels: columns: ticker (str), date (Timestamp), sector (str).
            Extracted from ``gics_sector`` rows in the CSV.  Yahoo-sourced
            labels are current/backfilled classifications — NOT PIT-safe
            historical membership.  An empty DataFrame means no sector data
            is present.  The PIT invariant is deliberately NOT enforced for
            this field; downstream code is responsible for treating
            sector-derived outputs as diagnostic-only when labels are not
            dated or are backfilled.
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    fundamentals: pd.DataFrame
    macro: pd.DataFrame
    corporate_actions: pd.DataFrame
    asof: pd.Timestamp
    sector_labels: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        asof_naive = self.asof.tz_localize(None) if self.asof.tzinfo else self.asof
        violations: list[str] = []

        # Check index-based frames: prices, volumes, macro
        for name, df in [("prices", self.prices), ("volumes", self.volumes), ("macro", self.macro)]:
            if not df.empty and len(df.index) > 0:
                max_date = pd.Timestamp(df.index.max())
                if max_date.tzinfo:
                    max_date = max_date.tz_localize(None)
                if max_date >= asof_naive:
                    violations.append(
                        f"{name}: max date {max_date} >= asof {self.asof}"
                    )

        # Check fundamentals (MultiIndex with publication_date level)
        if not self.fundamentals.empty and len(self.fundamentals.index) > 0:
            pub_dates = self.fundamentals.index.get_level_values("publication_date")
            max_pub = pd.Timestamp(pub_dates.max())
            if max_pub.tzinfo:
                max_pub = max_pub.tz_localize(None)
            if max_pub >= asof_naive:
                violations.append(
                    f"fundamentals: max publication_date {max_pub} >= asof {self.asof}"
                )

        # Check corporate_actions (ex_date column)
        if not self.corporate_actions.empty and "ex_date" in self.corporate_actions.columns:
            max_ex = pd.Timestamp(self.corporate_actions["ex_date"].max())
            if max_ex.tzinfo:
                max_ex = max_ex.tz_localize(None)
            if max_ex >= asof_naive:
                violations.append(
                    f"corporate_actions: max ex_date {max_ex} >= asof {self.asof}"
                )

        if violations:
            raise ValueError(
                "PIT invariant violated — no future data allowed.\n"
                + "\n".join(f"  - {v}" for v in violations)
            )


@dataclass(frozen=True)
class PreparedPanel:
    """Cleaned, aligned panel ready for factor computation.

    Attributes:
        returns_daily: index=date (business days), columns=ticker, total return
        returns_monthly: index=month-end date, columns=ticker, total return
        market_cap: index=date, columns=ticker, NZD
        sector: index=ticker, values=GICS sector or 'Unknown'
        universe_mask: index=rebalance_date, columns=ticker, bool
        macro: index=date, columns=macro_feature (e.g., interest rates)
        fundamentals: MultiIndex (ticker, publication_date), columns=feature
        prices: optional index=date, columns=ticker, adjusted close prices after
            any prepared-panel masking
        volumes: optional index=date, columns=ticker, daily trading volume aligned
            to the same calendar-daily index as prices
        corporate_actions: optional dividend/split event table filtered PIT
        asof: the timestamp this panel was built for
    """

    returns_daily: pd.DataFrame
    returns_monthly: pd.DataFrame
    market_cap: pd.DataFrame
    sector: pd.Series
    universe_mask: pd.DataFrame
    macro: pd.DataFrame
    asof: pd.Timestamp
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    volumes: pd.DataFrame = field(default_factory=pd.DataFrame)
    corporate_actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_cap_proxy: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        violations: list[str] = []

        # All wide frames must share the same ticker columns.
        expected_tickers = sorted(self.returns_daily.columns.tolist())
        for name, df in [
            ("returns_monthly", self.returns_monthly),
            ("market_cap", self.market_cap),
        ]:
            actual = sorted(df.columns.tolist())
            if actual != expected_tickers:
                violations.append(
                    f"{name}.columns != returns_daily.columns "
                    f"(extra: {set(actual) - set(expected_tickers)}, "
                    f"missing: {set(expected_tickers) - set(actual)})"
                )

        # sector must cover every ticker in the wide frames.
        sector_tickers = sorted(self.sector.index.tolist())
        if sector_tickers != expected_tickers:
            violations.append(
                f"sector.index != returns_daily.columns "
                f"(extra: {set(sector_tickers) - set(expected_tickers)}, "
                f"missing: {set(expected_tickers) - set(sector_tickers)})"
            )

        # universe_mask columns must be a subset of the wide-frame tickers.
        mask_tickers = set(self.universe_mask.columns.tolist())
        extra = mask_tickers - set(expected_tickers)
        if extra:
            violations.append(
                f"universe_mask.columns contains tickers not in returns_daily: {extra}"
            )

        if violations:
            raise ValueError(
                "PreparedPanel alignment invariant violated:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )


@dataclass(frozen=True)
class CombinedScores:
    """Combined cross-sectional scores output from Stage 4 (signal combiner).

    Attributes:
        scores: Combined z-score per ticker. NaN-free over the scored universe.
        component_scores: Per-factor post-shrinkage z-scores (NaN → 0 imputed),
            columns = factor names.
        asof: Rebalance date this was computed for.
    """

    scores: pd.Series
    component_scores: pd.DataFrame
    asof: pd.Timestamp

    def __post_init__(self) -> None:
        if self.scores.isna().any():
            bad = self.scores[self.scores.isna()].index.tolist()
            raise ValueError(
                f"CombinedScores.scores must be NaN-free; NaN found for: {bad}"
            )


@dataclass(frozen=True)
class TargetPortfolio:
    """Target portfolio weights from Stage 5 (portfolio constructor).

    Attributes:
        weights: Per-ticker target weights, summing to (1 - cash_weight).
            All values non-negative.
        cash_weight: Fraction held as cash, in [0, 1]. Always >= cash_floor.
        method: Optimisation method used (e.g., 'RiskParity', 'EqualWeight').
        asof: Rebalance date this was constructed for.
    """

    weights: pd.Series
    cash_weight: float
    method: str
    asof: pd.Timestamp

    def __post_init__(self) -> None:
        violations: list[str] = []
        if (self.weights < -1e-8).any():
            bad = self.weights[self.weights < -1e-8].index.tolist()
            violations.append(f"negative weights for: {bad}")
        total = self.weights.sum()
        expected = 1.0 - self.cash_weight
        if abs(total - expected) > 1e-4:
            violations.append(
                f"weights sum {total:.6f} != 1 - cash_weight ({expected:.6f})"
            )
        if violations:
            raise ValueError(
                "TargetPortfolio invariant violated:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )


@dataclass(frozen=True)
class BacktestResult:
    """Results from one period of backtesting (a fold or the full history).

    Attributes:
        returns: net monthly returns after costs (index = month-end date).
        costs_nzd: total transaction cost per period in NZD.
        turnover: one-sided portfolio turnover per period (0–1 fraction of NAV).
        drawdown: rolling drawdown from peak (0 at peak; negative below peak).
        sharpe_raw: annualised Sharpe ratio of net returns.
        sharpe_flat_haircut: Sharpe after subtracting 400 bps/yr flat haircut.
        start: first rebalance date included.
        end: last rebalance date included.
        n_periods: number of rebalance periods.
        avg_positions: average number of non-zero equity positions per period.
        hit_rate: fraction of rebalance periods with positive net return (0–1).
        skewness: sample skewness of net monthly returns (negative = left-tailed).
        calmar_ratio: annualised excess return divided by absolute max drawdown.
        period_n_positions: number of non-zero equity positions by period.
        gross_returns: pre-cost portfolio returns by period.
        spread_costs_nzd: spread cost component by period.
        sharesies_fee_nzd: Sharesies fee component by period.
        cost_drag: total cost divided by pre-period NAV by period.
        equity_weight: realised post-trade equity weight before return drift.
        cash_weight: realised post-trade cash weight before return drift.
        executed_volume_nzd: absolute trade volume executed by period.
        deferred_volume_nzd: absolute trade volume deferred by execution policy.
        excess_volume_nzd: executed volume above the monthly execution budget.
        cap_binding_count: number of tickers at or above max_position cap per period.
    """

    returns: pd.Series
    costs_nzd: pd.Series
    turnover: pd.Series
    drawdown: pd.Series
    sharpe_raw: float
    sharpe_flat_haircut: float
    start: pd.Timestamp
    end: pd.Timestamp
    n_periods: int
    avg_positions: float
    hit_rate: float = 0.0
    skewness: float = 0.0
    calmar_ratio: float = 0.0
    period_n_positions: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    gross_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    spread_costs_nzd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sharesies_fee_nzd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cost_drag: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_weight: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cash_weight: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    executed_volume_nzd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    deferred_volume_nzd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    excess_volume_nzd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cap_binding_count: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))

    def __post_init__(self) -> None:
        if self.n_periods < 0:
            raise ValueError("n_periods must be non-negative")
        if self.avg_positions < 0:
            raise ValueError("avg_positions must be non-negative")


@dataclass(frozen=True)
class FoldResult:
    """One OOS fold from a walk-forward evaluation.

    Attributes:
        fold_id: 0-based fold index.
        test_start: first rebalance date in the OOS window.
        test_end: last rebalance date in the OOS window.
        result: BacktestResult for the OOS period.
    """

    fold_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    result: BacktestResult

    def __post_init__(self) -> None:
        if self.fold_id < 0:
            raise ValueError("fold_id must be non-negative")
        if self.test_start >= self.test_end:
            raise ValueError("test_start must be before test_end")


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregated walk-forward results across all OOS folds.

    Attributes:
        folds: tuple of FoldResult, one per OOS window (chronological order).
        oos_returns: concatenated OOS net monthly returns.
        oos_sharpe_raw: Sharpe of oos_returns (annualised).
        oos_sharpe_flat_haircut: Sharpe after 400 bps/yr flat haircut.
        oos_sharpe_delisting_adjusted: Sharpe after probabilistic delisting drag.
        oos_drawdown_observed: rolling max-drawdown series of oos_returns.
        oos_max_drawdown_observed: worst observed drawdown (scalar, ≤ 0).
        oos_max_drawdown_augmented_median: median MC-augmented max drawdown (≤ 0).
        oos_max_drawdown_augmented_p90: 90th-pct MC-augmented max drawdown (≤ 0).
        oos_avg_turnover: average monthly one-sided turnover across all folds.
        oos_total_cost_nzd: total NZD cost over all OOS periods (at initial_nav assumption).
        oos_hit_rate: fraction of OOS rebalance periods with positive net return (0–1).
        oos_skewness: sample skewness of OOS net monthly returns (negative = left-tailed).
        oos_calmar_ratio: annualised OOS excess return divided by absolute max OOS drawdown.
    """

    folds: tuple[FoldResult, ...]
    oos_returns: pd.Series
    oos_sharpe_raw: float
    oos_sharpe_flat_haircut: float
    oos_sharpe_delisting_adjusted: float
    oos_drawdown_observed: pd.Series
    oos_max_drawdown_observed: float
    oos_max_drawdown_augmented_median: float
    oos_max_drawdown_augmented_p90: float
    oos_avg_turnover: float
    oos_total_cost_nzd: float
    oos_hit_rate: float = 0.0
    oos_skewness: float = 0.0
    oos_calmar_ratio: float = 0.0
    oos_sharpe_stationary_bootstrap_ci: tuple[float, float] = (float("nan"), float("nan"))
    oos_sharpe_deflated: float | None = None
    n_kept_folds: int = 0
    n_rejected_folds: int = 0
    rejection_reasons: tuple[str, ...] = ()
    oos_sharpe_by_regime: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CurrentPortfolio:
    """Current holdings portfolio for execution planning.

    Attributes:
        holdings: number of shares held per ticker (index=ticker, int shares).
        prices: last close price per ticker (index=ticker, NZD).
        cash_nzd: cash balance in NZD.
    """

    holdings: pd.Series
    prices: pd.Series
    cash_nzd: float

    def __post_init__(self) -> None:
        violations: list[str] = []

        # Holdings and prices indices must align
        holdings_tickers = set(self.holdings.index.tolist())
        prices_tickers = set(self.prices.index.tolist())
        if holdings_tickers != prices_tickers:
            extra_holdings = holdings_tickers - prices_tickers
            extra_prices = prices_tickers - holdings_tickers
            if extra_holdings:
                violations.append(f"holdings has tickers without prices: {extra_holdings}")
            if extra_prices:
                violations.append(f"prices has tickers without holdings: {extra_prices}")

        # No negative shares
        if (self.holdings < 0).any():
            bad = self.holdings[self.holdings < 0].index.tolist()
            violations.append(f"negative shares for: {bad}")

        # Cash non-negative
        if self.cash_nzd < 0:
            violations.append(f"cash_nzd must be >= 0, got {self.cash_nzd}")

        if violations:
            raise ValueError(
                "CurrentPortfolio invariant violated:\n"
                + "\n".join(f"  - {v}" for v in violations)
            )


@dataclass(frozen=True)
class TradeList:
    """Execution plan: one row per ticker plus metadata.

    Attributes:
        trades: DataFrame with columns per §3.8 of implementation plan.
            Required columns: ticker, action, current_shares, target_shares,
            delta_shares, current_value_nzd, target_value_nzd, delta_value_nzd,
            est_round_trip_cost_nzd, in_no_trade_region, below_size_floor,
            deferred_to_next_month, sharesies_fee_band.
        total_volume_nzd: sum of |delta_value_nzd|.
        total_estimated_cost_nzd: sum of est_round_trip_cost_nzd + Sharesies fee.
        asof: rebalance date this plan was generated for.
        config_hash: SHA-256 hash of the spec that produced this plan.
    """

    trades: pd.DataFrame
    total_volume_nzd: float
    total_estimated_cost_nzd: float
    asof: pd.Timestamp
    config_hash: str

    _REQUIRED_COLUMNS: frozenset[str] = frozenset([
        "ticker", "action", "current_shares", "target_shares", "delta_shares",
        "current_value_nzd", "target_value_nzd", "delta_value_nzd",
        "est_round_trip_cost_nzd", "in_no_trade_region", "below_size_floor",
        "deferred_to_next_month", "sharesies_fee_band",
    ])

    def __post_init__(self) -> None:
        actual_cols = set(self.trades.columns.tolist())
        missing = self._REQUIRED_COLUMNS - actual_cols
        if missing:
            raise ValueError(
                f"TradeList.trades missing required columns: {sorted(missing)}"
            )


@dataclass(frozen=True)
class ICReport:
    """Information coefficient report for one factor at one forward horizon.

    Attributes:
        factor_name: name of the factor scored.
        horizon_months: forward return horizon used (>=1).
        ic_series: index=rebalance_date, values=cross-sectional Spearman rank
            correlation between factor score at t and forward return over (t, t+H].
            NaN entries are dropped before aggregate stats.
        ic_mean: mean of ic_series.
        ic_std: standard deviation of ic_series (sample, ddof=1).
        ic_ir: ic_mean / ic_std * sqrt(12 / horizon_months) — annualised IR
            (matches Grinold/Kahn convention; horizon adjusts effective n).
        t_stat_newey_west: Newey–West t-stat on ic_mean with lag = horizon_months.
        n_obs: number of non-NaN monthly ICs.
        min_universe_per_date: minimum cross-sectional sample (n) used per IC obs;
            ICs computed on fewer than `min_cross_section` names are dropped.
    """

    factor_name: str
    horizon_months: int
    ic_series: pd.Series
    ic_mean: float
    ic_std: float
    ic_ir: float
    t_stat_newey_west: float
    n_obs: int
    min_universe_per_date: int


@dataclass(frozen=True)
class DecayReport:
    """Alpha decay across forward horizons for one factor.

    Attributes:
        factor_name: name of the factor.
        horizons: tuple of forward horizons in months (sorted ascending).
        ic_by_horizon: dict horizon -> ICReport.
        peak_horizon: argmax(ic_mean) across horizons.
    """

    factor_name: str
    horizons: tuple[int, ...]
    ic_by_horizon: dict[int, ICReport]
    peak_horizon: int


@dataclass(frozen=True)
class DecompositionReport:
    """OLS decomposition of strategy returns onto market + size + own factors.

    Attributes:
        regressors: column names of the design matrix (e.g. ['market', 'size_lmh',
            'momentum']).
        coefficients: dict regressor -> beta.
        t_stats: dict regressor -> Newey–West t-stat (lag = 3 monthly obs).
        residual_alpha_annualised: intercept × 12 (decimal).
        residual_alpha_t_stat: Newey–West t-stat on the intercept.
        r_squared: OLS R^2.
        n_obs: number of monthly observations regressed.
    """

    regressors: tuple[str, ...]
    coefficients: dict[str, float]
    t_stats: dict[str, float]
    residual_alpha_annualised: float
    residual_alpha_t_stat: float
    r_squared: float
    n_obs: int


@dataclass(frozen=True)
class BootstrapResult:
    """Stationary-bootstrap distribution of an annualised Sharpe.

    Attributes:
        point_estimate: annualised Sharpe of the original series.
        mean: mean of the bootstrap-resample Sharpes.
        ci_low_95: 2.5th percentile of bootstrap-resample Sharpes.
        ci_median: 50th percentile.
        ci_high_95: 97.5th percentile.
        n_resamples: number of bootstrap resamples performed.
        mean_block_len: geometric mean block length used.
    """

    point_estimate: float
    mean: float
    ci_low_95: float
    ci_median: float
    ci_high_95: float
    n_resamples: int
    mean_block_len: float


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Bailey & López de Prado deflated Sharpe.

    Attributes:
        sharpe_hat: input observed annualised Sharpe.
        sharpe_deflated: deflated annualised Sharpe estimate.
        p_value: two-sided p-value for the deflated estimate (Probabilistic Sharpe Ratio
            interpretation).
        n_obs: monthly OOS observations the input Sharpe was computed on.
        n_trials: total tried strategies (production_ledger + n_trials_prior).
        passes: True iff p_value ≤ alpha.
        alpha: gating significance level (default 0.05).
    """

    sharpe_hat: float
    sharpe_deflated: float
    p_value: float
    n_obs: int
    n_trials: int
    passes: bool
    alpha: float


@dataclass(frozen=True)
class ExcessReturnTestResult:
    """One-sided HAC test on strategy excess return versus a benchmark.

    Attributes:
        mean_excess_annual: annualised arithmetic mean of strategy minus benchmark.
        t_stat: HAC/Newey-West t-statistic for mean excess return.
        p_value: one-sided p-value for the alternative mean_excess > 0.
        passes: True iff the excess mean is positive and p_value <= alpha.
    """

    mean_excess_annual: float
    t_stat: float
    p_value: float
    passes: bool


@dataclass(frozen=True)
class DominanceResult:
    """Romano–Wolf stepwise multiple-testing for strategy-vs-benchmarks.

    Attributes:
        benchmark_names: names in order tested by Romano-Wolf. This excludes
            any TD/cash-hurdle benchmark routed through the dedicated
            excess-return gate.
        adjusted_p_values: dict benchmark -> Romano–Wolf adjusted p-value.
        dominates: dict benchmark -> bool (True iff strategy beats benchmark
            at family-wise alpha).
        alpha: family-wise significance level used.
        n_resamples: bootstrap resamples used in the step-down distribution.
    """

    benchmark_names: tuple[str, ...]
    adjusted_p_values: dict[str, float]
    dominates: dict[str, bool]
    alpha: float
    n_resamples: int


@dataclass(frozen=True)
class GatingDecision:
    """Result of `gating.evaluate(walk_forward_result, ledger, config)`.

    Attributes:
        passes: overall pass/fail.
        bars: dict bar_name -> (passed, reason). bar_name in
            {'sanity_floor', 'bootstrap_ci', 'deflated_sharpe',
             'td_excess_return', 'dominance_<benchmark>'}.
        deflated: DeflatedSharpeResult.
        bootstrap: BootstrapResult.
        dominance: DominanceResult | None. Contains only the risky benchmarks
            evaluated by Romano-Wolf, excluding the TD/cash-hurdle benchmark
            when that benchmark is split into `td_excess_return`.
        n_kept_folds: count from WalkForwardResult.
        n_rejected_folds: count from WalkForwardResult.
        rejection_reasons: tuple of reasons.
        notes: free-text diagnostic summary (deterministic).
    """

    passes: bool
    bars: dict[str, tuple[bool, str]]
    deflated: DeflatedSharpeResult
    bootstrap: BootstrapResult
    dominance: DominanceResult | None
    n_kept_folds: int
    n_rejected_folds: int
    rejection_reasons: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark run through the same metrics as the strategy.

    Attributes:
        name: human-readable label (e.g., "NZ TD floor (4% nominal)").
        wf_two_fold: WalkForwardResult from the existing 2-fold non-overlapping driver.
        wf_rolling: WalkForwardResult from rolling 5y/1y driver.
        coverage_start: first month in returns (may post-date panel start).
        coverage_end: last month in returns.
        notes: tuple of free-form caveats surfaced verbatim in the report
            (e.g., "FNZ.NZ history begins 2004-10; pre-2004 returns omitted").
    """

    name: str
    wf_two_fold: WalkForwardResult
    wf_rolling: WalkForwardResult
    coverage_start: pd.Timestamp
    coverage_end: pd.Timestamp
    notes: tuple[str, ...]


@dataclass(frozen=True)
class MethodologyReport:
    """Frozen aggregate of everything the markdown writer needs.

    Header (reproducibility):
        config_hash: SHA-256 placeholder until M7 lands ("pre-M7" until then).
        git_sha: short SHA from `git rev-parse --short HEAD`; "uncommitted" if dirty.
        asof: PIT cutoff used.
        panel_coverage_start / panel_coverage_end: panel returns_daily index bounds.
        master_seed: int; sub-seeds (bootstrap, MC delisting, optimiser tie-breakers)
            are derived deterministically from this via numpy.random.SeedSequence.
        n_trials_prior: copied from the gating config.
        rng_master_seed_note: the spawn formula used (committed to docs).

    Body:
        strategy_name: e.g., "mom-ar-spread".
        strategy_two_fold: WalkForwardResult.
        strategy_rolling: WalkForwardResult (gating reference).
        benchmarks: tuple of BenchmarkResult, one per benchmark, in display order.
        gating: GatingDecision (from M5).
        dominance: DominanceResult — Romano–Wolf adjusted p-values vs. the
            risky benchmarks only, computed on the rolling-driver OOS returns
            of the strategy and each risky benchmark (paired stationary
            bootstrap inside). The TD/cash-hurdle benchmark is reported via
            `gating.bars['td_excess_return']` instead.

    Footer:
        pass_fail: tuple of (bar_name, passed: bool, reason: str). Includes:
            "sanity floor (TD excess return)", "primary benchmark (NZX
            equal-weighted) via Romano–Wolf", "deflated Sharpe", and any
            other bars enabled.
    """

    config_hash: str
    git_sha: str
    asof: pd.Timestamp
    panel_coverage_start: pd.Timestamp
    panel_coverage_end: pd.Timestamp
    master_seed: int
    n_trials_prior: int
    rng_master_seed_note: str

    strategy_name: str
    strategy_two_fold: WalkForwardResult
    strategy_rolling: WalkForwardResult
    benchmarks: tuple[BenchmarkResult, ...]
    gating: GatingDecision
    dominance: DominanceResult

    pass_fail: tuple[tuple[str, bool, str], ...]
