"""Pydantic spec schema for pre-registered backtest configurations."""
from __future__ import annotations

import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UniverseSpec(BaseModel):
    """Universe filtering parameters."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_adv_dollars: float = 10_000.0
    # Mcap filter is OPT-IN. Default 0.0 disables the check so tickers without
    # shares-outstanding fundamentals (sparse pre-2022 in the NZX dataset) are
    # not silently excluded. Set > 0 to re-enable a hard size floor.
    min_market_cap_nzd: float = 0.0
    min_history_days: int = 126
    adv_window: int = 20
    mc_ffill_days: int = 5
    nzx_only: bool = True
    rebalance_freq: Literal["BME", "BQE"] = "BME"


class MomentumFactorSpec(BaseModel):
    """Momentum factor configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["momentum"] = "momentum"
    min_months: int = 11
    smoothing_months: int = 1


class LowVolatilityFactorSpec(BaseModel):
    """Low-volatility factor configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["low_vol"] = "low_vol"
    lookback_months: int = 12
    min_months: int = 6


class SizeFactorSpec(BaseModel):
    """Size factor configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["size"] = "size"


class DividendYieldFactorSpec(BaseModel):
    """Dividend-yield factor configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["dividend_yield"] = "dividend_yield"
    lookback_months: int = 12
    min_dividends: int = 1


class ReturnOnRiskFactorSpec(BaseModel):
    """Return-on-risk factor configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["return_on_risk"] = "return_on_risk"
    lookback_months: int = 12
    min_months: int = 6


# Discriminated union for factors (extensible for M8: value, quality, low_vol, size)
FactorSpec = Annotated[
    MomentumFactorSpec | LowVolatilityFactorSpec | SizeFactorSpec | DividendYieldFactorSpec | ReturnOnRiskFactorSpec,
    Field(discriminator="kind")
]


class CostSpec(BaseModel):
    """Transaction cost configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    spread_bps: float = 200.0
    sharesies_monthly_fee_nzd: float = 15.0
    sharesies_coverage_nzd: float = 5_000.0
    sharesies_excess_bps: float = 190.0
    # Spread modelling: "flat" uses spread_bps for every trade; "abdi_ranaldo"
    # estimates per-ticker per-side bps from raw OHLC via the Abdi-Ranaldo
    # (2017) estimator. Default "flat" preserves backward compatibility with
    # all existing pre-registered specs.
    spread_model: Literal["flat", "abdi_ranaldo"] = "flat"
    spread_estimator_window: int = 60
    spread_estimator_min_obs: int = 20
    spread_estimator_scale: float = 1.0
    spread_estimator_min_bps_per_side: float = 5.0


class BacktestEngineSpec(BaseModel):
    """Backtest engine parameters (mirrors BacktestConfig minus cost_config)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_nav_nzd: float = 10_000.0
    cash_floor: float = 0.05
    max_position: float = 0.25
    max_sector: float = 0.25
    min_names: int | None = Field(default=None, ge=1)
    score_lambda: float = 0.0
    no_trade_threshold_frac: float = 0.005
    size_floor_nzd: float = 50.0
    size_floor_cost_multiple: float = 5.0
    return_window_days: int = 252
    min_return_obs: int = 63
    adv_participation_cap: float | None = Field(default=0.01, ge=0.0, le=1.0)
    flat_haircut_bps: float = 400.0
    risk_free_annual: float = 0.0
    min_positions_per_month: int = 1
    degenerate_fold_max_empty_frac: float = 0.5
    turnover_budget_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    smoothing_alpha: float = Field(default=0.0, ge=0.0, lt=1.0)


class RollingDriverSpec(BaseModel):
    """Rolling walk-forward driver parameters."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    train_years: int = 5
    oos_years: int = 1
    step_years: int = 1


class WalkForwardSpec(BaseModel):
    """Walk-forward evaluation configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    two_fold_enabled: bool = True
    rolling: RollingDriverSpec = Field(default_factory=RollingDriverSpec)


class SurvivorshipSpec(BaseModel):
    """Survivorship bias adjustment parameters."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    monte_carlo_seeds: int = 500
    delisting_csv_relpath: str = "src/survivorship/nzx_delistings.csv"


class GatingSpec(BaseModel):
    """Statistical gating parameters."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = 0.05
    bootstrap_n_resamples: int = 2000
    dominance_n_resamples: int = 2000
    sanity_floor: float = 0.0


class BenchmarksSpec(BaseModel):
    """Benchmark configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    td_floor_default: float = 0.04
    nzx_eq_mcap_floor_nzd: float = 20e6
    nzx_eq_adv_floor_shares: int = 10_000
    sixty_forty_equity_proxy: str = "FNZ.NZ"
    sixty_forty_bond_macro_field: str = "long_term_interest_rates"
    sixty_forty_bond_duration_years: float = 0.0
    sixty_forty_flat_haircut_bps: float = 50.0


class OutputSpec(BaseModel):
    """Output and ledger configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    report_dir_relpath: str = "reports"
    ledger_scope: Literal["production", "exploration"] = "exploration"


class OverlayConfig(BaseModel):
    """Cash overlay configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["none", "nzx_ma200_agg_momentum"] = "none"
    defensive_cash_fraction: float = 0.30
    momentum_aggregate_lookback_months: int = 12


class ExecutionPolicySpec(BaseModel):
    """Cost-aware execution policy configuration."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["none", "volume_budget"] = "none"
    monthly_volume_budget_nzd: float | None = None
    min_trade_benefit_bps: float = 0.0
    excess_trade_benefit_bps: float = 190.0

    @model_validator(mode="after")
    def _require_budget_for_volume_policy(self) -> ExecutionPolicySpec:
        if self.kind == "volume_budget" and self.monthly_volume_budget_nzd is None:
            raise ValueError("monthly_volume_budget_nzd is required for volume_budget")
        return self


class ScrubbingSpec(BaseModel):
    """Daily-price scrubbing configuration.

    When `kind="round_trip"`, single-day prints whose return exceeds
    `threshold` AND whose two-day compounded return is within
    `reversal_tolerance` of zero are replaced with the geometric mean of
    the surrounding prints. When `kind="none"`, the scrubber is disabled
    and the spec field is omitted from the hash for backward compatibility.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["none", "round_trip"] = "none"
    threshold: float = 0.30
    reversal_tolerance: float = 0.10


class AnomalyFilterSpec(BaseModel):
    """Prepared-panel anomaly masking configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["none", "mask_extremes"] = "none"
    daily_abs_return_threshold: float = Field(default=2.0, ge=0.0)
    monthly_abs_return_threshold: float = Field(default=5.0, ge=0.0)
    volume_gate_threshold: float = Field(default=0.20, ge=0.0)
    require_volume_confirmation: bool = True
    corporate_action_buffer_days: int = Field(default=5, ge=0)
    # Tickers that still have more than this many extreme daily returns after
    # per-date masking are chronically mis-adjusted.  Their entire price series
    # is dropped to NaN so they cannot contaminate factor scores.
    # Set to 0 to disable the chronic-ticker pass.
    chronic_ticker_max_extreme_days: int = Field(default=5, ge=0)


class AdjustmentSpec(BaseModel):
    """Corporate-action adjustment audit/repair configuration.

    When ``kind="off"``, the adjustment layer is disabled entirely and the
    spec field is omitted from the hash for backward compatibility. When
    ``kind="audit"``, the loader runs :func:`audit_adjustments` and attaches
    the resulting :class:`AdjustmentAuditReport` to the returned ``RawData``
    without mutating prices. When ``kind="repair"``, the loader runs
    :func:`repair_adjustments` with ``policy`` (one of ``"off"``,
    ``"conservative"``, or ``"aggressive"``); the audit report is attached
    and the price panel is replaced with the repaired output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["off", "audit", "repair"] = "off"
    policy: Literal["off", "conservative", "aggressive"] = "conservative"
    dividend_residual_tol: float = 0.25
    split_residual_tol: float = 0.05
    unit_jump_tol: float = 0.02


class BacktestSpec(BaseModel):
    """Complete strategy backtest specification.

    Every field that influences numerical output is captured here.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    asof: datetime.date
    master_seed: int = 42
    n_trials_prior: int = 30
    passed_gating: bool = False

    universe: UniverseSpec = Field(default_factory=UniverseSpec)
    factors: list[FactorSpec] = Field(default_factory=list)
    cost: CostSpec = Field(default_factory=CostSpec)
    backtest: BacktestEngineSpec = Field(default_factory=BacktestEngineSpec)
    walk_forward: WalkForwardSpec = Field(default_factory=WalkForwardSpec)
    survivorship: SurvivorshipSpec = Field(default_factory=SurvivorshipSpec)
    gating: GatingSpec = Field(default_factory=GatingSpec)
    benchmarks: BenchmarksSpec = Field(default_factory=BenchmarksSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    overlay: OverlayConfig | None = None
    execution_policy: ExecutionPolicySpec = Field(default_factory=ExecutionPolicySpec)
    scrubbing: ScrubbingSpec | None = None
    anomaly_filter: AnomalyFilterSpec | None = None
    adjustments: AdjustmentSpec | None = None
