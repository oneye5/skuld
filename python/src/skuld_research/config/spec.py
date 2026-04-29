"""Pydantic spec schema for pre-registered backtest configurations."""
from __future__ import annotations

import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


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


# Discriminated union for factors (extensible for M8: value, quality, low_vol, size)
FactorSpec = Annotated[
    Union[MomentumFactorSpec, LowVolatilityFactorSpec, SizeFactorSpec],
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
    score_lambda: float = 0.0
    no_trade_threshold_frac: float = 0.005
    size_floor_nzd: float = 50.0
    size_floor_cost_multiple: float = 5.0
    return_window_days: int = 252
    min_return_obs: int = 63
    flat_haircut_bps: float = 400.0
    risk_free_annual: float = 0.0
    min_positions_per_month: int = 1
    degenerate_fold_max_empty_frac: float = 0.5


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


class BacktestSpec(BaseModel):
    """Complete pre-registered backtest specification.
    
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
