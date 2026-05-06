from __future__ import annotations

import csv
import datetime
from pathlib import Path

from skuld_research.config import BacktestSpec, CostSpec, GatingSpec, MomentumFactorSpec, OutputSpec
from skuld_research.config.spec import OverlayConfig
from skuld_research.experiments.factor_experiment import (
    ExperimentVariant,
    build_variants,
    run_variants,
)


def _spec(name: str) -> BacktestSpec:
    return BacktestSpec(
        name=name,
        asof=datetime.date(2026, 1, 1),
        factors=[MomentumFactorSpec()],
        gating=GatingSpec(bootstrap_n_resamples=10, dominance_n_resamples=10),
        output=OutputSpec(ledger_scope="exploration"),
    )


class _Rolling:
    n_kept_folds = 2
    n_rejected_folds = 0
    oos_sharpe_raw = 0.1
    oos_sharpe_delisting_adjusted = 0.2
    oos_returns = None
    oos_max_drawdown_observed = -0.1
    oos_avg_turnover = 0.4


class _Gating:
    passes = False
    bars = {
        "sanity_floor": (True, "Sharpe 0.10 > 0.00"),
        "deflated_sharpe": (False, "p=0.1234 > 0.05"),
    }


class _Result:
    spec_hash = "abcdef1234567890"
    strategy_rolling = _Rolling()
    gating = _Gating()


def test_run_variants_continues_after_one_variant_fails(tmp_path: Path):
    variants = [
        ExperimentVariant("ok-1", "lane", _spec("ok-1")),
        ExperimentVariant("bad", "lane", _spec("bad")),
        ExperimentVariant("ok-2", "lane", _spec("ok-2")),
    ]

    def runner(spec: BacktestSpec, **_kwargs: object) -> _Result:
        if spec.name == "bad":
            raise RuntimeError("synthetic failure")
        return _Result()

    summary = run_variants(
        variants,
        raw_csv_path=tmp_path / "data.csv",
        output_dir=tmp_path,
        runner=runner,
    )

    assert summary.completed == 2
    assert summary.failed == 1
    assert (tmp_path / "results.csv").exists()
    error_text = (tmp_path / "errors.log").read_text(encoding="utf-8")
    assert "RuntimeError: synthetic failure" in error_text
    assert error_text.count("[20") == 1


def test_run_variants_quick_limit_runs_only_first_variant(tmp_path: Path):
    calls: list[str] = []
    variants = [
        ExperimentVariant("first", "lane", _spec("first")),
        ExperimentVariant("second", "lane", _spec("second")),
    ]

    def runner(spec: BacktestSpec, **_kwargs: object) -> _Result:
        calls.append(spec.name)
        return _Result()

    summary = run_variants(
        variants,
        raw_csv_path=tmp_path / "data.csv",
        output_dir=tmp_path,
        runner=runner,
        max_variants=1,
    )

    assert calls == ["first"]
    assert summary.planned == 1
    assert summary.completed == 1


def test_run_variants_writes_gating_reason_and_bar_fields(tmp_path: Path):
    variants = [ExperimentVariant("first", "lane", _spec("first"))]

    def runner(spec: BacktestSpec, **_kwargs: object) -> _Result:
        return _Result()

    run_variants(
        variants,
        raw_csv_path=tmp_path / "data.csv",
        output_dir=tmp_path,
        runner=runner,
    )

    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert rows[0]["gating_reason"] == "deflated_sharpe: p=0.1234 > 0.05"
    assert rows[0]["gate_sanity_floor_pass"] == "True"
    assert rows[0]["gate_sanity_floor_reason"] == "Sharpe 0.10 > 0.00"
    assert rows[0]["gate_deflated_sharpe_pass"] == "False"
    assert rows[0]["gate_deflated_sharpe_reason"] == "p=0.1234 > 0.05"


def test_run_variants_writes_real_walk_forward_metric_fields(tmp_path: Path):
    variants = [ExperimentVariant("first", "lane", _spec("first"))]

    def runner(spec: BacktestSpec, **_kwargs: object) -> _Result:
        return _Result()

    run_variants(
        variants,
        raw_csv_path=tmp_path / "data.csv",
        output_dir=tmp_path,
        runner=runner,
    )

    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert rows[0]["max_drawdown"] == "-0.1"
    assert rows[0]["turnover_mean"] == "0.4"


def test_build_variants_does_not_duplicate_discovery_labels_without_overlay():
    variants = build_variants(_spec("base"), quick=False)
    semantic_specs = []
    for variant in variants:
        data = variant.spec.model_dump()
        data.pop("name", None)
        data.pop("description", None)
        semantic_specs.append(repr(data))

    assert len(semantic_specs) == len(set(semantic_specs))


def test_build_construction_variants_all_in_construction_lane():
    """Every variant produced by build_construction_variants should be in the 'construction' lane."""
    from skuld_research.experiments.factor_experiment import build_construction_variants

    base = _spec("cons-base")
    variants = build_construction_variants(base, quick=True)
    assert len(variants) > 0
    assert all(v.lane == "construction" for v in variants)


def test_build_construction_variants_no_duplicates():
    """build_construction_variants (quick) should produce no duplicate specs."""
    from skuld_research.experiments.factor_experiment import build_construction_variants

    base = _spec("cons-base")
    variants = build_construction_variants(base, quick=True)
    labels = [v.label for v in variants]
    assert len(labels) == len(set(labels))


def test_build_construction_variants_full_grid_count():
    """Full grid: 4×4×3×3×3×2 = 864 variants."""
    from skuld_research.experiments.factor_experiment import build_construction_variants

    base = _spec("cons-base")
    variants = build_construction_variants(base, quick=False)
    assert len(variants) == 4 * 4 * 3 * 3 * 3 * 2


def test_build_construction_variants_quick_grid_count():
    """Quick grid: 2×2×2×1×2×1 = 16 variants."""
    from skuld_research.experiments.factor_experiment import build_construction_variants

    base = _spec("cons-base")
    variants = build_construction_variants(base, quick=True)
    assert len(variants) == 2 * 2 * 2 * 1 * 2 * 1


def test_build_construction_variants_sweeps_max_position():
    """Variants should cover multiple distinct max_position values."""
    from skuld_research.experiments.factor_experiment import build_construction_variants

    base = _spec("cons-base")
    variants = build_construction_variants(base, quick=False)
    seen_max_pos = {v.spec.backtest.max_position for v in variants}
    assert len(seen_max_pos) == 4  # [0.15, 0.20, 0.25, 0.30]


def test_build_construction_variants_sweeps_rebalance_freq():
    """Variants should include both BME and BQE rebalance frequencies."""
    from skuld_research.experiments.factor_experiment import build_construction_variants

    base = _spec("cons-base")
    variants = build_construction_variants(base, quick=False)
    seen_freqs = {v.spec.universe.rebalance_freq for v in variants}
    assert seen_freqs == {"BME", "BQE"}


def test_build_variants_does_not_duplicate_refinement_specs_for_flat_spread_base():
    base = BacktestSpec(
        name="flat_base",
        asof=datetime.date(2026, 1, 1),
        factors=[MomentumFactorSpec()],
        cost=CostSpec(spread_model="flat"),
        overlay=OverlayConfig(kind="nzx_ma200_agg_momentum"),
        gating=GatingSpec(bootstrap_n_resamples=10, dominance_n_resamples=10),
        output=OutputSpec(ledger_scope="exploration"),
    )
    variants = build_variants(base, quick=False)
    refinement_specs = []
    for variant in variants:
        if variant.lane != "refinement":
            continue
        data = variant.spec.model_dump()
        data.pop("name", None)
        data.pop("description", None)
        refinement_specs.append(repr(data))

    assert len(refinement_specs) == len(set(refinement_specs))
