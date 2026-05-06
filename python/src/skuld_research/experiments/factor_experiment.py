from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from skuld_research.config import (
    BacktestSpec,
    DividendYieldFactorSpec,
    GatingSpec,
    LowVolatilityFactorSpec,
    MomentumFactorSpec,
    OutputSpec,
    ScrubbingSpec,
    SizeFactorSpec,
    load_spec,
    run_from_spec,
)
from skuld_research.config.spec import AdjustmentSpec


@dataclass(frozen=True)
class ExperimentVariant:
    label: str
    lane: str
    spec: BacktestSpec


@dataclass(frozen=True)
class ExperimentSummary:
    planned: int
    completed: int
    failed: int
    output_dir: Path


Runner = Callable[..., object]


RESULT_FIELDS = [
    "label",
    "lane",
    "status",
    "spec_hash",
    "oos_sharpe_raw",
    "oos_sharpe_delisting_adjusted",
    "total_return",
    "max_drawdown",
    "turnover_mean",
    "n_kept_folds",
    "n_rejected_folds",
    "gating_passes",
    "gating_reason",
    "gate_sanity_floor_pass",
    "gate_sanity_floor_reason",
    "gate_bootstrap_ci_pass",
    "gate_bootstrap_ci_reason",
    "gate_deflated_sharpe_pass",
    "gate_deflated_sharpe_reason",
    "gate_td_excess_return_pass",
    "gate_td_excess_return_reason",
    "gate_dominance_NZX_equal_weighted_pass",
    "gate_dominance_NZX_equal_weighted_reason",
    "gate_dominance_60_40_pass",
    "gate_dominance_60_40_reason",
    "error",
]


def _gate_field_prefix(bar_name: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in bar_name).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"gate_{safe}"


def _gating_fields(gating: object) -> dict[str, object]:
    bars = getattr(gating, "bars", {}) or {}
    failed = [f"{name}: {reason}" for name, (passed, reason) in bars.items() if not passed]
    fields: dict[str, object] = {"gating_reason": "; ".join(failed)}
    for name, (passed, reason) in bars.items():
        prefix = _gate_field_prefix(name)
        fields[f"{prefix}_pass"] = passed
        fields[f"{prefix}_reason"] = reason
    return fields


def _total_return(returns: object) -> object:
    if returns is None:
        return ""
    try:
        return float(np.prod(1.0 + returns) - 1.0)
    except Exception:  # noqa: BLE001 - CSV diagnostics should not abort completed variants.
        return ""


def _copy_spec(spec: BacktestSpec, name: str, **updates: object) -> BacktestSpec:
    data = spec.model_dump()
    data.update(updates)
    data["name"] = name
    data["description"] = f"Exploratory factor experiment variant: {name}"
    data["output"] = OutputSpec(ledger_scope="exploration").model_dump()
    return BacktestSpec.model_validate(data)


def _copy_model(model: object, **updates: object) -> object:
    return model.model_copy(update=updates)


def _fast_spec(spec: BacktestSpec) -> BacktestSpec:
    return _copy_spec(
        spec,
        spec.name,
        survivorship=_copy_model(spec.survivorship, monte_carlo_seeds=20),
        gating=GatingSpec(bootstrap_n_resamples=100, dominance_n_resamples=100),
    )


def build_variants(base_spec: BacktestSpec, quick: bool = False) -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = []
    base = _fast_spec(base_spec) if quick else base_spec

    factor_sets = [
        ("mom", [MomentumFactorSpec(min_months=11, smoothing_months=1)]),
        ("mom-s3", [MomentumFactorSpec(min_months=11, smoothing_months=3)]),
        ("mom-s6", [MomentumFactorSpec(min_months=11, smoothing_months=6)]),
        (
            "mom-lowvol",
            [MomentumFactorSpec(min_months=11, smoothing_months=3), LowVolatilityFactorSpec()],
        ),
        ("mom-size", [MomentumFactorSpec(min_months=11, smoothing_months=3), SizeFactorSpec()]),
        (
            "mom-divyield",
            [MomentumFactorSpec(min_months=11, smoothing_months=3), DividendYieldFactorSpec()],
        ),
        (
            "mom-lowvol-size",
            [
                MomentumFactorSpec(min_months=11, smoothing_months=3),
                LowVolatilityFactorSpec(),
                SizeFactorSpec(),
            ],
        ),
    ]
    rebalances = ["BME", "BQE"]
    mcap_floors = [0.0, 20_000_000.0]
    spread_models = ["flat", "abdi_ranaldo"]
    overlays = [None]
    if base.overlay is not None:
        overlays.append(base.overlay)

    if quick:
        factor_sets = factor_sets[:2]
        rebalances = ["BME"]
        mcap_floors = [0.0]
        spread_models = ["abdi_ranaldo"]
        overlays = [base.overlay]

    for factor_label, factors in factor_sets:
        for rebalance, mcap_floor, spread_model, overlay in itertools.product(
            rebalances, mcap_floors, spread_models, overlays
        ):
            mcap_label = int(mcap_floor / 1_000_000)
            label = f"disc-{factor_label}-{rebalance.lower()}-mcap{mcap_label}-{spread_model}"
            if overlay is not None:
                label += "-overlay"
            spec = _copy_spec(
                base,
                label,
                factors=[f.model_dump() for f in factors],
                universe=_copy_model(
                    base.universe,
                    rebalance_freq=rebalance,
                    min_market_cap_nzd=mcap_floor,
                ),
                cost=_copy_model(base.cost, spread_model=spread_model),
                overlay=overlay,
                scrubbing=ScrubbingSpec(kind="round_trip"),
                adjustments=base.adjustments or AdjustmentSpec(kind="audit"),
            )
            variants.append(ExperimentVariant(label, "discovery", spec))

    overlay_fractions = [0.15, 0.30, 0.45] if base.overlay is not None else []
    spread_scales = [0.75, 1.0, 1.25] if base.cost.spread_model == "abdi_ranaldo" else [1.0]
    no_trade_thresholds = [0.0025, 0.005, 0.01]
    if quick:
        overlay_fractions = [0.30] if base.overlay is not None else []
        spread_scales = [1.0]
        no_trade_thresholds = [0.005]

    for defensive_cash, spread_scale, no_trade in itertools.product(
        overlay_fractions, spread_scales, no_trade_thresholds
    ):
        label = f"refine-cash{int(defensive_cash * 100)}-spread{spread_scale:g}-nt{no_trade:g}"
        overlay = (
            _copy_model(base.overlay, defensive_cash_fraction=defensive_cash)
            if base.overlay
            else None
        )
        spec = _copy_spec(
            base,
            label,
            overlay=overlay,
            cost=_copy_model(base.cost, spread_estimator_scale=spread_scale),
            backtest=_copy_model(base.backtest, no_trade_threshold_frac=no_trade),
            scrubbing=ScrubbingSpec(kind="round_trip"),
            adjustments=base.adjustments or AdjustmentSpec(kind="audit"),
        )
        variants.append(ExperimentVariant(label, "refinement", spec))

    return variants


def build_construction_variants(
    base_spec: BacktestSpec,
    quick: bool = False,
) -> list[ExperimentVariant]:
    """Build a portfolio-construction sweep over ``mom-s8``-fixed factors.

    Keeps the base-spec factor signal frozen and varies only construction,
    sizing, and trading-rule parameters:

    - ``max_position``: per-name weight cap
    - ``score_lambda``: tilt weight toward higher scores
    - ``smoothing_alpha``: blend toward prior portfolio weights (reduces turnover)
    - ``no_trade_threshold_frac``: minimum position drift to trigger a rebalance
    - ``turnover_budget_frac``: one-sided monthly turnover budget (None = unlimited)
    - ``rebalance_freq``: BME (monthly) vs BQE (quarterly)

    Decision rule: improvements must clear the Phase 2 incremental bar
    (flat-haircut Sharpe +0.10 vs ``mom-s8``, paired CI lower bound >= 0)
    while not worsening turnover, capacity, or paired net-return stability.

    Args:
        base_spec: Frozen production spec (typically ``mom-s8``).  The factor
            list is carried over unchanged into every variant.
        quick: If True, use a reduced grid for fast iteration.

    Returns:
        List of ``ExperimentVariant`` objects, all in the ``"construction"`` lane.
    """
    variants: list[ExperimentVariant] = []
    base = _fast_spec(base_spec) if quick else base_spec

    max_positions = [0.15, 0.20, 0.25, 0.30] if not quick else [0.20, 0.25]
    score_lambdas = [0.0, 0.25, 0.5, 1.0] if not quick else [0.0, 0.5]
    smoothing_alphas = [0.0, 0.1, 0.2] if not quick else [0.0, 0.1]
    no_trade_thresholds = [0.0025, 0.005, 0.01] if not quick else [0.005]
    turnover_budgets: list[float | None] = [None, 0.20, 0.30] if not quick else [None, 0.30]
    rebalance_freqs = ["BME", "BQE"] if not quick else ["BME"]

    for max_pos, score_lam, smooth, no_trade, to_budget, rebalance in itertools.product(
        max_positions,
        score_lambdas,
        smoothing_alphas,
        no_trade_thresholds,
        turnover_budgets,
        rebalance_freqs,
    ):
        to_label = f"tb{int(to_budget * 100)}" if to_budget is not None else "tbNone"
        label = (
            f"cons"
            f"-maxpos{int(max_pos * 100)}"
            f"-lam{score_lam:g}"
            f"-smooth{smooth:g}"
            f"-nt{no_trade:g}"
            f"-{to_label}"
            f"-{rebalance.lower()}"
        )

        updated_backtest = _copy_model(
            base.backtest,
            max_position=max_pos,
            score_lambda=score_lam,
            smoothing_alpha=smooth,
            no_trade_threshold_frac=no_trade,
            turnover_budget_frac=to_budget,
        )
        updated_universe = _copy_model(base.universe, rebalance_freq=rebalance)

        spec = _copy_spec(
            base,
            label,
            backtest=updated_backtest,
            universe=updated_universe,
        )
        variants.append(ExperimentVariant(label, "construction", spec))

    return variants


def _ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_FIELDS).writeheader()


def _append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
        handle.flush()


def _result_row(variant: ExperimentVariant, result: object) -> dict[str, object]:
    rolling = result.strategy_rolling
    row = {
        "label": variant.label,
        "lane": variant.lane,
        "status": "ok",
        "spec_hash": result.spec_hash,
        "oos_sharpe_raw": rolling.oos_sharpe_raw,
        "oos_sharpe_delisting_adjusted": rolling.oos_sharpe_delisting_adjusted,
        "total_return": _total_return(getattr(rolling, "oos_returns", None)),
        "max_drawdown": getattr(rolling, "oos_max_drawdown_observed", ""),
        "turnover_mean": getattr(rolling, "oos_avg_turnover", ""),
        "n_kept_folds": rolling.n_kept_folds,
        "n_rejected_folds": rolling.n_rejected_folds,
        "gating_passes": result.gating.passes,
        "error": "",
    }
    row.update(_gating_fields(result.gating))
    return row


def run_variants(
    variants: Iterable[ExperimentVariant],
    *,
    raw_csv_path: Path,
    output_dir: Path,
    runner: Runner = run_from_spec,
    max_variants: int | None = None,
) -> ExperimentSummary:
    selected = list(variants)
    if max_variants is not None:
        selected = selected[:max_variants]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    errors_path = output_dir / "errors.log"
    manifest_path = output_dir / "manifest.json"
    _ensure_results_file(results_path)

    manifest_path.write_text(
        json.dumps(
            {
                "started_at": datetime.utcnow().isoformat() + "Z",
                "planned_variants": len(selected),
                "raw_csv_path": str(raw_csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    completed = 0
    failed = 0
    for index, variant in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {variant.lane}: {variant.label}", flush=True)
        try:
            result = runner(variant.spec, raw_csv_path=raw_csv_path, write_ledger=False)
            _append_row(results_path, _result_row(variant, result))
            completed += 1
        except Exception as exc:  # noqa: BLE001 - batch runner must isolate variant failures.
            failed += 1
            message = "".join(traceback.format_exception(exc))
            with errors_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{datetime.utcnow().isoformat()}Z] {variant.label}\n{message}\n")
            _append_row(
                results_path,
                {
                    "label": variant.label,
                    "lane": variant.lane,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    summary = ExperimentSummary(len(selected), completed, failed, output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary.__dict__ | {"output_dir": str(output_dir)}, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory factor-model experiment sweep")
    parser.add_argument(
        "--base-spec",
        type=Path,
        default=Path("configs/strategy-specs/candidates/mom-s6.yaml"),
    )
    parser.add_argument("--raw-csv", type=Path, default=Path("../data/data_long.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Small fail-fast variant set")
    parser.add_argument("--max-variants", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    python_root = Path(__file__).resolve().parents[3]
    base_spec_path = (
        args.base_spec if args.base_spec.is_absolute() else python_root / args.base_spec
    )
    raw_csv_path = args.raw_csv if args.raw_csv.is_absolute() else python_root / args.raw_csv
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir or python_root / "reports" / "experiments" / f"factor-sweep-{stamp}"
    )

    base_spec = load_spec(base_spec_path)
    variants = build_variants(base_spec, quick=args.quick)
    summary = run_variants(
        variants,
        raw_csv_path=raw_csv_path,
        output_dir=output_dir,
        max_variants=args.max_variants,
    )
    print(
        f"Complete: {summary.completed} ok, {summary.failed} failed, "
        f"results in {summary.output_dir / 'results.csv'}"
    )
    return 0 if summary.completed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
