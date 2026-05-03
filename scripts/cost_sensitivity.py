"""Cost sensitivity sweep for mom-s6 strategy.

Sweeps flat_haircut_bps × turnover_budget_frac and prints pivot tables
of key metrics plus break-even analysis.

Usage:
    uv run python scripts/cost_sensitivity.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd

# Repo root is parent of this script
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_ROOT = _REPO_ROOT / "python"
sys.path.insert(0, str(_PYTHON_ROOT / "src"))

from skuld_research.config.loader import load_spec
from skuld_research.config.runner import run_from_spec

SPEC_PATH = _PYTHON_ROOT / "configs" / "strategy-specs" / "candidates" / "mom-s6.yaml"
DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"

FLAT_HAIRCUTS = [0, 100, 200, 400]
TURNOVER_BUDGETS = [0.10, 0.20, 0.30, 0.50]


def main() -> None:
    print("Loading base spec …")
    base_spec = load_spec(SPEC_PATH)

    # Build panel once by running the base spec and extracting the panel,
    # but run_from_spec rebuilds the panel each time internally.
    # To avoid reloading the CSV each run we pre-build once; but the runner
    # doesn't expose a panel-reuse API, so we accept the per-run CSV reload.
    # (Data loading is fast relative to the backtest computation.)

    records: list[dict] = []
    total = len(FLAT_HAIRCUTS) * len(TURNOVER_BUDGETS)
    idx = 0

    for haircut, turnover in product(FLAT_HAIRCUTS, TURNOVER_BUDGETS):
        idx += 1
        print(
            f"[{idx:2d}/{total}] flat_haircut_bps={haircut:4d}  "
            f"turnover_budget_frac={turnover:.2f} …",
            flush=True,
        )

        # Build modified spec via model_copy (pydantic v2)
        modified_backtest = base_spec.backtest.model_copy(
            update={"flat_haircut_bps": float(haircut), "turnover_budget_frac": turnover}
        )
        spec = base_spec.model_copy(update={"backtest": modified_backtest})

        result = run_from_spec(
            spec,
            raw_csv_path=DATA_PATH,
            write_ledger=False,
        )
        wf = result.strategy_rolling

        # Annualised mean net return (monthly returns × 12)
        mean_net_annual = float(wf.oos_returns.mean() * 12)

        # Calmar
        calmar = wf.oos_calmar_ratio

        records.append(
            {
                "flat_haircut_bps": haircut,
                "turnover_budget_frac": turnover,
                "sharpe_raw": round(wf.oos_sharpe_raw, 4),
                "sharpe_flat_haircut": round(wf.oos_sharpe_flat_haircut, 4),
                "mean_turnover": round(wf.oos_avg_turnover, 4),
                "mean_net_return_annual": round(mean_net_annual, 4),
                "calmar": round(calmar, 4),
            }
        )

    df = pd.DataFrame(records)

    # ── Pivot: sharpe_raw ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIVOT: sharpe_raw  (rows=flat_haircut_bps, cols=turnover_budget_frac)")
    print("=" * 70)
    pivot_sharpe = df.pivot(
        index="flat_haircut_bps",
        columns="turnover_budget_frac",
        values="sharpe_raw",
    )
    pivot_sharpe.columns.name = "turnover_budget"
    print(pivot_sharpe.to_string())

    # ── Pivot: mean_turnover ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIVOT: mean_turnover  (rows=flat_haircut_bps, cols=turnover_budget_frac)")
    print("=" * 70)
    pivot_to = df.pivot(
        index="flat_haircut_bps",
        columns="turnover_budget_frac",
        values="mean_turnover",
    )
    pivot_to.columns.name = "turnover_budget"
    print(pivot_to.to_string())

    # ── Pivot: mean_net_return_annual ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIVOT: mean_net_return_annual  (annualised, rows=flat_haircut_bps)")
    print("=" * 70)
    pivot_ret = df.pivot(
        index="flat_haircut_bps",
        columns="turnover_budget_frac",
        values="mean_net_return_annual",
    )
    pivot_ret.columns.name = "turnover_budget"
    print(pivot_ret.to_string())

    # ── Pivot: calmar ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIVOT: calmar  (rows=flat_haircut_bps, cols=turnover_budget_frac)")
    print("=" * 70)
    pivot_calmar = df.pivot(
        index="flat_haircut_bps",
        columns="turnover_budget_frac",
        values="calmar",
    )
    pivot_calmar.columns.name = "turnover_budget"
    print(pivot_calmar.to_string())

    # ── Break-even analysis ─────────────────────────────────────────────────
    SHARPE_TARGET = 0.3
    print("\n" + "=" * 70)
    print(f"BREAK-EVEN: smallest flat_haircut_bps where sharpe_raw > {SHARPE_TARGET}")
    print("=" * 70)
    for tb in TURNOVER_BUDGETS:
        sub = df[df["turnover_budget_frac"] == tb].sort_values("flat_haircut_bps")
        candidates = sub[sub["sharpe_raw"] > SHARPE_TARGET]["flat_haircut_bps"]
        if candidates.empty:
            print(f"  turnover_budget={tb:.2f}: never exceeds {SHARPE_TARGET} in sweep")
        else:
            # Largest haircut that still clears the target
            max_ok = candidates.max()
            print(f"  turnover_budget={tb:.2f}: sharpe_raw > {SHARPE_TARGET} up to haircut = {max_ok} bps")

    print("\nDone.")


if __name__ == "__main__":
    main()
