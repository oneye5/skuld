"""Empirical diagnostic of the corp-action adjustment audit/repair layer.

Read-only research script. Runs:
  Part A: audit prevalence over the full data_long.csv panel.
  Part B: backtest impact of conservative repair on
          mom-ar-spread-scrubbed.yaml.

Usage (from python/):
    uv run python scripts/diag_adjustments.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY_ROOT / "src"))

from skuld_research.config import load_spec, run_from_spec  # noqa: E402
from skuld_research.config.spec import AdjustmentSpec, ScrubbingSpec  # noqa: E402
from skuld_research.data.adjustments import audit_adjustments  # noqa: E402
from skuld_research.data.csv_loader import load_raw_csv, load_raw_ohlc  # noqa: E402

DATA_CSV = Path(r"D:\Projects\StandAloneProjects\skuld\data\data_long.csv")
SPEC_PATH = (
    PY_ROOT / "configs" / "strategy-specs" / "candidates" / "mom-ar-spread-scrubbed.yaml"
)


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def part_a() -> None:
    hr("PART A: audit prevalence")
    t0 = time.time()
    scrub = ScrubbingSpec(kind="round_trip", threshold=0.30, reversal_tolerance=0.10)
    raw = load_raw_csv(DATA_CSV, scrub=scrub, adjustments=AdjustmentSpec(kind="audit"))
    print(f"loaded prices: shape={raw.prices.shape}  elapsed={time.time()-t0:.1f}s")
    print(f"corp_actions rows={len(raw.corporate_actions)}")

    # The loader passes raw_close internally (verified in csv_loader.py:133-145)
    # so raw.adjustment_report already includes bad_div_adjust. But to be
    # explicit and to allow direct inspection we re-run audit_adjustments
    # ourselves with the same inputs.
    _, _, raw_close = load_raw_ohlc(DATA_CSV, scrub=scrub)
    print(f"raw_close panel shape={raw_close.shape}")

    report = audit_adjustments(
        raw.prices, raw.corporate_actions, raw_close=raw_close
    )
    ev = report.events
    print(f"\nTotal detected events: {len(ev)}")

    if ev.empty:
        print("(empty events frame)")
        return

    # 1. kind x severity pivot
    hr("A.1  kind x severity pivot")
    pivot = (
        ev.groupby(["kind", "severity"]).size().unstack(fill_value=0)
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot.loc["TOTAL"] = pivot.sum(axis=0)
    print(pivot.to_string())

    # 2. top 10 tickers
    hr("A.2  top 10 tickers by total event count")
    top = ev.groupby("ticker").size().sort_values(ascending=False).head(10)
    print(top.to_string())

    # 3. missed_split breakdown
    hr("A.3  missed_split events")
    ms = ev[ev["kind"] == "missed_split"].copy()
    print(f"count={len(ms)}")
    if not ms.empty:
        ms_t = ms.groupby("ticker").size().sort_values(ascending=False)
        print("\nby ticker:")
        print(ms_t.to_string())
        print("\nall events (ticker, ex_date, observed_ratio, expected_ratio):")
        cols = ["ticker", "ex_date", "observed_ratio", "expected_ratio", "residual"]
        print(ms[cols].to_string(index=False))

    # 4. unit_jump breakdown
    hr("A.4  unit_jump events")
    uj = ev[ev["kind"] == "unit_jump"].copy()
    print(f"count={len(uj)}")
    if not uj.empty:
        print("\nby ticker:")
        print(uj.groupby("ticker").size().sort_values(ascending=False).to_string())
        print("\nall events:")
        cols = ["ticker", "ex_date", "observed_ratio", "expected_ratio", "residual"]
        print(uj[cols].to_string(index=False))

    # 5. bad_div_adjust
    hr("A.5  bad_div_adjust events")
    bd = ev[ev["kind"] == "bad_div_adjust"].copy()
    bd_err = bd[bd["severity"] == "error"]
    bd_skip = bd[bd["severity"] == "skipped_no_raw"]
    print(f"total={len(bd)}  error={len(bd_err)}  skipped_no_raw={len(bd_skip)}")
    if not bd_err.empty:
        print("\nby ticker (error severity only):")
        print(
            bd_err.groupby("ticker").size().sort_values(ascending=False).head(15).to_string()
        )
        r = bd_err["residual"].astype(float)
        print(
            f"\nresidual: median={r.median():.4f}  p90={r.quantile(0.9):.4f}  "
            f"max={r.max():.4f}"
        )
        print("\ntop 10 by residual:")
        cols = [
            "ticker", "ex_date", "observed_ratio", "expected_ratio", "residual",
            "raw_close_prev", "raw_close_ex", "corp_action_factor",
        ]
        print(bd_err.nlargest(10, "residual")[cols].to_string(index=False))

    # 6. orphan + duplicate
    hr("A.6  orphan_action & duplicate_action counts")
    orph = ev[ev["kind"] == "orphan_action"]
    dup = ev[ev["kind"] == "duplicate_action"]
    print(f"orphan_action={len(orph)}")
    print(f"duplicate_action={len(dup)}")
    if not orph.empty:
        print("\norphan_action by corp_action_type:")
        print(orph.groupby("corp_action_type").size().to_string())
    if not dup.empty:
        print("\nduplicate_action by corp_action_type:")
        print(dup.groupby("corp_action_type").size().to_string())

    # 7. Sanity check: top 3 bad_div by residual, manual recompute
    hr("A.7  sanity: manually recompute expected_ratio for top bad_div_adjust")
    if not bd_err.empty:
        for _, row in bd_err.nlargest(3, "residual").iterrows():
            t = row["ticker"]
            d = pd.Timestamp(row["ex_date"])
            div = float(row["corp_action_factor"])
            rp = float(row["raw_close_prev"])
            re = float(row["raw_close_ex"])
            manual = (re - div) / rp if rp else float("nan")
            obs = float(row["observed_ratio"])
            stored_exp = float(row["expected_ratio"])
            print(
                f"  {t} {d.date()}  raw_prev={rp:.4f} raw_ex={re:.4f} div={div:.4f}"
                f" -> manual_expected={manual:.6f}  layer_expected={stored_exp:.6f}"
                f"  observed={obs:.6f}  match={np.isclose(manual, stored_exp)}"
            )


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def _summarise(name: str, result) -> dict[str, float]:
    wf = result.strategy_rolling
    rets = wf.oos_returns.dropna()
    n = len(rets)
    ann_factor = 12.0  # monthly OOS returns
    ann_ret = float((1 + rets).prod() ** (ann_factor / max(n, 1)) - 1) if n else float("nan")
    ann_vol = float(rets.std() * np.sqrt(ann_factor)) if n else float("nan")
    return {
        "name": name,
        "wf_sharpe_raw": wf.oos_sharpe_raw,
        "wf_sharpe_haircut": wf.oos_sharpe_flat_haircut,
        "wf_sharpe_delisting": wf.oos_sharpe_delisting_adjusted,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "max_dd_observed": wf.oos_max_drawdown_observed,
        "max_dd_aug_p90": wf.oos_max_drawdown_augmented_p90,
        "n_oos_months": float(n),
        "n_kept_folds": float(wf.n_kept_folds),
        "n_rejected_folds": float(wf.n_rejected_folds),
    }


def part_b() -> None:
    hr("PART B: backtest impact (mom-ar-spread-scrubbed + conservative repair)")
    spec = load_spec(SPEC_PATH)
    print(f"spec={spec.name}  asof={spec.asof}")
    print(f"baseline scrubbing={spec.scrubbing}")
    print(f"baseline adjustments={spec.adjustments}")

    # Capture the input price panels (post-scrub, pre/post repair) to count
    # cells changed.
    scrub = spec.scrubbing
    raw_baseline = load_raw_csv(DATA_CSV, scrub=scrub)
    raw_repair = load_raw_csv(
        DATA_CSV,
        scrub=scrub,
        adjustments=AdjustmentSpec(kind="repair", policy="conservative"),
    )
    p0 = raw_baseline.prices
    p1 = raw_repair.prices.reindex_like(p0)
    # Cells where both non-nan and differ
    both = p0.notna() & p1.notna()
    diff = both & ~np.isclose(p0.values, p1.values, rtol=1e-9, atol=1e-12)
    n_cells = int(diff.values.sum())
    n_tickers = int(diff.any(axis=0).sum())
    print(f"\nrepair impact on price panel: cells_changed={n_cells} tickers_affected={n_tickers}")

    rep_report = raw_repair.adjustment_report
    if rep_report is not None and not rep_report.events.empty:
        # The RepairResult.repairs ledger is not exposed via RawData, but the
        # audit events are. Recreate the ledger by direct call.
        from skuld_research.data.adjustments import repair_adjustments, RepairPolicy
        rr = repair_adjustments(
            raw_baseline.prices,
            raw_baseline.corporate_actions,
            raw_close=load_raw_ohlc(DATA_CSV, scrub=scrub)[2],
            policy=RepairPolicy.CONSERVATIVE,
        )
        print(f"\nRepairResult.repairs ledger: {len(rr.repairs)} rows")
        if not rr.repairs.empty:
            print(rr.repairs.to_string(index=False))

    # Run baseline
    hr("PART B.1  running baseline (kind=off)")
    t0 = time.time()
    baseline = run_from_spec(spec, raw_csv_path=DATA_CSV, write_ledger=False)
    print(f"baseline elapsed={time.time()-t0:.1f}s")
    base = _summarise("baseline", baseline)

    # Treatment
    hr("PART B.2  running treatment (kind=repair, policy=conservative)")
    spec_t = spec.model_copy(
        update={"adjustments": AdjustmentSpec(kind="repair", policy="conservative")}
    )
    t0 = time.time()
    treatment = run_from_spec(spec_t, raw_csv_path=DATA_CSV, write_ledger=False)
    print(f"treatment elapsed={time.time()-t0:.1f}s")
    treat = _summarise("treatment", treatment)

    hr("PART B.3  side-by-side")
    keys = [k for k in base.keys() if k != "name"]
    print(f"{'metric':<28} {'baseline':>14} {'treatment':>14} {'delta':>14}")
    print("-" * 72)
    for k in keys:
        b = base[k]
        t = treat[k]
        d = t - b if isinstance(b, (int, float)) and isinstance(t, (int, float)) else float("nan")
        print(f"{k:<28} {b:>14.6f} {t:>14.6f} {d:>+14.6f}")

    # Average monthly turnover proxy: not directly exposed in WalkForwardResult.
    # Note this and skip.
    print(
        "\nNote: monthly turnover and total trades are not surfaced on "
        "WalkForwardResult; they live on per-fold backtest reports inside the "
        "engine and would require deeper plumbing to extract. Reporting "
        "Sharpe/return/vol/MDD is the headline summary the runner exposes."
    )


def main() -> None:
    if not DATA_CSV.exists():
        sys.exit(f"missing data file: {DATA_CSV}")
    part_a()
    part_b()


if __name__ == "__main__":
    main()
