"""
score_lambda sweep: tests whether tilting ERC weights toward higher-scoring
stocks improves risk-adjusted returns for mom-s6.

Usage (from repo root):
    uv run python scripts/score_lambda_sweep.py

Sweeps score_lambda over [0.0, 0.25, 0.5, 1.0, 2.0] and prints a results
table plus a recommendation.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

from skuld_research.config.loader import load_spec
from skuld_research.config.factors import build_factors_from_specs
from skuld_research.data.csv_loader import load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
from skuld_research.costs.model import CostConfig

DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"
SPEC_PATH = _REPO_ROOT / "python" / "configs" / "strategy-specs" / "candidates" / "mom-s6.yaml"

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def main() -> None:
    print("Loading spec and data...")
    spec = load_spec(SPEC_PATH)
    raw = load_raw_csv(DATA_PATH, scrub=spec.scrubbing, adjustments=spec.adjustments)

    print(f"Building PIT snapshot (asof={spec.asof})...")
    snap = PITLoader(raw).as_of(pd.Timestamp(spec.asof, tz="UTC"))

    print(f"Building prepared panel ({SPEC_PATH.name})...")
    panel = build_prepared_panel(
        snap,
        min_adv_dollars=spec.universe.min_adv_dollars,
        min_market_cap_nzd=spec.universe.min_market_cap_nzd,
        min_history_days=spec.universe.min_history_days,
        adv_window=spec.universe.adv_window,
        mc_ffill_days=spec.universe.mc_ffill_days,
        nzx_only=spec.universe.nzx_only,
        anomaly_filter=spec.anomaly_filter,
    )

    cost_config = CostConfig(
        spread_bps=spec.cost.spread_bps,
        sharesies_monthly_fee_nzd=spec.cost.sharesies_monthly_fee_nzd,
        sharesies_coverage_nzd=spec.cost.sharesies_coverage_nzd,
        sharesies_excess_bps=spec.cost.sharesies_excess_bps,
    )

    factor_instances = build_factors_from_specs(spec.factors)

    print(f"\nSweeping score_lambda over {LAMBDAS}...\n")

    rows = []
    for lam in LAMBDAS:
        print(f"  Running score_lambda={lam}...")
        config = BacktestConfig(
            initial_nav_nzd=spec.backtest.initial_nav_nzd,
            cash_floor=spec.backtest.cash_floor,
            max_position=spec.backtest.max_position,
            max_sector=spec.backtest.max_sector,
            min_names=spec.backtest.min_names,
            score_lambda=lam,
            no_trade_threshold_frac=spec.backtest.no_trade_threshold_frac,
            size_floor_nzd=spec.backtest.size_floor_nzd,
            size_floor_cost_multiple=spec.backtest.size_floor_cost_multiple,
            return_window_days=spec.backtest.return_window_days,
            min_return_obs=spec.backtest.min_return_obs,
            cost_config=cost_config,
            flat_haircut_bps=spec.backtest.flat_haircut_bps,
            risk_free_annual=spec.backtest.risk_free_annual,
            min_positions_per_month=spec.backtest.min_positions_per_month,
            degenerate_fold_max_empty_frac=spec.backtest.degenerate_fold_max_empty_frac,
            turnover_budget_frac=spec.backtest.turnover_budget_frac,
        )
        engine = BacktestEngine(factors=factor_instances, panel=panel, config=config)
        result = engine.run()

        mean_net_ann = float(result.returns.mean()) * 12
        mean_turnover = float(result.turnover.mean())

        rows.append({
            "score_lambda": lam,
            "sharpe_raw": result.sharpe_raw,
            "sharpe_flat_hc": result.sharpe_flat_haircut,
            "calmar": result.calmar_ratio,
            "mean_net_ret_ann": mean_net_ann,
            "mean_turnover": mean_turnover,
            "hit_rate": result.hit_rate,
        })

    df = pd.DataFrame(rows).sort_values("score_lambda").reset_index(drop=True)

    print("\n=== score_lambda Sweep Results (mom-s6.yaml) ===\n")
    col_fmt = {
        "score_lambda":    ("lambda", "{:.2f}"),
        "sharpe_raw":      ("sharpe_raw", "{:.3f}"),
        "sharpe_flat_hc":  ("sharpe_hc400", "{:.3f}"),
        "calmar":          ("calmar", "{:.3f}"),
        "mean_net_ret_ann":("net_ret_ann", "{:.2%}"),
        "mean_turnover":   ("turnover", "{:.2%}"),
        "hit_rate":        ("hit_rate", "{:.1%}"),
    }

    headers = [col_fmt[c][0] for c in col_fmt]
    widths = [max(len(h), 12) for h in headers]
    header_line = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
    sep_line = "  ".join("-" * w for w in widths)
    print(header_line)
    print(sep_line)
    for _, row in df.iterrows():
        vals = [col_fmt[c][1].format(row[c]) for c in col_fmt]
        print("  ".join(v.rjust(w) for v, w in zip(vals, widths)))

    # Recommendation
    best_idx = df["sharpe_raw"].idxmax()
    best = df.loc[best_idx]
    base = df.loc[df["score_lambda"] == 0.0].iloc[0]

    print("\n=== Recommendation ===\n")
    sharpe_delta = best["sharpe_raw"] - base["sharpe_raw"]
    print(f"  Best Sharpe (raw): {best['sharpe_raw']:.3f} at score_lambda={best['score_lambda']:.2f}")
    print(f"  Baseline (lambda=0): {base['sharpe_raw']:.3f}")
    print(f"  Delta: {sharpe_delta:+.3f}\n")

    # Detect elbow / diminishing returns
    sharpe_series = df["sharpe_raw"].values
    lam_series = df["score_lambda"].values
    gains = np.diff(sharpe_series)

    if best["score_lambda"] == 0.0:
        print("  Score tilting does NOT improve Sharpe. Recommend keeping score_lambda=0.0 (pure ERC).")
    else:
        # Find where gains become small (< 20% of max gain)
        max_gain = gains.max() if gains.max() > 0 else 1e-9
        elbow_candidates = [lam_series[i + 1] for i, g in enumerate(gains) if g >= 0.2 * max_gain]
        elbow = elbow_candidates[-1] if elbow_candidates else best["score_lambda"]

        print(f"  Optimal lambda: {best['score_lambda']:.2f}  (highest Sharpe raw = {best['sharpe_raw']:.3f})")
        if elbow < best["score_lambda"]:
            print(f"  Elbow / diminishing returns observed around lambda={elbow:.2f}; "
                  f"gains beyond that are marginal.")
            print(f"  Recommend score_lambda={elbow:.2f} as a conservative choice balancing performance vs. concentration risk.")
        else:
            print(f"  No clear elbow — returns scale monotonically to lambda={best['score_lambda']:.2f}.")
            print(f"  Recommend score_lambda={best['score_lambda']:.2f}, but monitor concentration and turnover.")

        to_best = best["mean_turnover"]
        base_to = base["mean_turnover"]
        if to_best > base_to * 1.15:
            print(f"\n  Warning: turnover rises from {base_to:.1%} to {to_best:.1%} — check cost sensitivity.")


if __name__ == "__main__":
    main()
