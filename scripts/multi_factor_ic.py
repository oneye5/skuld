"""
Multi-factor IC audit: momentum, size, low_volatility, dividend_yield.

Computes per-factor cross-sectional Spearman rank IC vs 1-month forward return,
then reports pairwise factor score correlations.

Usage (from repo root):
    uv run python scripts/multi_factor_ic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

from skuld_research.config.loader import load_spec
from skuld_research.data.csv_loader import load_raw_csv
from skuld_research.data.pit_loader import PITLoader
from skuld_research.data.prepared_panel import build_prepared_panel
from skuld_research.factors.momentum import MomentumFactor
from skuld_research.factors.size import SizeFactor
from skuld_research.factors.low_volatility import LowVolatilityFactor
from skuld_research.factors.dividend_yield import DividendYieldFactor
from skuld_research.factors.return_on_risk import ReturnOnRiskFactor

DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"
SPEC_PATH = _REPO_ROOT / "python" / "configs" / "strategy-specs" / "candidates" / "mom-s6.yaml"

FACTORS = [
    ("momentum", MomentumFactor()),
    ("size", SizeFactor()),
    ("low_volatility", LowVolatilityFactor()),
    ("dividend_yield", DividendYieldFactor()),
    ("return_on_risk", ReturnOnRiskFactor()),
]


def rank_ic(scores: pd.Series, fwd_returns: pd.Series) -> float | None:
    common = scores.dropna().index.intersection(fwd_returns.dropna().index)
    if len(common) < 5:
        return None
    ic, _ = spearmanr(scores.loc[common], fwd_returns.loc[common])
    return float(ic) if not np.isnan(ic) else None


def collect_scores_and_ics(
    panel,
    rebalance_dates: list,
    factor,
    fwd_returns_monthly: pd.DataFrame,
) -> tuple[list[float], pd.DataFrame]:
    """Return (ics, score_panel) where score_panel is date×ticker."""
    ics: list[float] = []
    score_rows: dict[pd.Timestamp, pd.Series] = {}

    for t in rebalance_dates:
        if t not in panel.universe_mask.index:
            continue
        mask = panel.universe_mask.loc[t]
        universe = mask[mask].index.tolist()
        if not universe:
            continue

        scores = factor.score(panel, t, universe)
        if scores is None or scores.empty:
            continue

        score_rows[t] = scores

        date_idx = fwd_returns_monthly.index.searchsorted(t)
        if date_idx + 1 >= len(fwd_returns_monthly):
            continue
        fwd_date = fwd_returns_monthly.index[date_idx + 1]
        fwd_ret = fwd_returns_monthly.loc[fwd_date].dropna()

        ic = rank_ic(scores, fwd_ret)
        if ic is not None:
            ics.append(ic)

    score_panel = pd.DataFrame(score_rows).T if score_rows else pd.DataFrame()
    return ics, score_panel


def print_ic_summary(name: str, ics: list[float]) -> None:
    if not ics:
        print(f"  {name:<20} no valid IC dates")
        return
    arr = np.array(ics)
    mean_ic = arr.mean()
    icir = mean_ic / arr.std() if arr.std() > 1e-9 else float("nan")
    hit = (arr > 0).mean()
    print(
        f"  {name:<20} mean_IC={mean_ic:+.4f}  ICIR={icir:+.3f}"
        f"  hit={hit:.1%}  n={len(arr)}"
    )


def main() -> None:
    print("Loading data...")
    spec = load_spec(SPEC_PATH)
    raw = load_raw_csv(DATA_PATH, scrub=spec.scrubbing, adjustments=spec.adjustments)

    print(f"Building PIT snapshot (asof={spec.asof})...")
    snap = PITLoader(raw).as_of(pd.Timestamp(spec.asof, tz="UTC"))

    print("Building prepared panel...")
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

    rebalance_dates = panel.universe_mask.index.tolist()
    fwd_returns_monthly = panel.returns_monthly
    print(f"  Rebalance dates: {len(rebalance_dates)}, tickers: {panel.universe_mask.shape[1]}\n")

    # --- IC per factor ---
    print("=== Factor IC (1-month forward return) ===")
    all_score_panels: dict[str, pd.DataFrame] = {}
    for name, factor in FACTORS:
        ics, score_panel = collect_scores_and_ics(panel, rebalance_dates, factor, fwd_returns_monthly)
        print_ic_summary(name, ics)
        if not score_panel.empty:
            all_score_panels[name] = score_panel

    # --- Pairwise factor correlation ---
    print("\n=== Pairwise Spearman Rank Correlation (factor scores) ===")
    factor_names = list(all_score_panels.keys())
    if len(factor_names) < 2:
        print("  Not enough factors with scores to compute correlation.")
        return

    # Align all score panels to common dates and stack ticker-dates
    common_dates = None
    for fp in all_score_panels.values():
        if common_dates is None:
            common_dates = fp.index
        else:
            common_dates = common_dates.intersection(fp.index)

    # Build stacked series per factor across common dates
    stacked: dict[str, pd.Series] = {}
    for name, fp in all_score_panels.items():
        s = fp.loc[common_dates].stack(future_stack=True).dropna()
        stacked[name] = s

    # Align on (date, ticker) multi-index
    combined = pd.DataFrame(stacked).dropna()

    corr_results: dict[tuple[str, str], float] = {}
    for i, a in enumerate(factor_names):
        for b in factor_names[i + 1 :]:
            if a in combined.columns and b in combined.columns:
                ab = combined[[a, b]].dropna()
                if len(ab) >= 10:
                    r, _ = spearmanr(ab[a], ab[b])
                    corr_results[(a, b)] = float(r)
                else:
                    corr_results[(a, b)] = float("nan")

    # Print correlation matrix
    header = f"  {'':20}" + "".join(f"{n:>16}" for n in factor_names)
    print(header)
    for a in factor_names:
        row = f"  {a:<20}"
        for b in factor_names:
            if a == b:
                row += f"{'1.000':>16}"
            elif (a, b) in corr_results:
                row += f"{corr_results[(a, b)]:>16.3f}"
            elif (b, a) in corr_results:
                row += f"{corr_results[(b, a)]:>16.3f}"
            else:
                row += f"{'  —':>16}"
        print(row)

    # --- Summary recommendation ---
    print("\n=== Recommendation ===")
    # Re-collect ICs for summary
    factor_ic_stats: dict[str, tuple[float, float]] = {}
    for name, factor in FACTORS:
        ics, _ = collect_scores_and_ics(panel, rebalance_dates, factor, fwd_returns_monthly)
        if ics:
            arr = np.array(ics)
            factor_ic_stats[name] = (float(arr.mean()), float(arr.std()))

    mom_corr = {b: corr_results.get(("momentum", b), corr_results.get((b, "momentum"), float("nan")))
                for b in factor_names if b != "momentum"}

    for name in factor_names:
        if name == "momentum":
            continue
        mean_ic, _ = factor_ic_stats.get(name, (float("nan"), float("nan")))
        corr_with_mom = mom_corr.get(name, float("nan"))
        ic_ok = mean_ic > 0.02
        corr_ok = abs(corr_with_mom) < 0.6
        verdict = "ADD" if ic_ok and corr_ok else "SKIP"
        print(
            f"  {name:<20} mean_IC={mean_ic:+.4f}  corr_mom={corr_with_mom:+.3f}"
            f"  ic_ok={ic_ok}  decorr_ok={corr_ok}  -> {verdict}"
        )


if __name__ == "__main__":
    main()
