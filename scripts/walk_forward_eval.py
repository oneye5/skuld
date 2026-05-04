"""Walk-forward evaluation using the canonical research runner."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python" / "src"))

from skuld_research.config.loader import load_spec
from skuld_research.config.runner import run_from_spec
from skuld_research.stats.bootstrap import stationary_bootstrap_sharpe

DATA_PATH = _REPO_ROOT / "data" / "data_long.csv"
_DEFAULT_SPEC = _REPO_ROOT / "python" / "configs" / "strategy-specs" / "candidates" / "mom-s7.yaml"


def _ascii_safe(text: str) -> str:
    return text.replace("≤", "<=").replace("≥", ">=")


def _iid_bootstrap_sharpe_ci(returns: pd.Series, n_boot: int = 2000, ci: float = 0.95, rf: float = 0.0) -> tuple[float, float]:
    """Percentile IID bootstrap CI for annualised Sharpe (ignores autocorrelation)."""
    rng = np.random.default_rng(42)
    n = len(returns)
    arr = returns.values
    boot_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        mu = sample.mean() * 12.0
        vol = sample.std(ddof=1) * (12.0 ** 0.5)
        s = (mu - rf) / vol if vol > 1e-12 else 0.0
        boot_sharpes.append(s)
    lo = np.percentile(boot_sharpes, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_sharpes, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def print_fold_table(wf_result) -> None:
    """Print per-fold results table."""
    print(f"\n{'Fold':>4}  {'Test Start':>10}  {'Test End':>10}  {'Sharpe':>7}  {'Hit%':>6}  {'AvgPos':>7}")
    print("-" * 60)
    for fr in wf_result.folds:
        r = fr.result
        sharpe = r.sharpe_raw
        hit = r.hit_rate
        avg_pos = r.avg_positions
        print(
            f"{fr.fold_id:>4}  "
            f"{fr.test_start:%Y-%m-%d}  "
            f"{fr.test_end:%Y-%m-%d}  "
            f"{sharpe:>7.3f}  "
            f"{hit:>5.1%}  "
            f"{avg_pos:>7.1f}"
        )


def print_aggregate_metrics(wf_result, rf: float = 0.0) -> None:
    """Print aggregate OOS metrics."""
    r = wf_result
    print(f"\n{'=' * 52}")
    print("  Aggregate OOS Metrics")
    print(f"{'=' * 52}")
    print(f"  OOS Sharpe (raw):               {r.oos_sharpe_raw:>8.3f}")
    print(f"  OOS Sharpe (flat haircut):       {r.oos_sharpe_flat_haircut:>8.3f}")
    print(f"  OOS Sharpe (delisting adj):      {r.oos_sharpe_delisting_adjusted:>8.3f}")
    print(f"  OOS Hit rate:                    {r.oos_hit_rate:>8.1%}")
    print(f"  OOS Calmar ratio:                {r.oos_calmar_ratio:>8.3f}")
    print(f"  OOS Max drawdown (observed):     {r.oos_max_drawdown_observed:>8.1%}")
    print(f"  OOS Max drawdown (MC median):    {r.oos_max_drawdown_augmented_median:>8.1%}")
    print(f"  OOS Avg turnover:                {r.oos_avg_turnover:>8.1%}")
    print(f"  N kept folds:                    {r.n_kept_folds:>8d}")
    print(f"  N rejected folds:                {r.n_rejected_folds:>8d}")

    if r.rejection_reasons:
        for reason in r.rejection_reasons:
            print(f"    Rejected: {reason}")

    # Bootstrap CI
    if not r.oos_returns.empty and len(r.oos_returns) >= 4:
        iid_lo, iid_hi = _iid_bootstrap_sharpe_ci(r.oos_returns, rf=rf)
        stat_result = stationary_bootstrap_sharpe(
            r.oos_returns.dropna(),
            n_resamples=2000,
            rf_annual=rf,
        )
        stat_lo, stat_hi = stat_result.ci_low_95, stat_result.ci_high_95
        print(f"\n  Bootstrap 95% CI for OOS Sharpe:")
        print(f"    IID (naive):       [{iid_lo:.3f}, {iid_hi:.3f}]  width={iid_hi - iid_lo:.3f}")
        print(f"    Stationary block:  [{stat_lo:.3f}, {stat_hi:.3f}]  width={stat_hi - stat_lo:.3f}")
        wider_pct = (stat_hi - stat_lo) / (iid_hi - iid_lo) * 100 - 100 if (iid_hi - iid_lo) > 1e-12 else float("nan")
        print(f"    Stationary CI is {wider_pct:+.1f}% wider than IID")

    # Per-regime Sharpes
    if r.oos_sharpe_by_regime:
        print(f"\n  Regime Sharpes:")
        for regime, sharpe in r.oos_sharpe_by_regime.items():
            print(f"    {regime:<8}: {sharpe:.3f}")
def main() -> None:
    SPEC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_SPEC
    print(f"=== Walk-Forward Evaluation: {SPEC_PATH.name} ===\n")

    spec = load_spec(SPEC_PATH)
    print("Running canonical research runner...")
    run_result = run_from_spec(spec, raw_csv_path=DATA_PATH, write_ledger=False)
    wf_result = run_result.strategy_rolling
    print(f"  Done — {len(wf_result.folds)} folds evaluated.")

    # --- Print fold table ---
    print_fold_table(wf_result)

    # --- Print aggregate metrics ---
    print_aggregate_metrics(wf_result, rf=spec.backtest.risk_free_annual)

    if run_result.gating is not None:
        print(f"\n{'=' * 52}")
        print("  Canonical Gating")
        print(f"{'=' * 52}")
        for bar_name, (passed, reason) in run_result.gating.bars.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {bar_name:<24} {status:>4}  {_ascii_safe(reason)}")

    print()


if __name__ == "__main__":
    main()
