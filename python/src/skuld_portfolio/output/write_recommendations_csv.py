"""Write recommendation CSV and sidecar metadata JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from skuld_common.contracts import CombinedScores, TradeList
from skuld_research.config.spec import BacktestSpec

# Maps spec factor kind → signal name used as component_scores column.
_FACTOR_KIND_TO_SIGNAL_NAME: dict[str, str] = {
    "momentum": "momentum",
    "low_vol": "low_volatility",
    "size": "size",
}


def write_recommendations_csv(
    trades: TradeList,
    spec: BacktestSpec,
    meta: dict,
    output_path: Path,
    combined_scores: CombinedScores | None = None,
) -> None:
    """Write TradeList to CSV with sidecar JSON and overrides log.
    
    Produces three files:
    - output_path: the main CSV with required columns + dynamic factor_*_z columns.
    - output_path.replace('.csv', '.meta.json'): sidecar metadata JSON.
    - output_dir/overrides_log_<YYYY-MM-DD>.csv: empty override log (header only).
    
    Args:
        trades: TradeList from execution planner.
        spec: BacktestSpec (for factor column names).
        meta: Metadata dict from recommend().
        output_path: Path to write main CSV (e.g., reports/recommendations_2026-01-01.csv).
        combined_scores: Optional CombinedScores from the signal combiner. When provided,
            combined_score_z and factor_*_z columns are populated with real values.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build output DataFrame
    df = trades.trades.copy()
    
    # Rebalance date (same for all rows)
    df.insert(0, "rebalance_date", trades.asof.date().isoformat())
    
    # Weights: use equity NAV (sum of target values) as denominator.
    equity_nav = df["target_value_nzd"].sum()
    df["current_weight"] = df["current_value_nzd"] / equity_nav if equity_nav > 0 else 0.0
    df["target_weight"] = df["target_value_nzd"] / equity_nav if equity_nav > 0 else 0.0
    
    # combined_score_z: populated from CombinedScores when available.
    if combined_scores is not None:
        df["combined_score_z"] = df["ticker"].map(combined_scores.scores).fillna(0.0)
    else:
        df["combined_score_z"] = 0.0
    
    # Dynamic factor columns: factor_<kind>_z, one per factor in spec.
    for factor_spec in spec.factors:
        col_name = f"factor_{factor_spec.kind}_z"
        if combined_scores is not None:
            signal_name = _FACTOR_KIND_TO_SIGNAL_NAME.get(factor_spec.kind, factor_spec.kind)
            if signal_name in combined_scores.component_scores.columns:
                df[col_name] = df["ticker"].map(
                    combined_scores.component_scores[signal_name]
                ).fillna(0.0)
            else:
                df[col_name] = 0.0
        else:
            df[col_name] = 0.0
    
    # Rationale (simplified)
    df["rationale"] = df["action"].apply(
        lambda a: f"{a} per target portfolio" if a in ["BUY", "SELL"] else "Hold or deferred"
    )
    
    # Write main CSV
    df.to_csv(output_path, index=False)
    
    # Write sidecar JSON
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    # Write empty overrides log
    asof_date = pd.Timestamp(trades.asof).date().isoformat()
    overrides_path = output_path.parent / f"overrides_log_{asof_date}.csv"
    if not overrides_path.exists():
        overrides_df = pd.DataFrame(columns=[
            "ticker", "override_action", "reason", "override_by", "override_at"
        ])
        overrides_df.to_csv(overrides_path, index=False)
