"""Report generation module."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.paths import get_run_dir, ensure_output_dirs
from config.settings import get_config_dict
from evaluation.metrics import ClassificationMetrics, metrics_to_dict as class_metrics_to_dict
from evaluation.simulator import (
    TradingMetrics,
    Trade,
    trades_to_dataframe,
    metrics_to_dict as trading_metrics_to_dict,
)


def generate_report(
    classification_metrics: ClassificationMetrics,
    trading_metrics: TradingMetrics,
    trades: list[Trade],
    window_summaries: list[dict] | None = None,
) -> Path:
    """Generate evaluation report and save all outputs.
    
    Creates:
    - report.md: Human-readable summary
    - results.json: Machine-readable full results
    - config.json: Configuration snapshot
    - trades.csv: All simulated trades
    
    Args:
        classification_metrics: Classification performance metrics.
        trading_metrics: Trading simulation metrics.
        trades: List of completed trades.
        window_summaries: Optional per-window summaries.
    
    Returns:
        Path to the run directory.
    """
    ensure_output_dirs()
    run_dir = get_run_dir()
    
    # Save config
    config = get_config_dict()
    _save_json(run_dir / "config.json", config)
    
    # Save trades CSV
    trades_df = trades_to_dataframe(trades)
    trades_df.to_csv(run_dir / "trades.csv", index=False)
    
    # Build results dict
    results = {
        "timestamp": datetime.now().isoformat(),
        "classification": class_metrics_to_dict(classification_metrics),
        "trading": trading_metrics_to_dict(trading_metrics),
    }
    
    if window_summaries:
        results["windows"] = window_summaries
    
    # Save JSON results
    _save_json(run_dir / "results.json", results)
    
    # Generate markdown report
    report_md = _generate_markdown_report(
        classification_metrics, trading_metrics, trades, window_summaries
    )
    (run_dir / "report.md").write_text(report_md)
    
    return run_dir


def _save_json(path: Path, data: dict) -> None:
    """Save dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _generate_markdown_report(
    classification: ClassificationMetrics,
    trading: TradingMetrics,
    trades: list[Trade],
    window_summaries: list[dict] | None,
) -> str:
    """Generate markdown report content."""
    lines = [
        "# ML Pipeline Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Classification Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {classification.accuracy:.4f} |",
        f"| Precision | {classification.precision:.4f} |",
        f"| Recall | {classification.recall:.4f} |",
        f"| F1 Score | {classification.f1:.4f} |",
        f"| AUC-ROC | {classification.auc_roc:.4f} |",
        "",
        "### Confusion Matrix",
        "",
        "| | Predicted Negative | Predicted Positive |",
        "|---|---|---|",
        f"| **Actual Negative** | {classification.true_negatives} | {classification.false_positives} |",
        f"| **Actual Positive** | {classification.false_negatives} | {classification.true_positives} |",
        "",
        f"**Total Samples:** {classification.total_samples} "
        f"(Positive: {classification.positive_samples}, Negative: {classification.negative_samples})",
        "",
        "---",
        "",
        "## Trading Simulation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Number of Trades | {trading.num_trades} |",
        f"| Mean Return | {trading.mean_return_pct:.2f}% |",
        f"| Median Return | {trading.median_return_pct:.2f}% |",
        f"| Std Return | {trading.std_return_pct:.2f}% |",
        f"| **Sharpe Ratio** | **{trading.sharpe_ratio:.3f}** |",
        f"| Min Return | {trading.min_return_pct:.2f}% |",
        f"| Max Return | {trading.max_return_pct:.2f}% |",
        f"| 25th Percentile | {trading.lqr_return_pct:.2f}% |",
        f"| 75th Percentile | {trading.uqr_return_pct:.2f}% |",
        "",
    ]
    
    if window_summaries:
        lines.extend([
            "---",
            "",
            "## Rolling Window Summary",
            "",
            "| Window | Train Period | Test Period | Test Samples | Positive Rate |",
            "|--------|--------------|-------------|--------------|---------------|",
        ])
        
        for ws in window_summaries:
            lines.append(
                f"| {ws.get('window_id', 'N/A')} | "
                f"{ws.get('train_period', 'N/A')} | "
                f"{ws.get('test_period', 'N/A')} | "
                f"{ws.get('test_samples', 'N/A')} | "
                f"{ws.get('positive_rate', 'N/A')} |"
            )
        
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "*Report generated by Skuld ML Pipeline*",
    ])
    
    return "\n".join(lines)
