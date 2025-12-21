"""Visualization module for evaluation results."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from config.column_names import TARGET, PREDICTION, PREDICTION_PROB, TIMESTAMP, TICKER
from config.file_paths import EVALUATION_DIR, get_run_evaluation_dir


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Path to save the figure. If None, just returns figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Gain (0)', 'Gain (1)'],
        yticklabels=['No Gain (0)', 'Gain (1)'],
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | None = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.3)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | None = None,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.fill_between(recall, precision, alpha=0.3)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | None = None,
    title: str = "Prediction Probability Distribution",
) -> plt.Figure:
    """
    Plot distribution of prediction probabilities by actual class.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate probabilities by class
    prob_class_0 = y_prob[y_true == 0]
    prob_class_1 = y_prob[y_true == 1]
    
    ax.hist(prob_class_0, bins=50, alpha=0.6, label='Actual: No Gain (0)', color='red')
    ax.hist(prob_class_1, bins=50, alpha=0.6, label='Actual: Gain (1)', color='green')
    
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Default Threshold')
    ax.axvline(x=0.7, color='orange', linestyle='--', linewidth=1, label='Trading Threshold (0.7)')
    
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_returns_distribution(
    returns: list[float],
    output_path: Path | None = None,
    title: str = "Trade Returns Distribution",
) -> plt.Figure:
    """
    Plot distribution of trade returns.
    
    Args:
        returns: List of return percentages per trade.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    returns_arr = np.array(returns)
    
    ax.hist(returns_arr, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
    ax.axvline(x=np.mean(returns_arr), color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(returns_arr):.2f}%')
    ax.axvline(x=np.median(returns_arr), color='orange', linestyle='-', linewidth=2,
               label=f'Median: {np.median(returns_arr):.2f}%')
    
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_cumulative_returns(
    trades_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    output_path: Path | None = None,
    title: str = "Cumulative Returns Over Time",
) -> plt.Figure:
    """
    Plot cumulative returns over time.
    
    Args:
        trades_df: DataFrame with trade information including sell_timestamp, return_pct.
        initial_capital: Starting capital.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    if trades_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center')
        return fig
    
    trades_sorted = trades_df.sort_values('sell_timestamp')
    
    # Calculate cumulative capital
    capital = initial_capital
    cumulative = [capital]
    timestamps = [trades_sorted['sell_timestamp'].min() - 86400000]  # Start 1 day before first trade
    
    for _, trade in trades_sorted.iterrows():
        capital = capital * (1 + trade['return_pct'] / 100)
        cumulative.append(capital)
        timestamps.append(trade['sell_timestamp'])
    
    # Convert timestamps to dates
    dates = pd.to_datetime(timestamps, unit='ms')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, cumulative, 'b-', linewidth=2)
    ax.axhline(y=initial_capital, color='red', linestyle='--', linewidth=1, 
               label=f'Initial Capital (${initial_capital:,.0f})')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_window_comparison(
    window_metrics: list[dict],
    metric_name: str,
    output_path: Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot comparison of a metric across windows.
    
    Args:
        window_metrics: List of dicts with window_id and metric values.
        metric_name: Name of the metric to plot.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    window_ids = [w['window_id'] for w in window_metrics]
    values = [w[metric_name] for w in window_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(window_ids, values, color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=np.mean(values), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(values):.2f}')
    
    ax.set_xlabel('Window ID')
    ax.set_ylabel(metric_name)
    ax.set_title(title or f'{metric_name} by Window')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_all_visualizations(
    combined_predictions: pd.DataFrame,
    combined_actuals: pd.DataFrame,
    trades: list[dict],
    window_metrics: list[dict] | None = None,
    output_dir: Path | None = None,
) -> dict[str, plt.Figure]:
    """
    Generate all evaluation visualizations.
    
    Args:
        combined_predictions: All predictions combined across windows.
        combined_actuals: All actuals combined across windows.
        trades: List of trade dictionaries.
        window_metrics: Optional list of per-window metrics for comparison plots.
        output_dir: Directory to save figures. If None, uses default.
    
    Returns:
        Dictionary of figure names to Figure objects.
    """
    output_dir = output_dir or get_run_evaluation_dir() / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge predictions with actuals
    merged = combined_predictions.merge(
        combined_actuals[[TIMESTAMP, TICKER, TARGET]],
        on=[TIMESTAMP, TICKER],
        how='inner'
    )
    
    y_true = merged[TARGET].values
    y_pred = merged[PREDICTION].values
    y_prob = merged[PREDICTION_PROB].values
    
    figures = {}
    
    # Classification plots
    print("  Generating confusion matrix...")
    figures['confusion_matrix'] = plot_confusion_matrix(
        y_true, y_pred,
        output_path=output_dir / "confusion_matrix.png",
        title="Combined Confusion Matrix (All Windows)"
    )
    
    print("  Generating ROC curve...")
    try:
        figures['roc_curve'] = plot_roc_curve(
            y_true, y_prob,
            output_path=output_dir / "roc_curve.png",
            title="Combined ROC Curve (All Windows)"
        )
    except ValueError as e:
        print(f"    Warning: Could not generate ROC curve: {e}")
    
    print("  Generating precision-recall curve...")
    figures['precision_recall'] = plot_precision_recall_curve(
        y_true, y_prob,
        output_path=output_dir / "precision_recall_curve.png",
        title="Combined Precision-Recall Curve (All Windows)"
    )
    
    print("  Generating probability distribution...")
    figures['prob_distribution'] = plot_probability_distribution(
        y_true, y_prob,
        output_path=output_dir / "probability_distribution.png",
        title="Prediction Probability Distribution (All Windows)"
    )
    
    # Trading plots
    if trades:
        returns = [t['return_pct'] for t in trades]
        
        print("  Generating returns distribution...")
        figures['returns_distribution'] = plot_returns_distribution(
            returns,
            output_path=output_dir / "returns_distribution.png",
            title="Trade Returns Distribution (All Windows)"
        )
        
        print("  Generating cumulative returns...")
        trades_df = pd.DataFrame(trades)
        figures['cumulative_returns'] = plot_cumulative_returns(
            trades_df,
            output_path=output_dir / "cumulative_returns.png",
            title="Cumulative Returns Over Time (All Windows)"
        )
    
    # Window comparison plots
    if window_metrics:
        print("  Generating window comparison plots...")
        figures['accuracy_by_window'] = plot_window_comparison(
            window_metrics, 'accuracy',
            output_path=output_dir / "accuracy_by_window.png",
            title="Classification Accuracy by Window"
        )
        
        figures['return_by_window'] = plot_window_comparison(
            window_metrics, 'trading_return',
            output_path=output_dir / "return_by_window.png",
            title="Trading Return (%) by Window"
        )
    
    plt.close('all')  # Close all figures to free memory
    
    print(f"  Saved {len(figures)} visualizations to {output_dir}")
    
    return figures
