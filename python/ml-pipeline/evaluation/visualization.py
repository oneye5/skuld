"""Visualization module for evaluation results."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

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


def plot_drawdown(
    trades_df: pd.DataFrame,
    initial_capital: float = 100000.0,
    output_path: Path | None = None,
    title: str = "Portfolio Drawdown Over Time",
) -> plt.Figure:
    """
    Plot drawdown from peak over time.
    
    Args:
        trades_df: DataFrame with trade information.
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
    timestamps = [trades_sorted['sell_timestamp'].min() - 86400000]
    
    for _, trade in trades_sorted.iterrows():
        capital = capital * (1 + trade['return_pct'] / 100)
        cumulative.append(capital)
        timestamps.append(trade['sell_timestamp'])
    
    cumulative = np.array(cumulative)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    
    dates = pd.to_datetime(timestamps, unit='ms')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(dates, drawdown, 0, alpha=0.5, color='red')
    ax.plot(dates, drawdown, 'r-', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add max drawdown annotation
    max_dd = drawdown.min()
    max_dd_idx = np.argmin(drawdown)
    ax.annotate(f'Max Drawdown: {max_dd:.1f}%',
                xy=(dates[max_dd_idx], max_dd),
                xytext=(10, -30), textcoords='offset points',
                ha='left', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | None = None,
    title: str = "Threshold Analysis",
) -> plt.Figure:
    """
    Plot precision, recall, and F1 at different probability thresholds.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    precisions = []
    recalls = []
    f1s = []
    trade_counts = []
    
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        trade_counts.append(preds.sum())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Precision, Recall, F1
    ax1.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax1.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
    ax1.plot(thresholds, f1s, 'r-', linewidth=2, label='F1 Score')
    ax1.axvline(x=0.79, color='orange', linestyle='--', linewidth=1.5, label='Current (0.79)')
    ax1.set_xlabel('Probability Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Trade count
    ax2.plot(thresholds, trade_counts, 'purple', linewidth=2)
    ax2.axvline(x=0.79, color='orange', linestyle='--', linewidth=1.5, label='Current (0.79)')
    ax2.set_xlabel('Probability Threshold')
    ax2.set_ylabel('Number of Trades (Predictions = 1)')
    ax2.set_title('Trade Count vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_monthly_returns(
    trades_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Monthly Returns Heatmap",
) -> plt.Figure:
    """
    Plot monthly returns as a heatmap.
    
    Args:
        trades_df: DataFrame with trade information.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    if trades_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center')
        return fig
    
    trades_df = trades_df.copy()
    trades_df['sell_date'] = pd.to_datetime(trades_df['sell_timestamp'], unit='ms')
    trades_df['year'] = trades_df['sell_date'].dt.year
    trades_df['month'] = trades_df['sell_date'].dt.month
    
    # Aggregate returns by year-month
    monthly_returns = trades_df.groupby(['year', 'month'])['return_pct'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(
        monthly_returns,
        annot=True, fmt='.1f', cmap='RdYlGn', center=0,
        xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        ax=ax, cbar_kws={'label': 'Mean Return (%)'}
    )
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ticker_performance(
    trades_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Path | None = None,
    title: str = "Top Ticker Performance",
) -> plt.Figure:
    """
    Plot performance by ticker.
    
    Args:
        trades_df: DataFrame with trade information.
        top_n: Number of top/bottom tickers to show.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    if trades_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center')
        return fig
    
    # Aggregate by ticker
    ticker_stats = trades_df.groupby('ticker').agg({
        'return_pct': ['mean', 'count', 'std']
    }).round(2)
    ticker_stats.columns = ['mean_return', 'num_trades', 'std_return']
    ticker_stats = ticker_stats.reset_index()
    
    # Filter tickers with at least 3 trades
    ticker_stats = ticker_stats[ticker_stats['num_trades'] >= 3]
    
    # Get top and bottom performers
    ticker_stats_sorted = ticker_stats.sort_values('mean_return', ascending=False)
    top_tickers = ticker_stats_sorted.head(top_n // 2)
    bottom_tickers = ticker_stats_sorted.tail(top_n // 2)
    display_tickers = pd.concat([top_tickers, bottom_tickers])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['green' if x >= 0 else 'red' for x in display_tickers['mean_return']]
    bars = ax.barh(display_tickers['ticker'], display_tickers['mean_return'], color=colors, alpha=0.7)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Mean Return (%)')
    ax.set_ylabel('Ticker')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add trade count labels
    for bar, count in zip(bars, display_tickers['num_trades']):
        width = bar.get_width()
        ax.annotate(f'n={int(count)}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rolling_sharpe(
    trades_df: pd.DataFrame,
    window: int = 50,
    output_path: Path | None = None,
    title: str = "Rolling Sharpe Ratio",
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio over time.
    
    Args:
        trades_df: DataFrame with trade information.
        window: Rolling window size.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    if trades_df.empty or len(trades_df) < window:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Not enough trades for rolling analysis', ha='center', va='center')
        return fig
    
    trades_sorted = trades_df.sort_values('sell_timestamp').copy()
    trades_sorted['sell_date'] = pd.to_datetime(trades_sorted['sell_timestamp'], unit='ms')
    
    # Calculate rolling Sharpe (annualized)
    returns = trades_sorted['return_pct']
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252 / 365)  # Annualized
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(trades_sorted['sell_date'].values[window-1:], 
            rolling_sharpe.values[window-1:], 'b-', linewidth=1.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
    ax.axhline(y=2, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Sharpe = 2')
    
    ax.fill_between(trades_sorted['sell_date'].values[window-1:],
                    rolling_sharpe.values[window-1:], 0, 
                    where=rolling_sharpe.values[window-1:] >= 0, 
                    alpha=0.3, color='green')
    ax.fill_between(trades_sorted['sell_date'].values[window-1:],
                    rolling_sharpe.values[window-1:], 0, 
                    where=rolling_sharpe.values[window-1:] < 0, 
                    alpha=0.3, color='red')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Sharpe Ratio (Annualized)')
    ax.set_title(f'{title} (Window={window} trades)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_win_loss_analysis(
    trades_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Win/Loss Analysis",
) -> plt.Figure:
    """
    Plot win/loss statistics.
    
    Args:
        trades_df: DataFrame with trade information.
        output_path: Path to save the figure.
        title: Title for the plot.
    
    Returns:
        matplotlib Figure object.
    """
    if trades_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center')
        return fig
    
    returns = trades_df['return_pct']
    winners = returns[returns > 0]
    losers = returns[returns <= 0]
    
    win_rate = len(winners) / len(returns) * 100
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0
    profit_factor = abs(winners.sum() / losers.sum()) if losers.sum() != 0 else np.inf
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Win rate pie chart
    ax1 = axes[0, 0]
    sizes = [len(winners), len(losers)]
    colors = ['green', 'red']
    labels = [f'Winners\n({len(winners)})', f'Losers\n({len(losers)})']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Win Rate: {win_rate:.1f}%')
    
    # Average win vs loss bar chart
    ax2 = axes[0, 1]
    bars = ax2.bar(['Avg Win', 'Avg Loss'], [avg_win, avg_loss], 
                   color=['green', 'red'], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Return (%)')
    ax2.set_title(f'Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%')
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords='offset points',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=12)
    
    # Distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(winners, bins=30, alpha=0.6, color='green', label='Winners')
    ax3.hist(losers, bins=30, alpha=0.6, color='red', label='Losers')
    ax3.set_xlabel('Return (%)')
    ax3.set_ylabel('Count')
    ax3.set_title('Return Distribution by Outcome')
    ax3.legend()
    
    # Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics_text = f"""
    Summary Statistics
    ═══════════════════════════
    
    Total Trades:     {len(returns)}
    Win Rate:         {win_rate:.1f}%
    
    Average Win:      +{avg_win:.2f}%
    Average Loss:     {avg_loss:.2f}%
    
    Best Trade:       +{returns.max():.2f}%
    Worst Trade:      {returns.min():.2f}%
    
    Profit Factor:    {profit_factor:.2f}
    
    Total Wins Sum:   +{winners.sum():.1f}%
    Total Loss Sum:   {losers.sum():.1f}%
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_all_visualizations_extended(
    combined_predictions: pd.DataFrame,
    combined_actuals: pd.DataFrame,
    trades: list[dict],
    window_metrics: list[dict] | None = None,
    output_dir: Path | None = None,
) -> dict[str, plt.Figure]:
    """
    Generate all evaluation visualizations including extended analysis.
    
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
    
    print("  Generating threshold analysis...")
    figures['threshold_analysis'] = plot_threshold_analysis(
        y_true, y_prob,
        output_path=output_dir / "threshold_analysis.png",
        title="Threshold Analysis"
    )
    
    # Trading plots
    if trades:
        returns = [t['return_pct'] for t in trades]
        trades_df = pd.DataFrame(trades)
        
        print("  Generating returns distribution...")
        figures['returns_distribution'] = plot_returns_distribution(
            returns,
            output_path=output_dir / "returns_distribution.png",
            title="Trade Returns Distribution (All Windows)"
        )
        
        print("  Generating cumulative returns...")
        figures['cumulative_returns'] = plot_cumulative_returns(
            trades_df,
            output_path=output_dir / "cumulative_returns.png",
            title="Cumulative Returns Over Time (All Windows)"
        )
        
        print("  Generating drawdown chart...")
        figures['drawdown'] = plot_drawdown(
            trades_df,
            output_path=output_dir / "drawdown.png",
            title="Portfolio Drawdown Over Time"
        )
        
        print("  Generating monthly returns heatmap...")
        figures['monthly_returns'] = plot_monthly_returns(
            trades_df,
            output_path=output_dir / "monthly_returns.png",
            title="Monthly Returns Heatmap"
        )
        
        print("  Generating ticker performance chart...")
        figures['ticker_performance'] = plot_ticker_performance(
            trades_df,
            output_path=output_dir / "ticker_performance.png",
            title="Top/Bottom Ticker Performance"
        )
        
        print("  Generating rolling Sharpe chart...")
        figures['rolling_sharpe'] = plot_rolling_sharpe(
            trades_df,
            output_path=output_dir / "rolling_sharpe.png",
            title="Rolling Sharpe Ratio"
        )
        
        print("  Generating win/loss analysis...")
        figures['win_loss_analysis'] = plot_win_loss_analysis(
            trades_df,
            output_path=output_dir / "win_loss_analysis.png",
            title="Win/Loss Analysis"
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
