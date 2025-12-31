"""Visualization module for ranking-based stock prediction.

This module provides comprehensive plotting functions for analyzing ranking model performance:

QUINTILE ANALYSIS:
- plot_quintile_returns: Bar chart of average returns by quintile
- plot_quintile_cumulative_returns: Line chart of cumulative returns over time
- plot_quintile_heatmap: Heatmap showing quintile returns over time

IC (INFORMATION COEFFICIENT) ANALYSIS:
- plot_ic_series: IC over time with rolling average
- plot_ic_distribution: Histogram of IC values
- plot_ic_by_window: IC comparison across rolling windows
- plot_ic_decay: IC at different forecast horizons

BACKTEST & PERFORMANCE:
- plot_cumulative_returns: Equity curve with benchmarks
- plot_drawdown: Drawdown over time
- plot_monthly_returns_heatmap: Monthly returns calendar view
- plot_rolling_sharpe: Rolling Sharpe ratio over time
- plot_returns_distribution: Distribution of portfolio returns

TURNOVER & COSTS:
- plot_turnover_histogram: Distribution of portfolio turnover
- plot_turnover_over_time: Turnover series plot
- plot_cost_impact: Pre vs post-fee returns and cost drag

FACTOR ANALYSIS:
- plot_feature_importance: Model feature importances
- plot_prediction_distribution: Distribution of predicted scores
- plot_prediction_vs_actual: Scatter plot with regression

COMPREHENSIVE DASHBOARDS:
- create_ranking_dashboard: Combined 2x2 dashboard
- generate_all_figures: Generate and save all figures to directory
"""

from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Define a consistent color palette
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'positive': '#2ca02c',     # Green
    'negative': '#d62728',     # Red
    'neutral': '#7f7f7f',      # Gray
    'highlight': '#9467bd',    # Purple
}

# Quintile color map (red to green)
QUINTILE_CMAP = LinearSegmentedColormap.from_list(
    'quintile', ['#d62728', '#ff7f0e', '#7f7f7f', '#98df8a', '#2ca02c']
) if MATPLOTLIB_AVAILABLE else None


def _set_style():
    """Set consistent matplotlib style for all plots."""
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'grid.color': '#e0e0e0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
    })


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )
    _set_style()


def _format_timestamp_axis(ax: "plt.Axes", timestamps: pd.Index) -> None:
    """Format x-axis for timestamp data."""
    try:
        if hasattr(timestamps, 'to_pydatetime'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    except Exception as e:
        import warnings
        warnings.warn(f"Could not format timestamp axis, using default: {e}")


def _save_figure(fig: "plt.Figure", save_path: Optional[str], dpi: int = 150) -> None:
    """Save figure to file if path provided."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')


# =============================================================================
# QUINTILE CHARTS
# =============================================================================

def plot_quintile_returns(
    quintile_df: pd.DataFrame,
    title: str = "Return by Predicted Quintile",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> "plt.Figure":
    """Bar chart of average returns by quintile.
    
    This is the most important chart for evaluating a ranking model.
    A good model should show monotonically increasing returns from Q1 to Q5.
    
    Args:
        quintile_df: DataFrame with columns Q1, Q2, ..., Q5 (from portfolio_simulator).
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size (width, height).
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    # Compute mean return per quintile
    quintile_cols = [col for col in quintile_df.columns if col.startswith("Q")]
    avg_returns = quintile_df[quintile_cols].mean()
    
    # Define colors (red for low, green for high)
    n_quintiles = len(quintile_cols)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_quintiles))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(quintile_cols, avg_returns.values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_returns.values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.2%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=9,
        )
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Quintile (Q1=Lowest Predicted, Q5=Highest Predicted)", fontsize=10)
    ax.set_ylabel("Average Return", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add spread annotation
    spread = avg_returns.iloc[-1] - avg_returns.iloc[0]
    ax.annotate(
        f'Q5-Q1 Spread: {spread:.2%}',
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        ha='right', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_quintile_cumulative_returns(
    quintile_df: pd.DataFrame,
    title: str = "Cumulative Returns by Quintile",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """Line chart of cumulative returns for each quintile over time.
    
    Args:
        quintile_df: DataFrame with columns Q1, Q2, ..., Q5.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    quintile_cols = [col for col in quintile_df.columns if col.startswith("Q")]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(quintile_cols)))
    
    for col, color in zip(quintile_cols, colors):
        cumulative = (1 + quintile_df[col]).cumprod() - 1
        ax.plot(cumulative.index, cumulative.values, label=col, color=color, linewidth=1.5)
    
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Cumulative Return", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# IC CHARTS
# =============================================================================

def plot_ic_series(
    ic_series: pd.Series,
    rolling_window: int = 20,
    title: str = "Information Coefficient Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> "plt.Figure":
    """Plot IC over time with rolling average.
    
    Args:
        ic_series: Series of IC values indexed by timestamp.
        rolling_window: Window size for rolling average.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw IC values
    ax.plot(
        ic_series.index, ic_series.values, 
        alpha=0.4, color='steelblue', linewidth=1,
        label="Daily IC"
    )
    
    # Plot rolling mean
    rolling_mean = ic_series.rolling(rolling_window, min_periods=1).mean()
    ax.plot(
        rolling_mean.index, rolling_mean.values,
        color='darkred', linewidth=2,
        label=f"{rolling_window}-Period Rolling Mean"
    )
    
    ax.axhline(0, color='black', linewidth=0.5)
    
    # Add mean IC annotation
    mean_ic = ic_series.mean()
    ax.axhline(mean_ic, color='green', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.annotate(
        f'Mean IC: {mean_ic:.4f}',
        xy=(ic_series.index[-1], mean_ic),
        xytext=(5, 0),
        textcoords='offset points',
        fontsize=9,
        color='green',
    )
    
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("IC", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ic_distribution(
    ic_series: pd.Series,
    title: str = "IC Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> "plt.Figure":
    """Histogram of IC values.
    
    Args:
        ic_series: Series of IC values.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(ic_series.dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Add vertical lines for mean and zero
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(ic_series.mean(), color='red', linewidth=2, label=f'Mean: {ic_series.mean():.4f}')
    
    ax.set_xlabel("IC", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    
    # Add statistics
    stats_text = f"Mean: {ic_series.mean():.4f}\nStd: {ic_series.std():.4f}\nHit Rate: {(ic_series > 0).mean():.1%}"
    ax.annotate(
        stats_text,
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        ha='right', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# BACKTEST CHARTS
# =============================================================================

def plot_cumulative_returns(
    returns_series: pd.Series,
    benchmark_series: Optional[pd.Series] = None,
    title: str = "Strategy Cumulative Returns",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """Plot cumulative returns (equity curve).
    
    Args:
        returns_series: Series of strategy returns.
        benchmark_series: Optional benchmark returns for comparison.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Strategy cumulative returns
    strategy_cum = (1 + returns_series).cumprod() - 1
    ax.plot(
        strategy_cum.index, strategy_cum.values,
        color='steelblue', linewidth=2,
        label="Strategy"
    )
    
    # Benchmark if provided
    if benchmark_series is not None:
        benchmark_cum = (1 + benchmark_series).cumprod() - 1
        ax.plot(
            benchmark_cum.index, benchmark_cum.values,
            color='gray', linewidth=1.5, linestyle='--',
            label="Benchmark"
        )
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(
        strategy_cum.index, 0, strategy_cum.values,
        where=(strategy_cum.values >= 0),
        color='green', alpha=0.1
    )
    ax.fill_between(
        strategy_cum.index, 0, strategy_cum.values,
        where=(strategy_cum.values < 0),
        color='red', alpha=0.1
    )
    
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Cumulative Return", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add final return annotation
    final_return = strategy_cum.iloc[-1]
    ax.annotate(
        f'Total: {final_return:.1%}',
        xy=(strategy_cum.index[-1], final_return),
        xytext=(5, 0),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drawdown(
    returns_series: pd.Series,
    title: str = "Drawdown Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> "plt.Figure":
    """Plot drawdown over time.
    
    Args:
        returns_series: Series of returns.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    # Compute drawdown
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.fill_between(
        drawdown.index, drawdown.values, 0,
        color='red', alpha=0.4
    )
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Drawdown", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Annotate max drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.annotate(
        f'Max DD: {max_dd:.1%}',
        xy=(max_dd_date, max_dd),
        xytext=(10, -20),
        textcoords='offset points',
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='black'),
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_turnover_histogram(
    turnovers: pd.Series,
    title: str = "Portfolio Turnover Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> "plt.Figure":
    """Histogram of portfolio turnover.
    
    Args:
        turnovers: Series of turnover values.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(turnovers, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(turnovers.mean(), color='red', linewidth=2, linestyle='--', label=f'Mean: {turnovers.mean():.1%}')
    
    ax.set_xlabel("Turnover", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_turnover_over_time(
    turnover_series: pd.Series,
    title: str = "Portfolio Turnover Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> "plt.Figure":
    """Plot turnover per rebalance with rolling mean.
    
    Args:
        turnover_series: Series of turnover values indexed by timestamp.
        title: Chart title.
        save_path: Optional path to save figure.
        figsize: Figure size.
    """
    _check_matplotlib()
    series = turnover_series.dropna()
    if series.empty:
        raise ValueError("turnover_series is empty")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values, color=COLORS['primary'], linewidth=1.4, label='Turnover')
    rolling = series.rolling(10, min_periods=1).mean()
    ax.plot(rolling.index, rolling.values, color=COLORS['secondary'], linestyle='--', linewidth=1.8, label='10-period mean')
    mean_turnover = series.mean()
    ax.axhline(mean_turnover, color='gray', linestyle=':', linewidth=1.2, label=f'Mean: {mean_turnover:.1%}')
    ax.set_ylabel("Turnover")
    ax.set_xlabel("Rebalance")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    _format_timestamp_axis(ax, series.index)
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_cost_impact(
    returns_post_fee: pd.Series,
    returns_pre_fee: pd.Series,
    turnover_series: Optional[pd.Series] = None,
    title: str = "Cost Impact (Pre vs Post Fee)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """Compare pre-fee vs post-fee performance and visualize cost drag.
    
    Args:
        returns_post_fee: Net returns after transaction costs.
        returns_pre_fee: Gross returns before costs.
        turnover_series: Optional turnover series to overlay.
        title: Chart title.
        save_path: Optional path to save figure.
        figsize: Figure size.
    """
    _check_matplotlib()
    aligned = pd.concat(
        [
            returns_pre_fee.rename("pre_fee"),
            returns_post_fee.rename("post_fee"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("No overlapping data for pre/post-fee returns")
    cost_drag = aligned["pre_fee"] - aligned["post_fee"]
    cum_pre = (1 + aligned["pre_fee"]).cumprod() - 1
    cum_post = (1 + aligned["post_fee"]).cumprod() - 1
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1, ax2 = axes
    ax1.plot(cum_pre.index, cum_pre.values, color=COLORS['primary'], linewidth=1.8, label='Pre-fee')
    ax1.plot(cum_post.index, cum_post.values, color=COLORS['secondary'], linewidth=1.8, label='Post-fee')
    ax1.fill_between(cum_post.index, cum_post.values, cum_pre.values, color='gray', alpha=0.2, label='Cost drag')
    final_drag = cum_pre.iloc[-1] - cum_post.iloc[-1]
    ax1.annotate(
        f'Drag: {final_drag:.2%}',
        xy=(cum_post.index[-1], cum_post.iloc[-1]),
        xytext=(5, -15),
        textcoords='offset points',
        fontsize=9,
    )
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Pre vs Post-fee Cumulative Returns", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    _format_timestamp_axis(ax1, cum_post.index)
    ax1.legend()
    colors = [COLORS['negative'] if v < 0 else COLORS['secondary'] for v in cost_drag]
    ax2.bar(aligned.index, cost_drag.values, color=colors, width=0.8)
    ax2.axhline(cost_drag.mean(), color='black', linestyle='--', linewidth=1.2, label=f'Mean drag: {cost_drag.mean():.3%}')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel("Cost Drag per Period")
    ax2.set_title("Cost Drag and Turnover", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
    _format_timestamp_axis(ax2, aligned.index)
    if turnover_series is not None and len(turnover_series) > 0:
        aligned_turnover = turnover_series.reindex(aligned.index)
        if aligned_turnover.notna().any():
            ax2b = ax2.twinx()
            ax2b.plot(aligned_turnover.index, aligned_turnover.values, color=COLORS['primary'], linewidth=1.3, label='Turnover')
            ax2b.set_ylabel("Turnover")
            ax2b.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax2b.legend(loc='upper right')
    ax2.legend(loc='lower left')
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


# =============================================================================
# COMBINED DASHBOARD
# =============================================================================

def create_ranking_dashboard(
    ic_series: pd.Series,
    quintile_df: pd.DataFrame,
    returns_series: pd.Series,
    title: str = "Ranking Model Performance Dashboard",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> "plt.Figure":
    """Create a combined dashboard with all key charts.
    
    Args:
        ic_series: Series of IC values.
        quintile_df: DataFrame with quintile returns.
        returns_series: Series of strategy returns.
        title: Dashboard title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)  # Quintile returns
    ax2 = fig.add_subplot(2, 2, 2)  # IC time series
    ax3 = fig.add_subplot(2, 2, 3)  # Cumulative returns
    ax4 = fig.add_subplot(2, 2, 4)  # IC distribution
    
    # 1. Quintile Returns
    quintile_cols = [col for col in quintile_df.columns if col.startswith("Q")]
    avg_returns = quintile_df[quintile_cols].mean()
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(quintile_cols)))
    ax1.bar(quintile_cols, avg_returns.values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel("Average Return")
    ax1.set_title("Return by Quintile")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # 2. IC Time Series
    ax2.plot(ic_series.index, ic_series.values, alpha=0.4, color='steelblue', linewidth=1)
    rolling_mean = ic_series.rolling(20, min_periods=1).mean()
    ax2.plot(rolling_mean.index, rolling_mean.values, color='darkred', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(ic_series.mean(), color='green', linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.set_ylabel("IC")
    ax2.set_title(f"IC Over Time (Mean: {ic_series.mean():.4f})")
    
    # 3. Cumulative Returns
    strategy_cum = (1 + returns_series).cumprod() - 1
    ax3.plot(strategy_cum.index, strategy_cum.values, color='steelblue', linewidth=2)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.fill_between(strategy_cum.index, 0, strategy_cum.values,
                     where=(strategy_cum.values >= 0), color='green', alpha=0.1)
    ax3.fill_between(strategy_cum.index, 0, strategy_cum.values,
                     where=(strategy_cum.values < 0), color='red', alpha=0.1)
    ax3.set_ylabel("Cumulative Return")
    ax3.set_title(f"Strategy Returns (Total: {strategy_cum.iloc[-1]:.1%})")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 4. IC Distribution
    ax4.hist(ic_series.dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='black', linewidth=1, linestyle='--')
    ax4.axvline(ic_series.mean(), color='red', linewidth=2)
    ax4.set_xlabel("IC")
    ax4.set_ylabel("Frequency")
    ax4.set_title(f"IC Distribution (Hit Rate: {(ic_series > 0).mean():.1%})")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# ADDITIONAL QUINTILE ANALYSIS
# =============================================================================

def plot_quintile_heatmap(
    quintile_df: pd.DataFrame,
    title: str = "Quintile Returns Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> "plt.Figure":
    """Heatmap showing quintile returns over time.
    
    Useful for seeing how quintile performance varies across different periods.
    
    Args:
        quintile_df: DataFrame with columns Q1, Q2, ..., Q5 and timestamp index.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    quintile_cols = [col for col in quintile_df.columns if col.startswith("Q")]
    data = quintile_df[quintile_cols].T
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(data.values, aspect='auto', cmap='RdYlGn', 
                   vmin=-data.abs().max().max(), vmax=data.abs().max().max())
    
    # Labels
    ax.set_yticks(range(len(quintile_cols)))
    ax.set_yticklabels(quintile_cols)
    
    # Reduce x-tick density
    n_ticks = min(20, len(data.columns))
    tick_positions = np.linspace(0, len(data.columns)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(data.columns[i])[:10] for i in tick_positions], rotation=45, ha='right')
    
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Quintile", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Return', format='%.2f')
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_quintile_spread_series(
    quintile_df: pd.DataFrame,
    title: str = "Long-Short Quintile Spread Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> "plt.Figure":
    """Plot the Q5-Q1 spread over time.
    
    Args:
        quintile_df: DataFrame with columns Q1, Q2, ..., Q5.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    spread = quintile_df['Q5'] - quintile_df['Q1']
    cumulative_spread = (1 + spread).cumprod() - 1
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Period spread
    ax1 = axes[0]
    colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in spread.values]
    ax1.bar(range(len(spread)), spread.values, color=colors, alpha=0.7, width=1.0)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axhline(spread.mean(), color=COLORS['secondary'], linewidth=1.5, linestyle='--', 
                label=f'Mean: {spread.mean():.2%}')
    ax1.set_ylabel("Q5-Q1 Spread")
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Cumulative spread
    ax2 = axes[1]
    ax2.fill_between(range(len(cumulative_spread)), 0, cumulative_spread.values,
                     where=(cumulative_spread.values >= 0), color=COLORS['positive'], alpha=0.3)
    ax2.fill_between(range(len(cumulative_spread)), 0, cumulative_spread.values,
                     where=(cumulative_spread.values < 0), color=COLORS['negative'], alpha=0.3)
    ax2.plot(range(len(cumulative_spread)), cumulative_spread.values, 
             color=COLORS['primary'], linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("Time Period")
    ax2.set_ylabel("Cumulative Spread")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# ADDITIONAL IC ANALYSIS
# =============================================================================

def plot_ic_by_window(
    window_ics: Dict[int, pd.Series],
    title: str = "IC by Rolling Window",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """Box plot of IC distribution by rolling window.
    
    Args:
        window_ics: Dictionary mapping window_id to IC series.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data = [window_ics[w].dropna().values for w in sorted(window_ics.keys())]
    labels = [f"Window {w+1}" for w in sorted(window_ics.keys())]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Rolling Window", fontsize=10)
    ax.set_ylabel("IC", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add mean IC annotation for each window
    means = [np.mean(d) for d in data]
    for i, m in enumerate(means):
        ax.annotate(f'{m:.3f}', xy=(i+1, m), xytext=(0, 5),
                   textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_rank_ic_comparison(
    ic_series: pd.Series,
    rank_ic_series: pd.Series,
    rolling_window: int = 20,
    title: str = "IC vs Rank IC Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """Compare Pearson IC and Spearman Rank IC over time.
    
    Args:
        ic_series: Pearson IC series.
        rank_ic_series: Spearman Rank IC series.
        rolling_window: Window for rolling average.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Rolling means
    ic_rolling = ic_series.rolling(rolling_window, min_periods=1).mean()
    rank_ic_rolling = rank_ic_series.rolling(rolling_window, min_periods=1).mean()
    
    # Top plot: Both IC series
    ax1 = axes[0]
    ax1.plot(ic_series.index, ic_rolling.values, color=COLORS['primary'], 
             linewidth=2, label=f'Pearson IC (Mean: {ic_series.mean():.4f})')
    ax1.plot(rank_ic_series.index, rank_ic_rolling.values, color=COLORS['secondary'],
             linewidth=2, label=f'Rank IC (Mean: {rank_ic_series.mean():.4f})')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel(f"{rolling_window}-Period Rolling IC")
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    
    # Bottom plot: Difference
    ax2 = axes[1]
    diff = rank_ic_series - ic_series
    ax2.fill_between(diff.index, 0, diff.values,
                     where=(diff.values >= 0), color=COLORS['positive'], alpha=0.5, label='Rank IC > IC')
    ax2.fill_between(diff.index, 0, diff.values,
                     where=(diff.values < 0), color=COLORS['negative'], alpha=0.5, label='Rank IC < IC')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Rank IC - IC")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_ic_rolling_stats(
    ic_series: pd.Series,
    windows: List[int] = [10, 20, 50],
    title: str = "IC Rolling Statistics",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> "plt.Figure":
    """Plot IC with multiple rolling windows and confidence bands.
    
    Args:
        ic_series: Series of IC values.
        windows: List of rolling window sizes.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Top: IC with rolling means
    ax1 = axes[0]
    ax1.plot(ic_series.index, ic_series.values, alpha=0.3, color=COLORS['neutral'], 
             linewidth=0.5, label='Daily IC')
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['highlight']]
    for window, color in zip(windows, colors):
        rolling = ic_series.rolling(window, min_periods=1).mean()
        ax1.plot(rolling.index, rolling.values, color=color, linewidth=1.5,
                label=f'{window}-period MA')
    
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axhline(ic_series.mean(), color=COLORS['positive'], linewidth=1.5, 
                linestyle='--', alpha=0.7, label=f'Overall Mean: {ic_series.mean():.4f}')
    ax1.set_ylabel("IC")
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', ncol=2)
    
    # Bottom: Rolling IC std (volatility)
    ax2 = axes[1]
    for window, color in zip(windows, colors):
        rolling_std = ic_series.rolling(window, min_periods=1).std()
        ax2.plot(rolling_std.index, rolling_std.values, color=color, linewidth=1.5,
                label=f'{window}-period Std')
    
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Rolling IC Std")
    ax2.set_title("IC Volatility Over Time")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================

def plot_monthly_returns_heatmap(
    returns_series: pd.Series,
    title: str = "Monthly Returns Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> "plt.Figure":
    """Create a calendar heatmap of monthly returns.
    
    Args:
        returns_series: Series of returns with datetime index.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    # Convert index to datetime if needed
    if not isinstance(returns_series.index, pd.DatetimeIndex):
        try:
            # Try treating as millisecond timestamps
            returns_series = returns_series.copy()
            returns_series.index = pd.to_datetime(returns_series.index, unit='ms')
        except Exception:
            # Create a simple sequential plot instead
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(range(len(returns_series)), returns_series.values)
            ax.set_title(f"{title} (non-datetime index)")
            _save_figure(fig, save_path)
            return fig
    
    # Resample to monthly (use 'ME' for month-end to avoid deprecation warning)
    monthly = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    # Create year/month pivot
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values,
    })
    
    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    max_val = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=-max_val, vmax=max_val)
    
    # Labels
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    
    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                text_color = 'white' if abs(val) > max_val * 0.5 else 'black'
                ax.text(j, i, f'{val:.1%}', ha='center', va='center', 
                       fontsize=8, color=text_color)
    
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Year", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, format='%.1%%', label='Return')
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_rolling_sharpe(
    returns_series: pd.Series,
    windows: List[int] = [63, 126, 252],  # ~3mo, 6mo, 1yr
    title: str = "Rolling Sharpe Ratio",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """Plot rolling Sharpe ratio over time.
    
    Args:
        returns_series: Series of returns.
        windows: List of rolling window sizes in periods.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['highlight']]
    labels = ['Short-term', 'Medium-term', 'Long-term']
    
    for window, color, label in zip(windows, colors, labels):
        rolling_mean = returns_series.rolling(window).mean()
        rolling_std = returns_series.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)  # Annualized
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=color, 
                linewidth=1.5, label=f'{label} ({window}d)')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(1.0, color=COLORS['positive'], linewidth=1, linestyle='--', alpha=0.5, label='Sharpe = 1')
    ax.axhline(-1.0, color=COLORS['negative'], linewidth=1, linestyle='--', alpha=0.5)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling Sharpe Ratio")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Shade regions
    ax.axhspan(1, 3, alpha=0.1, color=COLORS['positive'], label='Good (1-3)')
    ax.axhspan(-1, 1, alpha=0.05, color=COLORS['neutral'])
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_returns_distribution(
    returns_series: pd.Series,
    title: str = "Returns Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> "plt.Figure":
    """Distribution of portfolio returns with statistics.
    
    Args:
        returns_series: Series of returns.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    returns = returns_series.dropna()
    
    # Left: Histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(returns, bins=50, density=True, alpha=0.7, 
                                 color=COLORS['primary'], edgecolor='black', linewidth=0.5)
    
    # Color bars by sign
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor(COLORS['negative'])
        else:
            patch.set_facecolor(COLORS['positive'])
    
    # Add normal distribution overlay
    from scipy import stats
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, std), color='black', linewidth=2, 
             linestyle='--', label='Normal Fit')
    
    ax1.axvline(0, color='black', linewidth=1)
    ax1.axvline(mu, color=COLORS['secondary'], linewidth=2, label=f'Mean: {mu:.2%}')
    ax1.set_xlabel("Return")
    ax1.set_ylabel("Density")
    ax1.set_title("Return Distribution", fontsize=11)
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Right: Q-Q plot
    ax2 = axes[1]
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (vs Normal)", fontsize=11)
    ax2.get_lines()[0].set_markerfacecolor(COLORS['primary'])
    ax2.get_lines()[0].set_markersize(4)
    
    # Statistics text box
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    stats_text = f"Mean: {mu:.2%}\nStd: {std:.2%}\nSkew: {skew:.2f}\nKurtosis: {kurt:.2f}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, save_path)
    
    return fig


def plot_underwater(
    returns_series: pd.Series,
    title: str = "Underwater Plot (Time Spent in Drawdown)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> "plt.Figure":
    """Plot underwater chart showing time spent in drawdown.
    
    Args:
        returns_series: Series of returns.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.fill_between(drawdown.index, 0, drawdown.values, 
                    color=COLORS['negative'], alpha=0.6)
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=0.5)
    
    # Highlight max drawdown period
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.scatter([max_dd_idx], [max_dd_val], color='black', s=100, zorder=5, 
               marker='v', label=f'Max DD: {max_dd_val:.1%}')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower right')
    
    # Add recovery periods annotation
    in_drawdown = (drawdown < -0.01).sum() / len(drawdown) * 100
    ax.text(0.02, 0.02, f'Time in Drawdown (>1%): {in_drawdown:.1f}%', 
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# FACTOR ANALYSIS
# =============================================================================

def plot_feature_importance(
    importances: Dict[str, float],
    top_n: int = 20,
    title: str = "Top Feature Importances",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> "plt.Figure":
    """Horizontal bar chart of feature importances.
    
    Args:
        importances: Dictionary mapping feature name to importance.
        top_n: Number of top features to display.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    # Sort and take top N
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = range(len(features))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(features)))
    
    ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top feature at top
    
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add value labels
    for i, (feature, value) in enumerate(zip(features, values)):
        ax.text(value + max(values) * 0.01, i, f'{value:.4f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_prediction_vs_actual(
    predicted: pd.Series,
    actual: pd.Series,
    title: str = "Predicted Score vs Actual Return",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> "plt.Figure":
    """Scatter plot of predictions vs actual returns.
    
    Args:
        predicted: Predicted scores.
        actual: Actual returns.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sample if too many points
    n_points = len(predicted)
    if n_points > 5000:
        idx = np.random.choice(n_points, 5000, replace=False)
        pred_sample = predicted.iloc[idx]
        actual_sample = actual.iloc[idx]
    else:
        pred_sample = predicted
        actual_sample = actual
    
    # Scatter plot with transparency
    ax.scatter(pred_sample, actual_sample, alpha=0.2, s=10, color=COLORS['primary'])
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, _, _ = stats.linregress(pred_sample, actual_sample)
    x_line = np.array([pred_sample.min(), pred_sample.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=COLORS['negative'], linewidth=2, 
            label=f'Regression (R²={r_value**2:.4f})')
    
    # Add quintile means
    try:
        quintiles = pd.qcut(pred_sample, 5, labels=False, duplicates='drop')
        q_means = pd.DataFrame({'pred': pred_sample, 'actual': actual_sample, 'q': quintiles})
        q_summary = q_means.groupby('q').mean()
        ax.scatter(q_summary['pred'], q_summary['actual'], color=COLORS['secondary'],
                  s=200, marker='D', edgecolor='black', linewidth=2, zorder=5,
                  label='Quintile Means')
    except Exception as e:
        import warnings
        warnings.warn(f"Could not compute quintile means for scatter plot: {e}")
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    ax.set_xlabel("Predicted Score", fontsize=10)
    ax.set_ylabel("Actual Return", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_prediction_distribution(
    predictions: pd.Series,
    title: str = "Prediction Score Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> "plt.Figure":
    """Distribution of model prediction scores.
    
    Args:
        predictions: Series of prediction scores.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Histogram
    ax1 = axes[0]
    ax1.hist(predictions.dropna(), bins=50, color=COLORS['primary'], 
             edgecolor='black', alpha=0.7, density=True)
    ax1.axvline(predictions.mean(), color=COLORS['secondary'], linewidth=2, 
                label=f'Mean: {predictions.mean():.4f}')
    ax1.axvline(predictions.median(), color=COLORS['highlight'], linewidth=2, 
                linestyle='--', label=f'Median: {predictions.median():.4f}')
    ax1.set_xlabel("Prediction Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution", fontsize=11)
    ax1.legend()
    
    # Right: Box plot by timestamp (if index is available)
    ax2 = axes[1]
    ax2.boxplot([predictions.dropna().values], vert=True)
    ax2.set_ylabel("Prediction Score")
    ax2.set_title("Box Plot", fontsize=11)
    
    # Add statistics
    stats_text = (f"Min: {predictions.min():.4f}\n"
                  f"Q1: {predictions.quantile(0.25):.4f}\n"
                  f"Median: {predictions.median():.4f}\n"
                  f"Q3: {predictions.quantile(0.75):.4f}\n"
                  f"Max: {predictions.max():.4f}\n"
                  f"Std: {predictions.std():.4f}")
    ax2.text(1.3, predictions.median(), stats_text, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, save_path)
    
    return fig


def plot_hit_rate_over_time(
    predictions_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    predicted_col: str = "predicted_score",
    actual_col: str = "actual_return",
    top_n: int = 10,
    title: str = "Hit Rate Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> "plt.Figure":
    """Plot hit rate (% of top picks with positive returns) over time.
    
    Args:
        predictions_df: DataFrame with predictions and actuals.
        timestamp_col: Column name for timestamp.
        predicted_col: Column name for predicted scores.
        actual_col: Column name for actual returns.
        top_n: Number of top picks to consider.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    hit_rates = []
    timestamps = []
    
    for ts, group in predictions_df.groupby(timestamp_col):
        if len(group) >= top_n:
            top_picks = group.nlargest(top_n, predicted_col)
            hit_rate = (top_picks[actual_col] > 0).mean()
            hit_rates.append(hit_rate)
            timestamps.append(ts)
    
    hit_rate_series = pd.Series(hit_rates, index=timestamps)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bar plot
    colors = [COLORS['positive'] if hr > 0.5 else COLORS['negative'] for hr in hit_rates]
    ax.bar(range(len(hit_rates)), hit_rates, color=colors, alpha=0.7, width=1.0)
    
    # Rolling average
    rolling = hit_rate_series.rolling(20, min_periods=1).mean()
    ax.plot(range(len(rolling)), rolling.values, color='black', linewidth=2, 
            label=f'20-period MA')
    
    ax.axhline(0.5, color='black', linewidth=1, linestyle='--', label='50% (Random)')
    ax.axhline(hit_rate_series.mean(), color=COLORS['secondary'], linewidth=2, 
               linestyle='--', label=f'Mean: {hit_rate_series.mean():.1%}')
    
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Hit Rate")
    ax.set_title(f"{title} (Top {top_n} Picks)", fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# COMPREHENSIVE FIGURE GENERATION
# =============================================================================

def generate_all_figures(
    predictions_df: pd.DataFrame,
    ic_series: pd.Series,
    rank_ic_series: pd.Series,
    quintile_df: pd.DataFrame,
    returns_series: pd.Series,
    output_dir: str,
    feature_importances: Optional[Dict[str, float]] = None,
    timestamp_col: str = "timestamp",
    predicted_col: str = "predicted_score",
    actual_col: str = "actual_return",
) -> Dict[str, str]:
    """Generate all visualization figures and save to output directory.
    
    Args:
        predictions_df: DataFrame with timestamp, ticker, predicted_score, actual_return.
        ic_series: Pearson IC series.
        rank_ic_series: Spearman Rank IC series.
        quintile_df: DataFrame with quintile returns (Q1-Q5 columns).
        returns_series: Portfolio returns series.
        output_dir: Directory to save figures.
        feature_importances: Optional dictionary of feature importances.
        timestamp_col: Column name for timestamp.
        predicted_col: Column name for predicted scores.
        actual_col: Column name for actual returns.
    
    Returns:
        Dictionary mapping figure name to file path.
    """
    _check_matplotlib()
    
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    saved_figures = {}
    
    # 1. QUINTILE ANALYSIS
    print("  Generating quintile analysis figures...")
    
    # Quintile returns bar chart
    fig = plot_quintile_returns(quintile_df)
    path = str(figures_dir / "01_quintile_returns.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['quintile_returns'] = path
    
    # Quintile cumulative returns
    fig = plot_quintile_cumulative_returns(quintile_df)
    path = str(figures_dir / "02_quintile_cumulative.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['quintile_cumulative'] = path
    
    # Quintile heatmap
    if len(quintile_df) > 5:
        fig = plot_quintile_heatmap(quintile_df)
        path = str(figures_dir / "03_quintile_heatmap.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['quintile_heatmap'] = path
    
    # Quintile spread series
    if 'Q5' in quintile_df.columns and 'Q1' in quintile_df.columns:
        fig = plot_quintile_spread_series(quintile_df)
        path = str(figures_dir / "04_quintile_spread.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['quintile_spread'] = path
    
    # 2. IC ANALYSIS
    print("  Generating IC analysis figures...")
    
    # IC time series
    fig = plot_ic_series(ic_series)
    path = str(figures_dir / "05_ic_series.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['ic_series'] = path
    
    # IC distribution
    fig = plot_ic_distribution(ic_series)
    path = str(figures_dir / "06_ic_distribution.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['ic_distribution'] = path
    
    # IC vs Rank IC comparison
    if len(rank_ic_series) > 0:
        fig = plot_rank_ic_comparison(ic_series, rank_ic_series)
        path = str(figures_dir / "07_ic_vs_rank_ic.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['ic_vs_rank_ic'] = path
    
    # IC rolling stats
    fig = plot_ic_rolling_stats(ic_series)
    path = str(figures_dir / "08_ic_rolling_stats.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['ic_rolling_stats'] = path
    
    # 3. BACKTEST & PERFORMANCE
    print("  Generating performance figures...")
    
    # Cumulative returns
    fig = plot_cumulative_returns(returns_series)
    path = str(figures_dir / "09_cumulative_returns.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['cumulative_returns'] = path
    
    # Drawdown
    fig = plot_drawdown(returns_series)
    path = str(figures_dir / "10_drawdown.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['drawdown'] = path
    
    # Underwater plot
    fig = plot_underwater(returns_series)
    path = str(figures_dir / "11_underwater.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['underwater'] = path
    
    # Monthly returns heatmap
    try:
        fig = plot_monthly_returns_heatmap(returns_series)
        path = str(figures_dir / "12_monthly_heatmap.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['monthly_heatmap'] = path
    except Exception as e:
        print(f"    Warning: Could not generate monthly heatmap: {e}")
    
    # Rolling Sharpe
    if len(returns_series) > 63:
        fig = plot_rolling_sharpe(returns_series)
        path = str(figures_dir / "13_rolling_sharpe.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['rolling_sharpe'] = path
    
    # Returns distribution
    fig = plot_returns_distribution(returns_series)
    path = str(figures_dir / "14_returns_distribution.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['returns_distribution'] = path
    
    # 4. FACTOR ANALYSIS
    print("  Generating factor analysis figures...")
    
    # Prediction vs Actual scatter
    fig = plot_prediction_vs_actual(
        predictions_df[predicted_col],
        predictions_df[actual_col]
    )
    path = str(figures_dir / "15_prediction_vs_actual.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['prediction_vs_actual'] = path
    
    # Prediction distribution
    fig = plot_prediction_distribution(predictions_df[predicted_col])
    path = str(figures_dir / "16_prediction_distribution.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['prediction_distribution'] = path
    
    # Hit rate over time
    fig = plot_hit_rate_over_time(
        predictions_df, timestamp_col, predicted_col, actual_col
    )
    path = str(figures_dir / "17_hit_rate.png")
    _save_figure(fig, path)
    plt.close(fig)
    saved_figures['hit_rate'] = path
    
    # Feature importances (if provided)
    if feature_importances and len(feature_importances) > 0:
        fig = plot_feature_importance(feature_importances)
        path = str(figures_dir / "18_feature_importance.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['feature_importance'] = path
    
    # 5. DASHBOARDS
    print("  Generating dashboard...")
    
    # Main dashboard
    fig = create_ranking_dashboard(
        ic_series, quintile_df, returns_series,
        title="Ranking Model Performance Dashboard"
    )
    path = str(figures_dir / "00_dashboard.png")
    _save_figure(fig, path, dpi=200)
    plt.close(fig)
    saved_figures['dashboard'] = path
    
    print(f"  Generated {len(saved_figures)} figures in {figures_dir}")
    
    return saved_figures


def create_performance_summary_figure(
    metrics: Dict[str, Any],
    backtest_metrics: Dict[str, Any],
    title: str = "Performance Summary",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> "plt.Figure":
    """Create a summary figure with key metrics displayed as text.
    
    Args:
        metrics: Dictionary of ranking metrics (IC, ICIR, etc.).
        backtest_metrics: Dictionary of backtest metrics (Sharpe, etc.).
        title: Figure title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Build text content
    lines = []
    lines.append("═" * 50)
    lines.append(f"  {title.upper()}")
    lines.append("═" * 50)
    lines.append("")
    lines.append("  RANKING METRICS")
    lines.append("  " + "-" * 40)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key.lower() or 'spread' in key.lower():
                lines.append(f"  {key:.<30} {value:>10.2%}")
            else:
                lines.append(f"  {key:.<30} {value:>10.4f}")
        else:
            lines.append(f"  {key:.<30} {value}")
    
    lines.append("")
    lines.append("  BACKTEST RESULTS")
    lines.append("  " + "-" * 40)
    
    for key, value in backtest_metrics.items():
        if isinstance(value, float):
            if 'return' in key.lower() or 'drawdown' in key.lower() or 'turnover' in key.lower():
                lines.append(f"  {key:.<30} {value:>10.2%}")
            else:
                lines.append(f"  {key:.<30} {value:>10.2f}")
        else:
            lines.append(f"  {key:.<30} {value}")
    
    lines.append("")
    lines.append("═" * 50)
    
    text = "\n".join(lines)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# EXTENDED VISUALIZATIONS - DECILE ANALYSIS
# =============================================================================

def plot_decile_returns(
    decile_returns: Dict[int, float],
    title: str = "Return by Predicted Decile",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> "plt.Figure":
    """Bar chart of average returns by decile (10 groups).
    
    More granular than quintiles for detailed analysis.
    
    Args:
        decile_returns: Dictionary mapping decile (1-10) to average return.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    deciles = sorted(decile_returns.keys())
    returns = [decile_returns.get(d, 0) for d in deciles]
    labels = [f"D{d}" for d in deciles]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(deciles)))
    bars = ax.bar(labels, returns, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, returns):
        height = bar.get_height()
        ax.annotate(
            f'{val:.2%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=8,
        )
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Decile (D1=Lowest Predicted, D10=Highest Predicted)", fontsize=10)
    ax.set_ylabel("Average Return", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add spread annotation
    spread = returns[-1] - returns[0] if returns else 0
    ax.annotate(
        f'D10-D1 Spread: {spread:.2%}',
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        ha='right', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# WIN STREAK ANALYSIS
# =============================================================================

def plot_win_streak_analysis(
    returns_series: pd.Series,
    title: str = "Win/Loss Streak Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> "plt.Figure":
    """Visualize consecutive win/loss streaks over time.
    
    Args:
        returns_series: Series of returns.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    returns = returns_series.dropna()
    
    # Left: Streak bar chart over time
    ax1 = axes[0]
    
    streaks = []
    current_streak = 0
    streak_type = None  # 'win' or 'loss'
    
    for ret in returns:
        if ret > 0:
            if streak_type == 'win':
                current_streak += 1
            else:
                if current_streak != 0:
                    streaks.append((streak_type, current_streak))
                current_streak = 1
                streak_type = 'win'
        else:
            if streak_type == 'loss':
                current_streak += 1
            else:
                if current_streak != 0:
                    streaks.append((streak_type, current_streak))
                current_streak = 1
                streak_type = 'loss'
    
    if current_streak != 0:
        streaks.append((streak_type, current_streak))
    
    # Plot streaks
    win_streaks = [s[1] for s in streaks if s[0] == 'win']
    loss_streaks = [s[1] for s in streaks if s[0] == 'loss']
    
    x_positions = range(len(streaks))
    colors_streak = [COLORS['positive'] if s[0] == 'win' else COLORS['negative'] for s in streaks]
    heights = [s[1] if s[0] == 'win' else -s[1] for s in streaks]
    
    ax1.bar(x_positions, heights, color=colors_streak, alpha=0.7, width=0.8)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel("Streak Number")
    ax1.set_ylabel("Streak Length (+ = wins, - = losses)")
    ax1.set_title("Win/Loss Streaks Over Time", fontsize=11)
    
    # Add max streak annotations
    if win_streaks:
        max_win = max(win_streaks)
        ax1.axhline(max_win, color=COLORS['positive'], linestyle='--', alpha=0.5, 
                   label=f'Max Win Streak: {max_win}')
    if loss_streaks:
        max_loss = max(loss_streaks)
        ax1.axhline(-max_loss, color=COLORS['negative'], linestyle='--', alpha=0.5,
                   label=f'Max Loss Streak: {max_loss}')
    ax1.legend(loc='upper right')
    
    # Right: Streak length distribution
    ax2 = axes[1]
    
    if win_streaks and loss_streaks:
        ax2.hist([win_streaks, loss_streaks], bins=range(1, max(max(win_streaks), max(loss_streaks)) + 2),
                label=['Win Streaks', 'Loss Streaks'], color=[COLORS['positive'], COLORS['negative']],
                alpha=0.7, edgecolor='black')
    elif win_streaks:
        ax2.hist(win_streaks, bins=range(1, max(win_streaks) + 2), label='Win Streaks',
                color=COLORS['positive'], alpha=0.7, edgecolor='black')
    elif loss_streaks:
        ax2.hist(loss_streaks, bins=range(1, max(loss_streaks) + 2), label='Loss Streaks',
                color=COLORS['negative'], alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel("Streak Length")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Streak Length Distribution", fontsize=11)
    ax2.legend()
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# IC AUTOCORRELATION ANALYSIS
# =============================================================================

def plot_ic_autocorrelation(
    ic_series: pd.Series,
    max_lag: int = 20,
    title: str = "IC Autocorrelation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> "plt.Figure":
    """Plot autocorrelation of IC series.
    
    High autocorrelation suggests predictable IC patterns.
    
    Args:
        ic_series: Series of IC values.
        max_lag: Maximum lag to compute.
        title: Chart title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    ic_clean = ic_series.dropna()
    
    if len(ic_clean) < max_lag + 5:
        max_lag = max(len(ic_clean) - 5, 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    lags = range(1, max_lag + 1)
    autocorrs = [ic_clean.autocorr(lag=lag) for lag in lags]
    
    # Bar plot
    colors = [COLORS['positive'] if ac > 0 else COLORS['negative'] for ac in autocorrs]
    ax.bar(lags, autocorrs, color=colors, alpha=0.7, edgecolor='black')
    
    # Significance bands (approximate 95% CI)
    n = len(ic_clean)
    sig_level = 1.96 / np.sqrt(n)
    ax.axhline(sig_level, color='gray', linestyle='--', alpha=0.7, label='95% CI')
    ax.axhline(-sig_level, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Annotate first lag
    if autocorrs:
        ax.annotate(
            f'Lag-1: {autocorrs[0]:.3f}',
            xy=(1, autocorrs[0]),
            xytext=(5, 20 if autocorrs[0] > 0 else -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9,
        )
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# RISK METRICS VISUALIZATION
# =============================================================================

def plot_risk_metrics_dashboard(
    returns_series: pd.Series,
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Risk Metrics Dashboard",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
) -> "plt.Figure":
    """Comprehensive risk metrics dashboard.
    
    Args:
        returns_series: Series of returns.
        metrics: Pre-computed metrics dict (optional).
        title: Dashboard title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 3x2 grid
    ax1 = fig.add_subplot(2, 3, 1)  # Returns distribution
    ax2 = fig.add_subplot(2, 3, 2)  # Drawdown
    ax3 = fig.add_subplot(2, 3, 3)  # Rolling volatility
    ax4 = fig.add_subplot(2, 3, 4)  # VaR/CVaR
    ax5 = fig.add_subplot(2, 3, 5)  # Underwater
    ax6 = fig.add_subplot(2, 3, 6)  # Metrics summary
    
    returns = returns_series.dropna()
    
    # 1. Returns distribution with VaR markers
    ax1.hist(returns, bins=50, density=True, alpha=0.7, color=COLORS['primary'], 
             edgecolor='black', linewidth=0.3)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    ax1.axvline(var_95, color=COLORS['secondary'], linewidth=2, linestyle='--', 
                label=f'VaR 95%: {var_95:.2%}')
    ax1.axvline(var_99, color=COLORS['negative'], linewidth=2, linestyle='--',
                label=f'VaR 99%: {var_99:.2%}')
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.set_xlabel("Return")
    ax1.set_ylabel("Density")
    ax1.set_title("Returns Distribution with VaR")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # 2. Drawdown series
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    ax2.fill_between(range(len(drawdown)), 0, drawdown.values, color=COLORS['negative'], alpha=0.5)
    ax2.plot(drawdown.values, color='darkred', linewidth=0.5)
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Drawdown")
    ax2.set_title(f"Drawdown (Max: {drawdown.min():.1%})")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 3. Rolling volatility
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    ax3.plot(rolling_vol.values, color=COLORS['primary'], linewidth=1)
    ax3.axhline(rolling_vol.mean(), color=COLORS['secondary'], linestyle='--', 
                label=f'Mean: {rolling_vol.mean():.1%}')
    ax3.fill_between(range(len(rolling_vol)), 0, rolling_vol.values, 
                     color=COLORS['primary'], alpha=0.2)
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Annualized Volatility")
    ax3.set_title("Rolling 20-Period Volatility")
    ax3.legend()
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 4. VaR/CVaR analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    var_values = [np.percentile(returns, p) for p in percentiles]
    colors_var = [COLORS['negative'] if v < 0 else COLORS['positive'] for v in var_values]
    ax4.barh([str(p) + "%" for p in percentiles], var_values, color=colors_var, alpha=0.7)
    ax4.axvline(0, color='black', linewidth=0.5)
    ax4.set_xlabel("Return")
    ax4.set_title("Return Percentiles")
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # 5. Underwater equity curve
    wealth = cumulative
    ax5.fill_between(range(len(wealth)), 1, wealth.values, 
                     where=(wealth.values >= 1), color=COLORS['positive'], alpha=0.3)
    ax5.fill_between(range(len(wealth)), 1, wealth.values,
                     where=(wealth.values < 1), color=COLORS['negative'], alpha=0.3)
    ax5.plot(wealth.values, color=COLORS['primary'], linewidth=1.5)
    ax5.axhline(1, color='black', linewidth=0.5)
    ax5.set_xlabel("Period")
    ax5.set_ylabel("Wealth (Starting = 1)")
    ax5.set_title(f"Equity Curve (Final: {wealth.iloc[-1]:.2f})")
    
    # 6. Metrics summary text
    ax6.axis('off')
    
    # Use pre-computed metrics when available, otherwise compute from returns
    # Note: When computing from returns, assumes daily frequency (252 periods/year)
    from scipy import stats as scipy_stats
    mean_ret = returns.mean() * 252
    std_ret = returns.std() * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    skewness = scipy_stats.skew(returns)
    kurtosis = scipy_stats.kurtosis(returns)
    max_dd = drawdown.min()
    cvar_95 = returns[returns <= var_95].mean()
    
    metrics_text = (
        f"Annualized Return:  {mean_ret:>10.2%}\n"
        f"Annualized Vol:     {std_ret:>10.2%}\n"
        f"Sharpe Ratio:       {sharpe:>10.2f}\n"
        f"Max Drawdown:       {max_dd:>10.2%}\n"
        f"VaR (95%):          {var_95:>10.2%}\n"
        f"CVaR (95%):         {cvar_95:>10.2%}\n"
        f"Skewness:           {skewness:>10.2f}\n"
        f"Kurtosis:           {kurtosis:>10.2f}\n"
        f"Win Rate:           {(returns > 0).mean():>10.2%}\n"
        f"Periods:            {len(returns):>10d}\n"
        f"\n"
        f"Note: Metrics computed from returns\n"
        f"assuming daily frequency (252/year)."
    )
    
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    ax6.set_title("Summary Statistics")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, save_path)
    
    return fig


# =============================================================================
# EXTENDED COMPREHENSIVE DASHBOARD
# =============================================================================

def create_extended_dashboard(
    predictions_df: pd.DataFrame,
    ic_series: pd.Series,
    rank_ic_series: pd.Series,
    quintile_df: pd.DataFrame,
    returns_series: pd.Series,
    metrics_dict: Optional[Dict[str, Any]] = None,
    title: str = "Extended Performance Dashboard",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 16),
    timestamp_col: str = "timestamp",
    predicted_col: str = "predicted_score",
    actual_col: str = "actual_return",
) -> "plt.Figure":
    """Create comprehensive extended dashboard with all key visualizations.
    
    Args:
        predictions_df: DataFrame with predictions.
        ic_series: Pearson IC series.
        rank_ic_series: Spearman Rank IC series.
        quintile_df: DataFrame with quintile returns.
        returns_series: Portfolio returns series.
        metrics_dict: Pre-computed metrics dictionary.
        title: Dashboard title.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        timestamp_col: Column name for timestamp.
        predicted_col: Column name for predictions.
        actual_col: Column name for actual returns.
    
    Returns:
        Matplotlib figure object.
    """
    _check_matplotlib()
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 4x3 grid layout
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.25)
    
    # Row 1: Core metrics
    ax1 = fig.add_subplot(gs[0, 0])  # Quintile returns
    ax2 = fig.add_subplot(gs[0, 1])  # Cumulative returns
    ax3 = fig.add_subplot(gs[0, 2])  # IC series
    
    # Row 2: IC analysis
    ax4 = fig.add_subplot(gs[1, 0])  # IC distribution
    ax5 = fig.add_subplot(gs[1, 1])  # IC vs Rank IC
    ax6 = fig.add_subplot(gs[1, 2])  # IC rolling
    
    # Row 3: Risk/Performance
    ax7 = fig.add_subplot(gs[2, 0])  # Drawdown
    ax8 = fig.add_subplot(gs[2, 1])  # Returns distribution
    ax9 = fig.add_subplot(gs[2, 2])  # Hit rate over time
    
    # Row 4: Advanced
    ax10 = fig.add_subplot(gs[3, 0])  # Prediction vs Actual
    ax11 = fig.add_subplot(gs[3, 1])  # Quintile cumulative
    ax12 = fig.add_subplot(gs[3, 2])  # Metrics summary
    
    quintile_cols = [col for col in quintile_df.columns if col.startswith("Q")]
    returns = returns_series.dropna()
    
    # 1. Quintile Returns
    avg_returns = quintile_df[quintile_cols].mean()
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(quintile_cols)))
    ax1.bar(quintile_cols, avg_returns.values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel("Avg Return")
    ax1.set_title("Quintile Returns", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    spread = avg_returns.iloc[-1] - avg_returns.iloc[0]
    ax1.annotate(f'Spread: {spread:.2%}', xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, bbox=dict(facecolor='wheat', alpha=0.5))
    
    # 2. Cumulative Returns
    cum_ret = (1 + returns).cumprod() - 1
    ax2.fill_between(range(len(cum_ret)), 0, cum_ret.values, 
                     where=(cum_ret.values >= 0), color=COLORS['positive'], alpha=0.3)
    ax2.fill_between(range(len(cum_ret)), 0, cum_ret.values,
                     where=(cum_ret.values < 0), color=COLORS['negative'], alpha=0.3)
    ax2.plot(cum_ret.values, color=COLORS['primary'], linewidth=1.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel("Cumulative Return")
    ax2.set_title(f"Strategy Returns (Total: {cum_ret.iloc[-1]:.1%})", fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 3. IC Series
    ax3.plot(ic_series.values, alpha=0.4, color='steelblue', linewidth=0.5)
    rolling_ic = ic_series.rolling(20, min_periods=1).mean()
    ax3.plot(rolling_ic.values, color='darkred', linewidth=1.5)
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axhline(ic_series.mean(), color='green', linestyle='--', alpha=0.7)
    ax3.set_ylabel("IC")
    ax3.set_title(f"IC Over Time (Mean: {ic_series.mean():.4f})", fontsize=10)
    
    # 4. IC Distribution
    ax4.hist(ic_series.dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='black', linewidth=1, linestyle='--')
    ax4.axvline(ic_series.mean(), color='red', linewidth=2)
    ax4.set_xlabel("IC")
    ax4.set_title(f"IC Distribution (Hit: {(ic_series > 0).mean():.1%})", fontsize=10)
    
    # 5. IC vs Rank IC comparison
    if len(rank_ic_series) > 0:
        rolling_ic = ic_series.rolling(20, min_periods=1).mean()
        rolling_rank_ic = rank_ic_series.rolling(20, min_periods=1).mean()
        ax5.plot(rolling_ic.values, color=COLORS['primary'], linewidth=1.5, label='Pearson IC')
        ax5.plot(rolling_rank_ic.values, color=COLORS['secondary'], linewidth=1.5, label='Rank IC')
        ax5.axhline(0, color='black', linewidth=0.5)
        ax5.set_ylabel("Rolling IC")
        ax5.set_title("IC vs Rank IC (20-period)", fontsize=10)
        ax5.legend(fontsize=8)
    
    # 6. IC rolling with bands
    rolling_mean = ic_series.rolling(20).mean()
    rolling_std = ic_series.rolling(20).std()
    ax6.fill_between(range(len(rolling_mean)), 
                     (rolling_mean - 2*rolling_std).values, 
                     (rolling_mean + 2*rolling_std).values,
                     alpha=0.2, color=COLORS['primary'])
    ax6.plot(rolling_mean.values, color=COLORS['primary'], linewidth=1.5)
    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.set_ylabel("IC")
    ax6.set_title("Rolling IC with 2σ Bands", fontsize=10)
    
    # 7. Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    ax7.fill_between(range(len(drawdown)), 0, drawdown.values, color=COLORS['negative'], alpha=0.5)
    ax7.set_ylabel("Drawdown")
    ax7.set_title(f"Drawdown (Max: {drawdown.min():.1%})", fontsize=10)
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 8. Returns Distribution
    ax8.hist(returns, bins=40, density=True, alpha=0.7, color=COLORS['primary'], edgecolor='black')
    ax8.axvline(returns.mean(), color='red', linewidth=2, label=f'Mean: {returns.mean():.2%}')
    ax8.axvline(0, color='black', linewidth=0.5)
    ax8.set_xlabel("Return")
    ax8.set_title("Returns Distribution", fontsize=10)
    ax8.legend(fontsize=8)
    ax8.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # 9. Hit rate over time
    hit_rates = []
    for ts, group in predictions_df.groupby(timestamp_col):
        if len(group) >= 10:
            top_picks = group.nlargest(10, predicted_col)
            hit_rate = (top_picks[actual_col] > 0).mean()
            hit_rates.append(hit_rate)
    
    if hit_rates:
        colors_hr = [COLORS['positive'] if hr > 0.5 else COLORS['negative'] for hr in hit_rates]
        ax9.bar(range(len(hit_rates)), hit_rates, color=colors_hr, alpha=0.7, width=1.0)
        rolling_hr = pd.Series(hit_rates).rolling(20, min_periods=1).mean()
        ax9.plot(rolling_hr.values, color='black', linewidth=2)
        ax9.axhline(0.5, color='gray', linestyle='--')
        ax9.set_ylabel("Hit Rate")
        ax9.set_title(f"Hit Rate Top-10 (Mean: {np.mean(hit_rates):.1%})", fontsize=10)
        ax9.set_ylim(0, 1)
    
    # 10. Prediction vs Actual scatter (sampled)
    n_sample = min(2000, len(predictions_df))
    sample_idx = np.random.choice(len(predictions_df), n_sample, replace=False)
    pred_sample = predictions_df.iloc[sample_idx][predicted_col]
    actual_sample = predictions_df.iloc[sample_idx][actual_col]
    ax10.scatter(pred_sample, actual_sample, alpha=0.2, s=5, color=COLORS['primary'])
    ax10.axhline(0, color='black', linewidth=0.5)
    ax10.axvline(0, color='black', linewidth=0.5)
    ax10.set_xlabel("Predicted Score")
    ax10.set_ylabel("Actual Return")
    ax10.set_title("Prediction vs Actual", fontsize=10)
    
    # 11. Quintile cumulative returns
    for col, color in zip(quintile_cols, colors):
        q_cum = (1 + quintile_df[col]).cumprod() - 1
        ax11.plot(q_cum.values, label=col, color=color, linewidth=1.5)
    ax11.axhline(0, color='black', linewidth=0.5)
    ax11.set_ylabel("Cumulative Return")
    ax11.set_title("Quintile Cumulative Returns", fontsize=10)
    ax11.legend(fontsize=7, ncol=len(quintile_cols))
    ax11.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 12. Metrics summary
    ax12.axis('off')
    
    # Use pre-computed metrics when available to match console output
    # Otherwise compute basic metrics (for standalone visualization)
    if metrics_dict and 'sharpe_ratio' in metrics_dict:
        # Use pre-computed backtest metrics (correct for any return frequency)
        sharpe = metrics_dict.get('sharpe_ratio', 0)
        mean_return = metrics_dict.get('annualized_return', 0)
        std_return = metrics_dict.get('annualized_volatility', 0)
        max_dd = metrics_dict.get('max_drawdown', 0)
    else:
        # Fallback: compute from returns (assumes daily frequency)
        from scipy import stats as scipy_stats
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return > 0 else 0
        max_dd = drawdown.min()
    
    if metrics_dict:
        metrics_text = "\n".join([
            f"Mean IC:        {metrics_dict.get('mean_ic', 'N/A'):.4f}" if isinstance(metrics_dict.get('mean_ic'), (int, float)) else f"Mean IC: N/A",
            f"ICIR:           {metrics_dict.get('icir', 'N/A'):.4f}" if isinstance(metrics_dict.get('icir'), (int, float)) else f"ICIR: N/A",
            f"Q Spread:       {metrics_dict.get('quintile_spread', 'N/A'):.4f}" if isinstance(metrics_dict.get('quintile_spread'), (int, float)) else f"Q Spread: N/A",
            f"Hit Rate:       {metrics_dict.get('hit_rate_top_n', 'N/A'):.2%}" if isinstance(metrics_dict.get('hit_rate_top_n'), (int, float)) else f"Hit Rate: N/A",
            "-" * 25,
            f"Sharpe Ratio:   {sharpe:.2f}",
            f"Annual Return:  {mean_return:.2%}",
            f"Annual Vol:     {std_return:.2%}",
            f"Max Drawdown:   {max_dd:.2%}",
            f"Win Rate:       {(returns > 0).mean():.2%}",
            "-" * 25,
            f"Periods:        {len(returns)}",
            f"Timestamps:     {metrics_dict.get('num_timestamps', 'N/A')}",
        ])
    else:
        metrics_text = "\n".join([
            f"Mean IC:        {ic_series.mean():.4f}",
            f"IC Std:         {ic_series.std():.4f}",
            f"Q Spread:       {spread:.4f}",
            "-" * 25,
            f"Sharpe Ratio:   {sharpe:.2f}",
            f"Annual Return:  {mean_return:.2%}",
            f"Annual Vol:     {std_return:.2%}",
            f"Max Drawdown:   {max_dd:.2%}",
            f"Win Rate:       {(returns > 0).mean():.2%}",
            f"Periods:        {len(returns)}",
        ])
    
    ax12.text(0.1, 0.5, metrics_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    ax12.set_title("Key Metrics", fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, save_path, dpi=150)
    
    return fig


# =============================================================================
# UPDATED COMPREHENSIVE FIGURE GENERATION
# =============================================================================

def generate_all_figures_extended(
    predictions_df: pd.DataFrame,
    ic_series: pd.Series,
    rank_ic_series: pd.Series,
    quintile_df: pd.DataFrame,
    returns_series: pd.Series,
    output_dir: str,
    feature_importances: Optional[Dict[str, float]] = None,
    metrics_dict: Optional[Dict[str, Any]] = None,
    decile_returns: Optional[Dict[int, float]] = None,
    turnover_series: Optional[pd.Series] = None,
    pre_fee_returns_series: Optional[pd.Series] = None,
    timestamp_col: str = "timestamp",
    predicted_col: str = "predicted_score",
    actual_col: str = "actual_return",
) -> Dict[str, str]:
    """Generate all visualization figures including new extended visualizations.
    
    IMPORTANT: For accurate drawdown and volatility visualization, pass continuous
    daily returns (true_daily_returns) rather than period returns (daily_returns).
    This ensures plots reflect actual intra-period movements, not just rebalance points.
    
    Args:
        predictions_df: DataFrame with timestamp, ticker, predicted_score, actual_return.
        ic_series: Pearson IC series.
        rank_ic_series: Spearman Rank IC series.
        quintile_df: DataFrame with quintile returns (Q1-Q5 columns).
        returns_series: Portfolio returns series (prefer continuous daily returns).
        output_dir: Directory to save figures.
        feature_importances: Optional dictionary of feature importances.
        metrics_dict: Optional pre-computed metrics dictionary.
        decile_returns: Optional decile returns dictionary.
        turnover_series: Optional turnover series for cost/turnover plots.
        pre_fee_returns_series: Optional pre-fee returns series for cost impact.
        timestamp_col: Column name for timestamp.
        predicted_col: Column name for predicted scores.
        actual_col: Column name for actual returns.
    
    Returns:
        Dictionary mapping figure name to file path.
    """
    _check_matplotlib()
    
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    saved_figures = {}
    
    # Generate all standard figures
    print("  Generating standard figures...")
    standard_figures = generate_all_figures(
        predictions_df, ic_series, rank_ic_series, quintile_df, returns_series,
        output_dir, feature_importances, timestamp_col, predicted_col, actual_col
    )
    saved_figures.update(standard_figures)
    
    # Generate extended figures
    print("  Generating extended figures...")
    
    # Extended dashboard
    try:
        fig = create_extended_dashboard(
            predictions_df, ic_series, rank_ic_series, quintile_df, returns_series,
            metrics_dict=metrics_dict,
            title="Extended Performance Dashboard",
            timestamp_col=timestamp_col,
            predicted_col=predicted_col,
            actual_col=actual_col,
        )
        path = str(figures_dir / "00_extended_dashboard.png")
        _save_figure(fig, path, dpi=200)
        plt.close(fig)
        saved_figures['extended_dashboard'] = path
    except Exception as e:
        print(f"    Warning: Could not generate extended dashboard: {e}")
    
    # Decile returns
    if decile_returns:
        try:
            fig = plot_decile_returns(decile_returns)
            path = str(figures_dir / "19_decile_returns.png")
            _save_figure(fig, path)
            plt.close(fig)
            saved_figures['decile_returns'] = path
        except Exception as e:
            print(f"    Warning: Could not generate decile returns: {e}")
    
    # Win streak analysis
    try:
        fig = plot_win_streak_analysis(returns_series)
        path = str(figures_dir / "20_win_streak_analysis.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['win_streak_analysis'] = path
    except Exception as e:
        print(f"    Warning: Could not generate win streak analysis: {e}")
    
    # IC autocorrelation
    try:
        fig = plot_ic_autocorrelation(ic_series)
        path = str(figures_dir / "21_ic_autocorrelation.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['ic_autocorrelation'] = path
    except Exception as e:
        print(f"    Warning: Could not generate IC autocorrelation: {e}")
    
    # Risk metrics dashboard
    try:
        fig = plot_risk_metrics_dashboard(returns_series)
        path = str(figures_dir / "22_risk_dashboard.png")
        _save_figure(fig, path)
        plt.close(fig)
        saved_figures['risk_dashboard'] = path
    except Exception as e:
        print(f"    Warning: Could not generate risk dashboard: {e}")

    # Turnover-focused visuals
    if turnover_series is not None and len(turnover_series) > 0:
        try:
            fig = plot_turnover_histogram(turnover_series)
            path = str(figures_dir / "23_turnover_histogram.png")
            _save_figure(fig, path)
            plt.close(fig)
            saved_figures['turnover_histogram'] = path
        except Exception as e:
            print(f"    Warning: Could not generate turnover histogram: {e}")
        try:
            fig = plot_turnover_over_time(turnover_series)
            path = str(figures_dir / "24_turnover_over_time.png")
            _save_figure(fig, path)
            plt.close(fig)
            saved_figures['turnover_over_time'] = path
        except Exception as e:
            print(f"    Warning: Could not generate turnover over time: {e}")

    # Cost impact (pre vs post-fee)
    if pre_fee_returns_series is not None and len(pre_fee_returns_series) > 0:
        try:
            fig = plot_cost_impact(
                returns_post_fee=returns_series,
                returns_pre_fee=pre_fee_returns_series,
                turnover_series=turnover_series,
            )
            path = str(figures_dir / "25_cost_impact.png")
            _save_figure(fig, path)
            plt.close(fig)
            saved_figures['cost_impact'] = path
        except Exception as e:
            print(f"    Warning: Could not generate cost impact chart: {e}")
    
    print(f"  Generated {len(saved_figures)} figures total in {figures_dir}")
    
    return saved_figures