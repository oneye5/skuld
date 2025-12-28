"""Visualization module for ranking-based stock prediction.

This module provides plotting functions for analyzing ranking model performance:
- Quintile return bar charts
- IC time series plots
- Cumulative return equity curves
- Turnover histograms
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


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
