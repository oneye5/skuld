"""Generate comprehensive analysis report with visualizations.

This script analyzes the Sharpe optimization experiment results and generates
publication-quality visualizations for the documentation.

Usage:
    cd python/ml-pipeline
    uv run python scripts/generate_analysis_report.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RESULTS_PATH = SCRIPT_DIR.parent / "output" / "experiments" / "sharpe_optimization" / "results.json"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    """Load experiment results into a DataFrame."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    
    results = [r for r in data['results'] if r['success']]
    df = pd.DataFrame([{
        'forward_days': r['config']['forward_return_days'],
        'n_estimators': r['config']['n_estimators'],
        'num_leaves': r['config']['num_leaves'],
        'top_n': r['config']['top_n'],
        'sharpe': r['sharpe_ratio'],
        'total_return': r['total_return'],
        'max_drawdown': r['max_drawdown'],
        'mean_ic': r['mean_ic'],
        'std_ic': r['std_ic'],
        'icir': r['icir'],
        'mean_rank_ic': r['mean_rank_ic'],
        'hit_rate': r['hit_rate'],
        'quintile_spread': r['quintile_spread'],
        'runtime_seconds': r['runtime_seconds'],
    } for r in results])
    
    return df


def plot_sharpe_by_horizon(df: pd.DataFrame, output_dir: Path):
    """Box plot of Sharpe ratio distribution by forward return horizon."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create box plot with individual points
    horizon_order = sorted(df['forward_days'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(horizon_order)))
    
    bp = ax.boxplot(
        [df[df['forward_days'] == h]['sharpe'] for h in horizon_order],
        labels=[f'{h}d' for h in horizon_order],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='darkred', markersize=8)
    )
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add scatter points for individual runs
    for i, horizon in enumerate(horizon_order):
        y = df[df['forward_days'] == horizon]['sharpe']
        x = np.random.normal(i + 1, 0.06, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=30, color='black', zorder=3)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good Sharpe (0.5)')
    
    ax.set_xlabel('Forward Return Horizon (days)')
    ax.set_ylabel('Sharpe Ratio (Annualized)')
    ax.set_title('Sharpe Ratio Distribution by Prediction Horizon\n(NZX via Sharesies: 190 bps fees + 15 bps slippage)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sharpe_by_horizon.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sharpe_by_horizon.png")


def plot_return_vs_drawdown(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of total return vs max drawdown, colored by Sharpe."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        df['max_drawdown'] * 100,
        df['total_return'] * 100,
        c=df['sharpe'],
        cmap='RdYlGn',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio')
    
    # Highlight best configurations
    best_sharpe = df.nlargest(5, 'sharpe')
    ax.scatter(
        best_sharpe['max_drawdown'] * 100,
        best_sharpe['total_return'] * 100,
        s=200,
        facecolors='none',
        edgecolors='blue',
        linewidths=2,
        label='Top 5 Sharpe'
    )
    
    # Annotate the best one
    best = df.loc[df['sharpe'].idxmax()]
    ax.annotate(
        f"Best: {best['forward_days']}d, {best['n_estimators']}est, {best['num_leaves']}lv, top{best['top_n']}\n"
        f"Sharpe={best['sharpe']:.2f}, Return={best['total_return']*100:.1f}%",
        xy=(best['max_drawdown'] * 100, best['total_return'] * 100),
        xytext=(30, 10),
        textcoords='offset points',
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='blue'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
    )
    
    ax.set_xlabel('Maximum Drawdown (%)')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Risk-Return Trade-off Across 108 Configurations\n(Each point is a strategy configuration)')
    ax.legend(loc='upper left')
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='20% DD threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_vs_drawdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: return_vs_drawdown.png")


def plot_heatmap_sharpe(df: pd.DataFrame, output_dir: Path):
    """Heatmaps of Sharpe ratio across parameter combinations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Forward days vs num_leaves (averaged over other params)
    pivot1 = df.pivot_table(
        values='sharpe',
        index='forward_days',
        columns='num_leaves',
        aggfunc='mean'
    )
    sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                ax=axes[0, 0], cbar_kws={'label': 'Avg Sharpe'})
    axes[0, 0].set_title('Sharpe: Horizon vs Tree Complexity')
    axes[0, 0].set_xlabel('Number of Leaves')
    axes[0, 0].set_ylabel('Forward Days')
    
    # 2. Forward days vs top_n
    pivot2 = df.pivot_table(
        values='sharpe',
        index='forward_days',
        columns='top_n',
        aggfunc='mean'
    )
    sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                ax=axes[0, 1], cbar_kws={'label': 'Avg Sharpe'})
    axes[0, 1].set_title('Sharpe: Horizon vs Portfolio Size')
    axes[0, 1].set_xlabel('Top N Stocks')
    axes[0, 1].set_ylabel('Forward Days')
    
    # 3. Forward days vs n_estimators
    pivot3 = df.pivot_table(
        values='sharpe',
        index='forward_days',
        columns='n_estimators',
        aggfunc='mean'
    )
    sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                ax=axes[1, 0], cbar_kws={'label': 'Avg Sharpe'})
    axes[1, 0].set_title('Sharpe: Horizon vs Model Size')
    axes[1, 0].set_xlabel('Number of Estimators')
    axes[1, 0].set_ylabel('Forward Days')
    
    # 4. num_leaves vs top_n (at best horizon)
    best_horizon = df.groupby('forward_days')['sharpe'].mean().idxmax()
    df_best = df[df['forward_days'] == best_horizon]
    pivot4 = df_best.pivot_table(
        values='sharpe',
        index='num_leaves',
        columns='top_n',
        aggfunc='mean'
    )
    sns.heatmap(pivot4, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                ax=axes[1, 1], cbar_kws={'label': 'Sharpe'})
    axes[1, 1].set_title(f'Sharpe at Best Horizon ({best_horizon}d): Complexity vs Portfolio')
    axes[1, 1].set_xlabel('Top N Stocks')
    axes[1, 1].set_ylabel('Number of Leaves')
    
    plt.suptitle('Sharpe Ratio Heatmaps Across Parameter Space', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'sharpe_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sharpe_heatmaps.png")


def plot_ic_vs_sharpe(df: pd.DataFrame, output_dir: Path):
    """Scatter plot showing relationship between IC and Sharpe."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # IC vs Sharpe
    ax = axes[0]
    scatter = ax.scatter(
        df['mean_ic'],
        df['sharpe'],
        c=df['forward_days'],
        cmap='viridis',
        s=80,
        alpha=0.7
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Forward Days')
    
    # Add trend line
    z = np.polyfit(df['mean_ic'], df['sharpe'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['mean_ic'].min(), df['mean_ic'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Trend: y={z[0]:.1f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Mean Information Coefficient (IC)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('IC vs Sharpe Ratio\n(Higher IC generally leads to better Sharpe)')
    ax.legend()
    
    # ICIR vs Sharpe
    ax = axes[1]
    scatter = ax.scatter(
        df['icir'],
        df['sharpe'],
        c=df['forward_days'],
        cmap='viridis',
        s=80,
        alpha=0.7
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Forward Days')
    
    # Add trend line
    z = np.polyfit(df['icir'], df['sharpe'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['icir'].min(), df['icir'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Trend: y={z[0]:.1f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Information Coefficient IR (Annualized)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('ICIR vs Sharpe Ratio\n(ICIR measures IC consistency)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ic_vs_sharpe.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: ic_vs_sharpe.png")


def plot_parameter_importance(df: pd.DataFrame, output_dir: Path):
    """Bar chart showing impact of each parameter on Sharpe."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    params = ['forward_days', 'num_leaves', 'n_estimators', 'top_n']
    titles = ['Forward Return Horizon', 'Tree Complexity (Leaves)', 
              'Number of Estimators', 'Portfolio Size (Top N)']
    colors = ['steelblue', 'darkorange', 'forestgreen', 'mediumpurple']
    
    for ax, param, title, color in zip(axes.flat, params, titles, colors):
        agg = df.groupby(param).agg({
            'sharpe': ['mean', 'std', 'max'],
            'total_return': 'mean',
            'max_drawdown': 'mean'
        }).reset_index()
        agg.columns = [param, 'sharpe_mean', 'sharpe_std', 'sharpe_max', 'return_mean', 'dd_mean']
        
        x = np.arange(len(agg))
        width = 0.35
        
        bars = ax.bar(x, agg['sharpe_mean'], width, yerr=agg['sharpe_std'], 
                      capsize=5, color=color, alpha=0.7, label='Mean Sharpe')
        ax.bar(x + width, agg['sharpe_max'], width, color=color, alpha=0.4, 
               hatch='//', label='Max Sharpe')
        
        ax.set_xlabel(title)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'Impact of {title}')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(agg[param].astype(str))
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
        ax.legend(loc='best')
        
        # Add value labels on bars
        for i, (mean, mx) in enumerate(zip(agg['sharpe_mean'], agg['sharpe_max'])):
            ax.annotate(f'{mean:.2f}', xy=(i, mean), ha='center', va='bottom', fontsize=9)
            ax.annotate(f'{mx:.2f}', xy=(i + width, mx), ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Parameter Impact on Sharpe Ratio\n(Error bars show standard deviation across configurations)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: parameter_importance.png")


def plot_top_configurations(df: pd.DataFrame, output_dir: Path):
    """Detailed view of top 15 configurations."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top15 = df.nlargest(15, 'sharpe').reset_index(drop=True)
    
    # Create configuration labels
    labels = [
        f"{r['forward_days']}d-{r['n_estimators']}e-{r['num_leaves']}l-t{r['top_n']}"
        for _, r in top15.iterrows()
    ]
    
    x = np.arange(len(top15))
    width = 0.25
    
    # Normalize metrics for comparison
    bars1 = ax.bar(x - width, top15['sharpe'], width, label='Sharpe', color='steelblue')
    bars2 = ax.bar(x, top15['total_return'], width, label='Total Return', color='forestgreen')
    bars3 = ax.bar(x + width, top15['max_drawdown'], width, label='Max Drawdown', color='indianred')
    
    ax.set_xlabel('Configuration (days-estimators-leaves-topN)')
    ax.set_ylabel('Value')
    ax.set_title('Top 15 Configurations by Sharpe Ratio\nComparing Sharpe, Return, and Drawdown')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_configurations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: top_configurations.png")


def plot_quintile_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze quintile spread across configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Quintile spread by horizon
    ax = axes[0]
    agg = df.groupby('forward_days')['quintile_spread'].agg(['mean', 'std']).reset_index()
    ax.bar(range(len(agg)), agg['mean'] * 100, yerr=agg['std'] * 100, 
           capsize=5, color='teal', alpha=0.7)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([f"{d}d" for d in agg['forward_days']])
    ax.set_xlabel('Forward Return Horizon')
    ax.set_ylabel('Quintile Spread (%)')
    ax.set_title('Quintile Spread by Horizon\n(Q5 - Q1 return, higher is better)')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Quintile spread vs hit rate
    ax = axes[1]
    scatter = ax.scatter(
        df['quintile_spread'] * 100,
        df['hit_rate'] * 100,
        c=df['sharpe'],
        cmap='RdYlGn',
        s=80,
        alpha=0.7
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio')
    ax.set_xlabel('Quintile Spread (%)')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Quintile Spread vs Hit Rate\n(Both should be high for good models)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quintile_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: quintile_analysis.png")


def plot_efficiency_frontier(df: pd.DataFrame, output_dir: Path):
    """Plot efficiency frontier showing return vs risk trade-off."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate annualized return (assuming returns are over test period)
    # This is a simplification - actual annualization depends on test period
    df_plot = df.copy()
    
    scatter = ax.scatter(
        df_plot['max_drawdown'] * 100,
        df_plot['total_return'] * 100,
        c=df_plot['forward_days'],
        cmap='plasma',
        s=100,
        alpha=0.6
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Forward Days')
    
    # Find and plot the efficient frontier (Pareto optimal points)
    # A point is Pareto optimal if no other point has both higher return AND lower drawdown
    pareto_mask = np.ones(len(df_plot), dtype=bool)
    for i in range(len(df_plot)):
        for j in range(len(df_plot)):
            if i != j:
                if (df_plot.iloc[j]['total_return'] >= df_plot.iloc[i]['total_return'] and
                    df_plot.iloc[j]['max_drawdown'] <= df_plot.iloc[i]['max_drawdown'] and
                    (df_plot.iloc[j]['total_return'] > df_plot.iloc[i]['total_return'] or
                     df_plot.iloc[j]['max_drawdown'] < df_plot.iloc[i]['max_drawdown'])):
                    pareto_mask[i] = False
                    break
    
    pareto_df = df_plot[pareto_mask].sort_values('max_drawdown')
    ax.plot(pareto_df['max_drawdown'] * 100, pareto_df['total_return'] * 100, 
            'r-', linewidth=2, label='Efficient Frontier')
    ax.scatter(pareto_df['max_drawdown'] * 100, pareto_df['total_return'] * 100,
               s=200, facecolors='none', edgecolors='red', linewidths=2)
    
    ax.set_xlabel('Maximum Drawdown (%)')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Efficient Frontier: Return vs Drawdown\n(Red line shows Pareto-optimal configurations)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: efficiency_frontier.png")


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate markdown summary table."""
    
    # Best configuration
    best = df.loc[df['sharpe'].idxmax()]
    
    # Summary by horizon
    horizon_summary = df.groupby('forward_days').agg({
        'sharpe': ['mean', 'std', 'max'],
        'total_return': 'mean',
        'max_drawdown': 'mean',
        'mean_ic': 'mean'
    }).round(4)
    
    # Summary by top_n
    topn_summary = df.groupby('top_n').agg({
        'sharpe': ['mean', 'std', 'max'],
        'total_return': 'mean'
    }).round(4)
    
    markdown = f"""
## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Configurations Tested | {len(df)} |
| Best Sharpe Ratio | {best['sharpe']:.3f} |
| Best Total Return | {df['total_return'].max()*100:.1f}% |
| Lowest Max Drawdown | {df['max_drawdown'].min()*100:.1f}% |
| Average Sharpe | {df['sharpe'].mean():.3f} ± {df['sharpe'].std():.3f} |
| Average IC | {df['mean_ic'].mean():.4f} |

## Optimal Configuration

| Parameter | Value |
|-----------|-------|
| Forward Return Days | {int(best['forward_days'])} |
| Number of Estimators | {int(best['n_estimators'])} |
| Number of Leaves | {int(best['num_leaves'])} |
| Portfolio Top N | {int(best['top_n'])} |
| **Sharpe Ratio** | **{best['sharpe']:.3f}** |
| Total Return | {best['total_return']*100:.1f}% |
| Max Drawdown | {best['max_drawdown']*100:.1f}% |
| Mean IC | {best['mean_ic']:.4f} |
| Hit Rate | {best['hit_rate']*100:.1f}% |

## Performance by Forward Return Horizon

| Horizon | Avg Sharpe | Std | Max Sharpe | Avg Return | Avg Drawdown |
|---------|------------|-----|------------|------------|--------------|
"""
    
    for idx, row in horizon_summary.iterrows():
        markdown += f"| {idx}d | {row[('sharpe', 'mean')]:.3f} | {row[('sharpe', 'std')]:.3f} | {row[('sharpe', 'max')]:.3f} | {row[('total_return', 'mean')]*100:.1f}% | {row[('max_drawdown', 'mean')]*100:.1f}% |\n"
    
    markdown += """
## Performance by Portfolio Size

| Top N | Avg Sharpe | Std | Max Sharpe | Avg Return |
|-------|------------|-----|------------|------------|
"""
    
    for idx, row in topn_summary.iterrows():
        markdown += f"| {idx} | {row[('sharpe', 'mean')]:.3f} | {row[('sharpe', 'std')]:.3f} | {row[('sharpe', 'max')]:.3f} | {row[('total_return', 'mean')]*100:.1f}% |\n"
    
    return markdown


def main():
    """Generate all analysis outputs."""
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} successful configurations\n")
    
    print("Generating visualizations...")
    plot_sharpe_by_horizon(df, OUTPUT_DIR)
    plot_return_vs_drawdown(df, OUTPUT_DIR)
    plot_heatmap_sharpe(df, OUTPUT_DIR)
    plot_ic_vs_sharpe(df, OUTPUT_DIR)
    plot_parameter_importance(df, OUTPUT_DIR)
    plot_top_configurations(df, OUTPUT_DIR)
    plot_quintile_analysis(df, OUTPUT_DIR)
    plot_efficiency_frontier(df, OUTPUT_DIR)
    
    print("\nGenerating summary table...")
    summary_md = generate_summary_table(df, OUTPUT_DIR)
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    return df, summary_md


if __name__ == "__main__":
    df, summary = main()
    print("\n" + summary)
