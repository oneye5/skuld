"""Hyperparameter optimization with checkpoint-based resumption.

Optimizes for average annual Sharpe ratio across rolling windows.
Can be interrupted and resumed without losing progress.

Designed for overnight/unattended runs with robust error handling.

Usage:
    uv run python scripts/optimize_hyperparameters.py           # Full optimization (~60 configs)
    uv run python scripts/optimize_hyperparameters.py --fast    # Quick test (5 configs)
    uv run python scripts/optimize_hyperparameters.py --report  # Show results only
    uv run python scripts/optimize_hyperparameters.py --status  # Show progress status
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import gc
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

import numpy as np
import pandas as pd

from core.logging_config import setup_logging, get_logger
from pipeline.ranking_pipeline import run_ranking_pipeline


# =============================================================================
# JSON SERIALIZATION HELPERS
# =============================================================================

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


# =============================================================================
# CHECKPOINT MANAGER (with robust error handling)
# =============================================================================

class CheckpointManager:
    """Manages experiment checkpoints for resumable optimization."""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load existing checkpoint if available."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.results = data.get('results', [])
                    self.start_time = datetime.fromisoformat(data.get('start_time', datetime.now().isoformat()))
                logging.info(f"Loaded {len(self.results)} existing results from checkpoint")
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Checkpoint corrupted, backing up and starting fresh: {e}")
                # Backup corrupted file
                backup_path = self.checkpoint_file.with_suffix('.json.bak')
                if self.checkpoint_file.exists():
                    self.checkpoint_file.rename(backup_path)
                self.results = []
                self.start_time = datetime.now()
        else:
            logging.info("No checkpoint found, starting fresh")
            self.start_time = datetime.now()
    
    def save_checkpoint(self):
        """Save current results to checkpoint file with atomic write."""
        data = {
            'last_updated': datetime.now().isoformat(),
            'start_time': self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
            'num_experiments': len(self.results),
            'num_successful': len([r for r in self.results if 'sharpe_ratio' in r.get('metrics', {})]),
            'num_failed': len([r for r in self.results if 'error' in r.get('metrics', {})]),
            'results': convert_to_json_serializable(self.results),
        }
        
        # Atomic write: write to temp file first, then rename
        temp_file = self.checkpoint_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.checkpoint_file)
            logging.info(f"Checkpoint saved: {data['num_successful']} successful, {data['num_failed']} failed")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def is_completed(self, params: Dict[str, Any]) -> bool:
        """Check if experiment with these parameters was already run (success or fail)."""
        for result in self.results:
            if result['params'] == params:
                return True
        return False
    
    def is_successful(self, params: Dict[str, Any]) -> bool:
        """Check if experiment completed successfully."""
        for result in self.results:
            if result['params'] == params and 'sharpe_ratio' in result.get('metrics', {}):
                return True
        return False
    
    def add_result(self, params: Dict[str, Any], metrics: Dict[str, Any], 
                   run_dir: Optional[str] = None, duration_seconds: float = 0):
        """Add a new experiment result and save checkpoint."""
        result = {
            'params': params,
            'metrics': metrics,
            'run_dir': run_dir,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
        }
        self.results.append(result)
        self.save_checkpoint()
    
    def get_best_result(self, metric: str = 'sharpe_ratio') -> Optional[Dict]:
        """Get the best result by a given metric."""
        if not self.results:
            return None
        
        valid_results = [r for r in self.results 
                         if metric in r.get('metrics', {})]
        if not valid_results:
            return None
        
        return max(valid_results, key=lambda x: x['metrics'][metric])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        successful = [r for r in self.results if 'sharpe_ratio' in r.get('metrics', {})]
        failed = [r for r in self.results if 'error' in r.get('metrics', {})]
        
        avg_duration = 0
        if successful:
            durations = [r.get('duration_seconds', 600) for r in successful]
            avg_duration = sum(durations) / len(durations)
        
        return {
            'total': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'avg_duration_seconds': avg_duration,
            'start_time': self.start_time,
        }
    
    def get_results_df(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for result in self.results:
            row = {}
            # Flatten params
            for k, v in result['params'].items():
                row[f'param_{k}'] = v
            # Add metrics
            for k, v in result.get('metrics', {}).items():
                row[f'metric_{k}'] = v
            # Add metadata
            row['timestamp'] = result['timestamp']
            row['run_dir'] = result.get('run_dir')
            row['duration_seconds'] = result.get('duration_seconds', 0)
            rows.append(row)
        
        return pd.DataFrame(rows)


# =============================================================================
# PARAMETER GRID (expanded for overnight runs)
# =============================================================================

@dataclass
class ParamConfig:
    """Single parameter configuration to test."""
    # Model complexity
    n_estimators: int = 150
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1  # -1 = no limit
    min_child_samples: int = 20
    
    # Regularization
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 0.0  # L2 regularization
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Portfolio (long-only for NZX)
    top_n: int = 10
    
    # Target
    forward_days: int = 365
    return_type: str = "simple"  # "simple" or "log"
    
    # Evaluation
    num_windows: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def short_desc(self) -> str:
        """Short description for logging."""
        return f"est={self.n_estimators},lr={self.learning_rate},top={self.top_n},fwd={self.forward_days},ret={self.return_type}"


def generate_param_grid(fast_mode: bool = False) -> List[ParamConfig]:
    """Generate parameter configurations to test.
    
    Full mode: ~60 configurations optimized for overnight run (~10 hours)
    Fast mode: 5 configurations for quick testing (~1 hour)
    
    All configurations are LONG-ONLY (no shorting) for NZX compatibility.
    
    Args:
        fast_mode: If True, use smaller grid for quick testing
    
    Returns:
        List of parameter configurations (deduplicated)
    """
    if fast_mode:
        # Quick test mode: 5 configs, ~1 hour total
        return [
            ParamConfig(),  # Baseline
            ParamConfig(n_estimators=100),
            ParamConfig(n_estimators=200),
            ParamConfig(top_n=5),
            ParamConfig(top_n=20),
        ]
    
    configs = []
    
    # ==========================================================================
    # SINGLE PARAMETER VARIATIONS (understand individual effects)
    # ==========================================================================
    
    # 1. Baseline (current best settings)
    configs.append(ParamConfig())
    
    # 2. Number of estimators (6 configs) - key for overfitting control
    for n_est in [50, 75, 100, 200, 250, 300]:
        configs.append(ParamConfig(n_estimators=n_est))
    
    # 3. Learning rate (5 configs) - affects convergence
    for lr in [0.01, 0.02, 0.03, 0.07, 0.1]:
        configs.append(ParamConfig(learning_rate=lr))
    
    # 4. Tree complexity - num_leaves (4 configs)
    for num_leaves in [15, 20, 63, 127]:
        configs.append(ParamConfig(num_leaves=num_leaves))
    
    # 5. Tree depth limit (4 configs) - alternative complexity control
    for max_depth in [3, 5, 7, 10]:
        configs.append(ParamConfig(max_depth=max_depth))
    
    # 6. Regularization - min_child_samples (4 configs)
    for min_samples in [10, 30, 50, 100]:
        configs.append(ParamConfig(min_child_samples=min_samples))
    
    # 7. L1 regularization - reg_alpha (4 configs)
    for reg_alpha in [0.01, 0.1, 1.0, 10.0]:
        configs.append(ParamConfig(reg_alpha=reg_alpha))
    
    # 8. L2 regularization - reg_lambda (4 configs)
    for reg_lambda in [0.01, 0.1, 1.0, 10.0]:
        configs.append(ParamConfig(reg_lambda=reg_lambda))
    
    # 9. Subsampling/bagging (3 configs)
    for subsample in [0.5, 0.6, 1.0]:
        configs.append(ParamConfig(subsample=subsample, colsample_bytree=subsample))
    
    # ==========================================================================
    # PORTFOLIO SIZE VARIATIONS (LONG-ONLY)
    # ==========================================================================
    
    # 10. Different portfolio sizes (6 configs)
    for top_n in [3, 5, 8, 15, 20, 30]:
        configs.append(ParamConfig(top_n=top_n))
    
    # ==========================================================================
    # FORWARD RETURN HORIZON & TYPE VARIATIONS
    # ==========================================================================
    
    # 11. Different prediction horizons (5 configs)
    for days in [20, 60, 90, 180, 270]:
        configs.append(ParamConfig(forward_days=days))
    
    # 12. Log returns (may be better for longer horizons) (3 configs)
    for days in [90, 180, 365]:
        configs.append(ParamConfig(forward_days=days, return_type="log"))
    
    # ==========================================================================
    # COMBINED CONFIGURATIONS (based on research/intuition)
    # ==========================================================================
    
    configs.extend([
        # Conservative: more regularization, less overfitting risk
        ParamConfig(n_estimators=100, min_child_samples=50, subsample=0.6, colsample_bytree=0.6),
        ParamConfig(n_estimators=75, min_child_samples=30, num_leaves=20),
        ParamConfig(n_estimators=100, reg_alpha=0.1, reg_lambda=0.1),
        ParamConfig(n_estimators=100, max_depth=5, reg_lambda=1.0),
        
        # Aggressive: less regularization, higher capacity
        ParamConfig(n_estimators=200, num_leaves=63, min_child_samples=10),
        ParamConfig(n_estimators=250, learning_rate=0.03),
        
        # Concentrated portfolios with different model settings
        ParamConfig(n_estimators=100, top_n=5),
        ParamConfig(n_estimators=150, top_n=3),
        ParamConfig(n_estimators=200, top_n=5),
        
        # Short horizon with appropriate settings
        ParamConfig(forward_days=20, n_estimators=100, top_n=10),
        ParamConfig(forward_days=60, n_estimators=100, top_n=5),
        ParamConfig(forward_days=90, n_estimators=100, top_n=5),
        ParamConfig(forward_days=90, n_estimators=150, top_n=10),
        
        # Learning rate + estimator combinations
        ParamConfig(n_estimators=200, learning_rate=0.03),
        ParamConfig(n_estimators=300, learning_rate=0.02),
        ParamConfig(n_estimators=100, learning_rate=0.07),
        
        # Full regularization combo
        ParamConfig(n_estimators=100, learning_rate=0.03, num_leaves=20, 
                   min_child_samples=50, subsample=0.6, colsample_bytree=0.6,
                   reg_alpha=0.1, reg_lambda=0.1),
        
        # Depth-limited trees with regularization
        ParamConfig(n_estimators=150, max_depth=5, reg_lambda=1.0),
        ParamConfig(n_estimators=200, max_depth=7, reg_alpha=0.1),
        
        # Log returns with optimized settings
        ParamConfig(forward_days=365, return_type="log", n_estimators=150),
        ParamConfig(forward_days=180, return_type="log", n_estimators=100, top_n=5),
        
        # Balanced combo
        ParamConfig(n_estimators=150, learning_rate=0.05, num_leaves=31, 
                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                   top_n=10),
    ])
    
    # Deduplicate configs (same params = same dict)
    seen = set()
    unique_configs = []
    for config in configs:
        config_tuple = tuple(sorted(config.to_dict().items()))
        if config_tuple not in seen:
            seen.add(config_tuple)
            unique_configs.append(config)
    
    return unique_configs


# =============================================================================
# EXPERIMENT RUNNER (with robust error handling)
# =============================================================================

def run_single_experiment(
    config: ParamConfig,
    checkpoint_mgr: CheckpointManager,
    logger: logging.Logger,
    experiment_num: int,
    total_experiments: int,
) -> bool:
    """Run a single experiment with given parameters.
    
    Designed to never crash the main loop - all errors are caught and logged.
    
    Args:
        config: Parameter configuration
        checkpoint_mgr: Checkpoint manager for saving results
        logger: Logger instance
        experiment_num: Current experiment number (for progress display)
        total_experiments: Total number of experiments
    
    Returns:
        True if experiment completed successfully, False otherwise
    """
    params = config.to_dict()
    start_time = time.time()
    
    # Check if already completed
    if checkpoint_mgr.is_completed(params):
        logger.info(f"[{experiment_num}/{total_experiments}] Skipping (already done): {config.short_desc()}")
        return True
    
    # Calculate ETA
    stats = checkpoint_mgr.get_stats()
    remaining = total_experiments - experiment_num + 1
    if stats['avg_duration_seconds'] > 0:
        eta_seconds = remaining * stats['avg_duration_seconds']
        eta = timedelta(seconds=int(eta_seconds))
        eta_str = f"ETA: {eta}"
    else:
        eta_str = "ETA: calculating..."
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[{experiment_num}/{total_experiments}] Starting: {config.short_desc()}")
    logger.info(f"Progress: {stats['successful']} successful, {stats['failed']} failed | {eta_str}")
    logger.info(f"{'='*80}\n")
    
    try:
        # Run pipeline with explicit parameters (long-only, no shorting)
        summary = run_ranking_pipeline(
            num_windows=config.num_windows,
            forward_return_days=config.forward_days,
            return_type=config.return_type,
            portfolio_top_n=config.top_n,
            portfolio_bottom_n=0,  # Long-only for NZX
            ranker_n_estimators=config.n_estimators,
            ranker_learning_rate=config.learning_rate,
            ranker_num_leaves=config.num_leaves,
            ranker_max_depth=config.max_depth,
            ranker_min_child_samples=config.min_child_samples,
            ranker_reg_alpha=config.reg_alpha,
            ranker_reg_lambda=config.reg_lambda,
            ranker_subsample=config.subsample,
            ranker_colsample_bytree=config.colsample_bytree,
        )
        
        duration = time.time() - start_time
        
        # Extract key metrics safely
        metrics = {}
        try:
            metrics['sharpe_ratio'] = float(summary.backtest.sharpe_ratio)
            metrics['mean_ic'] = float(summary.metrics.mean_ic)
            metrics['icir'] = float(summary.metrics.icir)
            metrics['hit_rate'] = float(summary.metrics.hit_rate_top_n)
            metrics['quintile_spread'] = float(summary.metrics.quintile_spread)
            metrics['total_return'] = float(summary.backtest.total_return)
            metrics['max_drawdown'] = float(summary.backtest.max_drawdown)
            metrics['avg_turnover'] = float(summary.backtest.avg_turnover)
            metrics['num_windows'] = len(summary.window_summaries)
        except (AttributeError, TypeError) as e:
            logger.warning(f"Error extracting some metrics: {e}")
        
        # Get run directory
        run_dir = summary.config.get('output_dir', None)
        
        # Save result
        checkpoint_mgr.add_result(params, metrics, run_dir, duration)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ [{experiment_num}/{total_experiments}] SUCCESS: {config.short_desc()}")
        logger.info(f"   Sharpe: {metrics.get('sharpe_ratio', 'N/A'):.3f} | IC: {metrics.get('mean_ic', 'N/A'):.4f} | Duration: {duration/60:.1f}min")
        logger.info(f"{'='*80}\n")
        
        # Force garbage collection to free memory
        gc.collect()
        
        return True
        
    except KeyboardInterrupt:
        # Re-raise keyboard interrupt to allow clean shutdown
        logger.info("\n⚠️ Keyboard interrupt received, saving progress...")
        raise
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"\n{'='*80}")
        logger.error(f"❌ [{experiment_num}/{total_experiments}] FAILED: {config.short_desc()}")
        logger.error(f"   Error: {error_msg}")
        logger.error(f"   Duration: {duration/60:.1f}min")
        logger.error(f"{'='*80}\n")
        logger.debug(traceback.format_exc())
        
        # Save failed result with error info
        try:
            checkpoint_mgr.add_result(
                params, 
                {'error': error_msg, 'status': 'failed'},
                None,
                duration
            )
        except Exception as save_error:
            logger.error(f"Failed to save error result: {save_error}")
        
        # Force garbage collection
        gc.collect()
        
        return False


# =============================================================================
# REPORTING
# =============================================================================

def print_optimization_report(checkpoint_mgr: CheckpointManager):
    """Print summary report of optimization results."""
    df = checkpoint_mgr.get_results_df()
    
    if df.empty:
        print("\n❌ No results found. Run experiments first.")
        return
    
    # Check if we have any successful experiments
    if 'metric_sharpe_ratio' not in df.columns:
        print("\n❌ No successful experiments found. All experiments failed.")
        print("\n📋 Failed Experiments:")
        for idx, row in df.iterrows():
            print(f"\n   Experiment {idx + 1}:")
            params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_')}
            print(f"   Params: {params}")
            if 'metric_error' in df.columns and pd.notna(row.get('metric_error')):
                print(f"   Error: {row['metric_error']}")
        return
    
    # Filter to successful experiments
    df_success = df[~df['metric_sharpe_ratio'].isna()].copy()
    df_failed = df[df['metric_sharpe_ratio'].isna()].copy()
    
    if df_success.empty:
        print("\n❌ No successful experiments found.")
        return
    
    print("\n" + "="*100)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*100)
    
    stats = checkpoint_mgr.get_stats()
    print(f"\n📊 Total Experiments: {stats['total']}")
    print(f"✅ Successful: {stats['successful']}")
    print(f"❌ Failed: {stats['failed']}")
    if stats['avg_duration_seconds'] > 0:
        print(f"⏱️  Avg Duration: {stats['avg_duration_seconds']/60:.1f} minutes")
    
    # Sort by Sharpe ratio
    df_sorted = df_success.sort_values('metric_sharpe_ratio', ascending=False)
    
    print("\n" + "="*100)
    print("TOP 10 CONFIGURATIONS BY SHARPE RATIO")
    print("="*100)
    
    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"\n🏆 Rank {i}: Sharpe = {row['metric_sharpe_ratio']:.3f}")
        print(f"   IC = {row['metric_mean_ic']:.4f}, ICIR = {row['metric_icir']:.2f}, Hit Rate = {row['metric_hit_rate']:.2%}")
        print(f"   n_estimators={row['param_n_estimators']:.0f}, lr={row['param_learning_rate']:.3f}, "
              f"leaves={row['param_num_leaves']:.0f}, depth={row['param_max_depth']:.0f}")
        print(f"   min_samples={row['param_min_child_samples']:.0f}, reg_alpha={row['param_reg_alpha']:.2f}, "
              f"reg_lambda={row['param_reg_lambda']:.2f}, subsample={row['param_subsample']:.1f}")
        print(f"   top_n={row['param_top_n']:.0f}, forward_days={row['param_forward_days']:.0f}, "
              f"return_type={row['param_return_type']}")
        if pd.notna(row.get('run_dir')):
            print(f"   📁 {row['run_dir']}")
    
    print("\n" + "="*100)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*100)
    
    # Analyze each parameter's impact
    param_cols = [c for c in df_success.columns if c.startswith('param_')]
    
    for param_col in param_cols:
        param_name = param_col.replace('param_', '')
        unique_values = df_success[param_col].unique()
        
        if len(unique_values) > 1:
            print(f"\n📈 {param_name}:")
            grouped = df_success.groupby(param_col)['metric_sharpe_ratio'].agg(['mean', 'std', 'count'])
            grouped = grouped.sort_values('mean', ascending=False)
            
            for value, row_stats in grouped.iterrows():
                std_str = f"± {row_stats['std']:.3f}" if pd.notna(row_stats['std']) else ""
                print(f"   {value:>10} → Sharpe: {row_stats['mean']:6.3f} {std_str} (n={row_stats['count']:.0f})")
    
    print("\n" + "="*100)
    print("BEST CONFIGURATION SUMMARY")
    print("="*100)
    
    best = checkpoint_mgr.get_best_result('sharpe_ratio')
    if best:
        print(f"\n🎯 Best Sharpe Ratio: {best['metrics']['sharpe_ratio']:.3f}")
        print(f"\nOptimal Parameters:")
        for k, v in best['params'].items():
            print(f"   {k}: {v}")
        print(f"\nAll Metrics:")
        for k, v in best['metrics'].items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
            else:
                print(f"   {k}: {v}")
        if best.get('run_dir'):
            print(f"\n📁 Run Directory: {best['run_dir']}")
    
    print("\n" + "="*100)


def print_status(checkpoint_mgr: CheckpointManager, total_configs: int):
    """Print current optimization status."""
    stats = checkpoint_mgr.get_stats()
    
    print("\n" + "="*60)
    print("OPTIMIZATION STATUS")
    print("="*60)
    
    completed = stats['successful'] + stats['failed']
    remaining = total_configs - completed
    
    print(f"\n📊 Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%)")
    print(f"✅ Successful: {stats['successful']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⏳ Remaining: {remaining}")
    
    if stats['avg_duration_seconds'] > 0:
        eta_seconds = remaining * stats['avg_duration_seconds']
        eta = timedelta(seconds=int(eta_seconds))
        print(f"\n⏱️  Avg Duration: {stats['avg_duration_seconds']/60:.1f} minutes")
        print(f"🕐 Estimated Time Remaining: {eta}")
    
    if stats['start_time']:
        elapsed = datetime.now() - stats['start_time']
        print(f"⏰ Elapsed: {elapsed}")
    
    # Show best so far
    best = checkpoint_mgr.get_best_result('sharpe_ratio')
    if best:
        print(f"\n🏆 Best Sharpe So Far: {best['metrics']['sharpe_ratio']:.3f}")
        print(f"   Config: n_est={best['params']['n_estimators']}, top_n={best['params']['top_n']}, fwd={best['params']['forward_days']}")
    
    print("\n" + "="*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hyperparameter optimization with checkpointing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/optimize_hyperparameters.py           # Full overnight run (~60 configs)
  python scripts/optimize_hyperparameters.py --fast    # Quick test (~5 configs)
  python scripts/optimize_hyperparameters.py --report  # Show results
  python scripts/optimize_hyperparameters.py --status  # Check progress
        """
    )
    parser.add_argument('--fast', action='store_true', 
                        help='Fast mode: test fewer parameter combinations (~5 configs)')
    parser.add_argument('--report', action='store_true',
                        help='Only show report of existing results')
    parser.add_argument('--status', action='store_true',
                        help='Show current optimization status')
    parser.add_argument('--checkpoint-file', type=str,
                        default='output/optimization/checkpoint.json',
                        help='Path to checkpoint file')
    args = parser.parse_args()
    
    # Setup
    checkpoint_file = Path(__file__).parent.parent / args.checkpoint_file
    checkpoint_mgr = CheckpointManager(checkpoint_file)
    
    # Generate parameter grid first (needed for status)
    param_grid = generate_param_grid(fast_mode=args.fast)
    
    # Status mode
    if args.status:
        print_status(checkpoint_mgr, len(param_grid))
        return
    
    # Report mode
    if args.report:
        print_optimization_report(checkpoint_mgr)
        return
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = get_logger(__name__)
    
    # Calculate time estimates
    estimated_minutes = len(param_grid) * 10  # ~10 min per experiment
    estimated_hours = estimated_minutes / 60
    
    logger.info(f"\n{'='*80}")
    logger.info(f"HYPERPARAMETER OPTIMIZATION")
    logger.info(f"{'='*80}")
    logger.info(f"Mode: {'FAST' if args.fast else 'FULL'}")
    logger.info(f"Total configurations: {len(param_grid)}")
    logger.info(f"Already completed: {len(checkpoint_mgr.results)}")
    logger.info(f"Estimated total time: {estimated_hours:.1f} hours")
    logger.info(f"Checkpoint file: {checkpoint_file}")
    logger.info(f"{'='*80}\n")
    
    # Run experiments
    successful = 0
    failed = 0
    skipped = 0
    
    try:
        for i, config in enumerate(param_grid, 1):
            if checkpoint_mgr.is_completed(config.to_dict()):
                skipped += 1
                logger.info(f"[{i}/{len(param_grid)}] Skipping (already done): {config.short_desc()}")
                continue
            
            success = run_single_experiment(config, checkpoint_mgr, logger, i, len(param_grid))
            
            if success:
                successful += 1
            else:
                failed += 1
                
    except KeyboardInterrupt:
        logger.info("\n" + "="*80)
        logger.info("⚠️ OPTIMIZATION INTERRUPTED BY USER")
        logger.info("Progress has been saved. Run the same command to resume.")
        logger.info("="*80 + "\n")
    
    # Final report
    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total: {len(param_grid)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already done): {skipped}")
    logger.info(f"{'='*80}\n")
    
    print_optimization_report(checkpoint_mgr)


if __name__ == "__main__":
    main()
