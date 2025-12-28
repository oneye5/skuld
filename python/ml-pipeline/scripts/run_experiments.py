"""Experiment runner for hyperparameter grid search.

Iterates over different configurations (lookahead days, model params, etc.)
with graceful interruption - if interrupted, only the current run is lost.

Usage:
    uv run python scripts/run_experiments.py
    uv run python scripts/run_experiments.py --resume  # Resume from last checkpoint
    uv run python scripts/run_experiments.py --dry-run  # Show configs without running
"""

import argparse
import json
import signal
import sys
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.ranking_pipeline import run_ranking_pipeline
from config.paths import OUTPUT_DIR


# =============================================================================
# EXPERIMENT CONFIGURATION - EDIT THIS SECTION
# =============================================================================

EXPERIMENT_NAME = "sharpe_optimization"

# Parameter grid - optimizing for Sharpe ratio
# Based on prior experiments: 63-189 day horizons performed best (Sharpe 0.57-0.90)
# Total configs: 4 horizons × 3 estimators × 3 leaves × 3 top_n = 108 configs
PARAM_GRID = {
    # Target settings - best performing horizons from prior experiment
    "forward_return_days": [63, 105, 126, 168],  # 3mo, 4.2mo, 6mo, 6.7mo
    "return_type": ["simple"],
    
    # Model settings - vary to find optimal
    "n_estimators": [75, 100, 150],
    "learning_rate": [0.05],  # Fixed - 0.05 worked well
    "num_leaves": [23, 31, 47],
    
    # Portfolio settings - test different concentrations (long-only)
    "top_n": [5, 10, 15],
    "bottom_n": [0],  # Long-only (Sharesies can't short)
    
    # Rolling window settings
    "num_windows": [10],
    "test_period_years": [0.5],
}

# Subset grid for quick testing - uncomment to use
PARAM_GRID_QUICK = {
    "forward_return_days": [63, 126],
    "return_type": ["simple"],
    "n_estimators": [100],
    "learning_rate": [0.05],
    "num_leaves": [31],
    "top_n": [10],
    "bottom_n": [0],
    "num_windows": [5],
    "test_period_years": [0.5],
}


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    forward_return_days: int = 5
    return_type: str = "simple"
    n_estimators: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    top_n: int = 10
    bottom_n: int = 10
    num_windows: int = 5
    test_period_years: float = 0.5
    
    def to_hash(self) -> str:
        """Generate unique hash for this config."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    config: dict
    config_hash: str
    timestamp: str
    
    # Metrics
    mean_ic: float | None = None
    std_ic: float | None = None
    icir: float | None = None
    mean_rank_ic: float | None = None
    rank_icir: float | None = None
    hit_rate: float | None = None
    quintile_spread: float | None = None
    sharpe_ratio: float | None = None
    total_return: float | None = None
    max_drawdown: float | None = None
    
    # Meta
    runtime_seconds: float | None = None
    error: str | None = None
    success: bool = False


class ExperimentRunner:
    """Manages experiment grid search with checkpointing."""
    
    def __init__(self, experiment_name: str, output_dir: Path | None = None):
        self.experiment_name = experiment_name
        self.output_dir = output_dir or OUTPUT_DIR / "experiments" / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.output_dir / "results.json"
        self.completed_file = self.output_dir / "completed_hashes.txt"
        
        self.results: list[ExperimentResult] = []
        self.completed_hashes: set[str] = set()
        self._interrupted = False
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal gracefully."""
        print("\n\n⚠️  Interrupt received! Finishing current operation...")
        print("   (Results from completed runs have been saved)")
        self._interrupted = True
    
    def load_checkpoint(self) -> None:
        """Load completed experiment hashes from checkpoint."""
        if self.completed_file.exists():
            with open(self.completed_file, "r") as f:
                self.completed_hashes = set(line.strip() for line in f if line.strip())
            print(f"📂 Loaded {len(self.completed_hashes)} completed experiment(s) from checkpoint")
        
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                data = json.load(f)
                self.results = [ExperimentResult(**r) for r in data.get("results", [])]
    
    def save_result(self, result: ExperimentResult) -> None:
        """Save a single result immediately (for crash safety)."""
        self.results.append(result)
        
        # Append hash to completed file (fast, atomic-ish)
        with open(self.completed_file, "a") as f:
            f.write(result.config_hash + "\n")
        self.completed_hashes.add(result.config_hash)
        
        # Save full results JSON
        self._save_results_json()
    
    def _save_results_json(self) -> None:
        """Save all results to JSON file."""
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        results_data = []
        for r in self.results:
            r_dict = asdict(r)
            results_data.append(convert_to_serializable(r_dict))
        
        data = {
            "experiment_name": self.experiment_name,
            "last_updated": datetime.now().isoformat(),
            "total_runs": len(self.results),
            "successful_runs": sum(1 for r in self.results if r.success),
            "results": results_data,
        }
        
        # Write to temp file first, then rename (atomic on most systems)
        temp_file = self.results_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.results_file)
    
    def generate_configs(self, param_grid: dict) -> list[ExperimentConfig]:
        """Generate all config combinations from parameter grid."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        configs = []
        for combo in product(*values):
            config_dict = dict(zip(keys, combo))
            configs.append(ExperimentConfig(**config_dict))
        
        return configs
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with the given config."""
        import time
        
        config_hash = config.to_hash()
        timestamp = datetime.now().isoformat()
        
        result = ExperimentResult(
            config=config.to_dict(),
            config_hash=config_hash,
            timestamp=timestamp,
        )
        
        start_time = time.time()
        
        try:
            # Run the ranking pipeline
            pipeline_result = run_ranking_pipeline(
                forward_return_days=config.forward_return_days,
                return_type=config.return_type,
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
                portfolio_top_n=config.top_n,
                portfolio_bottom_n=config.bottom_n,
                num_windows=config.num_windows,
                test_period_years=config.test_period_years,
                save_results=False,  # Don't save individual run outputs
                save_outputs=False,
            )
            
            # Extract metrics
            if pipeline_result.metrics:
                metrics = pipeline_result.metrics
                result.mean_ic = metrics.mean_ic
                result.std_ic = metrics.std_ic
                result.icir = metrics.icir
                result.mean_rank_ic = metrics.mean_rank_ic
                result.rank_icir = metrics.rank_icir
                result.hit_rate = metrics.hit_rate_top_n
                result.quintile_spread = metrics.quintile_spread
            
            if pipeline_result.backtest:
                backtest = pipeline_result.backtest
                result.sharpe_ratio = backtest.sharpe_ratio
                result.total_return = backtest.total_return
                result.max_drawdown = backtest.max_drawdown
            
            result.success = True
            
        except Exception as e:
            result.error = str(e)
            result.success = False
            print(f"      ❌ Error: {e}")
        
        result.runtime_seconds = time.time() - start_time
        return result
    
    def run_grid_search(
        self, 
        param_grid: dict, 
        resume: bool = True,
        dry_run: bool = False,
    ) -> None:
        """Run full grid search over parameter combinations."""
        configs = self.generate_configs(param_grid)
        total_configs = len(configs)
        
        print(f"\n{'='*60}")
        print(f"🔬 EXPERIMENT: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"📊 Total configurations: {total_configs}")
        print(f"📁 Output directory: {self.output_dir}")
        
        if resume:
            self.load_checkpoint()
        
        # Filter out completed configs
        pending_configs = [
            c for c in configs 
            if c.to_hash() not in self.completed_hashes
        ]
        
        print(f"✅ Already completed: {len(configs) - len(pending_configs)}")
        print(f"⏳ Remaining: {len(pending_configs)}")
        
        if dry_run:
            print(f"\n🔍 DRY RUN - Configurations to test:")
            for i, config in enumerate(pending_configs[:10], 1):
                print(f"   {i}. {config.to_dict()}")
            if len(pending_configs) > 10:
                print(f"   ... and {len(pending_configs) - 10} more")
            return
        
        if not pending_configs:
            print("\n✨ All experiments already completed!")
            self._print_summary()
            return
        
        print(f"\n{'='*60}")
        print("Starting experiments... (Ctrl+C to stop gracefully)")
        print(f"{'='*60}\n")
        
        for i, config in enumerate(pending_configs, 1):
            if self._interrupted:
                print("\n🛑 Stopping due to interrupt.")
                break
            
            config_hash = config.to_hash()
            print(f"[{i}/{len(pending_configs)}] Running config {config_hash}:")
            print(f"      forward_days={config.forward_return_days}, "
                  f"n_est={config.n_estimators}, lr={config.learning_rate}, "
                  f"leaves={config.num_leaves}")
            
            result = self.run_single_experiment(config)
            
            if result.success:
                print(f"      ✅ IC={result.mean_ic:.4f}, "
                      f"Sharpe={result.sharpe_ratio:.3f}, "
                      f"Time={result.runtime_seconds:.1f}s")
            
            # Save immediately after each run
            self.save_result(result)
        
        print(f"\n{'='*60}")
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print summary of all results."""
        if not self.results:
            print("No results yet.")
            return
        
        successful = [r for r in self.results if r.success]
        
        print(f"\n📊 EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total runs: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(self.results) - len(successful)}")
        
        if successful:
            # Find best by different metrics
            best_ic = max(successful, key=lambda r: r.mean_ic or -999)
            best_sharpe = max(successful, key=lambda r: r.sharpe_ratio or -999)
            best_icir = max(successful, key=lambda r: r.icir or -999)
            
            print(f"\n🏆 BEST RESULTS:")
            print(f"\n   Best Mean IC ({best_ic.mean_ic:.4f}):")
            print(f"      {best_ic.config}")
            
            print(f"\n   Best Sharpe ({best_sharpe.sharpe_ratio:.3f}):")
            print(f"      {best_sharpe.config}")
            
            print(f"\n   Best ICIR ({best_icir.icir:.3f}):")
            print(f"      {best_icir.config}")
        
        print(f"\n📁 Full results saved to: {self.results_file}")
        
        # Also save a summary CSV for easy analysis
        self._save_summary_csv()
    
    def _save_summary_csv(self) -> None:
        """Save results as CSV for easy analysis."""
        import csv
        
        csv_file = self.output_dir / "results_summary.csv"
        
        if not self.results:
            return
        
        # Flatten config into columns
        fieldnames = [
            "config_hash", "success", "runtime_seconds",
            "forward_return_days", "return_type", "n_estimators", 
            "learning_rate", "num_leaves", "top_n", "bottom_n",
            "num_windows", "test_period_years",
            "mean_ic", "std_ic", "icir", "mean_rank_ic", "rank_icir",
            "hit_rate", "quintile_spread", "sharpe_ratio", 
            "total_return", "max_drawdown", "error",
        ]
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in self.results:
                row = {
                    "config_hash": r.config_hash,
                    "success": r.success,
                    "runtime_seconds": r.runtime_seconds,
                    **r.config,
                    "mean_ic": r.mean_ic,
                    "std_ic": r.std_ic,
                    "icir": r.icir,
                    "mean_rank_ic": r.mean_rank_ic,
                    "rank_icir": r.rank_icir,
                    "hit_rate": r.hit_rate,
                    "quintile_spread": r.quintile_spread,
                    "sharpe_ratio": r.sharpe_ratio,
                    "total_return": r.total_return,
                    "max_drawdown": r.max_drawdown,
                    "error": r.error,
                }
                writer.writerow(row)
        
        print(f"📄 CSV summary saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter grid search for ranking pipeline"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true", 
        help="Start fresh, ignore previous results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configurations without running experiments"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller parameter grid for quick testing"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=EXPERIMENT_NAME,
        help=f"Experiment name (default: {EXPERIMENT_NAME})"
    )
    
    args = parser.parse_args()
    
    param_grid = PARAM_GRID_QUICK if args.quick else PARAM_GRID
    resume = not args.no_resume
    
    runner = ExperimentRunner(args.name)
    runner.run_grid_search(
        param_grid=param_grid,
        resume=resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
