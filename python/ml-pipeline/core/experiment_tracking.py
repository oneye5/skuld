"""Experiment tracking and reproducibility utilities.

This module provides:
- Experiment manifest generation for reproducibility
- Git commit tracking
- Config serialization
- Result comparison utilities
"""

import json
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib


@dataclass
class ExperimentManifest:
    """Manifest containing all information needed to reproduce an experiment.
    
    Attributes:
        experiment_id: Unique identifier for this experiment.
        timestamp: When the experiment was run.
        git_commit: Git commit hash (if in a git repo).
        git_branch: Git branch name.
        git_dirty: Whether there were uncommitted changes.
        config: Configuration parameters used.
        data_hash: Hash of input data for verification.
        feature_columns: List of feature columns used.
        metrics: Performance metrics from the experiment.
        notes: Optional notes about the experiment.
    """
    experiment_id: str
    timestamp: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    data_hash: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, output_dir: Path) -> Path:
        """Save manifest to JSON file.
        
        Args:
            output_dir: Directory to save the manifest.
        
        Returns:
            Path to the saved manifest file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        return manifest_path
    
    @classmethod
    def load(cls, manifest_path: Path) -> "ExperimentManifest":
        """Load manifest from JSON file.
        
        Args:
            manifest_path: Path to manifest JSON file.
        
        Returns:
            ExperimentManifest instance.
        """
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def get_git_info() -> Dict[str, Any]:
    """Get current git repository information.
    
    Returns:
        Dictionary with git_commit, git_branch, and git_dirty.
        Returns empty values if not in a git repo.
    """
    info = {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": False,
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()[:12]  # Short hash
        
        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_branch"] = result.stdout.strip()
        
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_dirty"] = bool(result.stdout.strip())
            
    except (subprocess.SubprocessError, FileNotFoundError):
        # Not in a git repo or git not available
        pass
    
    return info


def compute_data_hash(df, columns: Optional[List[str]] = None) -> str:
    """Compute a hash of DataFrame contents for verification.
    
    Args:
        df: DataFrame to hash.
        columns: Specific columns to include (default: all).
    
    Returns:
        MD5 hash string of the data.
    """
    import pandas as pd
    
    if columns:
        df = df[columns]
    
    # Convert to bytes and hash
    data_bytes = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.md5(data_bytes).hexdigest()[:16]


def create_experiment_manifest(
    config: Dict[str, Any],
    feature_columns: List[str],
    metrics: Optional[Dict[str, float]] = None,
    data_hash: Optional[str] = None,
    notes: str = "",
) -> ExperimentManifest:
    """Create a new experiment manifest.
    
    Args:
        config: Configuration parameters used.
        feature_columns: List of feature columns used.
        metrics: Performance metrics (can be added later).
        data_hash: Hash of input data.
        notes: Optional notes about the experiment.
    
    Returns:
        ExperimentManifest instance.
    """
    git_info = get_git_info()
    
    # Generate unique experiment ID
    timestamp = datetime.now()
    experiment_id = f"exp_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    return ExperimentManifest(
        experiment_id=experiment_id,
        timestamp=timestamp.isoformat(),
        git_commit=git_info["git_commit"],
        git_branch=git_info["git_branch"],
        git_dirty=git_info["git_dirty"],
        config=config,
        data_hash=data_hash,
        feature_columns=feature_columns,
        metrics=metrics or {},
        notes=notes,
    )


def compare_experiments(
    manifest_a: ExperimentManifest,
    manifest_b: ExperimentManifest,
) -> Dict[str, Any]:
    """Compare two experiment manifests.
    
    Args:
        manifest_a: First experiment manifest.
        manifest_b: Second experiment manifest.
    
    Returns:
        Dictionary with differences and metric comparisons.
    """
    comparison = {
        "same_git_commit": manifest_a.git_commit == manifest_b.git_commit,
        "same_data": manifest_a.data_hash == manifest_b.data_hash,
        "config_differences": {},
        "metric_differences": {},
        "feature_differences": {
            "added": [],
            "removed": [],
        },
    }
    
    # Compare configs
    all_keys = set(manifest_a.config.keys()) | set(manifest_b.config.keys())
    for key in all_keys:
        val_a = manifest_a.config.get(key)
        val_b = manifest_b.config.get(key)
        if val_a != val_b:
            comparison["config_differences"][key] = {"a": val_a, "b": val_b}
    
    # Compare metrics
    all_metrics = set(manifest_a.metrics.keys()) | set(manifest_b.metrics.keys())
    for metric in all_metrics:
        val_a = manifest_a.metrics.get(metric, 0)
        val_b = manifest_b.metrics.get(metric, 0)
        if val_a != val_b:
            diff = val_b - val_a if isinstance(val_a, (int, float)) else None
            comparison["metric_differences"][metric] = {
                "a": val_a, 
                "b": val_b,
                "diff": diff,
            }
    
    # Compare features
    features_a = set(manifest_a.feature_columns)
    features_b = set(manifest_b.feature_columns)
    comparison["feature_differences"]["added"] = list(features_b - features_a)
    comparison["feature_differences"]["removed"] = list(features_a - features_b)
    
    return comparison


def find_best_experiment(
    experiment_dir: Path,
    metric: str = "sharpe_ratio",
    higher_is_better: bool = True,
) -> Optional[ExperimentManifest]:
    """Find the best experiment by a given metric.
    
    Args:
        experiment_dir: Directory containing experiment folders.
        metric: Metric name to optimize.
        higher_is_better: If True, higher metric values are better.
    
    Returns:
        ExperimentManifest of the best experiment, or None if no experiments found.
    """
    experiment_dir = Path(experiment_dir)
    best_manifest = None
    best_value = float('-inf') if higher_is_better else float('inf')
    
    for manifest_path in experiment_dir.glob("*/manifest.json"):
        try:
            manifest = ExperimentManifest.load(manifest_path)
            value = manifest.metrics.get(metric)
            
            if value is not None:
                is_better = (value > best_value) if higher_is_better else (value < best_value)
                if is_better:
                    best_value = value
                    best_manifest = manifest
        except Exception:
            continue
    
    return best_manifest


def generate_experiment_summary(experiment_dir: Path) -> Dict[str, Any]:
    """Generate a summary of all experiments in a directory.
    
    Args:
        experiment_dir: Directory containing experiment folders.
    
    Returns:
        Dictionary with experiment statistics and rankings.
    """
    experiment_dir = Path(experiment_dir)
    experiments = []
    
    for manifest_path in experiment_dir.glob("*/manifest.json"):
        try:
            manifest = ExperimentManifest.load(manifest_path)
            experiments.append({
                "id": manifest.experiment_id,
                "timestamp": manifest.timestamp,
                "git_commit": manifest.git_commit,
                **manifest.metrics,
            })
        except Exception:
            continue
    
    if not experiments:
        return {"n_experiments": 0, "experiments": []}
    
    # Sort by timestamp (most recent first)
    experiments.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "n_experiments": len(experiments),
        "experiments": experiments,
        "latest": experiments[0] if experiments else None,
    }
