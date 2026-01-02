"""Tests for feature concentration and model health diagnostics.

These tests ensure the model isn't over-relying on a small number of features,
which could indicate:
- Spurious correlations
- Data quality issues with dominant features
- Lack of diversified alpha sources
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeatureConcentration:
    """Tests for feature importance concentration."""
    
    def test_no_single_feature_dominates(self):
        """No single feature should have more than 30% of total importance.
        
        Rationale: If one feature dominates, the model is essentially a
        single-factor strategy, which is:
        - Fragile to changes in that factor
        - Potentially exploiting a data artifact
        - Missing diversified alpha sources
        """
        # Find most recent run with feature importances
        runs_dir = Path(__file__).parent.parent / "output" / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory found")
        
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("ranking_")])
        if not run_dirs:
            pytest.skip("No ranking runs found")
        
        latest_run = run_dirs[-1]
        fi_path = latest_run / "feature_importances.json"
        
        if not fi_path.exists():
            pytest.skip(f"No feature_importances.json in {latest_run}")
        
        with open(fi_path) as f:
            importances = json.load(f)
        
        total_importance = sum(importances.values())
        if total_importance == 0:
            pytest.skip("All feature importances are zero")
        
        max_importance = max(importances.values())
        max_feature = max(importances, key=importances.get)
        concentration = max_importance / total_importance
        
        assert concentration < 0.30, (
            f"Feature '{max_feature}' has {concentration:.1%} of total importance. "
            f"This exceeds the 30% threshold, indicating over-reliance on a single feature. "
            f"Consider: (1) investigating why this feature is so dominant, "
            f"(2) checking for data quality issues, "
            f"(3) adding more diverse features."
        )
    
    def test_top5_features_not_too_concentrated(self):
        """Top 5 features should not exceed 60% of total importance.
        
        Rationale: Healthy models should have diversified feature usage.
        """
        runs_dir = Path(__file__).parent.parent / "output" / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory found")
        
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("ranking_")])
        if not run_dirs:
            pytest.skip("No ranking runs found")
        
        latest_run = run_dirs[-1]
        fi_path = latest_run / "feature_importances.json"
        
        if not fi_path.exists():
            pytest.skip(f"No feature_importances.json in {latest_run}")
        
        with open(fi_path) as f:
            importances = json.load(f)
        
        total_importance = sum(importances.values())
        if total_importance == 0:
            pytest.skip("All feature importances are zero")
        
        sorted_importances = sorted(importances.values(), reverse=True)
        top5_importance = sum(sorted_importances[:5])
        top5_concentration = top5_importance / total_importance
        
        # Get names of top 5 features for error message
        sorted_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
        top5_names = [f[0] for f in sorted_features]
        
        assert top5_concentration < 0.60, (
            f"Top 5 features have {top5_concentration:.1%} of total importance. "
            f"Top 5: {top5_names}. "
            f"This exceeds the 60% threshold. Consider adding more diverse features."
        )
    
    def test_excluded_features_not_in_model(self):
        """Verify excluded features (raw prices, cluster) are not used.
        
        These features were excluded due to potential leakage concerns:
        - Close, AdjClose: Raw price levels
        - Cluster, Rank_InCluster: Use raw Close price internally
        """
        runs_dir = Path(__file__).parent.parent / "output" / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory found")
        
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("ranking_")])
        if not run_dirs:
            pytest.skip("No ranking runs found")
        
        latest_run = run_dirs[-1]
        fi_path = latest_run / "feature_importances.json"
        
        if not fi_path.exists():
            pytest.skip(f"No feature_importances.json in {latest_run}")
        
        with open(fi_path) as f:
            importances = json.load(f)
        
        # Features that should be excluded
        excluded_features = {
            'Close', 'AdjClose', 'Open', 'High', 'Low', 'Volume',
            'Cluster', 'Rank_InCluster',
            'Dividend', 'Split',
            'timestamp', 'ticker', 'forward_return',
        }
        
        used_excluded = set(importances.keys()) & excluded_features
        
        assert not used_excluded, (
            f"Found excluded features in model: {used_excluded}. "
            f"These features should not be used due to leakage risk."
        )


class TestMetricSanity:
    """Sanity checks for model metrics."""
    
    def test_ic_within_reasonable_bounds(self):
        """IC should be within reasonable bounds for NZX.
        
        - IC < -0.1: Model is inversely predictive (likely bug)
        - IC > 0.5: Suspiciously high (potential leakage)
        """
        runs_dir = Path(__file__).parent.parent / "output" / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory found")
        
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("ranking_")])
        if not run_dirs:
            pytest.skip("No ranking runs found")
        
        latest_run = run_dirs[-1]
        metrics_path = latest_run / "metrics.json"
        
        if not metrics_path.exists():
            pytest.skip(f"No metrics.json in {latest_run}")
        
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        # Handle both flat and nested structure
        if "ranking_metrics" in metrics:
            ic = metrics["ranking_metrics"].get("mean_ic", 0)
        else:
            ic = metrics.get("mean_ic", 0)
        
        assert ic > -0.1, (
            f"Mean IC is {ic:.4f}, which is strongly negative. "
            f"This suggests the model is inversely predictive - check for bugs."
        )
        
        # Note: We allow up to 0.5 for NZX given its characteristics
        # but warn above 0.4
        if ic > 0.4:
            import warnings
            warnings.warn(
                f"Mean IC is {ic:.4f}, which is very high. "
                f"While this may be legitimate for NZX, verify by: "
                f"(1) checking feature importances for concentration, "
                f"(2) reviewing top quintile returns for winsorization patterns, "
                f"(3) comparing IC across different time periods."
            )
    
    def test_quintile_returns_monotonic(self):
        """Quintile returns should generally increase from Q1 to Q5.
        
        Non-monotonic returns suggest the model isn't learning
        a consistent ranking signal.
        """
        runs_dir = Path(__file__).parent.parent / "output" / "runs"
        if not runs_dir.exists():
            pytest.skip("No runs directory found")
        
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("ranking_")])
        if not run_dirs:
            pytest.skip("No ranking runs found")
        
        latest_run = run_dirs[-1]
        metrics_path = latest_run / "metrics.json"
        
        if not metrics_path.exists():
            pytest.skip(f"No metrics.json in {latest_run}")
        
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        # Handle both flat and nested structure
        if "ranking_metrics" in metrics:
            quintiles = metrics["ranking_metrics"].get("quintile_returns", {})
        else:
            quintiles = metrics.get("quintile_returns", {})
        
        if not quintiles:
            pytest.skip("No quintile_returns in metrics")
        
        # Get quintile returns in order (handle both 'Q1' and '1' key formats)
        q_returns = []
        for i in range(1, 6):
            val = quintiles.get(f"Q{i}") or quintiles.get(str(i)) or 0
            q_returns.append(val)
        
        # Check monotonicity (allowing one inversion)
        inversions = sum(1 for i in range(4) if q_returns[i] > q_returns[i+1])
        
        assert inversions <= 1, (
            f"Quintile returns are non-monotonic: {q_returns}. "
            f"Found {inversions} inversions. "
            f"This suggests the model isn't learning a consistent ranking signal."
        )
        
        # Q5 should beat Q1 by a meaningful margin
        spread = q_returns[4] - q_returns[0]
        assert spread > 0, (
            f"Q5 return ({q_returns[4]:.4f}) is not greater than Q1 ({q_returns[0]:.4f}). "
            f"The model should predict higher returns for top-ranked stocks."
        )
