"""Model persistence utilities for saving and loading trained models.

This module provides functions to save trained ranking models along with
all necessary components (scaler, feature columns, configuration) for
making predictions on new data.

Usage:
    # Save after training
    from core.model_persistence import save_model, load_model
    
    model_bundle = ModelBundle(
        ranker=trained_ranker,
        scaler=fitted_scaler,
        feature_columns=feature_cols,
        config=training_config,
    )
    save_model(model_bundle, "output/models/ranking_model.pkl")
    
    # Load for prediction
    loaded_bundle = load_model("output/models/ranking_model.pkl")
    predictions = loaded_bundle.ranker.predict(X_new[loaded_bundle.feature_columns])
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import pickle

from core.scaler import ScalerSet
from learner.ranking import LightGBMRankerWrapper, RankerConfig


@dataclass
class ModelBundle:
    """Container for all components needed for prediction.
    
    This bundles together everything required to make predictions on new data:
    - The trained ranking model
    - The fitted scaler (must be applied to features before prediction)
    - The list of feature columns (in correct order)
    - Training configuration (for reference/reproducibility)
    - Metadata (creation time, data hash, git commit, etc.)
    
    Attributes:
        ranker: Trained LightGBMRankerWrapper instance.
        scaler: Fitted ScalerSet for feature scaling.
        feature_columns: Ordered list of feature column names.
        config: Dictionary with training configuration.
        metadata: Dictionary with creation metadata.
    """
    ranker: LightGBMRankerWrapper
    scaler: ScalerSet
    feature_columns: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Add creation timestamp if not present."""
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
    
    @property
    def n_features(self) -> int:
        """Number of features expected by the model."""
        return len(self.feature_columns)
    
    @property
    def created_at(self) -> str:
        """When the model was created."""
        return self.metadata.get("created_at", "unknown")
    
    def summary(self) -> str:
        """Return a summary string of the model bundle."""
        lines = [
            "=" * 50,
            "MODEL BUNDLE SUMMARY",
            "=" * 50,
            f"Created:     {self.created_at}",
            f"Features:    {self.n_features}",
            f"Ranker:      LightGBM ({self.config.get('n_estimators', '?')} estimators)",
        ]
        
        if "forward_return_days" in self.config:
            lines.append(f"Horizon:     {self.config['forward_return_days']} days")
        
        if "data_hash" in self.metadata:
            lines.append(f"Data hash:   {self.metadata['data_hash'][:12]}...")
        
        if "git_commit" in self.metadata:
            lines.append(f"Git commit:  {self.metadata['git_commit'][:8]}...")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def save_model(
    bundle: ModelBundle,
    filepath: str | Path,
    create_dirs: bool = True,
) -> Path:
    """Save a model bundle to disk.
    
    The bundle is saved as a pickle file containing all components needed
    for prediction. A JSON sidecar file is also created with human-readable
    metadata.
    
    Args:
        bundle: ModelBundle to save.
        filepath: Path to save the model (should end in .pkl).
        create_dirs: If True, create parent directories if needed.
    
    Returns:
        Path to the saved model file.
    
    Example:
        >>> bundle = ModelBundle(ranker=ranker, scaler=scaler, feature_columns=cols)
        >>> save_model(bundle, "models/my_model.pkl")
        PosixPath('models/my_model.pkl')
    """
    filepath = Path(filepath)
    
    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the pickle file
    with open(filepath, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save metadata as JSON sidecar for easy inspection
    meta_path = filepath.with_suffix(".meta.json")
    meta_info = {
        "created_at": bundle.metadata.get("created_at"),
        "n_features": bundle.n_features,
        "feature_columns": bundle.feature_columns,
        "config": bundle.config,
        "metadata": bundle.metadata,
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2, default=str)
    
    return filepath


def load_model(filepath: str | Path) -> ModelBundle:
    """Load a model bundle from disk.
    
    Args:
        filepath: Path to the saved model (.pkl file).
    
    Returns:
        Loaded ModelBundle ready for predictions.
    
    Raises:
        FileNotFoundError: If the model file doesn't exist.
        ValueError: If the file is not a valid ModelBundle.
    
    Example:
        >>> bundle = load_model("models/my_model.pkl")
        >>> predictions = bundle.ranker.predict(X_new[bundle.feature_columns])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, "rb") as f:
        bundle = pickle.load(f)
    
    if not isinstance(bundle, ModelBundle):
        raise ValueError(
            f"Expected ModelBundle, got {type(bundle).__name__}. "
            "File may be corrupted or from an incompatible version."
        )
    
    return bundle


def get_latest_model(models_dir: str | Path) -> Optional[Path]:
    """Get the path to the most recently created model in a directory.
    
    Args:
        models_dir: Directory containing saved models.
    
    Returns:
        Path to the most recent .pkl model file, or None if no models found.
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        return None
    
    pkl_files = list(models_dir.glob("*.pkl"))
    
    if not pkl_files:
        return None
    
    # Sort by modification time, most recent first
    pkl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return pkl_files[0]


def list_models(models_dir: str | Path) -> List[Dict[str, Any]]:
    """List all models in a directory with their metadata.
    
    Args:
        models_dir: Directory containing saved models.
    
    Returns:
        List of dicts with model info (path, created_at, n_features, etc.)
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        return []
    
    models = []
    
    for pkl_file in models_dir.glob("*.pkl"):
        meta_file = pkl_file.with_suffix(".meta.json")
        
        info = {
            "path": str(pkl_file),
            "filename": pkl_file.name,
            "size_mb": pkl_file.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(pkl_file.stat().st_mtime).isoformat(),
        }
        
        # Load metadata if available
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                info["created_at"] = meta.get("created_at")
                info["n_features"] = meta.get("n_features")
                info["config"] = meta.get("config", {})
            except Exception:
                pass
        
        models.append(info)
    
    # Sort by creation/modification time
    models.sort(key=lambda m: m.get("created_at", m.get("modified", "")), reverse=True)
    
    return models


def compute_data_fingerprint(df, columns: List[str] = None) -> str:
    """Compute a hash fingerprint of DataFrame for reproducibility tracking.
    
    Args:
        df: DataFrame to hash.
        columns: Specific columns to include (all if None).
    
    Returns:
        SHA256 hash string (first 16 chars).
    """
    import pandas as pd
    import numpy as np
    
    if columns:
        df = df[columns]
    
    # Hash shape and column names (all columns)
    content = f"shape={df.shape};cols={sorted(df.columns.tolist())}"
    
    # Add sample of data values for uniqueness
    # Use more rows and columns for better discrimination
    sample_size = min(1000, len(df))
    if sample_size > 0:
        # Sample from different parts of the dataframe
        head_sample = df.head(sample_size // 2)
        tail_sample = df.tail(sample_size // 2)
        sample = pd.concat([head_sample, tail_sample], ignore_index=True)
        
        # Use up to 50 columns spread across the dataframe
        n_cols = min(50, len(sample.columns))
        col_indices = np.linspace(0, len(sample.columns) - 1, n_cols, dtype=int)
        cols_to_sample = [sample.columns[i] for i in col_indices]
        
        for col in cols_to_sample:
            vals = sample[col].dropna().head(100).tolist()
            content += f";{col}={vals}"
        
        # Also include basic statistics for numeric columns
        numeric_cols = sample.select_dtypes(include=[np.number]).columns[:20]
        for col in numeric_cols:
            if col in sample.columns:
                col_data = sample[col].dropna()
                if len(col_data) > 0:
                    content += f";{col}_stats=({col_data.mean():.6f},{col_data.std():.6f})"
    
    return hashlib.sha256(content.encode()).hexdigest()[:16]
