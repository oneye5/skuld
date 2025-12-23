"""Tests for feature selection module - base classes."""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add paths for imports
_ml_pipeline = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_ml_pipeline))
sys.path.insert(0, str(_ml_pipeline / "data-preparation" / "feature-selection"))

from base import BaseFeatureSelector, FeatureSelectionResult, select_features_with_selector


class TestFeatureSelectionResult:
    """Tests for FeatureSelectionResult dataclass."""
    
    def test_create_result(self):
        """Test creating a result object."""
        result = FeatureSelectionResult(
            selected_features=["a", "b"],
            dropped_features=["c"],
            metadata={"method": "test"},
        )
        
        assert result.selected_features == ["a", "b"]
        assert result.dropped_features == ["c"]
        assert result.metadata["method"] == "test"


class TestBaseFeatureSelectorInterface:
    """Tests for BaseFeatureSelector abstract interface."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseFeatureSelector()
