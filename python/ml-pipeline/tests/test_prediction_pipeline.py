"""Tests for model persistence and prediction pipeline.

These tests verify that:
1. Models can be saved and loaded correctly
2. Predictions can be generated from loaded models
3. Feature consistency is maintained between training and prediction
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from config.columns import TIMESTAMP, TICKER


# =============================================================================
# MODEL PERSISTENCE TESTS
# =============================================================================

class TestModelBundle:
    """Tests for ModelBundle dataclass."""
    
    def test_model_bundle_creation(self):
        """Test creating a ModelBundle."""
        from core.model_persistence import ModelBundle
        from core.scaler import ScalerSet
        from sklearn.preprocessing import RobustScaler
        
        # Create minimal mock objects
        scaler_set = ScalerSet(
            continuous_scaler=RobustScaler(),
            continuous_cols=["feature_1", "feature_2"],
            binary_cols=["is_flag"],
            excluded_cols=[TIMESTAMP, TICKER],
        )
        
        # Create bundle without a real ranker (just test the dataclass)
        bundle = ModelBundle(
            ranker=None,  # Would be LightGBMRankerWrapper in production
            scaler=scaler_set,
            feature_columns=["feature_1", "feature_2", "is_flag"],
            config={"n_estimators": 100},
            metadata={"note": "test"},
        )
        
        assert bundle.n_features == 3
        assert "created_at" in bundle.metadata  # Auto-added
        assert bundle.config["n_estimators"] == 100
    
    def test_model_bundle_summary(self):
        """Test ModelBundle summary method."""
        from core.model_persistence import ModelBundle
        from core.scaler import ScalerSet
        from sklearn.preprocessing import RobustScaler
        
        scaler_set = ScalerSet(
            continuous_scaler=RobustScaler(),
            continuous_cols=["f1"],
            binary_cols=[],
            excluded_cols=[],
        )
        
        bundle = ModelBundle(
            ranker=None,
            scaler=scaler_set,
            feature_columns=["f1", "f2"],
            config={"n_estimators": 50, "forward_return_days": 365},
        )
        
        summary = bundle.summary()
        assert "MODEL BUNDLE" in summary
        assert "Features" in summary


class TestModelPersistence:
    """Tests for save_model and load_model functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        dirpath = tempfile.mkdtemp()
        yield Path(dirpath)
        shutil.rmtree(dirpath)
    
    def test_save_and_load_model(self, temp_dir):
        """Test saving and loading a model bundle."""
        from core.model_persistence import ModelBundle, save_model, load_model
        from core.scaler import ScalerSet
        from sklearn.preprocessing import RobustScaler
        
        # Create a simple bundle
        scaler = RobustScaler()
        scaler.fit([[1, 2], [3, 4], [5, 6]])  # Fit scaler
        
        scaler_set = ScalerSet(
            continuous_scaler=scaler,
            continuous_cols=["col1", "col2"],
            binary_cols=[],
            excluded_cols=[TIMESTAMP],
        )
        
        bundle = ModelBundle(
            ranker=None,  # Skip ranker for this basic test
            scaler=scaler_set,
            feature_columns=["col1", "col2"],
            config={"test_param": 42},
        )
        
        # Save
        model_path = temp_dir / "test_model.pkl"
        saved_path = save_model(bundle, model_path)
        
        assert saved_path.exists()
        assert (model_path.with_suffix(".meta.json")).exists()
        
        # Load
        loaded = load_model(model_path)
        
        assert isinstance(loaded, ModelBundle)
        assert loaded.n_features == 2
        assert loaded.config["test_param"] == 42
        assert loaded.feature_columns == ["col1", "col2"]
    
    def test_load_nonexistent_model(self, temp_dir):
        """Test loading a model that doesn't exist."""
        from core.model_persistence import load_model
        
        with pytest.raises(FileNotFoundError):
            load_model(temp_dir / "nonexistent.pkl")
    
    def test_get_latest_model(self, temp_dir):
        """Test getting the most recent model from a directory."""
        from core.model_persistence import ModelBundle, save_model, get_latest_model
        from core.scaler import ScalerSet
        from sklearn.preprocessing import RobustScaler
        import time
        
        scaler_set = ScalerSet(
            continuous_scaler=RobustScaler(),
            continuous_cols=["x"],
            binary_cols=[],
            excluded_cols=[],
        )
        
        # Save two models
        bundle1 = ModelBundle(ranker=None, scaler=scaler_set, feature_columns=["x"])
        save_model(bundle1, temp_dir / "model_old.pkl")
        
        time.sleep(0.1)  # Ensure different timestamps
        
        bundle2 = ModelBundle(ranker=None, scaler=scaler_set, feature_columns=["x", "y"])
        save_model(bundle2, temp_dir / "model_new.pkl")
        
        # Get latest
        latest = get_latest_model(temp_dir)
        
        assert latest is not None
        assert latest.name == "model_new.pkl"
    
    def test_list_models(self, temp_dir):
        """Test listing all models in a directory."""
        from core.model_persistence import ModelBundle, save_model, list_models
        from core.scaler import ScalerSet
        from sklearn.preprocessing import RobustScaler
        
        scaler_set = ScalerSet(
            continuous_scaler=RobustScaler(),
            continuous_cols=["x"],
            binary_cols=[],
            excluded_cols=[],
        )
        
        # Save two models
        save_model(
            ModelBundle(ranker=None, scaler=scaler_set, feature_columns=["a"]),
            temp_dir / "model_a.pkl"
        )
        save_model(
            ModelBundle(ranker=None, scaler=scaler_set, feature_columns=["b"]),
            temp_dir / "model_b.pkl"
        )
        
        models = list_models(temp_dir)
        
        assert len(models) == 2
        assert all("path" in m for m in models)
        assert all("filename" in m for m in models)


class TestDataFingerprint:
    """Tests for data fingerprinting."""
    
    def test_fingerprint_same_data(self):
        """Test that same data produces same fingerprint."""
        from core.model_persistence import compute_data_fingerprint
        
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })
        
        fp1 = compute_data_fingerprint(df)
        fp2 = compute_data_fingerprint(df)
        
        assert fp1 == fp2
    
    def test_fingerprint_different_data(self):
        """Test that different data produces different fingerprint."""
        from core.model_persistence import compute_data_fingerprint
        
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        
        fp1 = compute_data_fingerprint(df1)
        fp2 = compute_data_fingerprint(df2)
        
        assert fp1 != fp2


# =============================================================================
# PREDICTION RESULT TESTS
# =============================================================================

class TestPredictionResult:
    """Tests for PredictionResult from the pipeline module."""
    
    def test_prediction_result_top_picks(self):
        """Test getting top picks from prediction result."""
        from pipeline.ranking_pipeline import PredictionResult
        from datetime import datetime
        
        predictions = pd.DataFrame({
            TIMESTAMP: [1000000] * 20,
            TICKER: [f"STOCK_{i}" for i in range(20)],
            "predicted_score": list(range(20, 0, -1)),  # 20 down to 1
            "rank": list(range(1, 21)),
        })
        
        result = PredictionResult(
            predictions=predictions,
            prediction_date=datetime.now(),
            forward_days=365,
            n_stocks=20,
            feature_columns=["f1", "f2"],
            model_config={"n_estimators": 100},
        )
        
        top = result.top_picks
        assert len(top) == 10
        assert top.iloc[0][TICKER] == "STOCK_0"  # Highest score
    
    def test_prediction_result_get_stock_rank(self):
        """Test getting rank info for a specific stock."""
        from pipeline.ranking_pipeline import PredictionResult
        from datetime import datetime
        
        predictions = pd.DataFrame({
            TIMESTAMP: [1000000] * 10,
            TICKER: ["AAPL", "GOOG", "MSFT", "AMZN", "META", 
                     "NVDA", "TSLA", "JPM", "BAC", "WMT"],
            "predicted_score": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        
        result = PredictionResult(
            predictions=predictions,
            prediction_date=datetime.now(),
            forward_days=365,
            n_stocks=10,
            feature_columns=["f1"],
            model_config={},
        )
        
        # Test finding a stock
        info = result.get_stock_rank("MSFT")
        assert info is not None
        assert info["rank"] == 3
        assert info["score"] == 8
        assert info["percentile"] == 70.0  # Rank 3 of 10 = top 30% = 70th percentile
        
        # Test stock not found
        info = result.get_stock_rank("NONEXISTENT")
        assert info is None


# =============================================================================
# INTEGRATION TESTS (require full pipeline setup)
# =============================================================================

@pytest.mark.slow
class TestPredictionPipelineIntegration:
    """Integration tests for the full prediction pipeline.
    
    These tests are marked slow as they require loading real data.
    Run with: pytest -m slow
    """
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for models."""
        dirpath = tempfile.mkdtemp()
        yield Path(dirpath)
        shutil.rmtree(dirpath)
    
    def test_train_and_save_model(self, temp_model_dir):
        """Test training and saving a model end-to-end.
        
        Note: This test requires the actual data file to be present.
        """
        pytest.importorskip("lightgbm")
        
        from pipeline.ranking_pipeline import train_and_save_model
        from core.model_persistence import load_model
        
        model_path = temp_model_dir / "test_model.pkl"
        
        try:
            bundle = train_and_save_model(
                output_path=model_path,
                forward_days=30,  # Short horizon for faster test
                min_stocks=5,
            )
            
            assert model_path.exists()
            assert bundle.n_features > 0
            assert bundle.ranker is not None
            
            # Verify we can load it back
            loaded = load_model(model_path)
            assert loaded.n_features == bundle.n_features
            
        except FileNotFoundError:
            pytest.skip("Data file not available for integration test")
    
    def test_generate_predictions(self, temp_model_dir):
        """Test generating predictions from a saved model.
        
        Note: This test requires the actual data file to be present.
        """
        pytest.importorskip("lightgbm")
        
        from pipeline.ranking_pipeline import train_and_save_model, generate_predictions
        
        model_path = temp_model_dir / "test_model.pkl"
        
        try:
            # First train and save
            train_and_save_model(
                output_path=model_path,
                forward_days=30,
                min_stocks=5,
            )
            
            # Then generate predictions
            result = generate_predictions(model_path=model_path)
            
            assert result.n_stocks > 0
            assert len(result.predictions) > 0
            assert "predicted_score" in result.predictions.columns
            assert "rank" in result.predictions.columns
            
        except FileNotFoundError:
            pytest.skip("Data file not available for integration test")


# =============================================================================
# SCRIPT INTERFACE TESTS
# =============================================================================

class TestPredictionScriptInterface:
    """Tests for the prediction script command-line interface."""
    
    def test_script_imports(self):
        """Test that the prediction script can be imported."""
        # This verifies all imports work
        import scripts.run_predictions as script
        
        assert hasattr(script, "train_and_predict")
        assert hasattr(script, "load_and_predict")
        assert hasattr(script, "PredictionResult")
        assert hasattr(script, "main")
    
    def test_prediction_result_class(self):
        """Test PredictionResult class from script."""
        from scripts.run_predictions import PredictionResult
        from datetime import datetime
        
        predictions = pd.DataFrame({
            TIMESTAMP: [1000],
            TICKER: ["TEST"],
            "predicted_score": [1.0],
            "rank": [1],
        })
        
        result = PredictionResult(
            predictions=predictions,
            prediction_date=datetime.now(),
            forward_days=365,
            n_stocks=1,
            feature_columns=["f1"],
            model_config={"n_estimators": 100},
            training_samples=1000,
        )
        
        assert result.forward_days == 365
        assert result.training_samples == 1000
