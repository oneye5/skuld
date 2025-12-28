"""Tests for the ranking model module - LGBMRanker wrapper."""

import pandas as pd
import numpy as np
import pytest

from config.columns import TIMESTAMP, TICKER


class TestBuildGroupFromTimestamps:
    """Tests for build_group_from_timestamps utility."""
    
    def test_basic_group_building(self):
        """Build group sizes from sorted DataFrame."""
        from learner.ranking import build_group_from_timestamps
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 1, 2, 2, 3, 3, 3, 3],
            TICKER: ["A", "B", "C", "A", "B", "A", "B", "C", "D"],
        })
        
        groups = build_group_from_timestamps(df, timestamp_col=TIMESTAMP)
        
        assert groups == [3, 2, 4]  # 3 at ts=1, 2 at ts=2, 4 at ts=3
        assert sum(groups) == len(df)
    
    def test_single_timestamp(self):
        """Single timestamp should produce single group."""
        from learner.ranking import build_group_from_timestamps
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 1, 1, 1],
            TICKER: ["A", "B", "C", "D", "E"],
        })
        
        groups = build_group_from_timestamps(df, timestamp_col=TIMESTAMP)
        
        assert groups == [5]
    
    def test_raises_on_unsorted_data(self):
        """Should raise ValueError if DataFrame is not sorted by timestamp."""
        from learner.ranking import build_group_from_timestamps
        
        df = pd.DataFrame({
            TIMESTAMP: [2, 1, 3, 1, 2],  # Not sorted
            TICKER: ["A", "B", "C", "D", "E"],
        })
        
        with pytest.raises(ValueError, match="must be sorted"):
            build_group_from_timestamps(df, timestamp_col=TIMESTAMP)


class TestRankerConfig:
    """Tests for RankerConfig dataclass."""
    
    def test_default_values(self):
        """RankerConfig should have sensible defaults."""
        from learner.ranking import RankerConfig
        
        config = RankerConfig()
        
        assert config.n_estimators == 100
        assert config.learning_rate == 0.05
        assert config.objective == "lambdarank"
        assert config.metric == "ndcg"
    
    def test_custom_values(self):
        """RankerConfig should accept custom values."""
        from learner.ranking import RankerConfig
        
        config = RankerConfig(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
        )
        
        assert config.n_estimators == 200
        assert config.learning_rate == 0.1
        assert config.max_depth == 10


class TestLightGBMRankerWrapper:
    """Tests for LightGBMRankerWrapper."""
    
    @pytest.fixture
    def synthetic_ranking_data(self):
        """Create synthetic data for ranking tests."""
        np.random.seed(42)
        
        n_timestamps = 10
        n_stocks = 20
        n_features = 5
        
        data = []
        for ts in range(n_timestamps):
            for stock in range(n_stocks):
                features = np.random.randn(n_features)
                # Target is correlated with first feature
                target = features[0] * 0.5 + np.random.randn() * 0.1
                
                row = {TIMESTAMP: ts, TICKER: f"STOCK_{stock}"}
                for f in range(n_features):
                    row[f"feature_{f}"] = features[f]
                row["target"] = target
                data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values(TIMESTAMP).reset_index(drop=True)
        
        feature_cols = [f"feature_{f}" for f in range(n_features)]
        X = df[feature_cols]
        y = df["target"]
        
        return df, X, y, feature_cols
    
    def test_fit_predict_smoke(self, synthetic_ranking_data):
        """Basic fit/predict smoke test with synthetic data."""
        from learner.ranking import LightGBMRankerWrapper, build_group_from_timestamps, RankerConfig
        
        df, X, y, _ = synthetic_ranking_data
        groups = build_group_from_timestamps(df, timestamp_col=TIMESTAMP)
        
        config = RankerConfig(n_estimators=10)  # Small for speed
        ranker = LightGBMRankerWrapper(config)
        
        ranker.fit(X, y, groups)
        predictions = ranker.predict(X)
        
        assert len(predictions) == len(X)
        assert not np.any(np.isnan(predictions))
    
    def test_group_validation(self, synthetic_ranking_data):
        """Should raise ValueError if sum(group) != len(X)."""
        from learner.ranking import LightGBMRankerWrapper, RankerConfig
        
        df, X, y, _ = synthetic_ranking_data
        wrong_groups = [10, 10, 10]  # Sum = 30, but len(X) = 200
        
        config = RankerConfig(n_estimators=10)
        ranker = LightGBMRankerWrapper(config)
        
        with pytest.raises(ValueError, match="sum.*!=.*len"):
            ranker.fit(X, y, wrong_groups)
    
    def test_predict_without_fit_raises(self, synthetic_ranking_data):
        """Should raise RuntimeError if predict called before fit."""
        from learner.ranking import LightGBMRankerWrapper, RankerConfig
        
        _, X, _, _ = synthetic_ranking_data
        
        config = RankerConfig()
        ranker = LightGBMRankerWrapper(config)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            ranker.predict(X)
    
    def test_predictions_have_ranking_property(self, synthetic_ranking_data):
        """Higher predictions should generally correspond to higher targets."""
        from learner.ranking import LightGBMRankerWrapper, build_group_from_timestamps, RankerConfig
        from scipy.stats import spearmanr
        
        df, X, y, _ = synthetic_ranking_data
        groups = build_group_from_timestamps(df, timestamp_col=TIMESTAMP)
        
        config = RankerConfig(n_estimators=50)  # Enough to learn pattern
        ranker = LightGBMRankerWrapper(config)
        
        ranker.fit(X, y, groups)
        predictions = ranker.predict(X)
        
        # Should have positive rank correlation (model learns signal)
        corr, _ = spearmanr(predictions, y)
        assert corr > 0.3  # Reasonably correlated
    
    def test_feature_importances(self, synthetic_ranking_data):
        """Should be able to get feature importances after fitting."""
        from learner.ranking import LightGBMRankerWrapper, build_group_from_timestamps, RankerConfig
        
        df, X, y, feature_cols = synthetic_ranking_data
        groups = build_group_from_timestamps(df, timestamp_col=TIMESTAMP)
        
        config = RankerConfig(n_estimators=10)
        ranker = LightGBMRankerWrapper(config)
        
        ranker.fit(X, y, groups)
        importances = ranker.feature_importances()
        
        assert len(importances) == len(feature_cols)
        assert all(imp >= 0 for imp in importances)


class TestPrepareRankingData:
    """Tests for prepare_ranking_data utility."""
    
    def test_prepares_data_for_lgbm(self):
        """Should prepare DataFrame for LGBMRanker training."""
        from learner.ranking import prepare_ranking_data
        
        df = pd.DataFrame({
            TIMESTAMP: [1, 1, 2, 2, 2],
            TICKER: ["A", "B", "A", "B", "C"],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0.1, 0.2, 0.15, 0.25, 0.05],
        })
        
        X, y, groups = prepare_ranking_data(
            df, 
            feature_cols=["feature_1"],
            target_col="target",
            timestamp_col=TIMESTAMP,
        )
        
        assert len(X) == 5
        assert len(y) == 5
        assert groups == [2, 3]
        assert sum(groups) == len(X)
    
    def test_sorts_by_timestamp(self):
        """Should sort DataFrame by timestamp."""
        from learner.ranking import prepare_ranking_data
        
        df = pd.DataFrame({
            TIMESTAMP: [2, 1, 2, 1],  # Unsorted
            TICKER: ["A", "A", "B", "B"],
            "feature_1": [3.0, 1.0, 4.0, 2.0],
            "target": [0.3, 0.1, 0.4, 0.2],
        })
        
        X, y, groups = prepare_ranking_data(
            df,
            feature_cols=["feature_1"],
            target_col="target",
            timestamp_col=TIMESTAMP,
        )
        
        # Data should now be sorted: ts=1 first, then ts=2
        assert groups == [2, 2]
        assert y.iloc[0] == 0.1 or y.iloc[0] == 0.2  # One of the ts=1 values
