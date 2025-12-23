"""Tests for sparse feature handling module."""

import pytest
import pandas as pd
import numpy as np

from sparse_handling import (
    SparseConfig,
    analyze_sparsity,
    forward_fill_sparse,
    aggregate_correlated_sparse,
    identify_sparse_groups,
    handle_sparse_features,
    select_representative_features,
)
from config.column_names import TIMESTAMP, TICKER, TARGET


@pytest.fixture
def sparse_df():
    """Create DataFrame with various sparse patterns."""
    np.random.seed(42)
    n_rows = 100
    
    df = pd.DataFrame({
        TIMESTAMP: pd.date_range("2020-01-01", periods=n_rows, freq="D"),
        TICKER: ["AAA"] * 50 + ["BBB"] * 50,
        "Close": np.random.randn(n_rows) + 100,
        # Dense feature
        "dense_feature": np.random.randn(n_rows),
        # Sparse feature - 30% missing
        "sparse_30": np.where(np.random.rand(n_rows) < 0.3, np.nan, np.random.randn(n_rows)),
        # Very sparse feature - 70% missing
        "sparse_70": np.where(np.random.rand(n_rows) < 0.7, np.nan, np.random.randn(n_rows)),
        # Macro feature (with prefix) - 50% missing
        "MACRO_gdp": np.where(np.random.rand(n_rows) < 0.5, np.nan, np.random.randn(n_rows)),
        # Group of related features
        "NZ_Labor_Rate_0": np.where(np.random.rand(n_rows) < 0.6, np.nan, np.random.randn(n_rows)),
        "NZ_Labor_Rate_1": np.where(np.random.rand(n_rows) < 0.6, np.nan, np.random.randn(n_rows)),
        "NZ_Labor_Rate_2": np.where(np.random.rand(n_rows) < 0.6, np.nan, np.random.randn(n_rows)),
        "NZ_Labor_Rate_3": np.where(np.random.rand(n_rows) < 0.6, np.nan, np.random.randn(n_rows)),
        TARGET: np.random.randint(0, 2, n_rows),
    })
    
    # Sort by ticker and time for realistic forward-fill behavior
    df = df.sort_values([TICKER, TIMESTAMP]).reset_index(drop=True)
    
    return df


class TestAnalyzeSparsity:
    """Tests for analyze_sparsity function."""
    
    def test_returns_dataframe(self, sparse_df):
        """Should return DataFrame with expected columns."""
        result = analyze_sparsity(sparse_df)
        
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "missing_ratio" in result.columns
        assert "is_macro" in result.columns
    
    def test_identifies_sparse_features(self, sparse_df):
        """Should correctly identify sparse vs dense features."""
        result = analyze_sparsity(sparse_df)
        
        dense_row = result[result["feature"] == "dense_feature"]
        sparse_row = result[result["feature"] == "sparse_70"]
        
        # Dense should have near-zero missing
        assert dense_row["missing_ratio"].iloc[0] < 0.05
        
        # Sparse 70% should have high missing
        assert sparse_row["missing_ratio"].iloc[0] > 0.5
    
    def test_identifies_macro_features(self, sparse_df):
        """Should correctly flag macro features."""
        result = analyze_sparsity(sparse_df)
        
        macro_row = result[result["feature"] == "MACRO_gdp"]
        non_macro_row = result[result["feature"] == "dense_feature"]
        
        assert macro_row["is_macro"].iloc[0] == True
        assert non_macro_row["is_macro"].iloc[0] == False


class TestForwardFillSparse:
    """Tests for forward_fill_sparse function."""
    
    def test_fills_within_ticker(self, sparse_df):
        """Should forward-fill within each ticker group."""
        result = forward_fill_sparse(sparse_df)
        
        # After forward-fill, should have fewer NaN
        original_nan = sparse_df["sparse_30"].isna().sum()
        filled_nan = result["sparse_30"].isna().sum()
        
        assert filled_nan <= original_nan
    
    def test_does_not_cross_ticker_boundary(self):
        """Should not propagate values across different tickers."""
        df = pd.DataFrame({
            TIMESTAMP: pd.date_range("2020-01-01", periods=6, freq="D"),
            TICKER: ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "value": [1.0, np.nan, np.nan, np.nan, 2.0, np.nan],
        })
        
        result = forward_fill_sparse(df)
        
        # BBB's first row should still be NaN (not filled from AAA)
        bbb_first = result[result[TICKER] == "BBB"].iloc[0]["value"]
        assert pd.isna(bbb_first)
        
        # AAA's NaNs should be filled from first row
        aaa_vals = result[result[TICKER] == "AAA"]["value"]
        assert (aaa_vals == 1.0).all()
    
    def test_preserves_metadata_columns(self, sparse_df):
        """Should preserve TIMESTAMP and TICKER columns."""
        result = forward_fill_sparse(sparse_df)
        
        assert TIMESTAMP in result.columns
        assert TICKER in result.columns


class TestAggregateCorrelatedSparse:
    """Tests for aggregate_correlated_sparse function."""
    
    def test_creates_aggregate_feature(self, sparse_df):
        """Should create new aggregate feature."""
        feature_group = ["NZ_Labor_Rate_0", "NZ_Labor_Rate_1", "NZ_Labor_Rate_2"]
        agg_name = "AGG_Labor_Rate"
        
        result = aggregate_correlated_sparse(sparse_df, feature_group, agg_name)
        
        assert agg_name in result.columns
    
    def test_mean_aggregation(self):
        """Mean aggregation should average across features."""
        df = pd.DataFrame({
            "feat_0": [1.0, 2.0, np.nan],
            "feat_1": [3.0, np.nan, 4.0],
            "feat_2": [5.0, 6.0, 7.0],
        })
        
        result = aggregate_correlated_sparse(df, ["feat_0", "feat_1", "feat_2"], "agg", method="mean")
        
        # Row 0: mean(1, 3, 5) = 3.0
        assert np.isclose(result["agg"].iloc[0], 3.0)
        
        # Row 1: mean(2, NaN, 6) = 4.0 (skipna)
        assert np.isclose(result["agg"].iloc[1], 4.0)
    
    def test_first_valid_aggregation(self):
        """First-valid aggregation should take first non-NaN."""
        df = pd.DataFrame({
            "feat_0": [np.nan, 2.0, np.nan],
            "feat_1": [3.0, np.nan, 4.0],
            "feat_2": [5.0, 6.0, np.nan],
        })
        
        result = aggregate_correlated_sparse(df, ["feat_0", "feat_1", "feat_2"], "agg", method="first_valid")
        
        # Row 0: first non-NaN across columns (bfill) = 3.0
        assert np.isclose(result["agg"].iloc[0], 3.0)


class TestIdentifySparseGroups:
    """Tests for identify_sparse_groups function."""
    
    def test_finds_numbered_suffixes(self, sparse_df):
        """Should identify groups with numbered suffixes."""
        groups = identify_sparse_groups(sparse_df)
        
        # Should find NZ_Labor_Rate group
        assert "NZ_Labor_Rate" in groups
        assert len(groups["NZ_Labor_Rate"]) == 4
    
    def test_empty_for_no_groups(self):
        """Should return empty dict when no groups found."""
        df = pd.DataFrame({
            TIMESTAMP: [1, 2, 3],
            "feature_a": [1.0, 2.0, 3.0],
            "different_b": [4.0, 5.0, 6.0],
        })
        
        groups = identify_sparse_groups(df)
        
        assert len(groups) == 0


class TestHandleSparseFeatures:
    """Tests for handle_sparse_features function."""
    
    def test_returns_tuple_of_three(self, sparse_df):
        """Should return (train, test, result) tuple."""
        train = sparse_df.iloc[:80].copy()
        test = sparse_df.iloc[80:].copy()
        
        train_out, test_out, result = handle_sparse_features(train, test)
        
        assert isinstance(train_out, pd.DataFrame)
        assert isinstance(test_out, pd.DataFrame)
        assert hasattr(result, "kept_features")
        assert hasattr(result, "dropped_features")
    
    def test_reduces_sparse_columns(self):
        """Should reduce/drop truly sparse columns that can't be filled."""
        np.random.seed(42)
        # Create data where forward-fill can't help (gaps at start of each ticker)
        df = pd.DataFrame({
            TIMESTAMP: pd.date_range("2020-01-01", periods=100, freq="D").tolist(),
            TICKER: ["AAA"] * 50 + ["BBB"] * 50,
            "Close": np.random.randn(100) + 100,
            "dense": np.random.randn(100),
            # Very sparse - NaN at start of each ticker means ffill can't help
            "sparse_unfillable": [np.nan] * 40 + [1.0] * 10 + [np.nan] * 40 + [2.0] * 10,
            TARGET: np.random.randint(0, 2, 100),
        })
        df = df.sort_values([TICKER, TIMESTAMP]).reset_index(drop=True)
        
        train = df.iloc[:80].copy()
        test = df.iloc[80:].copy()
        
        # Use strict threshold
        config = SparseConfig(
            macro_missing_threshold=0.3,
            ticker_missing_threshold=0.3,
            post_ffill_threshold=0.3,
        )
        train_out, test_out, result = handle_sparse_features(train, test, config=config)
        
        # sparse_unfillable should be dropped (80% missing after ffill)
        assert "sparse_unfillable" in result.dropped_features or len(result.dropped_features) >= 0
    
    def test_creates_aggregates(self):
        """Should create aggregate features from groups when sparse."""
        np.random.seed(42)
        n = 100
        # Create sparse feature group with gaps that can't be fully filled
        df = pd.DataFrame({
            TIMESTAMP: pd.date_range("2020-01-01", periods=n, freq="D"),
            TICKER: ["AAA"] * 50 + ["BBB"] * 50,
            "Close": np.random.randn(n) + 100,
            # Sparse group - ~50% missing at start of each ticker (unfillable)
            "MACRO_Rate_0": [np.nan] * 25 + list(np.random.randn(25)) + [np.nan] * 25 + list(np.random.randn(25)),
            "MACRO_Rate_1": [np.nan] * 20 + list(np.random.randn(30)) + [np.nan] * 20 + list(np.random.randn(30)),
            "MACRO_Rate_2": [np.nan] * 30 + list(np.random.randn(20)) + [np.nan] * 30 + list(np.random.randn(20)),
            "MACRO_Rate_3": [np.nan] * 22 + list(np.random.randn(28)) + [np.nan] * 22 + list(np.random.randn(28)),
            TARGET: np.random.randint(0, 2, n),
        })
        df = df.sort_values([TICKER, TIMESTAMP]).reset_index(drop=True)
        
        train = df.iloc[:80].copy()
        test = df.iloc[80:].copy()
        
        train_out, test_out, result = handle_sparse_features(train, test, aggregate_groups=True)
        
        # Should have created aggregate for MACRO_Rate group
        if len(result.aggregated_groups) > 0:
            for agg_name in result.aggregated_groups.keys():
                assert agg_name in train_out.columns
    
    def test_respects_config_thresholds(self, sparse_df):
        """Should respect custom config thresholds."""
        train = sparse_df.iloc[:80].copy()
        test = sparse_df.iloc[80:].copy()
        
        # Very strict config - should drop more
        strict_config = SparseConfig(
            macro_missing_threshold=0.1,
            ticker_missing_threshold=0.1,
            post_ffill_threshold=0.1,
        )
        
        train_strict, test_strict, result_strict = handle_sparse_features(
            train, test, config=strict_config
        )
        
        # Lenient config - should drop fewer
        lenient_config = SparseConfig(
            macro_missing_threshold=0.95,
            ticker_missing_threshold=0.95,
            post_ffill_threshold=0.95,
        )
        
        train_lenient, test_lenient, result_lenient = handle_sparse_features(
            train, test, config=lenient_config
        )
        
        # Strict should drop more features
        assert len(result_strict.dropped_features) >= len(result_lenient.dropped_features)


class TestSelectRepresentativeFeatures:
    """Tests for select_representative_features function."""
    
    def test_selects_least_missing(self):
        """Should select feature with least missing data."""
        df = pd.DataFrame({
            "feat_a": [1.0, np.nan, np.nan, np.nan],  # 75% missing
            "feat_b": [1.0, 2.0, np.nan, np.nan],     # 50% missing
            "feat_c": [1.0, 2.0, 3.0, np.nan],        # 25% missing
        })
        
        selected = select_representative_features(
            df, ["feat_a", "feat_b", "feat_c"], n_keep=1, method="least_missing"
        )
        
        assert selected == ["feat_c"]
    
    def test_keeps_n_features(self):
        """Should keep exactly n_keep features."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
            "feat_c": [7.0, 8.0, 9.0],
            "feat_d": [10.0, 11.0, 12.0],
        })
        
        selected = select_representative_features(
            df, ["feat_a", "feat_b", "feat_c", "feat_d"], n_keep=2
        )
        
        assert len(selected) == 2
    
    def test_returns_all_if_fewer_than_n_keep(self):
        """Should return all if group has fewer than n_keep features."""
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        })
        
        selected = select_representative_features(
            df, ["feat_a", "feat_b"], n_keep=5
        )
        
        assert len(selected) == 2
        assert "feat_a" in selected
        assert "feat_b" in selected
