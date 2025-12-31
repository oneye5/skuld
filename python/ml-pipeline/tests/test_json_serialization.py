"""Tests for JSON serialization of numpy types."""

import json
import numpy as np
import pytest

from pipeline.ranking_pipeline import _to_json_serializable


class TestToJsonSerializable:
    """Test the _to_json_serializable helper function."""

    def test_float32_conversion(self):
        """float32 should be converted to Python float."""
        val = np.float32(3.14)
        result = _to_json_serializable(val)
        assert isinstance(result, float)
        assert result == pytest.approx(3.14, rel=1e-5)
        # Should be JSON serializable
        json.dumps(result)

    def test_float64_conversion(self):
        """float64 should be converted to Python float."""
        val = np.float64(2.718)
        result = _to_json_serializable(val)
        assert isinstance(result, float)
        assert result == pytest.approx(2.718)
        json.dumps(result)

    def test_int32_conversion(self):
        """int32 should be converted to Python int."""
        val = np.int32(42)
        result = _to_json_serializable(val)
        assert isinstance(result, int)
        assert result == 42
        json.dumps(result)

    def test_int64_conversion(self):
        """int64 should be converted to Python int."""
        val = np.int64(123456789)
        result = _to_json_serializable(val)
        assert isinstance(result, int)
        assert result == 123456789
        json.dumps(result)

    def test_bool_conversion(self):
        """numpy bool_ should be converted to Python bool."""
        val_true = np.bool_(True)
        val_false = np.bool_(False)
        assert _to_json_serializable(val_true) is True
        assert _to_json_serializable(val_false) is False
        json.dumps(_to_json_serializable(val_true))

    def test_ndarray_conversion(self):
        """numpy array should be converted to Python list."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _to_json_serializable(arr)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]
        json.dumps(result)

    def test_nested_dict_conversion(self):
        """Nested dict with numpy types should be fully converted."""
        data = {
            "metrics": {
                "mean_ic": np.float32(0.372),
                "icir": np.float64(2.15),
                "count": np.int32(80),
            },
            "portfolio": {
                "sharpe": np.float32(2.30),
                "positions": np.int64(10),
            },
        }
        result = _to_json_serializable(data)
        
        # Check types are converted
        assert isinstance(result["metrics"]["mean_ic"], float)
        assert isinstance(result["metrics"]["icir"], float)
        assert isinstance(result["metrics"]["count"], int)
        assert isinstance(result["portfolio"]["sharpe"], float)
        assert isinstance(result["portfolio"]["positions"], int)
        
        # Should be fully JSON serializable
        json_str = json.dumps(result)
        assert "0.37" in json_str  # Verify value is present

    def test_nested_list_conversion(self):
        """Nested list with numpy types should be fully converted."""
        data = [
            {"value": np.float32(1.5)},
            {"value": np.float32(2.5)},
            {"array": np.array([1, 2, 3])},
        ]
        result = _to_json_serializable(data)
        
        assert isinstance(result[0]["value"], float)
        assert isinstance(result[1]["value"], float)
        assert isinstance(result[2]["array"], list)
        
        json.dumps(result)

    def test_mixed_types_preserved(self):
        """Native Python types should be preserved unchanged."""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
        }
        result = _to_json_serializable(data)
        
        assert result["string"] == "hello"
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None
        assert result["list"] == [1, 2, 3]

    def test_realistic_backtest_metrics(self):
        """Test with realistic backtest metrics structure (the actual failure case)."""
        backtest_metrics = {
            "returns": {
                "total_return": np.float32(0.1565),
                "annualized_return": np.float64(0.0823),
                "sharpe_ratio": np.float32(2.30),
                "sortino_ratio": np.float64(3.15),
                "calmar_ratio": np.float32(1.85),
                "max_drawdown": np.float32(-0.0845),
            },
            "implementation": {
                "avg_turnover_per_rebalance": np.float32(0.45),
                "avg_cost_per_rebalance": np.float64(0.0019),
                "total_cost_drag": np.float32(0.0152),
                "return_per_unit_turnover": np.float64(0.183),
                "num_rebalances": np.int32(80),
                "avg_holding_period_years": np.float32(0.25),
            },
            "portfolio": {
                "long_positions": 10,
                "short_positions": 0,
                "transaction_cost_bps": 190.0,
                "slippage_bps": np.float32(15.0),
            },
        }
        
        result = _to_json_serializable(backtest_metrics)
        
        # This was the actual failure - should now work
        json_str = json.dumps(result, indent=2)
        
        # Verify structure is preserved
        parsed = json.loads(json_str)
        assert parsed["returns"]["sharpe_ratio"] == pytest.approx(2.30, rel=1e-2)
        assert parsed["implementation"]["num_rebalances"] == 80
        assert parsed["portfolio"]["long_positions"] == 10
