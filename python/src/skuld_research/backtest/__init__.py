from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from skuld_research.backtest.engine import BacktestConfig, BacktestEngine
	from skuld_research.backtest.metrics import compute_drawdown_series, compute_max_drawdown
	from skuld_research.backtest.walk_forward import FoldSpec, WalkForwardEngine

_EXPORTS = {
	"BacktestConfig": ("skuld_research.backtest.engine", "BacktestConfig"),
	"BacktestEngine": ("skuld_research.backtest.engine", "BacktestEngine"),
	"compute_drawdown_series": (
		"skuld_research.backtest.metrics",
		"compute_drawdown_series",
	),
	"compute_max_drawdown": ("skuld_research.backtest.metrics", "compute_max_drawdown"),
	"FoldSpec": ("skuld_research.backtest.walk_forward", "FoldSpec"),
	"WalkForwardEngine": ("skuld_research.backtest.walk_forward", "WalkForwardEngine"),
}

__all__ = ["BacktestConfig", "BacktestEngine", "compute_drawdown_series", "compute_max_drawdown", "FoldSpec", "WalkForwardEngine"]


def __getattr__(name: str) -> Any:
	if name not in _EXPORTS:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module_name, attr_name = _EXPORTS[name]
	value = getattr(import_module(module_name), attr_name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
	return sorted(set(globals()) | set(__all__))
