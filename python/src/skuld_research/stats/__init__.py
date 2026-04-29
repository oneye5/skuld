"""Statistical gating infrastructure."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skuld_research.stats.bootstrap import stationary_bootstrap_sharpe
    from skuld_research.stats.deflated import deflated_sharpe
    from skuld_research.stats.dominance import romano_wolf_stepwise
    from skuld_research.stats.gating import evaluate
    from skuld_research.stats.ledger import (
        ExplorationTrialLedger,
        ProductionTrialLedger,
        n_trials_prior,
    )
    from skuld_research.stats.regimes import label_regimes
    from skuld_research.stats.rolling_walk_forward import RollingWalkForwardEngine

_EXPORTS = {
    "stationary_bootstrap_sharpe": (
        "skuld_research.stats.bootstrap",
        "stationary_bootstrap_sharpe",
    ),
    "deflated_sharpe": ("skuld_research.stats.deflated", "deflated_sharpe"),
    "romano_wolf_stepwise": (
        "skuld_research.stats.dominance",
        "romano_wolf_stepwise",
    ),
    "label_regimes": ("skuld_research.stats.regimes", "label_regimes"),
    "RollingWalkForwardEngine": (
        "skuld_research.stats.rolling_walk_forward",
        "RollingWalkForwardEngine",
    ),
    "ProductionTrialLedger": (
        "skuld_research.stats.ledger",
        "ProductionTrialLedger",
    ),
    "ExplorationTrialLedger": (
        "skuld_research.stats.ledger",
        "ExplorationTrialLedger",
    ),
    "n_trials_prior": ("skuld_research.stats.ledger", "n_trials_prior"),
    "evaluate": ("skuld_research.stats.gating", "evaluate"),
}

__all__ = [
    "stationary_bootstrap_sharpe",
    "deflated_sharpe",
    "romano_wolf_stepwise",
    "label_regimes",
    "RollingWalkForwardEngine",
    "ProductionTrialLedger",
    "ExplorationTrialLedger",
    "n_trials_prior",
    "evaluate",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
