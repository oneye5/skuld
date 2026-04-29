"""RNG seed derivation for reproducibility."""
from __future__ import annotations

import numpy as np


# Module-level constant: documents the spawn order used to derive child seeds
SPAWN_ORDER_DOC = """Child RNG seeds derived via numpy.random.SeedSequence(master_seed).spawn(4):
1. bootstrap (stationary bootstrap for Sharpe CI)
2. mc_delisting (Monte Carlo delisting simulation for survivorship adjustment)
3. optimiser_tiebreak (portfolio optimizer tie-breaking when multiple solutions exist)
4. dominance (Romano-Wolf stepwise procedure bootstrap resampling)

Changing this order is a breaking change to reproducibility."""


def derive_child_seeds(master_seed: int) -> dict[str, int]:
    """Derive child RNG seeds from master seed using SeedSequence.
    
    Args:
        master_seed: master RNG seed.
    
    Returns:
        Dict mapping subsystem name to int seed:
        {"bootstrap": int, "mc_delisting": int, "optimiser_tiebreak": int, "dominance": int}
    """
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(4)
    
    # Convert each SeedSequence to an int via generate_state(1)[0]
    bootstrap_seed = int(children[0].generate_state(1)[0])
    mc_delisting_seed = int(children[1].generate_state(1)[0])
    optimiser_tiebreak_seed = int(children[2].generate_state(1)[0])
    dominance_seed = int(children[3].generate_state(1)[0])
    
    return {
        "bootstrap": bootstrap_seed,
        "mc_delisting": mc_delisting_seed,
        "optimiser_tiebreak": optimiser_tiebreak_seed,
        "dominance": dominance_seed,
    }
