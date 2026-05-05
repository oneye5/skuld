"""Canonical feature-name constants used across the pipeline.

Feature names are stable identifiers that appear as values in the `feature`
column of `data_long.csv` and as column names in the resulting DataFrames.
They are referenced explicitly by feature-engineering and signal code, so
having a single source of truth here avoids string typos scattered through
the codebase.

The loader categorises rows by *feature name* (and ticker presence), not by
the originating source — consumers should not need to know which upstream
provider supplied a given observation.

Add new constants here as new features are introduced, then import the
constant rather than re-typing the string literal.
"""

from __future__ import annotations

# --- Per-ticker daily observations --------------------------------------
ADJ_CLOSE = "adj_close"
VOLUME = "volume"

# Raw (unadjusted) OHLC; used by cost-modelling components such as the
# Abdi-Ranaldo spread estimator. Adjusted prices distort H-L ranges
# retroactively when splits/dividends occur, so we keep these separate from
# `adj_close` (which is what the return calculations use).
HIGH = "high"
LOW = "low"
CLOSE = "close"
OPEN = "open"

# Price/volume features go into the wide `prices` / `volumes` panels.
PRICE_FEATURES = frozenset({ADJ_CLOSE, VOLUME})

# OHLC features used by cost models (loaded on demand, not part of the
# core PIT contract).
OHLC_FEATURES = frozenset({HIGH, LOW, CLOSE, OPEN})

# --- Per-ticker corporate-action events ---------------------------------
DIVIDEND = "dividend"
SPLIT = "split"

CORPORATE_ACTIONS = frozenset({DIVIDEND, SPLIT})

# --- Per-ticker sector classification -----------------------------------
# GICS sector label rows have ticker present and a string value (e.g.
# "Technology").  They are routed separately from fundamentals because the
# pipeline must preserve the string before the numeric coercion step.
# Yahoo-sourced labels are current/backfilled classifications, not dated
# PIT-safe historical membership; downstream code must check PIT safety.
GICS_SECTOR = "gics_sector"

# Features that carry string (non-numeric) values and must be extracted
# before the value column is coerced to float.
STRING_FEATURES = frozenset({GICS_SECTOR})

# --- Per-ticker fundamentals (publication-date-indexed) -----------------
# The loader does not enumerate fundamental features — any ticker-bearing
# feature that is not a price feature or corporate action is treated as a
# fundamental. List names here only when downstream code needs to reference
# them by constant.
ANNUAL_NET_INCOME = "annual_net_income_common_stockholders"

# --- Macro features (no ticker) -----------------------------------------
# Macro rows are identified by an empty ticker; feature names listed here
# are for downstream reference only.
OECD_BCICP = "oecd_bcicp"
