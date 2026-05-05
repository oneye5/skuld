"""Point-in-time loader.

Wraps RawData and produces PITSnapshot instances filtered to strictly-before
a given timestamp. This is the central anti-lookahead control.
"""

from __future__ import annotations

import pandas as pd

from skuld_common.contracts import PITSnapshot
from skuld_research.data.csv_loader import RawData


class PITLoader:
    """Produces point-in-time snapshots from raw data.

    Usage:
        raw = load_raw_csv(path)
        loader = PITLoader(raw)
        snap = loader.as_of(pd.Timestamp("2025-01-15", tz="UTC"))
    """

    def __init__(self, raw: RawData) -> None:
        self._raw = raw

    def as_of(self, t: pd.Timestamp) -> PITSnapshot:
        """Return all data knowable strictly before `t`.

        Args:
            t: The as-of timestamp. Must be timezone-aware (UTC).

        Returns:
            PITSnapshot with all frames filtered to dates < t.
        """
        t = pd.Timestamp(t)
        t_naive = t.tz_localize(None) if t.tzinfo else t

        prices = self._filter_by_index(self._raw.prices, t_naive)
        prices = self._remove_negative_prices(prices)
        volumes = self._filter_by_index(self._raw.volumes, t_naive)
        fundamentals = self._filter_fundamentals(self._raw.fundamentals, t_naive)
        macro = self._filter_by_index(self._raw.macro, t_naive)
        corporate_actions = self._filter_corporate_actions(
            self._raw.corporate_actions, t_naive
        )

        # Sector labels are passed through without date filtering.  Yahoo-
        # sourced labels are current/backfilled classifications and carry no
        # meaningful PIT date.  The PITSnapshot docstring documents the
        # non-PIT-safe status; downstream SectorNeutraliser and sector-
        # relative factor code must treat sector-derived outputs as
        # diagnostic-only (exploration scope) when labels are not independently
        # dated or verified as PIT-safe.
        sector_labels = self._raw.sector_labels.copy() if not self._raw.sector_labels.empty else self._raw.sector_labels

        return PITSnapshot(
            prices=prices,
            volumes=volumes,
            fundamentals=fundamentals,
            macro=macro,
            corporate_actions=corporate_actions,
            asof=t,
            sector_labels=sector_labels,
        )

    @staticmethod
    def _filter_by_index(df: pd.DataFrame, t_naive: pd.Timestamp) -> pd.DataFrame:
        """Keep only rows where index < t_naive."""
        if df.empty:
            return df
        return df.loc[df.index < t_naive]

    @staticmethod
    def _remove_negative_prices(prices: pd.DataFrame) -> pd.DataFrame:
        """Replace negative prices with NaN, then drop all-NaN rows."""
        if prices.empty:
            return prices
        cleaned = prices.where(prices >= 0)
        return cleaned.dropna(how="all")

    @staticmethod
    def _filter_fundamentals(df: pd.DataFrame, t_naive: pd.Timestamp) -> pd.DataFrame:
        """Keep fundamentals where publication_date < t_naive."""
        if df.empty:
            return df
        pub_dates = df.index.get_level_values("publication_date")
        mask = pub_dates < t_naive
        return df.loc[mask]

    @staticmethod
    def _filter_corporate_actions(df: pd.DataFrame, t_naive: pd.Timestamp) -> pd.DataFrame:
        """Keep corporate actions where ex_date < t_naive."""
        if df.empty:
            return df
        return df.loc[df["ex_date"] < t_naive].reset_index(drop=True)
