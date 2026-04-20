"""Core data contract types for Skuld pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PITSnapshot:
    """All values knowable strictly before `asof`. Enforced, not asked nicely.

    Attributes:
        prices: index=date, columns=ticker, values=adj_close
        volumes: index=date, columns=ticker, values=volume
        fundamentals: MultiIndex (ticker, publication_date), columns=feature
        macro: index=date, columns=macro_feature
        corporate_actions: columns: ticker, ex_date, type, factor
        asof: the timestamp this snapshot was built for
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    fundamentals: pd.DataFrame
    macro: pd.DataFrame
    corporate_actions: pd.DataFrame
    asof: pd.Timestamp

    def __post_init__(self) -> None:
        asof_naive = self.asof.tz_localize(None) if self.asof.tzinfo else self.asof
        violations: list[str] = []

        # Check index-based frames: prices, volumes, macro
        for name, df in [("prices", self.prices), ("volumes", self.volumes), ("macro", self.macro)]:
            if not df.empty and len(df.index) > 0:
                max_date = pd.Timestamp(df.index.max())
                if max_date.tzinfo:
                    max_date = max_date.tz_localize(None)
                if max_date >= asof_naive:
                    violations.append(
                        f"{name}: max date {max_date} >= asof {self.asof}"
                    )

        # Check fundamentals (MultiIndex with publication_date level)
        if not self.fundamentals.empty and len(self.fundamentals.index) > 0:
            pub_dates = self.fundamentals.index.get_level_values("publication_date")
            max_pub = pd.Timestamp(pub_dates.max())
            if max_pub.tzinfo:
                max_pub = max_pub.tz_localize(None)
            if max_pub >= asof_naive:
                violations.append(
                    f"fundamentals: max publication_date {max_pub} >= asof {self.asof}"
                )

        # Check corporate_actions (ex_date column)
        if not self.corporate_actions.empty and "ex_date" in self.corporate_actions.columns:
            max_ex = pd.Timestamp(self.corporate_actions["ex_date"].max())
            if max_ex.tzinfo:
                max_ex = max_ex.tz_localize(None)
            if max_ex >= asof_naive:
                violations.append(
                    f"corporate_actions: max ex_date {max_ex} >= asof {self.asof}"
                )

        if violations:
            raise ValueError(
                f"PIT invariant violated — no future data allowed.\n"
                + "\n".join(f"  - {v}" for v in violations)
            )
