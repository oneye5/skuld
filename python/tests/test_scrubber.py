"""Tests for raw daily-price scrubber.

The scrubber detects single-day price prints that round-trip the next day
(e.g. SKT.NZ 2010-01-04 close=5.05 surrounded by ~$32) and replaces them
with the geometric mean of the surrounding prints. This eliminates the
artificial jump+rebound from the compounded monthly return without
introducing NaN gaps that downstream rolling computations would have to
tolerate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skuld_research.data.scrubber import (
    ScrubReport,
    ScrubResult,
    scrub_daily_prices,
)


def _make_prices(values: dict[str, list[float]]) -> pd.DataFrame:
    """Helper: build a daily-indexed price frame from per-ticker lists."""
    n = max(len(v) for v in values.values())
    idx = pd.date_range("2010-01-04", periods=n, freq="B")
    return pd.DataFrame(values, index=idx).astype(float)


class TestScrubDailyPrices:
    def test_flags_single_day_round_trip_jump(self) -> None:
        # SKT.NZ-style: a print of 5.05 between two prints near 32.
        prices = _make_prices(
            {"SKT.NZ": [32.10, 5.05, 32.61, 33.06, 33.57]}
        )

        result = scrub_daily_prices(prices, threshold=0.30, reversal_tolerance=0.10)

        assert isinstance(result, ScrubResult)
        report: ScrubReport = result.report
        assert len(report.events) == 1
        event = report.events.iloc[0]
        assert event["ticker"] == "SKT.NZ"
        assert event["date"] == prices.index[1]
        assert pytest.approx(event["original"], rel=1e-9) == 5.05
        # Geometric mean of 32.10 and 32.61 ≈ 32.355
        assert pytest.approx(event["replacement"], rel=1e-3) == np.sqrt(32.10 * 32.61)

    def test_replaces_with_geometric_mean_of_neighbours(self) -> None:
        prices = _make_prices(
            {"SKT.NZ": [32.10, 5.05, 32.61, 33.06]}
        )

        result = scrub_daily_prices(prices)

        cleaned_value = result.prices["SKT.NZ"].iloc[1]
        assert pytest.approx(cleaned_value, rel=1e-9) == np.sqrt(32.10 * 32.61)

    def test_does_not_flag_real_sustained_move(self) -> None:
        # Real +50% move that is *not* reversed the next day.
        prices = _make_prices(
            {"FOO.NZ": [10.0, 15.0, 15.5, 15.2, 15.6]}
        )

        result = scrub_daily_prices(prices)

        assert len(result.report.events) == 0
        # Series unchanged.
        pd.testing.assert_series_equal(
            result.prices["FOO.NZ"], prices["FOO.NZ"], check_names=False
        )

    def test_returns_input_unchanged_when_no_events(self) -> None:
        prices = _make_prices(
            {"FOO.NZ": [10.0, 10.1, 10.2, 10.15, 10.3]}
        )

        result = scrub_daily_prices(prices)

        pd.testing.assert_frame_equal(result.prices, prices)
        assert result.report.events.empty

    def test_threshold_controls_sensitivity(self) -> None:
        # 35% jump that round-trips. With threshold=0.40 it should NOT flag.
        prices = _make_prices(
            {"FOO.NZ": [10.0, 13.5, 10.0, 10.1]}
        )

        loose = scrub_daily_prices(prices, threshold=0.40, reversal_tolerance=0.10)
        tight = scrub_daily_prices(prices, threshold=0.30, reversal_tolerance=0.10)

        assert len(loose.report.events) == 0
        assert len(tight.report.events) == 1

    def test_reversal_tolerance_controls_round_trip_strictness(self) -> None:
        # Big jump, partial reversal that leaves a 15% net change.
        # 10 -> 15 (+50%), 15 -> 11.5 (-23.3%), combined ≈ +15%.
        prices = _make_prices(
            {"FOO.NZ": [10.0, 15.0, 11.5, 11.6]}
        )

        strict = scrub_daily_prices(prices, threshold=0.30, reversal_tolerance=0.10)
        loose = scrub_daily_prices(prices, threshold=0.30, reversal_tolerance=0.20)

        assert len(strict.report.events) == 0  # 15% net > 10% tolerance
        assert len(loose.report.events) == 1   # 15% net < 20% tolerance

    def test_handles_nan_neighbours_per_ticker(self) -> None:
        # Cross-ticker holiday creates a NaN row for FOO.NZ but not for BAR.NZ.
        # The scrubber must compute returns per-ticker after dropna so the
        # NaN row doesn't make pct_change spurious.
        idx = pd.DatetimeIndex(
            ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07", "2010-01-08"]
        )
        prices = pd.DataFrame(
            {
                "FOO.NZ": [32.10, np.nan, 5.05, 32.61, 33.06],
                "BAR.NZ": [10.00, 10.10, 10.20, 10.15, 10.30],
            },
            index=idx,
        )

        result = scrub_daily_prices(prices)

        # The 5.05 should be flagged using FOO.NZ's *own* prior valid print (32.10).
        events = result.report.events
        assert len(events) == 1
        assert events.iloc[0]["ticker"] == "FOO.NZ"
        assert events.iloc[0]["date"] == pd.Timestamp("2010-01-06")

    def test_does_not_touch_other_tickers(self) -> None:
        prices = _make_prices(
            {
                "BAD.NZ": [32.10, 5.05, 32.61, 33.06],
                "GOOD.NZ": [10.0, 10.1, 10.2, 10.15],
            }
        )

        result = scrub_daily_prices(prices)

        pd.testing.assert_series_equal(
            result.prices["GOOD.NZ"], prices["GOOD.NZ"], check_names=False
        )

    def test_idempotent(self) -> None:
        prices = _make_prices(
            {"SKT.NZ": [32.10, 5.05, 32.61, 33.06, 33.57]}
        )

        first = scrub_daily_prices(prices)
        second = scrub_daily_prices(first.prices)

        pd.testing.assert_frame_equal(first.prices, second.prices)
        assert second.report.events.empty

    def test_skips_first_and_last_observations(self) -> None:
        # An anomaly at the first or last day has no surrounding pair to
        # validate against; it must not be flagged.
        prices = _make_prices(
            {
                "EDGE.NZ": [5.05, 32.10, 32.20, 32.30, 5.05],
            }
        )

        result = scrub_daily_prices(prices)

        assert result.report.events.empty
