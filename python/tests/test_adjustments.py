"""Tests for the corporate-action adjustment audit/repair layer.

See `docs/specs/2026-04-30-corporate-action-adjustments.md`.

The audit detects discrepancies between Yahoo's `adj_close` and a
corporate-action ledger; repair optionally back-scales the price chain for
high-confidence detections (missed split, unit jump) and, under
``RepairPolicy.AGGRESSIVE`` with a ``raw_close`` panel supplied, re-derives
the entire dividend back-adjustment chain.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from skuld_research.data.adjustments import (
    AdjustmentAuditReport,
    RepairPolicy,
    RepairResult,
    audit_adjustments,
    repair_adjustments,
)


def _at(df: pd.DataFrame, i: int) -> pd.Timestamp:
    """Narrow ``df.index[i]`` to ``pd.Timestamp`` for static type checkers."""
    return cast(pd.Timestamp, cast(pd.DatetimeIndex, df.index)[i])

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_panel(rows: dict[str, list[float]], start: str = "2010-01-04") -> pd.DataFrame:
    """Wide date-by-ticker panel with a business-day index starting `start`."""
    n = max(len(v) for v in rows.values())
    idx = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame(rows, index=idx).astype(float)


def _make_actions(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Long-form corp-action frame: (ticker, ex_date, type, factor)."""
    df = pd.DataFrame(rows, columns=["ticker", "ex_date", "type", "factor"])
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    df["factor"] = df["factor"].astype(float)
    return df


def _empty_actions() -> pd.DataFrame:
    return _make_actions([])


# ---------------------------------------------------------------------------
# Audit-only behaviour
# ---------------------------------------------------------------------------


class TestAuditNoEvents:
    def test_audit_no_actions_no_events(self) -> None:
        prices = _make_panel({"FOO.NZ": [10.0, 10.1, 10.2, 10.3, 10.4]})

        report = audit_adjustments(prices, _empty_actions())

        assert isinstance(report, AdjustmentAuditReport)
        assert report.events.empty

    def test_audit_clean_panel_no_events(self) -> None:
        # adj_close ratio on split day equals 1/factor=0.5 (clean per spec model).
        prices = _make_panel({"FOO.NZ": [10.0, 10.0, 5.0, 5.0, 5.0]})
        # raw close consistent with the dividend-drop model: drops by exactly D.
        # Day 5 has a $0.50 dividend; expected ratio = (raw[5]-0.5)/raw[4] = 4.5/5 = 0.9.
        # Make adj match: adj[5]/adj[4] = 4.5/5.0 = 0.9.
        prices_div = _make_panel({"FOO.NZ": [10.0, 10.0, 5.0, 5.0, 4.5]})
        raw = _make_panel({"FOO.NZ": [10.0, 10.0, 5.0, 5.0, 5.0]})
        actions = _make_actions(
            [
                ("FOO.NZ", str(_at(prices, 2).date()), "split", 2.0),
                ("FOO.NZ", str(_at(prices, 4).date()), "dividend", 0.5),
            ]
        )

        # Use the dividend-bearing adj panel so both events have matching ratios.
        report = audit_adjustments(prices_div, actions, raw_close=raw)

        # All events should be either absent or matched within tolerances.
        # Spec convention: clean panel ⇒ no events.
        assert report.events.empty


# ---------------------------------------------------------------------------
# Detection categories
# ---------------------------------------------------------------------------


class TestDetectMissedSplit:
    def test_detect_missed_split_2for1(self) -> None:
        # adj_close drops 50% on day 3 with NO split row — Yahoo missed it.
        prices = _make_panel({"FOO.NZ": [10.0, 10.0, 5.0, 5.0, 5.0]})

        report = audit_adjustments(prices, _empty_actions())

        events = report.events
        assert len(events) == 1
        ev = events.iloc[0]
        assert ev["ticker"] == "FOO.NZ"
        assert ev["kind"] == "missed_split"
        assert ev["severity"] == "error"
        assert ev["ex_date"] == prices.index[2]
        assert pytest.approx(ev["observed_ratio"], rel=1e-9) == 0.5

    def test_detect_missed_reverse_split_1for10(self) -> None:
        # adj_close jumps 10x on day 3 — missed reverse split.
        prices = _make_panel({"FOO.NZ": [1.0, 1.0, 10.0, 10.0, 10.0]})

        report = audit_adjustments(prices, _empty_actions())

        kinds = report.events["kind"].tolist()
        assert "missed_split" in kinds

    def test_missed_split_suppressed_when_split_row_present(self) -> None:
        prices = _make_panel({"FOO.NZ": [10.0, 10.0, 5.0, 5.0, 5.0]})
        actions = _make_actions(
            [("FOO.NZ", str(_at(prices, 2).date()), "split", 2.0)]
        )

        report = audit_adjustments(prices, actions)

        # No missed_split should fire because a split row exists at the ex-date.
        kinds = report.events["kind"].tolist()
        assert "missed_split" not in kinds


class TestDetectUnitJump:
    def test_detect_unit_jump_100x(self) -> None:
        prices = _make_panel({"FOO.NZ": [1.0, 1.0, 100.0, 100.0, 100.0]})

        report = audit_adjustments(prices, _empty_actions())

        events = report.events
        # Could be detected as unit_jump (100x) — both candidate sets contain it.
        # missed_split_ratios doesn't include 100, only unit_jump_ratios does.
        assert (events["kind"] == "unit_jump").any()
        ev = events[events["kind"] == "unit_jump"].iloc[0]
        assert ev["severity"] == "error"
        assert pytest.approx(ev["observed_ratio"], rel=1e-9) == 100.0

    def test_detect_unit_jump_001(self) -> None:
        prices = _make_panel({"FOO.NZ": [100.0, 100.0, 1.0, 1.0, 1.0]})

        report = audit_adjustments(prices, _empty_actions())

        assert (report.events["kind"] == "unit_jump").any()


class TestDetectBadDivAdjust:
    def test_detect_bad_div_adjust_when_raw_close_provided(self) -> None:
        # Dividend of 10 on day 5; raw flat at 100. Expected ratio = 90/100=0.9.
        # Adj ratio is 0.5 (huge mismatch) -> flag as error.
        adj = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0, 100.0, 50.0]})
        raw = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0, 100.0, 100.0]})
        actions = _make_actions(
            [("FOO.NZ", str(_at(adj, 4).date()), "dividend", 10.0)]
        )

        report = audit_adjustments(adj, actions, raw_close=raw)

        ev = report.events
        bad = ev[ev["kind"] == "bad_div_adjust"]
        assert len(bad) == 1
        row = bad.iloc[0]
        assert row["severity"] == "error"
        assert pytest.approx(row["expected_ratio"], rel=1e-9) == 0.9
        assert pytest.approx(row["observed_ratio"], rel=1e-9) == 0.5

    def test_skip_bad_div_adjust_without_raw_close(self) -> None:
        adj = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0, 100.0, 50.0]})
        actions = _make_actions(
            [("FOO.NZ", str(_at(adj, 4).date()), "dividend", 10.0)]
        )

        report = audit_adjustments(adj, actions, raw_close=None)

        ev = report.events
        bad = ev[ev["kind"] == "bad_div_adjust"]
        assert len(bad) == 1
        row = bad.iloc[0]
        assert row["severity"] == "skipped_no_raw"
        assert np.isnan(row["expected_ratio"])
        assert np.isnan(row["residual"])

    def test_skip_bad_div_adjust_when_ticker_missing_from_raw(self) -> None:
        adj = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0, 100.0, 50.0]})
        raw = _make_panel({"BAR.NZ": [100.0, 100.0, 100.0, 100.0, 100.0]})
        actions = _make_actions(
            [("FOO.NZ", str(_at(adj, 4).date()), "dividend", 10.0)]
        )

        report = audit_adjustments(adj, actions, raw_close=raw)

        bad = report.events[report.events["kind"] == "bad_div_adjust"]
        assert len(bad) == 1
        assert bad.iloc[0]["severity"] == "skipped_no_raw"

    def test_bad_div_adjust_within_tolerance_no_flag(self) -> None:
        # adj ratio 0.95 vs expected 0.9 -> residual=|0.95/0.9 -1|≈0.055 < 0.25.
        adj = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0, 100.0, 95.0]})
        raw = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0, 100.0, 100.0]})
        actions = _make_actions(
            [("FOO.NZ", str(_at(adj, 4).date()), "dividend", 10.0)]
        )

        report = audit_adjustments(adj, actions, raw_close=raw)

        bad = report.events[report.events["kind"] == "bad_div_adjust"]
        assert bad.empty


class TestDetectSplitMismatch:
    def test_detect_split_mismatch(self) -> None:
        # Split row exists with factor 2.0 (expected ratio = 1/2 = 0.5),
        # but observed adj ratio is 1.0 (Yahoo did NOT halve history).
        prices = _make_panel({"FOO.NZ": [10.0, 10.0, 10.0, 10.0, 10.0]})
        actions = _make_actions(
            [("FOO.NZ", str(_at(prices, 2).date()), "split", 2.0)]
        )

        report = audit_adjustments(prices, actions)

        ev = report.events
        sm = ev[ev["kind"] == "split_mismatch"]
        assert len(sm) == 1
        row = sm.iloc[0]
        assert row["severity"] == "warn"
        assert pytest.approx(row["expected_ratio"], rel=1e-9) == 0.5
        assert pytest.approx(row["observed_ratio"], rel=1e-9) == 1.0


class TestDetectOrphanAction:
    def test_detect_orphan_action_after_last_price(self) -> None:
        prices = _make_panel({"FOO.NZ": [10.0, 10.1, 10.2, 10.3]})
        ex_date = (_at(prices, -1) + pd.Timedelta(days=30)).date()
        actions = _make_actions([("FOO.NZ", str(ex_date), "dividend", 0.5)])

        report = audit_adjustments(prices, actions)

        ev = report.events
        orphan = ev[ev["kind"] == "orphan_action"]
        assert len(orphan) == 1
        assert orphan.iloc[0]["severity"] == "info"

    def test_detect_orphan_action_before_first_price(self) -> None:
        prices = _make_panel({"FOO.NZ": [10.0, 10.1, 10.2, 10.3]})
        ex_date = (_at(prices, 0) - pd.Timedelta(days=30)).date()
        actions = _make_actions([("FOO.NZ", str(ex_date), "split", 2.0)])

        report = audit_adjustments(prices, actions)

        orphan = report.events[report.events["kind"] == "orphan_action"]
        assert len(orphan) == 1


class TestDetectDuplicateAction:
    def test_detect_duplicate_action_same_day(self) -> None:
        prices = _make_panel({"FOO.NZ": [10.0, 10.0, 9.5, 9.5, 9.5]})
        d = str(_at(prices, 2).date())
        actions = _make_actions(
            [
                ("FOO.NZ", d, "dividend", 0.5),
                ("FOO.NZ", d, "dividend", 0.5),
            ]
        )

        report = audit_adjustments(prices, actions)

        dup = report.events[report.events["kind"] == "duplicate_action"]
        assert len(dup) == 1
        assert dup.iloc[0]["severity"] == "warn"


# ---------------------------------------------------------------------------
# Repair behaviour
# ---------------------------------------------------------------------------


class TestRepairOff:
    def test_repair_off_is_identity(self) -> None:
        prices = _make_panel({"FOO.NZ": [10.0, 10.0, 5.0, 5.0, 5.0]})

        result = repair_adjustments(
            prices, _empty_actions(), policy=RepairPolicy.OFF
        )

        assert isinstance(result, RepairResult)
        pd.testing.assert_frame_equal(result.prices, prices)
        assert result.repairs.empty
        # Audit still ran.
        assert not result.report.events.empty


class TestRepairConservative:
    def test_repair_conservative_fixes_missed_split(self) -> None:
        # 2-for-1 missed split on day 3: pre-split prices halved post-repair.
        prices = _make_panel({"FOO.NZ": [10.0, 10.0, 5.0, 5.0, 5.0]})

        result = repair_adjustments(
            prices, _empty_actions(), policy=RepairPolicy.CONSERVATIVE
        )

        repaired = result.prices["FOO.NZ"]
        # Pre-ex (indices 0,1) divided by 2; post-ex (2,3,4) untouched.
        assert pytest.approx(repaired.iloc[0], rel=1e-9) == 5.0
        assert pytest.approx(repaired.iloc[1], rel=1e-9) == 5.0
        assert pytest.approx(repaired.iloc[2], rel=1e-9) == 5.0
        assert pytest.approx(repaired.iloc[3], rel=1e-9) == 5.0
        # Repairs ledger contains one row for the missed split.
        assert len(result.repairs) == 1
        rep = result.repairs.iloc[0]
        assert rep["ticker"] == "FOO.NZ"
        assert rep["kind"] == "missed_split"
        assert rep["action"] == "back_scale"

    def test_repair_conservative_fixes_unit_jump_100x(self) -> None:
        prices = _make_panel({"FOO.NZ": [1.0, 1.0, 100.0, 100.0, 100.0]})

        result = repair_adjustments(
            prices, _empty_actions(), policy=RepairPolicy.CONSERVATIVE
        )

        repaired = result.prices["FOO.NZ"]
        # Pre-jump prices multiplied by 100 to align with post-jump scale.
        assert pytest.approx(repaired.iloc[0], rel=1e-9) == 100.0
        assert pytest.approx(repaired.iloc[1], rel=1e-9) == 100.0
        assert pytest.approx(repaired.iloc[2], rel=1e-9) == 100.0
        assert any(result.repairs["kind"] == "unit_jump")

    def test_repair_conservative_skips_bad_div_adjust(self) -> None:
        adj = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0, 100.0, 50.0]})
        raw = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0, 100.0, 100.0]})
        actions = _make_actions(
            [("FOO.NZ", str(_at(adj, 4).date()), "dividend", 10.0)]
        )

        result = repair_adjustments(
            adj, actions, raw_close=raw, policy=RepairPolicy.CONSERVATIVE
        )

        # bad_div_adjust appears in audit but NOT in repairs ledger.
        assert (result.report.events["kind"] == "bad_div_adjust").any()
        assert not (result.repairs["kind"] == "bad_div_adjust").any()


class TestRepairAggressive:
    def test_repair_aggressive_rederives_dividend_chain(self) -> None:
        # Hand-computed CRSP backward chain.
        # raw=[100,100,100], div=1.0 on day 3, no splits.
        # Day3: factor=1.0 -> adj[3] = 100.
        # Day2: dividend on day 3 with D=1, P_prev=raw[2]=100.
        #       factor *= (1 - 1/100) = 0.99 -> adj[2] = 100*0.99 = 99.
        # Day1: no action between day2 and day1 -> factor=0.99, adj[1]=99.
        raw = _make_panel({"FOO.NZ": [100.0, 100.0, 100.0]})
        # Start with a clearly broken adj panel.
        adj = _make_panel({"FOO.NZ": [100.0, 100.0, 50.0]})
        actions = _make_actions(
            [("FOO.NZ", str(_at(adj, 2).date()), "dividend", 1.0)]
        )

        result = repair_adjustments(
            adj, actions, raw_close=raw, policy=RepairPolicy.AGGRESSIVE
        )

        rep = result.prices["FOO.NZ"]
        assert pytest.approx(rep.iloc[2], rel=1e-9) == 100.0
        assert pytest.approx(rep.iloc[1], rel=1e-9) == 99.0
        assert pytest.approx(rep.iloc[0], rel=1e-9) == 99.0
        assert (result.repairs["kind"] == "bad_div_adjust").any()

    def test_repair_aggressive_with_split_and_dividend(self) -> None:
        # 4 days. Split factor 2 on day 3. Dividend 1.0 on day 4.
        # raw = [50, 50, 100, 100].
        # Backward chain:
        #   Day4: factor=1, adj[4]=100.
        #   Day3 -> Day2: dividend on day4 with D=1, P_prev=raw[3]=100.
        #            factor=1*(1-1/100)=0.99 (applied for day3 onward going back)
        #            ... actually dividend on day 4 affects day 3 and earlier.
        # Restart cleanly: walk last-to-first, on visiting day t check actions
        # AT day t and modify factor for days t-1 and earlier:
        #   - dividend on day t with D, P_prev=raw[t-1] -> factor *= (1-D/raw[t-1])
        #   - split on day t with F -> factor /= F
        # Visit day 4: actions: dividend(D=1). factor stays 1 for day4.
        #   adj[4]=raw[4]*1=100. Then update factor for prior days:
        #   factor *= (1-1/raw[3]) = (1-1/100) = 0.99.
        # Visit day 3: actions: split(F=2). factor stays 0.99 for day3.
        #   adj[3]=raw[3]*0.99=99. Then update for prior: factor /= 2 -> 0.495.
        # Visit day 2: no actions. adj[2]=raw[2]*0.495=50*0.495=24.75.
        # Visit day 1: no actions. adj[1]=raw[1]*0.495=24.75.
        raw = _make_panel({"FOO.NZ": [50.0, 50.0, 100.0, 100.0]})
        adj = _make_panel({"FOO.NZ": [50.0, 50.0, 100.0, 50.0]})  # broken
        actions = _make_actions(
            [
                ("FOO.NZ", str(_at(adj, 2).date()), "split", 2.0),
                ("FOO.NZ", str(_at(adj, 3).date()), "dividend", 1.0),
            ]
        )

        result = repair_adjustments(
            adj, actions, raw_close=raw, policy=RepairPolicy.AGGRESSIVE
        )

        rep = result.prices["FOO.NZ"]
        assert pytest.approx(rep.iloc[3], rel=1e-9) == 100.0
        assert pytest.approx(rep.iloc[2], rel=1e-9) == 99.0
        assert pytest.approx(rep.iloc[1], rel=1e-9) == 24.75
        assert pytest.approx(rep.iloc[0], rel=1e-9) == 24.75


class TestRepairIdempotent:
    def test_repair_idempotent_missed_split(self) -> None:
        prices = _make_panel({"FOO.NZ": [10.0, 10.0, 5.0, 5.0, 5.0]})

        first = repair_adjustments(
            prices, _empty_actions(), policy=RepairPolicy.CONSERVATIVE
        )
        second = repair_adjustments(
            first.prices, _empty_actions(), policy=RepairPolicy.CONSERVATIVE
        )

        pd.testing.assert_frame_equal(first.prices, second.prices)
        assert second.repairs.empty
        # No error-severity events left after repair.
        assert not (second.report.events["severity"] == "error").any()


# ---------------------------------------------------------------------------
# Orthogonality / regression
# ---------------------------------------------------------------------------


class TestOrthogonality:
    def test_skt_2010_corruption_not_falsely_flagged_as_split(self) -> None:
        # Single-day round-trip print (the scrubber's domain).
        # 32.10 -> 5.05 -> 32.61. Ratio ≈ 0.157. NOT in missed_split_ratios
        # (0.5, 2, 3, 4, 5, 10, 0.1, 0.2, 0.25, 1/3) within 5% tol.
        prices = _make_panel(
            {"SKT.NZ": [32.10, 5.05, 32.61, 33.06, 33.57]}
        )

        report = audit_adjustments(prices, _empty_actions())

        kinds = report.events["kind"].tolist()
        assert "missed_split" not in kinds
        # And not a unit_jump either (0.157 ≠ 0.01 within 2% tol).
        assert "unit_jump" not in kinds

    def test_does_not_touch_other_tickers(self) -> None:
        prices = _make_panel(
            {
                "BAD.NZ": [10.0, 10.0, 5.0, 5.0, 5.0],
                "GOOD.NZ": [20.0, 20.1, 20.2, 20.15, 20.3],
            }
        )

        result = repair_adjustments(
            prices, _empty_actions(), policy=RepairPolicy.CONSERVATIVE
        )

        pd.testing.assert_series_equal(
            result.prices["GOOD.NZ"], prices["GOOD.NZ"], check_names=False
        )


# ---------------------------------------------------------------------------
# Decoupling guarantee (spec §4.5)
# ---------------------------------------------------------------------------


class TestDecoupling:
    def test_module_does_not_import_pipeline_contracts(self) -> None:
        import skuld_research.data.adjustments as mod

        src = mod.__file__
        assert src is not None
        with open(src, encoding="utf-8") as fh:
            text = fh.read()

        forbidden = [
            "from skuld_research.data.csv_loader",
            "from skuld_research.data.pit_loader",
            "from skuld_research.data.prepared_panel",
            "from skuld_common",
            "import skuld_common",
        ]
        for needle in forbidden:
            assert needle not in text, f"forbidden import found: {needle}"
