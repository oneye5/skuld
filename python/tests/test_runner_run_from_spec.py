"""Tests for run_from_spec."""
from __future__ import annotations

import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skuld_research.config import BacktestSpec, MomentumFactorSpec, ScrubbingSpec, run_from_spec
from skuld_research.config.spec import AdjustmentSpec


@pytest.fixture
def tiny_synthetic_panel_csv(tmp_path: Path) -> Path:
    """Create a tiny synthetic panel CSV for fast testing."""
    # 5 tickers, ~400 business days
    np.random.seed(12345)

    tickers = ["TK1.NZ", "TK2.NZ", "TK3.NZ", "TK4.NZ", "TK5.NZ"]
    dates = pd.bdate_range("2020-01-01", "2021-12-31", freq="B", tz="UTC")

    rows = []

    # Add price data
    for ticker in tickers:
        for date in dates:
            price = 100 + np.random.randn() * 10
            volume = int(10_000 + np.random.randn() * 1000)
            timestamp_ms = int(date.value // 1_000_000)  # nanoseconds to milliseconds

            rows.append({
                "timestamp": timestamp_ms,
                "ticker": ticker,
                "feature": "adj_close",
                "value": max(price, 1.0),
            })

            rows.append({
                "timestamp": timestamp_ms,
                "ticker": ticker,
                "feature": "volume",
                "value": max(volume, 1),
            })

    # Add fundamentals for market cap calculation
    for ticker in tickers:
        for date in pd.date_range("2020-01-01", "2021-12-31", freq="QE", tz="UTC"):
            shares = 1_000_000
            timestamp_ms = int(date.value // 1_000_000)

            rows.append({
                "timestamp": timestamp_ms,
                "ticker": ticker,
                "feature": "trailing_basic_average_shares",
                "value": shares,
            })

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def test_run_from_spec_byte_identical_on_reruns(tiny_synthetic_panel_csv: Path, tmp_path: Path):
    """Same spec produces byte-identical numeric tables across two runs (done-when a)."""
    from skuld_research.config import (
        GatingSpec,
        RollingDriverSpec,
        SurvivorshipSpec,
        WalkForwardSpec,
    )

    spec = BacktestSpec(
        name="test_determinism",
        asof=datetime.date(2022, 1, 1),
        master_seed=42,
        factors=[MomentumFactorSpec(min_months=11)],
        survivorship=SurvivorshipSpec(monte_carlo_seeds=10),
        gating=GatingSpec(
            bootstrap_n_resamples=50,
            dominance_n_resamples=50,
        ),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,  # not enough data for 2-fold
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    # Run twice
    result1 = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    result2 = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    # Check byte-equality of OOS returns
    assert result1.strategy_rolling.oos_returns.values.tobytes() == \
           result2.strategy_rolling.oos_returns.values.tobytes()

    # Check float fields
    assert result1.strategy_rolling.oos_sharpe_raw == result2.strategy_rolling.oos_sharpe_raw
    assert result1.strategy_rolling.oos_sharpe_delisting_adjusted == \
           result2.strategy_rolling.oos_sharpe_delisting_adjusted


def test_run_from_spec_ledger_deduplication(tiny_synthetic_panel_csv: Path, tmp_path: Path):
    """Ledger entry is deduplicated correctly on second run (done-when c)."""
    from skuld_research.config import (
        GatingSpec,
        RollingDriverSpec,
        SurvivorshipSpec,
        WalkForwardSpec,
    )

    ledger_root = tmp_path / "ledger"

    spec = BacktestSpec(
        name="test_ledger_dedup",
        asof=datetime.date(2022, 1, 1),
        master_seed=42,
        factors=[MomentumFactorSpec(min_months=11)],
        survivorship=SurvivorshipSpec(monte_carlo_seeds=10),
        gating=GatingSpec(
            bootstrap_n_resamples=50,
            dominance_n_resamples=50,
        ),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,  # not enough data for 2-fold
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    # Run twice with write_ledger=True
    result1 = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=True,
        ledger_root=ledger_root,
    )

    result2 = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=True,
        ledger_root=ledger_root,
    )

    # Check that both runs have the same spec_hash
    assert result1.spec_hash == result2.spec_hash

    # Load ledger and check for unique spec_hash
    from skuld_research.stats.ledger import TrialLedger

    ledger = TrialLedger(ledger_root, spec.output.ledger_scope)
    entries = ledger.all_entries()

    spec_hashes = [e["spec_hash"] for e in entries]
    unique_hashes = set(spec_hashes)

    # Should have exactly one unique hash (deduplication worked)
    assert len(unique_hashes) == 1
    assert result1.spec_hash in unique_hashes


def test_run_from_spec_passes_share_adv_to_nzx_benchmark(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-default NZX ADV floors should receive PIT-derived share ADV."""
    from skuld_research.config import BenchmarksSpec, RollingDriverSpec, WalkForwardSpec

    captured: dict[str, object] = {}

    def fake_nzx_benchmark(*args, **kwargs):
        captured["adv_floor_shares"] = kwargs["adv_floor_shares"]
        captured["share_adv"] = kwargs["share_adv"]
        index = pd.DatetimeIndex([pd.Timestamp("2021-12-31")])
        return SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        )

    monkeypatch.setattr(
        "skuld_research.config.runner.nzx_equal_weighted_fixed_universe",
        fake_nzx_benchmark,
    )

    spec = BacktestSpec(
        name="test_nzx_adv_floor",
        asof=datetime.date(2022, 1, 1),
        factors=[MomentumFactorSpec(min_months=11)],
        benchmarks=BenchmarksSpec(nzx_eq_adv_floor_shares=25_000),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    assert captured["adv_floor_shares"] == 25_000
    assert isinstance(captured["share_adv"], pd.DataFrame)
    assert not captured["share_adv"].empty


def test_run_from_spec_uses_gating_dominance_as_single_source(
    tiny_synthetic_panel_csv: Path,
) -> None:
    """Reported dominance must match the dominance bars used for gating."""
    from skuld_research.config import GatingSpec, RollingDriverSpec, WalkForwardSpec

    spec = BacktestSpec(
        name="test_single_source_dominance",
        asof=datetime.date(2022, 1, 1),
        factors=[MomentumFactorSpec(min_months=11)],
        gating=GatingSpec(
            bootstrap_n_resamples=30,
            dominance_n_resamples=50,
        ),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    result = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    assert result.gating.dominance is not None
    assert result.dominance is result.gating.dominance


def test_run_from_spec_passes_explicit_td_benchmark_name_into_gating(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner should identify the TD benchmark explicitly for gating."""
    from types import SimpleNamespace

    from skuld_research.config import RollingDriverSpec, WalkForwardSpec

    captured: dict[str, object] = {}
    index = pd.to_datetime(["2021-12-31"])

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self):
            return SimpleNamespace(
                oos_returns=pd.Series([0.0], index=index),
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    def fake_evaluate_gating(*args, **kwargs):
        captured["td_benchmark_name"] = kwargs.get("td_benchmark_name")
        captured["benchmarks"] = kwargs.get("benchmarks")
        return SimpleNamespace(dominance=object())

    monkeypatch.setattr(
        "skuld_research.config.runner.RollingWalkForwardEngine",
        DummyRollingWalkForwardEngine,
    )
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr("skuld_research.config.runner.evaluate_gating", fake_evaluate_gating)
    monkeypatch.setattr(
        "skuld_research.config.runner.nz_td_floor",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nzx_equal_weighted_fixed_universe",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.sixty_forty",
        lambda *args, **kwargs: (
            pd.Series([0.0], index=index),
            index[0],
            index[0],
            (),
        ),
    )

    spec = BacktestSpec(
        name="test_explicit_td_benchmark_name",
        asof=datetime.date(2022, 1, 1),
        factors=[MomentumFactorSpec(min_months=11)],
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    assert captured["td_benchmark_name"] == "NZ TD floor"
    assert "NZ TD floor" in captured["benchmarks"]


def test_run_from_spec_exposes_scrub_report(tiny_synthetic_panel_csv: Path) -> None:
    """Scrubbed runs should expose the raw-data scrub audit report."""
    from skuld_research.config import GatingSpec, RollingDriverSpec, WalkForwardSpec

    df = pd.read_csv(tiny_synthetic_panel_csv)
    price_mask = df["feature"] == "adj_close"
    for ticker, idx in df[price_mask].groupby("ticker", sort=False).groups.items():
        steps = np.arange(len(idx), dtype=float)
        drift = 0.0002 + 0.00005 * (int(ticker[2]) - 1)
        df.loc[idx, "value"] = 100.0 * (1.0 + drift) ** steps
    tk1_prices = df[(df["ticker"] == "TK1.NZ") & (df["feature"] == "adj_close")].index[:3]
    df.loc[tk1_prices, "value"] = [100.0, 50.0, 100.0]
    df.to_csv(tiny_synthetic_panel_csv, index=False)

    spec = BacktestSpec(
        name="test_scrub_report_exposed",
        asof=datetime.date(2022, 1, 1),
        factors=[MomentumFactorSpec(min_months=11)],
        scrubbing=ScrubbingSpec(kind="round_trip", threshold=0.30, reversal_tolerance=0.10),
        gating=GatingSpec(
            bootstrap_n_resamples=30,
            dominance_n_resamples=50,
        ),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    result = run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    assert len(result.scrub_report.events) == 1
    event = result.scrub_report.events.iloc[0]
    assert event["ticker"] == "TK1.NZ"
    assert event["original"] == 50.0
    assert event["replacement"] == 100.0


def test_run_from_spec_threads_adv_participation_cap_into_backtest_config(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec ADV participation cap should reach the backtest engine config."""
    from types import SimpleNamespace

    from skuld_research.config import (
        BacktestEngineSpec,
        RollingDriverSpec,
        WalkForwardSpec,
    )

    captured: dict[str, object] = {}
    index = pd.to_datetime(["2021-12-31"])

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            captured["backtest_config"] = kwargs["backtest_config"]

        def run(self):
            return SimpleNamespace(
                oos_returns=pd.Series([0.0], index=index),
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    monkeypatch.setattr(
        "skuld_research.config.runner.RollingWalkForwardEngine",
        DummyRollingWalkForwardEngine,
    )
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr(
        "skuld_research.config.runner.evaluate_gating",
        lambda *args, **kwargs: SimpleNamespace(dominance=object()),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nz_td_floor",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nzx_equal_weighted_fixed_universe",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.sixty_forty",
        lambda *args, **kwargs: (
            pd.Series([0.0], index=index),
            index[0],
            index[0],
            (),
        ),
    )

    spec = BacktestSpec(
        name="test_adv_participation_cap",
        asof=datetime.date(2022, 1, 1),
        factors=[MomentumFactorSpec(min_months=11)],
        backtest=BacktestEngineSpec(adv_participation_cap=0.03),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    assert captured["backtest_config"].adv_participation_cap == pytest.approx(0.03)


def test_run_from_spec_passes_anomaly_filter_into_prepared_panel(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner should thread anomaly_filter into prepared-panel construction."""
    from types import SimpleNamespace

    from skuld_research.config import RollingDriverSpec, WalkForwardSpec
    from skuld_research.config.spec import AnomalyFilterSpec

    captured: dict[str, object] = {}
    index = pd.to_datetime(["2021-12-31"])
    panel = SimpleNamespace(
        prices=pd.DataFrame({"TK1.NZ": [10.0]}, index=index),
        returns_daily=pd.DataFrame({"TK1.NZ": [0.0]}, index=index),
        returns_monthly=pd.DataFrame({"TK1.NZ": [0.0]}, index=index),
        market_cap=pd.DataFrame({"TK1.NZ": [1.0]}, index=index),
        universe_mask=pd.DataFrame(True, index=index, columns=["TK1.NZ"]),
        sector=pd.Series("Unknown", index=["TK1.NZ"]),
        macro=pd.DataFrame(index=index),
        asof=pd.Timestamp("2022-01-01"),
    )

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self):
            return SimpleNamespace(
                oos_returns=pd.Series([0.0], index=index),
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    monkeypatch.setattr(
        "skuld_research.config.runner.RollingWalkForwardEngine",
        DummyRollingWalkForwardEngine,
    )
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr(
        "skuld_research.config.runner.evaluate_gating",
        lambda *args, **kwargs: SimpleNamespace(dominance=object()),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nz_td_floor",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nzx_equal_weighted_fixed_universe",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.sixty_forty",
        lambda *args, **kwargs: (
            pd.Series([0.0], index=index),
            index[0],
            index[0],
            (),
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.load_raw_csv",
        lambda *args, **kwargs: SimpleNamespace(
            scrub_report=SimpleNamespace(events=pd.DataFrame())
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.PITLoader",
        lambda raw: SimpleNamespace(
            as_of=lambda asof: SimpleNamespace(
                prices=pd.DataFrame(index=pd.DatetimeIndex([])),
                volumes=pd.DataFrame(index=pd.DatetimeIndex([])),
                asof=asof,
            )
        ),
    )
    def fake_build_prepared_panel(*args, **kwargs):
        captured["anomaly_filter"] = kwargs.get("anomaly_filter")
        return panel

    monkeypatch.setattr(
        "skuld_research.config.runner.build_prepared_panel",
        fake_build_prepared_panel,
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.build_factors_from_specs",
        lambda specs: [],
    )

    anomaly_filter = AnomalyFilterSpec(kind="mask_extremes", daily_abs_return_threshold=1.5)
    spec = BacktestSpec(
        name="test_anomaly_filter_passthrough",
        asof=datetime.date(2022, 1, 1),
        factors=[],
        anomaly_filter=anomaly_filter,
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(spec, raw_csv_path=tiny_synthetic_panel_csv, write_ledger=False)

    assert captured["anomaly_filter"] == anomaly_filter


def test_run_from_spec_passes_adjustments_into_raw_loader(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner should thread adjustment audit/repair config into raw loading."""
    from types import SimpleNamespace

    from skuld_research.config import RollingDriverSpec, WalkForwardSpec

    captured: dict[str, object] = {}
    index = pd.to_datetime(["2021-12-31"])
    panel = SimpleNamespace(
        prices=pd.DataFrame({"TK1.NZ": [10.0]}, index=index),
        returns_daily=pd.DataFrame({"TK1.NZ": [0.0]}, index=index),
        returns_monthly=pd.DataFrame({"TK1.NZ": [0.0]}, index=index),
        market_cap=pd.DataFrame({"TK1.NZ": [1.0]}, index=index),
        universe_mask=pd.DataFrame(True, index=index, columns=["TK1.NZ"]),
        sector=pd.Series("Unknown", index=["TK1.NZ"]),
        macro=pd.DataFrame(index=index),
        asof=pd.Timestamp("2022-01-01"),
    )

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self):
            return SimpleNamespace(
                oos_returns=pd.Series([0.0], index=index),
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    def fake_load_raw_csv(*args, **kwargs):
        captured["adjustments"] = kwargs.get("adjustments")
        return SimpleNamespace(scrub_report=SimpleNamespace(events=pd.DataFrame()))

    monkeypatch.setattr("skuld_research.config.runner.RollingWalkForwardEngine", DummyRollingWalkForwardEngine)
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr("skuld_research.config.runner.evaluate_gating", lambda *args, **kwargs: SimpleNamespace(dominance=object()))
    monkeypatch.setattr(
        "skuld_research.config.runner.nz_td_floor",
        lambda *args, **kwargs: SimpleNamespace(returns=pd.Series([0.0], index=index), start=index[0], end=index[0]),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nzx_equal_weighted_fixed_universe",
        lambda *args, **kwargs: SimpleNamespace(returns=pd.Series([0.0], index=index), start=index[0], end=index[0]),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.sixty_forty",
        lambda *args, **kwargs: (pd.Series([0.0], index=index), index[0], index[0], ()),
    )
    monkeypatch.setattr("skuld_research.config.runner.load_raw_csv", fake_load_raw_csv)
    monkeypatch.setattr(
        "skuld_research.config.runner.PITLoader",
        lambda raw: SimpleNamespace(
            as_of=lambda asof: SimpleNamespace(
                prices=pd.DataFrame(index=pd.DatetimeIndex([])),
                volumes=pd.DataFrame(index=pd.DatetimeIndex([])),
                asof=asof,
            )
        ),
    )
    monkeypatch.setattr("skuld_research.config.runner.build_prepared_panel", lambda *args, **kwargs: panel)
    monkeypatch.setattr("skuld_research.config.runner.build_factors_from_specs", lambda specs: [])

    adjustments = AdjustmentSpec(kind="repair", policy="conservative")
    spec = BacktestSpec(
        name="test_adjustments_passthrough",
        asof=datetime.date(2022, 1, 1),
        factors=[],
        adjustments=adjustments,
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(spec, raw_csv_path=tiny_synthetic_panel_csv, write_ledger=False)

    assert captured["adjustments"] == adjustments


def test_run_from_spec_passes_adjustments_into_ohlc_loader_for_ar_spread(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AR spread estimation should use the same repair config as return data."""
    from types import SimpleNamespace

    from skuld_research.config import CostSpec, RollingDriverSpec, WalkForwardSpec

    captured: dict[str, object] = {}
    index = pd.to_datetime(["2021-12-31"])
    panel = SimpleNamespace(
        prices=pd.DataFrame({"TK1.NZ": [10.0]}, index=index),
        returns_daily=pd.DataFrame({"TK1.NZ": [0.0]}, index=index),
        returns_monthly=pd.DataFrame({"TK1.NZ": [0.0]}, index=index),
        market_cap=pd.DataFrame({"TK1.NZ": [1.0]}, index=index),
        universe_mask=pd.DataFrame(True, index=index, columns=["TK1.NZ"]),
        sector=pd.Series("Unknown", index=["TK1.NZ"]),
        macro=pd.DataFrame(index=index),
        asof=pd.Timestamp("2022-01-01"),
    )

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self):
            return SimpleNamespace(
                oos_returns=pd.Series([0.0], index=index),
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    def fake_load_raw_ohlc(*args, **kwargs):
        captured["adjustments"] = kwargs.get("adjustments")
        ohlc = pd.DataFrame({"TK1.NZ": [10.0]}, index=index)
        return ohlc, ohlc, ohlc

    monkeypatch.setattr("skuld_research.config.runner.RollingWalkForwardEngine", DummyRollingWalkForwardEngine)
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr("skuld_research.config.runner.evaluate_gating", lambda *args, **kwargs: SimpleNamespace(dominance=object()))
    monkeypatch.setattr("skuld_research.config.runner.compute_abdi_ranaldo_spread_panel", lambda *args, **kwargs: pd.DataFrame({"TK1.NZ": [5.0]}, index=index))
    monkeypatch.setattr("skuld_research.config.runner.load_raw_csv", lambda *args, **kwargs: SimpleNamespace(scrub_report=SimpleNamespace(events=pd.DataFrame())))
    monkeypatch.setattr("skuld_research.config.runner.load_raw_ohlc", fake_load_raw_ohlc)
    monkeypatch.setattr("skuld_research.config.runner.PITLoader", lambda raw: SimpleNamespace(as_of=lambda asof: SimpleNamespace(prices=pd.DataFrame(index=pd.DatetimeIndex([])), volumes=pd.DataFrame(index=pd.DatetimeIndex([])), asof=asof)))
    monkeypatch.setattr("skuld_research.config.runner.build_prepared_panel", lambda *args, **kwargs: panel)
    monkeypatch.setattr("skuld_research.config.runner.build_factors_from_specs", lambda specs: [])
    monkeypatch.setattr("skuld_research.config.runner.nz_td_floor", lambda *args, **kwargs: SimpleNamespace(returns=pd.Series([0.0], index=index), start=index[0], end=index[0]))
    monkeypatch.setattr("skuld_research.config.runner.nzx_equal_weighted_fixed_universe", lambda *args, **kwargs: SimpleNamespace(returns=pd.Series([0.0], index=index), start=index[0], end=index[0]))
    monkeypatch.setattr("skuld_research.config.runner.sixty_forty", lambda *args, **kwargs: (pd.Series([0.0], index=index), index[0], index[0], ()))

    adjustments = AdjustmentSpec(kind="repair", policy="conservative")
    spec = BacktestSpec(
        name="test_ohlc_adjustments_passthrough",
        asof=datetime.date(2022, 1, 1),
        factors=[],
        cost=CostSpec(spread_model="abdi_ranaldo"),
        adjustments=adjustments,
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(spec, raw_csv_path=tiny_synthetic_panel_csv, write_ledger=False)

    assert captured["adjustments"] == adjustments


def test_run_from_spec_uses_strictly_prior_spread_row_for_rebalance(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spread lookup should avoid same-day rows because AR uses eta[t+1]."""
    from types import SimpleNamespace

    from skuld_research.config import CostSpec, RollingDriverSpec, WalkForwardSpec

    captured: dict[str, object] = {}
    rebalance = pd.Timestamp("2021-12-31")
    panel = SimpleNamespace(
        prices=pd.DataFrame({"TK1.NZ": [10.0]}, index=[rebalance]),
        returns_daily=pd.DataFrame({"TK1.NZ": [0.0]}, index=[rebalance]),
        returns_monthly=pd.DataFrame({"TK1.NZ": [0.0]}, index=[rebalance]),
        market_cap=pd.DataFrame({"TK1.NZ": [1.0]}, index=[rebalance]),
        universe_mask=pd.DataFrame(True, index=[rebalance], columns=["TK1.NZ"]),
        sector=pd.Series("Unknown", index=["TK1.NZ"]),
        macro=pd.DataFrame(index=[rebalance]),
        asof=pd.Timestamp("2022-01-01"),
    )

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            spread_panel = kwargs.get("spread_panel")
            if spread_panel is not None:
                captured["spread_index"] = spread_panel.index

        def run(self):
            return SimpleNamespace(
                oos_returns=pd.Series([0.0], index=[rebalance]),
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    spread_panel = pd.DataFrame(
        {"TK1.NZ": [20.0, 99.0]},
        index=[pd.Timestamp("2021-12-24"), rebalance],
    )

    monkeypatch.setattr("skuld_research.config.runner.RollingWalkForwardEngine", DummyRollingWalkForwardEngine)
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr("skuld_research.config.runner.evaluate_gating", lambda *args, **kwargs: SimpleNamespace(dominance=object()))
    monkeypatch.setattr("skuld_research.config.runner.compute_abdi_ranaldo_spread_panel", lambda *args, **kwargs: spread_panel)
    monkeypatch.setattr("skuld_research.config.runner.load_raw_csv", lambda *args, **kwargs: SimpleNamespace(scrub_report=SimpleNamespace(events=pd.DataFrame())))
    monkeypatch.setattr("skuld_research.config.runner.load_raw_ohlc", lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr("skuld_research.config.runner.PITLoader", lambda raw: SimpleNamespace(as_of=lambda asof: SimpleNamespace(prices=pd.DataFrame(index=pd.DatetimeIndex([])), volumes=pd.DataFrame(index=pd.DatetimeIndex([])), asof=asof)))
    monkeypatch.setattr("skuld_research.config.runner.build_prepared_panel", lambda *args, **kwargs: panel)
    monkeypatch.setattr("skuld_research.config.runner.build_factors_from_specs", lambda specs: [])
    monkeypatch.setattr("skuld_research.config.runner.nz_td_floor", lambda *args, **kwargs: SimpleNamespace(returns=pd.Series([0.0], index=[rebalance]), start=rebalance, end=rebalance))
    monkeypatch.setattr("skuld_research.config.runner.nzx_equal_weighted_fixed_universe", lambda *args, **kwargs: SimpleNamespace(returns=pd.Series([0.0], index=[rebalance]), start=rebalance, end=rebalance))
    monkeypatch.setattr("skuld_research.config.runner.sixty_forty", lambda *args, **kwargs: (pd.Series([0.0], index=[rebalance]), rebalance, rebalance, ()))

    spec = BacktestSpec(
        name="test_prior_spread_lookup",
        asof=datetime.date(2022, 1, 1),
        factors=[],
        cost=CostSpec(spread_model="abdi_ranaldo"),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(spec, raw_csv_path=tiny_synthetic_panel_csv, write_ledger=False)

    assert list(captured["spread_index"]) == [rebalance]


def test_run_from_spec_passes_sixty_forty_duration_years_into_benchmark(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner should thread configured 60/40 bond duration into the benchmark."""
    from types import SimpleNamespace

    from skuld_research.config import BenchmarksSpec, RollingDriverSpec, WalkForwardSpec

    captured: dict[str, object] = {}
    index = pd.to_datetime(["2021-12-31"])

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self):
            return SimpleNamespace(
                oos_returns=pd.Series([0.0], index=index),
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    monkeypatch.setattr(
        "skuld_research.config.runner.RollingWalkForwardEngine",
        DummyRollingWalkForwardEngine,
    )
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr(
        "skuld_research.config.runner.evaluate_gating",
        lambda *args, **kwargs: SimpleNamespace(dominance=object()),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nz_td_floor",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nzx_equal_weighted_fixed_universe",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )

    def fake_sixty_forty(*args, **kwargs):
        captured["duration_years"] = kwargs.get("duration_years")
        return (
            pd.Series([0.0], index=index),
            index[0],
            index[0],
            (),
        )

    monkeypatch.setattr("skuld_research.config.runner.sixty_forty", fake_sixty_forty)

    spec = BacktestSpec(
        name="test_sixty_forty_duration_passthrough",
        asof=datetime.date(2022, 1, 1),
        factors=[MomentumFactorSpec(min_months=11)],
        benchmarks=BenchmarksSpec(sixty_forty_bond_duration_years=6.5),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    assert captured["duration_years"] == pytest.approx(6.5)


def test_run_from_spec_excludes_empty_sixty_forty_from_gating_benchmarks(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty 60/40 OOS returns should not collapse dominance alignment for other benchmarks."""
    from types import SimpleNamespace

    from skuld_research.config import RollingDriverSpec, WalkForwardSpec

    captured: dict[str, object] = {}
    index = pd.to_datetime(["2021-12-31"])

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            precomputed = kwargs.get("precomputed_returns")
            self._returns = (
                precomputed
                if precomputed is not None
                else pd.Series([0.0], index=index)
            )

        def run(self):
            return SimpleNamespace(
                oos_returns=self._returns,
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    def fake_evaluate_gating(*args, **kwargs):
        captured["benchmarks"] = kwargs.get("benchmarks")
        captured["td_benchmark_name"] = kwargs.get("td_benchmark_name")
        return SimpleNamespace(dominance=object())

    monkeypatch.setattr(
        "skuld_research.config.runner.RollingWalkForwardEngine",
        DummyRollingWalkForwardEngine,
    )
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr("skuld_research.config.runner.evaluate_gating", fake_evaluate_gating)
    monkeypatch.setattr(
        "skuld_research.config.runner.nz_td_floor",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nzx_equal_weighted_fixed_universe",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.sixty_forty",
        lambda *args, **kwargs: (
            pd.Series([], dtype=float),
            pd.Timestamp("1970-01-01"),
            pd.Timestamp("1970-01-01"),
            ("No overlapping months with both equity and bond data",),
        ),
    )

    spec = BacktestSpec(
        name="test_empty_sixty_forty_filtered_from_gating",
        asof=datetime.date(2022, 1, 1),
        factors=[MomentumFactorSpec(min_months=11)],
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    assert captured["td_benchmark_name"] == "NZ TD floor"
    assert set(captured["benchmarks"]) == {"NZ TD floor", "NZX equal-weighted"}


def test_run_from_spec_adv_panel_normalizes_intraday_raw_data_before_rolling(
    tiny_synthetic_panel_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADV panel should normalize intraday raw prices and volumes onto daily dates."""
    from types import SimpleNamespace

    from skuld_research.config import (
        BacktestEngineSpec,
        RollingDriverSpec,
        UniverseSpec,
        WalkForwardSpec,
    )

    captured: dict[str, object] = {}
    index = pd.to_datetime(["2021-12-31"])

    class DummyRollingWalkForwardEngine:
        def __init__(self, *args, **kwargs) -> None:
            captured["backtest_config"] = kwargs["backtest_config"]

        def run(self):
            return SimpleNamespace(
                oos_returns=pd.Series([0.0], index=index),
                oos_sharpe_raw=0.0,
                oos_sharpe_delisting_adjusted=0.0,
                n_kept_folds=1,
                n_rejected_folds=0,
            )

    class DummyTrialLedger:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contains(self, spec_hash: str) -> bool:
            return True

    def fake_load_raw_csv(*args, **kwargs):
        return SimpleNamespace(scrub_report=SimpleNamespace(events=pd.DataFrame()))

    snap_dates = pd.to_datetime([
        "2021-01-07 10:00:00",
        "2021-01-08 10:00:00",
        "2021-01-11 10:00:00",
    ])
    snap = SimpleNamespace(
        prices=pd.DataFrame({"TK1.NZ": [10.0, 20.0, 30.0]}, index=snap_dates),
        volumes=pd.DataFrame({"TK1.NZ": [1.0, 1.0, 1.0]}, index=snap_dates),
        asof=pd.Timestamp("2021-01-12", tz="UTC"),
    )
    panel = SimpleNamespace(
        prices=pd.DataFrame({"TK1.NZ": [10.0, 20.0, 30.0]}, index=snap_dates),
        returns_daily=pd.DataFrame({"TK1.NZ": [0.0, 0.0, 0.0]}, index=snap_dates),
        returns_monthly=pd.DataFrame({"TK1.NZ": [0.0]}, index=index),
        market_cap=pd.DataFrame({"TK1.NZ": [1.0, 1.0, 1.0]}, index=snap_dates),
        universe_mask=pd.DataFrame(True, index=index, columns=["TK1.NZ"]),
        sector=pd.Series("Unknown", index=["TK1.NZ"]),
        macro=pd.DataFrame(index=snap_dates),
        asof=pd.Timestamp("2021-01-12"),
    )

    monkeypatch.setattr(
        "skuld_research.config.runner.RollingWalkForwardEngine",
        DummyRollingWalkForwardEngine,
    )
    monkeypatch.setattr("skuld_research.config.runner.TrialLedger", DummyTrialLedger)
    monkeypatch.setattr(
        "skuld_research.config.runner.evaluate_gating",
        lambda *args, **kwargs: SimpleNamespace(dominance=object()),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nz_td_floor",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.nzx_equal_weighted_fixed_universe",
        lambda *args, **kwargs: SimpleNamespace(
            returns=pd.Series([0.0], index=index),
            start=index[0],
            end=index[0],
        ),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.sixty_forty",
        lambda *args, **kwargs: (
            pd.Series([0.0], index=index),
            index[0],
            index[0],
            (),
        ),
    )
    monkeypatch.setattr("skuld_research.config.runner.load_raw_csv", fake_load_raw_csv)
    monkeypatch.setattr(
        "skuld_research.config.runner.PITLoader",
        lambda raw: SimpleNamespace(as_of=lambda asof: snap),
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.build_prepared_panel",
        lambda *args, **kwargs: panel,
    )
    monkeypatch.setattr(
        "skuld_research.config.runner.build_factors_from_specs",
        lambda specs: [],
    )

    spec = BacktestSpec(
        name="test_adv_panel_daily_normalization",
        asof=datetime.date(2021, 1, 12),
        factors=[],
        universe=UniverseSpec(adv_window=2),
        backtest=BacktestEngineSpec(adv_participation_cap=0.03),
        walk_forward=WalkForwardSpec(
            two_fold_enabled=False,
            rolling=RollingDriverSpec(train_years=1, oos_years=1, step_years=1),
        ),
    )

    run_from_spec(
        spec,
        raw_csv_path=tiny_synthetic_panel_csv,
        write_ledger=False,
    )

    adv_panel = captured["backtest_config"].adv_panel
    assert adv_panel.loc[pd.Timestamp("2021-01-11"), "TK1.NZ"] == pytest.approx(30.0)
    assert not pd.isna(adv_panel.loc[pd.Timestamp("2021-01-11"), "TK1.NZ"])
