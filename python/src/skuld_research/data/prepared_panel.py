"""Stage 2: build a PreparedPanel from a PITSnapshot.

Takes a point-in-time snapshot and produces cleaned, aligned series ready
for factor computation: total-return daily/monthly, market cap, sector,
and per-rebalance-date universe masks driven by liquidity + history filters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skuld_common.contracts import PITSnapshot, PreparedPanel
from skuld_research.config.spec import AnomalyFilterSpec

# Preferred order for share-count fundamentals (most recent / most accurate first).
_SHARE_FIELDS_PREFERENCE = (
    "trailing_basic_average_shares",
    "trailing_diluted_average_shares",
    "annual_basic_average_shares",
    "annual_diluted_average_shares",
)


def build_prepared_panel(
    snap: PITSnapshot,
    *,
    rebalance_dates: pd.DatetimeIndex | None = None,
    rebalance_start: pd.Timestamp | str | None = None,
    min_adv_dollars: float = 10_000.0,
    min_market_cap_nzd: float = 0.0,
    min_history_days: int = 126,
    adv_window: int = 20,
    mc_ffill_days: int = 5,
    nzx_only: bool = True,
    rebalance_freq: str = "BME",
    anomaly_filter: AnomalyFilterSpec | None = None,
) -> PreparedPanel:
    """Build a PreparedPanel from a PITSnapshot.

    Args:
        snap: Point-in-time snapshot (PIT contract enforced upstream).
        rebalance_dates: Dates at which to evaluate the universe mask. Defaults
            to all business month-ends strictly before `snap.asof`.
        min_adv_dollars: Liquidity floor (trailing-window mean of non-zero-volume
            $-ADV).
        min_market_cap_nzd: Size floor. **Opt-in.** Default 0.0 disables the check
            entirely; tickers with NaN market cap (no shares-outstanding data) are
            not filtered out. Set > 0 to apply a hard size floor — note this will
            silently exclude any ticker without fundamentals coverage.
        min_history_days: Minimum non-NaN return observations before a ticker is
            eligible for the universe.
        adv_window: Lookback (trading days) for the ADV calculation.
        rebalance_start: Earliest rebalance date to generate. Defaults to the first
            date when at least 10 tickers have non-NaN prices, which avoids the
            sparse pre-equity rows that appear when macro/international series
            extend the price index back to the 1970s.
        mc_ffill_days: Maximum gap (trading days) to forward-fill market cap before
            applying the size filter. Handles missing prints near rebalance dates
            for thinly-traded names. Default 5 corresponds to one calendar week.
        nzx_only: If True (default), restrict the panel to NZX-listed tickers
            (those whose symbol ends with `.NZ`). Non-NZX tickers (e.g. `%5ETNX`,
            `%5EFTSE`, `ZS=F`) are dropped before any further computation.
            Currency normalisation (FX) is not performed; the assumption that
            all in-universe instruments are NZD-denominated is enforced by the
            ticker filter, not by an FX layer. Set False to retain non-NZX
            tickers (only safe for NZD-denominated instruments).
        rebalance_freq: pandas frequency alias for the default rebalance schedule.
            Must be one of {"BME", "BQE"} (business month-end / quarter-end).
            Ignored when `rebalance_dates` is provided explicitly.
    """
    if rebalance_freq not in {"BME", "BQE"}:
        raise ValueError(
            f"rebalance_freq must be one of {{'BME', 'BQE'}}, got {rebalance_freq!r}"
        )
    prices = snap.prices.sort_index()
    if nzx_only:
        nzx_cols = [c for c in prices.columns if str(c).endswith(".NZ")]
        prices = prices[nzx_cols]
    volumes = snap.volumes.reindex_like(prices)

    # The raw price index may have intraday timestamps (e.g. "10:00:00 NZST")
    # when prices come from Yahoo Finance or similar intraday sources.  We need
    # one observation per calendar day before computing pct_change; otherwise
    # consecutive rows from different intraday timestamps produce spurious
    # returns and almost all daily returns become NaN.
    prices_daily = prices.resample("D").last()
    volumes_daily = volumes.resample("D").sum(min_count=1)

    filtered_prices, masked_month_ends = _apply_anomaly_mask(
        prices_daily,
        volumes_daily,
        snap.corporate_actions,
        anomaly_filter,
    )

    returns_daily = filtered_prices.pct_change(fill_method=None)
    # min_count=1: months where a ticker had no price data at all produce NaN
    # (not 0.0, which would wrongly inflate the valid-observation count in
    # history-length checks such as the momentum factor's min_months guard).
    returns_monthly = (1.0 + returns_daily).resample("BME").prod(min_count=1) - 1.0
    for ticker, dates in masked_month_ends.items():
        returns_monthly.loc[returns_monthly.index.intersection(dates), ticker] = np.nan

    shares = _build_share_count_series(
        snap.fundamentals,
        filtered_prices.index,
        filtered_prices.columns,
    )
    market_cap = filtered_prices * shares

    # Build market-cap proxy from shares × adj_close for pre-publication coverage.
    # _build_share_count_series already forward-fills from the first publication date.
    # Additionally backward-fill so dates before the earliest publication also get
    # a share count (using the earliest known value as a best-effort estimate).
    # This extends coverage into the pre-2022 period where market_cap is NaN.
    proxy_shares_filled = shares.ffill().bfill()
    market_cap_proxy = (filtered_prices * proxy_shares_filled).reindex_like(market_cap)

    sector = _build_sector_series(snap.sector_labels, filtered_prices.columns)

    asof_naive = snap.asof.tz_localize(None) if snap.asof.tzinfo else snap.asof

    if rebalance_start is not None:
        _rs = pd.Timestamp(rebalance_start)
        if _rs.tzinfo is not None:
            _rs = _rs.tz_localize(None)
    else:
        # Default: first date when ≥10 tickers have data, avoiding sparse
        # pre-equity rows from macro/international series.
        n_req = min(10, max(1, len(prices_daily.columns)))
        _has = filtered_prices.notna().sum(axis=1).ge(n_req)
        _first = _has.idxmax() if _has.any() else filtered_prices.index.min()
        _rs = _first.tz_localize(None) if _first.tzinfo else _first

    if rebalance_dates is None:
        if len(filtered_prices.index) == 0:
            rebalance_dates = pd.DatetimeIndex([])
        else:
            rebalance_dates = pd.date_range(
                start=_rs,
                end=asof_naive - pd.Timedelta(days=1),
                freq=rebalance_freq,
            )
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    universe_mask = _build_universe_mask(
        prices=filtered_prices,
        volumes=volumes_daily,
        market_cap=market_cap,
        rebalance_dates=rebalance_dates,
        min_adv_dollars=min_adv_dollars,
        min_market_cap_nzd=min_market_cap_nzd,
        min_history_days=min_history_days,
        adv_window=adv_window,
        mc_ffill_days=mc_ffill_days,
    )

    return PreparedPanel(
        returns_daily=returns_daily,
        returns_monthly=returns_monthly,
        market_cap=market_cap,
        sector=sector,
        universe_mask=universe_mask,
        macro=snap.macro,
        fundamentals=snap.fundamentals,
        asof=snap.asof,
        prices=filtered_prices,
        corporate_actions=snap.corporate_actions,
        market_cap_proxy=market_cap_proxy,
    )


def _apply_anomaly_mask(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    anomaly_filter: AnomalyFilterSpec | None,
) -> tuple[pd.DataFrame, dict[str, pd.DatetimeIndex]]:
    """Mask suspicious prints to NaN before return construction."""
    if anomaly_filter is None or anomaly_filter.kind == "none" or prices.empty:
        return prices, {}

    masked = prices.copy()
    masked_month_ends: dict[str, pd.DatetimeIndex] = {}
    corporate_action_dates = _build_corporate_action_dates(corporate_actions)

    for ticker in masked.columns:
        price_series = masked[ticker]
        volume_series = volumes[ticker].reindex(masked.index)

        trading_prices = price_series.dropna()
        if trading_prices.empty:
            continue

        trading_volumes = volume_series.reindex(trading_prices.index)
        trading_returns = trading_prices.pct_change(fill_method=None)

        if anomaly_filter.require_volume_confirmation:
            next_volumes = trading_volumes.shift(-1)
            bad_volume = (
                trading_returns.abs() > anomaly_filter.volume_gate_threshold
            ) & next_volumes.notna() & ~((trading_volumes > 0) & (next_volumes > 0))
            bad_dates = [
                date
                for date in bad_volume[bad_volume].index
                if not _has_corporate_action_near(
                    corporate_action_dates,
                    ticker,
                    date,
                    anomaly_filter.corporate_action_buffer_days,
                )
            ]
            if len(bad_dates) > 0:
                masked.loc[bad_dates, ticker] = np.nan

        trading_prices = masked[ticker].dropna()
        trading_returns = trading_prices.pct_change(fill_method=None)
        extreme_daily = (
            trading_returns.abs() > anomaly_filter.daily_abs_return_threshold
        ) & _is_one_sided_daily_move(trading_returns)
        for date in extreme_daily[extreme_daily].index:
            if not _has_corporate_action_near(
                corporate_action_dates,
                ticker,
                date,
                anomaly_filter.corporate_action_buffer_days,
            ):
                masked.loc[date, ticker] = np.nan

        trading_prices = masked[ticker].dropna()
        trading_returns = trading_prices.pct_change(fill_method=None)
        monthly_returns = (1.0 + trading_returns).resample("BME").prod(min_count=1) - 1.0
        extreme_monthly = monthly_returns.abs() > anomaly_filter.monthly_abs_return_threshold
        month_end_dates: list[pd.Timestamp] = []
        for date in extreme_monthly[extreme_monthly].index:
            offending_date = _monthly_offending_trading_date(trading_returns, date)
            check_date = offending_date or date
            if not _has_corporate_action_near(
                corporate_action_dates,
                ticker,
                check_date,
                anomaly_filter.corporate_action_buffer_days,
            ):
                month_end_dates.append(date)

        if month_end_dates:
            masked_month_ends[ticker] = pd.DatetimeIndex(month_end_dates)

    # Chronic-ticker pass (PIT-correct, expanding window): at each date t, a
    # ticker is excluded if its cumulative count of extreme daily returns up to
    # t exceeds the threshold.  This avoids the look-ahead bias of the old
    # full-history count, which would retroactively exclude tickers from early
    # rebalances based on future extreme prints.
    #
    # Note: `masked` is on a calendar-daily index (after resample("D").last()
    # upstream). Non-trading days are NaN, so pct_change() on the full index
    # would produce NaN for every post-weekend/holiday day.  We compute
    # trading-day returns per ticker (dropna), align back to the full index,
    # then cumsum on the aligned mask.
    max_days = anomaly_filter.chronic_ticker_max_extreme_days
    if max_days > 0:
        threshold = anomaly_filter.daily_abs_return_threshold
        # Build a boolean extreme-return mask aligned to the full calendar index
        extreme_aligned = pd.DataFrame(False, index=masked.index, columns=masked.columns)
        for ticker in masked.columns:
            trading_prices = masked[ticker].dropna()
            if len(trading_prices) < 2:
                continue
            trading_returns = trading_prices.pct_change(fill_method=None)
            extreme_trading = trading_returns.abs() > threshold
            extreme_aligned.loc[extreme_trading.index, ticker] = extreme_trading.values

        # cumsum gives running count of extreme days per ticker (calendar index)
        cumulative_extreme = extreme_aligned.cumsum(axis=0)
        # pit_ok is True while cumulative count has not yet exceeded threshold
        pit_ok = cumulative_extreme <= max_days
        # Set prices to NaN from the first date the threshold is exceeded
        n_excluded = int((~pit_ok & masked.notna()).sum().sum())
        masked = masked.where(pit_ok, other=np.nan)

        import logging as _logging
        _logging.getLogger(__name__).warning(
            "chronic-ticker PIT pass: %d ticker-date pairs excluded "
            "(cumulative extreme days > %d)",
            n_excluded,
            max_days,
        )

    return masked, masked_month_ends


def _is_one_sided_daily_move(returns: pd.Series) -> pd.Series:
    direction = np.sign(returns)
    prev_direction = np.sign(returns.shift(1))
    next_direction = np.sign(returns.shift(-1))
    return (
        (prev_direction.ne(-direction) | prev_direction.eq(0) | prev_direction.isna())
        & (next_direction.ne(-direction) | next_direction.eq(0) | next_direction.isna())
    ).fillna(False)


def _is_one_sided_monthly_move(
    daily_returns: pd.Series,
    monthly_returns: pd.Series,
) -> pd.Series:
    one_sided = pd.Series(False, index=monthly_returns.index)
    for month_end, monthly_return in monthly_returns.dropna().items():
        direction = np.sign(monthly_return)
        if direction == 0:
            continue

        month_mask = daily_returns.index.to_period("M") == month_end.to_period("M")
        month_slice = daily_returns.loc[month_mask].dropna()
        if month_slice.empty:
            continue

        month_directions = np.sign(month_slice)
        if ((month_directions == direction) | (month_directions == 0)).all():
            one_sided.loc[month_end] = True
    return one_sided


def _monthly_offending_trading_date(
    daily_returns: pd.Series,
    month_end: pd.Timestamp,
) -> pd.Timestamp | None:
    month_mask = daily_returns.index.to_period("M") == month_end.to_period("M")
    month_slice = daily_returns.loc[month_mask].dropna()
    if month_slice.empty:
        return None
    return pd.Timestamp(month_slice.abs().idxmax())


def _build_corporate_action_dates(corporate_actions: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    if corporate_actions.empty or "ticker" not in corporate_actions.columns or "ex_date" not in corporate_actions.columns:
        return {}

    actions = corporate_actions[["ticker", "ex_date"]].dropna()
    if actions.empty:
        return {}

    ex_dates = pd.to_datetime(actions["ex_date"])
    if getattr(ex_dates.dt, "tz", None) is not None:
        ex_dates = ex_dates.dt.tz_localize(None)
    actions = actions.assign(ex_date=ex_dates)
    return {
        str(ticker): pd.DatetimeIndex(group["ex_date"].sort_values().unique())
        for ticker, group in actions.groupby("ticker", sort=False)
    }


def _has_corporate_action_near(
    corporate_action_dates: dict[str, pd.DatetimeIndex],
    ticker: str,
    date: pd.Timestamp,
    buffer_days: int,
) -> bool:
    dates = corporate_action_dates.get(str(ticker))
    if dates is None or len(dates) == 0:
        return False

    delta = (dates - pd.Timestamp(date)).days
    return bool((abs(delta) <= buffer_days).any())


def _build_universe_mask(
    *,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    market_cap: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    min_adv_dollars: float,
    min_market_cap_nzd: float,
    min_history_days: int,
    adv_window: int,
    mc_ffill_days: int,
) -> pd.DataFrame:
    """For each rebalance date, mark tickers that pass all filters.

    Vectorised: computes rolling ADV and cumulative history across the full
    time series, then reindexes to rebalance dates. Avoids a Python loop over
    dates so performance scales with tickers rather than rebalance periods.
    """
    if len(rebalance_dates) == 0:
        return pd.DataFrame(False, index=rebalance_dates, columns=prices.columns)

    # Dollar volume per day; zero-volume days become NaN so rolling().mean()
    # naturally excludes them, satisfying the "non-zero ADV" intent.
    dollar_volume = (prices * volumes).where(volumes > 0)

    # Forward-fill market cap up to mc_ffill_days trading days so a single
    # missing print at a rebalance date doesn't drop an eligible ticker.
    mc_ffill = market_cap.ffill(limit=mc_ffill_days)

    # Cumulative count of non-NaN price observations as the history measure.
    history = prices.notna().cumsum()

    # Rolling ADV over the full series; NaN days (zero-volume) are excluded.
    adv = dollar_volume.rolling(adv_window, min_periods=1).mean()

    # Align to rebalance dates via ffill so weekend/holiday dates use the
    # last available trading-day value.
    adv_at = adv.reindex(rebalance_dates, method="ffill")
    hist_at = history.reindex(rebalance_dates, method="ffill").fillna(0)

    mask = (adv_at >= min_adv_dollars) & (hist_at >= min_history_days)

    # Mcap filter is opt-in. When disabled (<= 0), tickers without shares
    # fundamentals are NOT silently dropped; they pass on ADV + history alone.
    if min_market_cap_nzd > 0:
        mc_at = mc_ffill.reindex(rebalance_dates, method="ffill")
        mask = mask & (mc_at >= min_market_cap_nzd)

    mask = mask.fillna(False)
    mask.index.name = "rebalance_date"
    return mask


def _build_sector_series(sector_labels: pd.DataFrame, tickers: pd.Index) -> pd.Series:
    """Build a ticker→sector Series from a sector_labels DataFrame.

    Uses the most-recently-dated label per ticker.  Tickers with no entry
    remain as ``None`` (not silently bucketed into ``"Unknown"``); callers
    that need the combiner's single-group behaviour must handle ``None``
    explicitly — the combiner already does this via ``fillna("Unknown")``.

    Args:
        sector_labels: DataFrame with columns ``["ticker", "date", "sector"]``,
            as produced by ``csv_loader._build_sector_labels``.
        tickers: The ticker universe (from ``filtered_prices.columns``).

    Returns:
        pd.Series indexed by ticker, values = GICS sector string or ``None``.
    """
    if (
        sector_labels.empty
        or "ticker" not in sector_labels.columns
        or "sector" not in sector_labels.columns
    ):
        return pd.Series(None, index=tickers, name="sector", dtype=object)

    # Take the most recent label per ticker (sort ascending so last = newest).
    if "date" in sector_labels.columns:
        latest = (
            sector_labels.sort_values("date", ascending=True, na_position="first")
            .groupby("ticker")["sector"]
            .last()
        )
    else:
        latest = sector_labels.groupby("ticker")["sector"].last()

    result = latest.reindex(tickers)
    result.name = "sector"
    return result


def _build_share_count_series(
    fundamentals: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    """Build a date×ticker DataFrame of share counts, forward-filled.

    Picks the first non-null value per ticker/date across the preference list.
    Tickers with no available shares data get NaN (which propagates to market_cap
    and excludes them from the size filter).
    """
    empty = pd.DataFrame(float("nan"), index=date_index, columns=tickers)
    if fundamentals.empty:
        return empty

    available = [f for f in _SHARE_FIELDS_PREFERENCE if f in fundamentals.columns]
    if not available:
        return empty

    coalesced = fundamentals[available[0]].copy()
    for col in available[1:]:
        coalesced = coalesced.combine_first(fundamentals[col])

    coalesced = coalesced.dropna()
    if coalesced.empty:
        return empty

    frames: dict = {}
    for ticker, ts in coalesced.groupby(level="ticker"):
        if ticker not in tickers:
            continue
        s = ts.droplevel("ticker").sort_index()
        s.index = pd.to_datetime(s.index)
        frames[ticker] = s.reindex(date_index, method="ffill")

    if not frames:
        return empty

    out = pd.DataFrame(frames).reindex(columns=tickers)
    return out
