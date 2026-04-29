"""Stage 2: build a PreparedPanel from a PITSnapshot.

Takes a point-in-time snapshot and produces cleaned, aligned series ready
for factor computation: total-return daily/monthly, market cap, sector,
and per-rebalance-date universe masks driven by liquidity + history filters.
"""

from __future__ import annotations

import pandas as pd

from skuld_common.contracts import PITSnapshot, PreparedPanel

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
    volumes_daily = volumes.resample("D").last()

    returns_daily = prices_daily.pct_change(fill_method=None)
    # min_count=1: months where a ticker had no price data at all produce NaN
    # (not 0.0, which would wrongly inflate the valid-observation count in
    # history-length checks such as the momentum factor's min_months guard).
    returns_monthly = (1.0 + returns_daily).resample("BME").prod(min_count=1) - 1.0

    shares = _build_share_count_series(snap.fundamentals, prices_daily.index, prices_daily.columns)
    market_cap = prices_daily * shares

    sector = pd.Series("Unknown", index=prices_daily.columns, name="sector")

    asof_naive = snap.asof.tz_localize(None) if snap.asof.tzinfo else snap.asof

    if rebalance_start is not None:
        _rs = pd.Timestamp(rebalance_start)
        if _rs.tzinfo is not None:
            _rs = _rs.tz_localize(None)
    else:
        # Default: first date when ≥10 tickers have data, avoiding sparse
        # pre-equity rows from macro/international series.
        n_req = min(10, max(1, len(prices_daily.columns)))
        _has = prices_daily.notna().sum(axis=1).ge(n_req)
        _first = _has.idxmax() if _has.any() else prices_daily.index.min()
        _rs = _first.tz_localize(None) if _first.tzinfo else _first

    if rebalance_dates is None:
        if len(prices_daily.index) == 0:
            rebalance_dates = pd.DatetimeIndex([])
        else:
            rebalance_dates = pd.date_range(
                start=_rs,
                end=asof_naive - pd.Timedelta(days=1),
                freq=rebalance_freq,
            )
    rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    universe_mask = _build_universe_mask(
        prices=prices_daily,
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
        asof=snap.asof,
    )


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
