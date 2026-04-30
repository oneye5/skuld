# Data Pipeline — Quality, Preprocessing, and Contracts

## 1. Raw Data Quality

The raw `data/data_long.csv` produced by the Java ingestion layer has several quality characteristics that are important to understand before working with it. The overarching theme is: **the raw data is not safe to consume directly**. It requires the full preprocessing pipeline before any analysis.

### 1.1 No Publication Dates

The `timestamp` column represents the observation or period date, not the date the data became publicly available. This creates different risks depending on source type:

- **Price data (`yf_prices`):** Timestamp is the trading date. The close price is known at market close on that date — accurate.
- **Macro data:** Timestamps approximate publication dates from the publishing agencies' release schedules — close enough for Phase 1.
- **Fundamental data (`yf_finances`):** Yahoo Finance's time series API returns period-end dates, not filing dates. A company's year-end result with `timestamp = 2024-06-30` was not publicly available on that date — it was published weeks to months later. The PIT loader treats `timestamp` as the publication date, which means fundamentals appear *later* than actually available (by the filing lag), not earlier. This is a conservative bias (performance drag, not lookahead leakage). The correct fix — sourcing actual filing dates per company — is a future data-quality improvement.

### 1.2 Undefined Row Order

Rows are not sorted. Data is collected via parallel streams in the Java ingestion layer, so row order reflects source execution timing, not date or ticker order. Downstream consumers must sort explicitly as needed. The Python CSV loader does not impose a sort order at load time.

### 1.3 Sparse Fundamental Coverage

Only 69.9% of NZX tickers have any fundamental data from `yf_finances`. 30.1% have none. Of those with fundamentals, not all fields are present for all companies: book equity, gross profit, total assets, and operating cash flow are frequently absent, which blocks value and quality factor computation for those names. Imputation policy (exclusion from ranking, sector-median fill, or forward-fill within a max-staleness window) is specified per factor in the pre-registered spec — the default is exclusion.

### 1.4 Missing Price Data for Some NZX Tickers

9 NZX tickers in the universe (ARV.NZ, FRE.NZ, THL.NZ, TPW.NZ, TRA.NZ, TWR.NZ, VCT.NZ, VGL.NZ, ZEL.NZ) have no price data — only fundamental data from `yf_finances`. These are excluded from the investable universe but may carry fundamental fields. Likely caused by ticker symbol changes, delistings during the data collection window, or Yahoo Finance API limitations.

### 1.5 Uneven History Depth

History length varies widely across the NZX universe: 53 tickers have >20 years of price data; 5 have <2 years. The minimum-history filter in universe construction excludes short-history names from factor rankings that require long lookbacks (e.g., 12-month momentum requires ≥11 months of clean price history). Short-history names still appear in the raw CSV — they are excluded downstream, not filtered at the ingestion layer.

### 1.6 Stale or Intermittent Fundamentals

Yahoo Finance's fundamental data refresh is not guaranteed to be regular or timely. A fundamental snapshot may be months or years old. The data validation layer flags any ticker whose most recent fundamental observation is >18 months old and excludes it from value and quality factor rankings for that rebalance date. Stale fundamentals are worse than missing ones because forward-fill across a stale gap produces a silently incorrect factor score.

### 1.7 Incomplete Corporate Action Records

Split data is present for only 31.5% of NZX tickers; dividend data for 78.8%. A price series without its corresponding split adjustments will show artificial price discontinuities, inflating or inverting return signals. `PreparedPanel` does **not** apply the corporate-action frame to prices — it forwards it to consumers and otherwise relies on Yahoo's `adj_close` to already encode splits and dividends. The optional adjustments layer (`skuld_research.data.adjustments`, see `docs/ADJUSTMENTS.md`) cross-validates `adj_close` against the ledger and can repair detected issues; in particular its `missed_split` detector flags split-shaped jumps in `adj_close` that have no corresponding record, which is the closest the pipeline currently comes to surfacing the never-recorded-split failure mode. Splits that are *also* missing from `adj_close` itself (i.e. Yahoo's chain failed silently and no ledger row exists) remain unaddressed.

### 1.8 No Schema Versioning

The raw CSV carries no version tag. If the Java ingestion layer changes column semantics — repurposing `src` IDs, switching `timestamp` encoding, or renaming features — the Python loader will silently misread it. The implementation plan (§6.6) specifies a `data/data_long.schema.json` sidecar to address this; until it is implemented, any change to the Java output schema requires a matched Python loader update.

### 1.9 Value Column is a String

The `value` column is written as a plain decimal string (no scientific notation). It is parsed to `float64` at load time by the CSV loader. Nulls are filtered upstream by the Java `IngestManager` (rows with `null` value are dropped before CSV write), so the loaded frame should contain no NaN values in the value column before the validation layer runs.

### 1.10 Wikimedia Pageviews Signal Uncertainty

At 1.17M rows, `wikimedia_pageviews` is the second-largest source (24.6% of all rows after `yf_prices`). Its predictive value for NZX equity returns is unvalidated. It is present in the pipeline but no signal module currently uses it. Treat any factor using Wikipedia pageviews as experimental and subject to a full gating evaluation before deployment.

---

## 2. Preprocessing Pipeline

The Python pipeline addresses the raw data quality issues through a layered approach. Each layer has a well-defined responsibility boundary; no layer reaches across it.

### 2.1 CSV Loader → Raw Data

Loads the raw long-format CSV into memory and applies validation before any downstream consumption:

- **Negative prices:** flagged and excluded. Indicates a data error (Yahoo Finance occasionally returns negative adjusted prices from corporate action miscalculation).
- **Zero-volume ADV:** the trailing 20-day ADV used in the liquidity filter is computed as the mean of the 20 most recent *trading days with non-zero volume*, not calendar days. This prevents NZX small-caps with intermittent zero-volume days from having their ADV diluted to near-zero.
- **Gap detection:** if a ticker has >5 consecutive missing trading days (excluding known holidays), the ticker is excluded from the investable universe for that rebalance date. Prices in the gap are not forward-filled.
- **Stale fundamentals:** tickers whose most recent fundamental snapshot is >18 months old are excluded from value and quality factor rankings. Stale data is not forward-filled.

The `src` column is used for staleness reporting and audit only. The loader routes rows to the appropriate frame by `feature` name and ticker presence, not by `src` ID.

### 2.2 PIT Loader → PITSnapshot

The point-in-time loader (`pit_loader.py`) enforces the no-lookahead invariant. `PITLoader.as_of(t)` returns a `PITSnapshot` containing only observations with `timestamp < t` (strictly less than). This is enforced in code, not by convention, and tested in CI with a synthetic future-dated row that must not appear in the resulting snapshot.

**Fundamental data note:** As described in §1.1, `timestamp` for fundamentals is the period-end date, not the filing date. Using it as-is means fundamentals arrive somewhat later in the backtest than they were actually available — a conservative bias, not a leakage risk.

### 2.3 PreparedPanel Construction

`PreparedPanel` is built from a `PITSnapshot` and applies corporate actions, computes derived series, and constructs the investable universe mask:

- **Total-return series:** derived directly from Yahoo Finance's `adj_close`, which is *expected* to already incorporate splits, dividends, and capital returns. Skuld trusts that adjustment by default — no explicit re-application of the corporate-action frame is performed at panel-construction time. An optional audit/repair layer (`skuld_research.data.adjustments`) can be enabled by callers to cross-validate `adj_close` against the corporate-action ledger and, if requested, repair detected discrepancies (missed splits, unit jumps, bad dividend back-adjustments, etc.). It is off by default; see `docs/ADJUSTMENTS.md` for the detection taxonomy, repair policies, and integration patterns.
- **Market cap:** price × shares outstanding as of the snapshot date.
- **Sector:** populated from sector source if available; currently all "Unknown" (sector data is deferred). Cross-sectional z-scoring degenerates to universe-wide z-scoring until real sector data is added.
- **Universe mask:** `universe_mask[t, ticker] = True` only when all of the following hold as of `PITSnapshot.as_of(t)`:
  - Price history is present
  - At least 2 years of history
  - Intended position size would not exceed 1% of trailing 20-day ADV
  - No price-gap exclusion triggered in the lookback window

A `LookaheadError` is raised (not a warning) if any future-dated input is detected. This is tested in CI.

---

## 3. Data Contracts

These are the typed contracts that flow between pipeline stages. They are defined in `skuld_common` and consumed by both `skuld_research` and `skuld_portfolio`. A change to any contract is a breaking change for both consumers.

### PITSnapshot — all values knowable strictly before `t`

| Field | Type | Description |
|---|---|---|
| `prices` | `DataFrame` | index=date, columns=ticker, values=adjusted close |
| `volumes` | `DataFrame` | index=date, columns=ticker, values=ADV in NZD |
| `fundamentals` | `DataFrame` | MultiIndex (ticker, publication_date), columns=field |
| `macro` | `DataFrame` | index=date, columns=macro_feature |
| `corporate_actions` | `DataFrame` | columns: ticker, ex_date, type, factor |
| `asof` | `Timestamp` | the `t` this snapshot was built for |

### PreparedPanel — cleaned, aligned, investment-ready

| Field | Type | Description |
|---|---|---|
| `returns_daily` | `DataFrame` | index=date, columns=ticker, total-return |
| `returns_monthly` | `DataFrame` | month-end total-return |
| `market_cap` | `DataFrame` | index=date, columns=ticker, NZD |
| `sector` | `Series` | index=ticker, GICS sector or 'Unknown' |
| `universe_mask` | `DataFrame` | index=rebalance_date, columns=ticker, bool |
| `asof` | `Timestamp` | |

### SignalGenerator — protocol implemented by each factor module

Each factor module implements:

```python
class SignalGenerator(Protocol):
    name: str
    required_data: list[DataRequest]
    def score(self, panel: PreparedPanel, t: Timestamp,
              universe: list[str]) -> pd.Series:
        """index=ticker, values=raw factor score. NaN = excluded from ranking."""
```

Output is a `pd.Series` with `index = universe`, `dtype = float64`. Missing values are NaN; the combiner handles them per the imputation policy in the spec.

### CombinedScores — output of signal combiner

| Field | Type | Description |
|---|---|---|
| `scores` | `Series` | index=ticker, combined z-score, NaN-free over universe |
| `component_scores` | `DataFrame` | index=ticker, columns=factor name, post-shrinkage z |
| `asof` | `Timestamp` | |

Combiner pipeline per date: cross-sectional z within sector → winsorise at ±3 → shrink toward sector mean (degenerates to universe mean until sector data is available) → equal-weighted average → re-z the combined score.

### TargetPortfolio — output of portfolio constructor

| Field | Type | Description |
|---|---|---|
| `weights` | `Series` | index=ticker, sums to (1 − cash_weight) |
| `cash_weight` | `float` | in [0, 1] |
| `method` | `str` | `'HRP'`, `'RiskParity'`, or `'InverseVol'` (small-N fallback) |
| `asof` | `Timestamp` | |

HRP is the default optimiser. `InverseVol` is substituted automatically when the post-filter universe has <15 names — HRP clustering is uninformative below that threshold.

### TradeList — execution plan

`trades` DataFrame columns:

| Column | Description |
|---|---|
| `ticker` | Equity ticker or CASH |
| `action` | `BUY` / `SELL` / `HOLD` / `DEFER` |
| `current_shares`, `target_shares`, `delta_shares` | Position change |
| `current_value_nzd`, `target_value_nzd`, `delta_value_nzd` | NZD equivalents |
| `est_round_trip_cost_nzd` | Spread + Sharesies fee estimate |
| `sharesies_fee_band` | `flat_15` or `percent_19bps` |
| `in_no_trade_region` | True if expected alpha < 2× round-trip cost |
| `below_size_floor` | True if trade value < max($50, 5× round-trip cost) |
| `deferred_to_next_month` | True if trade pushed volume past the $5k Sharesies cliff |

Additional metadata fields on `TradeList`: `total_volume_nzd`, `total_estimated_cost_nzd`, `asof`, `config_hash`.

For the complete output CSV schema (recommendation file columns including per-factor z-scores), see the implementation plan §3.9 (not currently available as a document).

---

## 4. Key Invariants

These invariants are enforced in code and tested in CI. Violating them silently would produce backtests with lookahead bias or corrupted portfolio weights.

1. **No-lookahead:** every column in every frame returned by `PITLoader.as_of(t)` has `publication_date < t`. A CI test with a synthetic future-dated row must produce a snapshot where that row is absent.
2. **Universe mask PIT-safety:** every cell `universe_mask[t, ticker] = True` must be derivable solely from `PITSnapshot.as_of(t)`. Tested with a future-dated volume row that would qualify the ticker — the mask must produce `False`.
3. **Portfolio weights sum:** all weights ≥ 0 and `sum(weights) + cash_weight` is within floating-point tolerance of 1.0. Enforced by assertion in the portfolio constructor.
4. **Config hash in every output:** every backtest report and recommendation CSV embeds the SHA-256 hash of the spec file used to produce it. Any run without a valid spec hash is not a valid result.
