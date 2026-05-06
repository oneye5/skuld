# WS1 Attribution + WS3 Investability Filter Analysis

Scope: exploration. No production spec is modified.

## WS1 — Return Attribution (mom-s8)

### Headline decomposition

| Component | Annualised CAGR |
|---|---:|
| Market proxy (EW universe) | +8.4% |
| Signal EW (zero-cost top-50%) | +14.8% |
| Production (net of costs) | +13.4% |
| **Signal contribution (signal − mkt)** | **+5.9%** |
| **Construction + cost drag (prod − signal)** | **-1.0%** |
| **Total alpha (prod − mkt)** | **+4.9%** |

### Factor-leg standalone alpha vs market proxy

| Factor | Standalone EW alpha (ann.) |
|---|---:|
| momentum | +4.0% |
| return_on_risk | +6.8% |

### Universe breadth (tickers passing all filters per rebalance)

| Statistic | Value |
|---|---:|
| Mean | 71.1 |
| Min | 0 |
| Max | 131 |
| Median | 60 |

### Position cap binding (max_position=0.25)

| Metric | Value |
|---|---:|
| Mean bound tickers per period | 0.00 |
| Max bound tickers in one period | 0 |
| % periods with any binding | 0% |

### Top-5 contributors to signal-EW (mean per-period weighted return)

| Ticker | Mean contribution |
|---|---:|
| BLT.NZ | +0.0028 |
| BFG.NZ | +0.0025 |
| SKT.NZ | +0.0025 |
| GEN.NZ | +0.0012 |
| MKR.NZ | +0.0012 |

### Bottom-5 contributors to signal-EW

| Ticker | Mean contribution |
|---|---:|
| SVR.NZ | -0.0008 |
| WCO.NZ | -0.0006 |
| WIN.NZ | -0.0005 |
| MFB.NZ | -0.0004 |
| TWL.NZ | -0.0003 |

### OOS Sharpe by regime

| Regime | Sharpe |
|---|---:|
| bull | 0.844 |
| bear | -1.702 |
| chop | 1.391 |

## WS3 — Investability Filter Variants

Decision criterion: universe breadth must stay >= 6 names on average and flat-haircut Sharpe must not worsen by more than 0.05 vs baseline.

Baseline `mom-s8`: flat-haircut Sharpe 0.532, turnover 17.8%, OOS n=234.

| Variant | Universe breadth (mean) | Sharpe HC | Delta HC | Paired delta ann. | Paired CI monthly | Turnover | Assessment |
|---|---:|---:|---:|---:|---|---:|---|
| `ws3-mom-s8-adv25k` | 59.4 | 0.486 | -0.046 | -0.2% | [-0.13%, +0.09%] | 20.0% | pass |
| `ws3-mom-s8-hist180` | 69.6 | 0.531 | -0.001 | -0.0% | [-0.08%, +0.09%] | 17.3% | pass |
| `ws3-mom-s8-chronic3` | 69.4 | 0.452 | -0.080 | -0.9% | [-0.18%, +0.03%] | 17.4% | worse |
| `ws3-mom-s8-strict` | 57.4 | 0.426 | -0.106 | -1.0% | [-0.23%, +0.06%] | 19.8% | worse |

