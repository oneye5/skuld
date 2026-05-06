# WS2 Construction Sweep + WS4 Regime Overlay + WS5 New Alpha Factors

Scope: exploration. No production spec is modified.

Baseline `mom-s8`: flat-haircut Sharpe 0.532, turnover 17.8%.

## WS2 — Portfolio Construction Sweep (quick=True, 16 variants)

Variants completed: 16, failed: 0.

Decision criterion: flat-haircut Sharpe ≥ baseline + 0.10 AND paired CI lower bound ≥ 0.

Top-10 by OOS Sharpe (raw, before haircut):

| Variant | Sharpe raw | Δ vs baseline | Turnover | Max DD |
|---|---:|---:|---:|---:|
| `cons-maxpos20-lam0-smooth0-nt0.005-tb30-bme` | 0.914 | +0.009 | 18.3% | -27.5% |
| `cons-maxpos25-lam0-smooth0-nt0.005-tb30-bme` | 0.914 | +0.009 | 18.3% | -27.5% |
| `cons-maxpos25-lam0.5-smooth0-nt0.005-tb30-bme` | 0.906 | +0.000 | 17.8% | -27.5% |
| `cons-maxpos20-lam0.5-smooth0-nt0.005-tb30-bme` | 0.906 | +0.000 | 17.8% | -27.5% |
| `cons-maxpos20-lam0-smooth0.1-nt0.005-tb30-bme` | 0.866 | -0.039 | 16.8% | -29.5% |
| `cons-maxpos25-lam0-smooth0.1-nt0.005-tb30-bme` | 0.866 | -0.039 | 16.8% | -29.5% |
| `cons-maxpos25-lam0.5-smooth0.1-nt0.005-tb30-bme` | 0.850 | -0.055 | 16.3% | -29.5% |
| `cons-maxpos20-lam0.5-smooth0.1-nt0.005-tb30-bme` | 0.850 | -0.055 | 16.3% | -29.5% |
| `cons-maxpos25-lam0-smooth0-nt0.005-tbNone-bme` | 0.806 | -0.100 | 21.2% | -26.2% |
| `cons-maxpos20-lam0-smooth0-nt0.005-tbNone-bme` | 0.806 | -0.100 | 21.2% | -26.2% |

**Best variant**: `cons-maxpos20-lam0-smooth0-nt0.005-tb30-bme` — Sharpe raw 0.914, delta +0.009.

**WS2 verdict**: FAIL — no variant clears +0.10 threshold over baseline.

## WS4 — Regime Overlay (NZX MA-200 + aggregate momentum)

| Metric | Value |
|---|---:|
| Sharpe HC | 0.513 |
| Delta HC vs baseline | -0.019 |
| Paired ann. delta | -0.2% |
| Paired 95% CI (monthly) | [-0.10%, +0.08%] |
| Turnover | 17.4% |
| Max drawdown | -25.3% |
| CAGR | +13.2% |

**WS4 verdict**: FAIL — overlay does not improve risk-adjusted return.

## WS5 — New Alpha Factors (EPS momentum, volume trend)

Decision criterion: flat-haircut Sharpe ≥ baseline (no regression) AND paired ann. delta ≥ 0 (positive contribution).

| Variant | Sharpe HC | Δ HC | Paired ann Δ | Paired 95% CI | Turnover | Verdict |
|---|---:|---:|---:|---|---:|---|
| `ws5-mom-s8-eps` | 0.518 | -0.014 | +0.3% | [-0.07%, +0.13%] | 17.2% | fail |
| `ws5-mom-s8-voltrd` | 0.513 | -0.019 | -0.2% | [-0.10%, +0.08%] | 17.4% | fail |

