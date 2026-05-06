# Phase 2 Exploration Summary

Scope: exploration. These runs do not promote production specs or increment production trial count.

Baseline: `mom-s8` flat-haircut Sharpe 0.532, turnover 17.8%.

| Candidate | Factors | Sharpe HC | Delta HC | Paired Delta Ann | Paired CI Monthly | Paired N | Turnover | Recommendation |
|---|---|---:|---:|---:|---|---:|---:|---|
| `phase2-mom-resid` | residual_momentum, return_on_risk | 0.541 | +0.009 | +0.1% | [-0.12%, +0.13%] | 234 | 19.7% | exclude |
| `phase2-mom-max-avoid` | momentum, return_on_risk, max_daily_return_avoidance | 0.532 | +0.000 | -0.0% | [-0.00%, +0.00%] | 234 | 17.8% | exclude |
| `phase2-mom-betaadj` | beta_adjusted_momentum, return_on_risk | 0.525 | -0.007 | -0.1% | [-0.14%, +0.12%] | 234 | 19.6% | exclude |
| `phase2-mom-ex-short-spike` | momentum_ex_short_spike, return_on_risk | 0.491 | -0.041 | -0.5% | [-0.12%, +0.04%] | 234 | 19.5% | exclude |
| `phase2-mom-dual-horizon` | dual_horizon_momentum, return_on_risk | 0.460 | -0.072 | -0.7% | [-0.17%, +0.08%] | 234 | 19.6% | exclude |
| `phase2-mom-ts-filter` | time_series_filtered_momentum, return_on_risk | 0.439 | -0.093 | -1.1% | [-0.22%, +0.03%] | 234 | 21.1% | exclude |
| `phase2-mom-drawdownaware` | momentum_drawdown_aware, return_on_risk | 0.404 | -0.128 | -1.7% | [-0.29%, +0.02%] | 234 | 22.6% | exclude |
| `phase2-mom-vol-penalized` | momentum_vol_penalized, return_on_risk | 0.404 | -0.128 | -1.7% | [-0.29%, +0.02%] | 234 | 22.6% | exclude |
| `phase2-mom-reversal-adjusted` | reversal_adjusted_momentum, return_on_risk | 0.402 | -0.131 | -1.3% | [-0.21%, -0.02%] | 234 | 19.6% | exclude |
| `phase2-mom-consistency` | momentum_consistency, return_on_risk | 0.384 | -0.148 | -2.0% | [-0.30%, -0.04%] | 234 | 18.6% | exclude |
| `phase2-mom-resid-52wh` | residual_momentum, high_52_week, return_on_risk | 0.379 | -0.153 | -1.9% | [-0.31%, -0.00%] | 234 | 22.3% | exclude |
| `phase2-mom-stack-52wh` | momentum, return_on_risk, high_52_week | 0.361 | -0.171 | -2.0% | [-0.29%, -0.04%] | 234 | 21.1% | exclude |
| `phase2-mom-acceleration` | momentum_acceleration, return_on_risk | 0.186 | -0.347 | -4.1% | [-0.74%, -0.02%] | 234 | 25.5% | exclude |
| `phase2-mom-52wh` | high_52_week, return_on_risk | 0.164 | -0.368 | -4.3% | [-0.55%, -0.15%] | 234 | 26.5% | exclude |

## Notes

- Shortlist review requires flat-haircut Sharpe at least +0.10 above `mom-s8`, positive paired-delta median, and paired CI lower bound >= 0.
- Sector-dependent candidates are intentionally absent; all listed candidates use price/return-derived inputs only.
- `watch` means directionally useful but below the formal incremental shortlist bar.
