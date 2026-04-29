# Sharesies NZ Pricing Reference

**Source:** Sharesies pricing page, last updated 28 November 2025.  
**Retrieved via:** Wayback Machine snapshot (2026-01-29).

---

## Subscription Plans

| Plan | Monthly | Annual | Buy/Sell coverage/month |
|------|---------|--------|-------------------------|
| $3 plan | $3 NZD | $32.40 NZD | $500 NZD |
| **$7 plan** | **$7 NZD** | **$75.60 NZD** | **$1,000 NZD** |
| **$15 plan** | **$15 NZD** | **$162 NZD** | **$5,000 NZD** |
| Tailored | Contact | — | $1M+ portfolios |

Coverage means transaction fees are waived on buy/sell volume up to that amount each month.

---

## Pay-As-You-Go (no subscription, or volume exceeding plan coverage)

- **1.9% of order value** (190 bps)
- Capped per order:
  - NZ shares: **$25 NZD** (cap reached at orders ≥ $1,316 NZD)
  - AU shares: $15 AUD
  - US shares: $5 USD
- No transaction fee on managed funds (unlisted)

---

## Currency Exchange

Charged separately on top of transaction fees when converting from NZD. Rate not published as a fixed figure.

---

## Model Notes for Backtesting

The Skuld cost model (`CostConfig`) uses the **$15/month plan** as its baseline:

- `sharesies_monthly_fee_nzd = 15.0` — charged every month regardless of trading activity
- `sharesies_coverage_nzd = 5_000.0` — $5k of trades per month covered by subscription
- `sharesies_excess_bps = 190.0` — **1.9% on volume ABOVE coverage only** (not on total)
- `spread_bps = 200.0` — 200 bps round-trip bid/ask spread (separate from Sharesies fee)

**Key implication:** the subscription is charged even in months with no trades (e.g. when holding all cash).

**NAV break-even:** the $15/month flat fee costs 1.8% of NAV/year at $10K NAV, falling to 0.36%/year at $50K NAV.

**Known simplification:** per-order $25 NZD cap is not modelled; the model applies 1.9% to all excess volume. Conservative for large single orders (≥$1,316 NZD each).
