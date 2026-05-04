"""
Data source viability probe for NZX balance sheet, cash flow, and sector classification.

Tests:
1. yfinance - balance_sheet, cashflow, sector via info
2. Yahoo Finance fundamentals-timeseries API (direct) - balance sheet + cash flow types
3. Financial Modeling Prep (FMP) free tier
4. OpenFIGI for sector/GICS mapping

Run: uv run python scripts/data_source_probe.py
"""

import json
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime

# Representative NZX tickers for testing (large caps with known financials)
TEST_TICKERS = ["FPH.NZ", "SPK.NZ", "RYM.NZ", "MEL.NZ", "AIR.NZ"]

# Reference values for cross-validation (approximate, from public sources)
# FPH.NZ FY2023 (March year-end): TotalAssets ~2.2B NZD, OperatingCashFlow ~400M NZD
# SPK.NZ FY2023 (June year-end): TotalAssets ~4.5B NZD, OperatingCashFlow ~540M NZD
# RYM.NZ FY2023: TotalAssets ~8.9B NZD
REFERENCE = {
    "FPH.NZ": {"year": 2023, "total_assets_approx_B_NZD": 2.2, "ocf_approx_B_NZD": 0.40},
    "SPK.NZ": {"year": 2023, "total_assets_approx_B_NZD": 4.5, "ocf_approx_B_NZD": 0.54},
    "RYM.NZ": {"year": 2023, "total_assets_approx_B_NZD": 8.9, "ocf_approx_B_NZD": 0.12},
}

SEP = "=" * 70


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: yfinance Python library
# ─────────────────────────────────────────────────────────────────────────────

def probe_yfinance():
    section("SOURCE 1: yfinance (Python library)")

    for ticker_str in TEST_TICKERS[:3]:
        print(f"\n--- {ticker_str} ---")
        t = yf.Ticker(ticker_str)

        # 1a. Sector / industry classification
        try:
            info = t.info
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            print(f"  Sector: {sector}  |  Industry: {industry}")
        except Exception as e:
            print(f"  [sector] ERROR: {e}")

        # 1b. Annual balance sheet
        try:
            bs = t.balance_sheet  # columns = fiscal year dates
            if bs is not None and not bs.empty:
                print(f"  Balance sheet: {bs.shape[1]} years x {bs.shape[0]} items")
                print(f"  Years available: {[str(c.date()) for c in bs.columns]}")
                key_rows = ["Total Assets", "Total Debt", "Stockholders Equity",
                            "Cash And Cash Equivalents", "Total Liabilities Net Minority Interest"]
                for row in key_rows:
                    if row in bs.index:
                        vals = {str(c.date()): f"{v/1e9:.3f}B" for c, v in bs.loc[row].items() if pd.notna(v)}
                        print(f"    {row}: {vals}")
            else:
                print("  Balance sheet: EMPTY")
        except Exception as e:
            print(f"  [balance_sheet] ERROR: {e}")

        # 1c. Annual cash flow statement
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                print(f"  Cash flow: {cf.shape[1]} years x {cf.shape[0]} items")
                key_rows = ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow",
                            "Investing Cash Flow", "Financing Cash Flow"]
                for row in key_rows:
                    if row in cf.index:
                        vals = {str(c.date()): f"{v/1e9:.3f}B" for c, v in cf.loc[row].items() if pd.notna(v)}
                        print(f"    {row}: {vals}")
            else:
                print("  Cash flow: EMPTY")
        except Exception as e:
            print(f"  [cashflow] ERROR: {e}")

        # 1d. Quarterly balance sheet (history depth check)
        try:
            qbs = t.quarterly_balance_sheet
            if qbs is not None and not qbs.empty:
                dates = sorted([c for c in qbs.columns], reverse=True)
                print(f"  Quarterly BS: {len(dates)} quarters, oldest: {str(dates[-1].date())}")
            else:
                print("  Quarterly BS: EMPTY")
        except Exception as e:
            print(f"  [quarterly_balance_sheet] ERROR: {e}")

    # Check coverage across a broader set
    print(f"\n--- Coverage check across {len(TEST_TICKERS)} tickers ---")
    results = {}
    for ticker_str in TEST_TICKERS:
        t = yf.Ticker(ticker_str)
        try:
            bs = t.balance_sheet
            cf = t.cashflow
            info = t.info
            results[ticker_str] = {
                "bs_years": bs.shape[1] if bs is not None and not bs.empty else 0,
                "cf_years": cf.shape[1] if cf is not None and not cf.empty else 0,
                "sector": info.get("sector", "N/A"),
            }
        except Exception as e:
            results[ticker_str] = {"error": str(e)}

    for ticker, r in results.items():
        print(f"  {ticker}: {r}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: Yahoo Finance fundamentals-timeseries API (direct HTTP)
# Tests whether balance sheet + cash flow types are available in the existing API
# ─────────────────────────────────────────────────────────────────────────────

BALANCE_SHEET_TYPES = [
    "annualTotalAssets", "annualTotalLiabilitiesNetMinorityInterest",
    "annualStockholdersEquity", "annualCashAndCashEquivalents",
    "annualTotalDebt", "annualNetPPE", "annualCurrentAssets",
    "annualCurrentLiabilities", "annualInventory",
    "annualAccountsReceivable", "annualGoodwill",
    "annualTotalEquityGrossMinorityInterest",
    "annualRetainedEarnings", "annualCommonStock",
]

CASHFLOW_TYPES = [
    "annualOperatingCashFlow", "annualCapitalExpenditure",
    "annualFreeCashFlow", "annualCashFlowFromContinuingOperatingActivities",
    "annualCashFlowFromContinuingInvestingActivities",
    "annualCashFlowFromContinuingFinancingActivities",
    "annualDepreciationAndAmortization", "annualChangeInWorkingCapital",
    "annualDividendsPaid", "annualRepurchaseOfCapitalStock",
    "annualIssuanceOfDebt", "annualRepaymentOfDebt",
]

def probe_yf_timeseries_api():
    section("SOURCE 2: Yahoo Finance fundamentals-timeseries API (direct)")

    ticker = "FPH.NZ"
    all_types = BALANCE_SHEET_TYPES + CASHFLOW_TYPES
    type_str = ",".join(all_types)

    url = (
        f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}"
        f"?merge=false&padTimeSeries=true&period1=493590046&period2=2750557599"
        f"&type={type_str}&lang=en-NZ&region=NZ"
    )

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    print(f"\nFetching balance sheet + cash flow types for {ticker}...")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results_arr = data.get("timeseries", {}).get("result", [])
        print(f"  API returned {len(results_arr)} result objects")

        found_types = {}
        for result in results_arr:
            meta = result.get("meta", {})
            type_list = meta.get("type", [])
            if not type_list:
                continue
            feat_type = type_list[0]
            if feat_type in result and result[feat_type]:
                data_points = [p for p in result[feat_type] if p is not None]
                if data_points:
                    dates = [p["asOfDate"] for p in data_points]
                    values = [p.get("reportedValue", {}).get("raw") for p in data_points]
                    found_types[feat_type] = {
                        "count": len(data_points),
                        "earliest": min(dates),
                        "latest": max(dates),
                        "latest_value": values[-1] if values else None,
                    }

        print(f"\n  Balance Sheet types found ({sum(1 for k in found_types if 'Asset' in k or 'Equit' in k or 'Liab' in k or 'Cash' in k or 'Debt' in k or 'PPE' in k or 'Inventor' in k or 'Receivabl' in k or 'Goodwill' in k or 'Retained' in k or 'Common' in k)}):")
        bs_found = [t for t in BALANCE_SHEET_TYPES if t in found_types]
        bs_missing = [t for t in BALANCE_SHEET_TYPES if t not in found_types]
        for t_name in bs_found:
            info = found_types[t_name]
            val = info['latest_value']
            val_str = f"{val/1e9:.3f}B" if val else "N/A"
            print(f"    [OK] {t_name}: {info['count']} obs, {info['earliest']}–{info['latest']}, latest={val_str}")
        for t_name in bs_missing:
            print(f"    [--] {t_name}: NOT FOUND")

        print(f"\n  Cash Flow types found:")
        cf_found = [t for t in CASHFLOW_TYPES if t in found_types]
        cf_missing = [t for t in CASHFLOW_TYPES if t not in found_types]
        for t_name in cf_found:
            info = found_types[t_name]
            val = info['latest_value']
            val_str = f"{val/1e9:.3f}B" if val else "N/A"
            print(f"    [OK] {t_name}: {info['count']} obs, {info['earliest']}–{info['latest']}, latest={val_str}")
        for t_name in cf_missing:
            print(f"    [--] {t_name}: NOT FOUND")

        return found_types

    except Exception as e:
        print(f"  ERROR: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: Financial Modeling Prep (FMP) - free tier
# ─────────────────────────────────────────────────────────────────────────────

def probe_fmp():
    section("SOURCE 3: Financial Modeling Prep (FMP) - free tier")
    # FMP free tier uses API key; demo key has very limited access
    # Test with demo key first to check endpoint structure and NZX coverage
    FMP_BASE = "https://financialmodelingprep.com/api/v3"
    DEMO_KEY = "demo"  # FMP demo key - limited to AAPL, MSFT etc.

    # Check if NZX tickers are accessible (they typically aren't on demo)
    # But we can verify the API structure and try anyway
    ticker = "FPH.NZ"

    print(f"\nTesting FMP balance sheet for {ticker} (demo key - may be restricted)...")
    try:
        url = f"{FMP_BASE}/balance-sheet-statement/{ticker}?limit=10&apikey={DEMO_KEY}"
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"  Got {len(data)} years of data")
            print(f"  Keys: {list(data[0].keys())[:15]}")
            print(f"  Dates: {[d['date'] for d in data]}")
            fph_ta = data[0].get("totalAssets")
            print(f"  Latest totalAssets: {fph_ta/1e9:.3f}B" if fph_ta else "  totalAssets: N/A")
        elif isinstance(data, dict) and "Error Message" in data:
            print(f"  FMP error: {data['Error Message']}")
        else:
            print(f"  Response: {str(data)[:300]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Check FMP company profile for sector
    print(f"\nTesting FMP profile/sector for {ticker}...")
    try:
        url = f"{FMP_BASE}/profile/{ticker}?apikey={DEMO_KEY}"
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            profile = data[0]
            print(f"  sector: {profile.get('sector')}")
            print(f"  industry: {profile.get('industry')}")
            print(f"  exchange: {profile.get('exchangeShortName')}")
        elif isinstance(data, dict) and "Error Message" in data:
            print(f"  FMP error: {data['Error Message']}")
        else:
            print(f"  Response: {str(data)[:300]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Also test with AAPL to confirm the API structure works
    print(f"\nTesting FMP balance sheet for AAPL (demo key reference test)...")
    try:
        url = f"{FMP_BASE}/balance-sheet-statement/AAPL?limit=3&apikey={DEMO_KEY}"
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"  Got {len(data)} years for AAPL - API structure working")
            print(f"  Available keys: {[k for k in data[0].keys()]}")
        else:
            print(f"  Unexpected: {str(data)[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4: OpenFIGI - sector/GICS mapping
# ─────────────────────────────────────────────────────────────────────────────

def probe_openfigi():
    section("SOURCE 4: OpenFIGI - sector/GICS classification")

    # OpenFIGI maps tickers to FIGIs and returns security metadata
    # Free tier: 25 req/min, 250/day without API key; with free key: 25 req/min, 25000/day
    url = "https://api.openfigi.com/v3/mapping"
    headers = {"Content-Type": "application/json"}

    # Test NZX tickers - use ticker + exchCode NZE (NZX)
    payload = [
        {"idType": "TICKER", "idValue": "FPH", "exchCode": "NZ"},
        {"idType": "TICKER", "idValue": "SPK", "exchCode": "NZ"},
        {"idType": "TICKER", "idValue": "RYM", "exchCode": "NZ"},
        {"idType": "TICKER", "idValue": "MEL", "exchCode": "NZ"},
        {"idType": "TICKER", "idValue": "AIR", "exchCode": "NZ"},
    ]

    print(f"\nQuerying OpenFIGI for {len(payload)} NZX tickers...")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for i, result in enumerate(data):
            ticker_in = payload[i]["idValue"]
            if "data" in result and result["data"]:
                for item in result["data"]:
                    figi = item.get("figi")
                    sec_type = item.get("securityType")
                    name = item.get("name")
                    exch = item.get("exchCode")
                    print(f"  {ticker_in}: figi={figi}, name={name}, secType={sec_type}, exch={exch}")
            elif "error" in result:
                print(f"  {ticker_in}: ERROR - {result['error']}")
            else:
                print(f"  {ticker_in}: {result}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Note: OpenFIGI does not return GICS sector - it returns security type
    # For GICS, we need a different source
    print("\n  Note: OpenFIGI returns FIGI + security type, not GICS sector classification")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 5: NZX own data via public API / website
# ─────────────────────────────────────────────────────────────────────────────

def probe_nzx_api():
    section("SOURCE 5: NZX public API")
    # NZX has a public API at https://www.nzx.com/
    # Check if instrument data (sector) is available
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    # NZX API instrument endpoint
    ticker = "FPH"
    url = f"https://www.nzx.com/api/instruments/{ticker}"
    print(f"\nTesting NZX API for {ticker}...")
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Keys: {list(data.keys())[:20]}")
            # Look for sector
            for key in ["sector", "industry", "gics", "classification", "type"]:
                if key in data:
                    print(f"  {key}: {data[key]}")
        else:
            print(f"  Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Also try NZX securities endpoint
    url2 = f"https://api.nzx.com/v1/securities/{ticker}"
    print(f"\nTesting NZX securities API for {ticker}...")
    try:
        resp = requests.get(url2, headers=headers, timeout=15)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Keys: {list(data.keys())[:20]}")
        else:
            print(f"  Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation summary
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(yf_results, yf_ts_results):
    section("CROSS-VALIDATION vs Known Reference Values")

    print("\nFPH.NZ (Fisher & Paykel Healthcare):")
    print("  Reference FY2023: TotalAssets ~2.2B NZD, OperatingCashFlow ~400M NZD")

    # From yfinance
    ticker = yf.Ticker("FPH.NZ")
    try:
        bs = ticker.balance_sheet
        cf = ticker.cashflow
        if bs is not None and not bs.empty and "Total Assets" in bs.index:
            latest_col = bs.columns[0]
            ta = bs.loc["Total Assets", latest_col]
            print(f"  yfinance TotalAssets ({str(latest_col.date())}): {ta/1e9:.3f}B NZD ({'~OK' if 1.8 < ta/1e9 < 2.6 else 'MISMATCH'})")
        if cf is not None and not cf.empty and "Operating Cash Flow" in cf.index:
            latest_col = cf.columns[0]
            ocf = cf.loc["Operating Cash Flow", latest_col]
            print(f"  yfinance OCF ({str(latest_col.date())}): {ocf/1e9:.3f}B NZD ({'~OK' if 0.2 < ocf/1e9 < 0.6 else 'MISMATCH'})")
    except Exception as e:
        print(f"  yfinance validation ERROR: {e}")

    # From YF timeseries API
    if "annualTotalAssets" in yf_ts_results:
        ta = yf_ts_results["annualTotalAssets"]["latest_value"]
        print(f"  YF timeseries TotalAssets (latest): {ta/1e9:.3f}B NZD ({'~OK' if ta and 1.8 < ta/1e9 < 2.6 else 'MISMATCH'})")
    if "annualOperatingCashFlow" in yf_ts_results:
        ocf = yf_ts_results["annualOperatingCashFlow"]["latest_value"]
        print(f"  YF timeseries OCF (latest): {ocf/1e9:.3f}B NZD ({'~OK' if ocf and 0.2 < ocf/1e9 < 0.6 else 'MISMATCH'})")

    print("\nSPK.NZ (Spark NZ):")
    print("  Reference FY2023: TotalAssets ~4.5B NZD, OperatingCashFlow ~540M NZD")
    ticker2 = yf.Ticker("SPK.NZ")
    try:
        bs2 = ticker2.balance_sheet
        cf2 = ticker2.cashflow
        if bs2 is not None and not bs2.empty and "Total Assets" in bs2.index:
            latest_col = bs2.columns[0]
            ta2 = bs2.loc["Total Assets", latest_col]
            print(f"  yfinance TotalAssets ({str(latest_col.date())}): {ta2/1e9:.3f}B NZD ({'~OK' if 3.5 < ta2/1e9 < 5.5 else 'MISMATCH'})")
        if cf2 is not None and not cf2.empty and "Operating Cash Flow" in cf2.index:
            latest_col = cf2.columns[0]
            ocf2 = cf2.loc["Operating Cash Flow", latest_col]
            print(f"  yfinance OCF ({str(latest_col.date())}): {ocf2/1e9:.3f}B NZD ({'~OK' if 0.3 < ocf2/1e9 < 0.8 else 'MISMATCH'})")
    except Exception as e:
        print(f"  yfinance validation ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"NZX Data Source Viability Probe — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    yf_results = probe_yfinance()
    yf_ts_results = probe_yf_timeseries_api()
    probe_fmp()
    probe_openfigi()
    probe_nzx_api()
    cross_validate(yf_results, yf_ts_results)

    print(f"\n{SEP}\nProbe complete.\n{SEP}\n")
