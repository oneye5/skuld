"""
Deep probe: history depth, OCF type names, full universe coverage, SimFin + Alpha Vantage.

Run: uv run python scripts/data_source_probe2.py
"""

import requests
import yfinance as yf
import pandas as pd
from datetime import datetime

SEP = "=" * 70

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBE A: Discover actual OCF type name in YF timeseries API
# Try many possible names for operating cash flow
# ─────────────────────────────────────────────────────────────────────────────

def probe_ocf_type_names():
    section("PROBE A: Discover correct OCF type name in YF timeseries API")

    ticker = "FPH.NZ"
    headers = {"User-Agent": "Mozilla/5.0"}

    # Broad fetch - get ALL available types for this ticker
    # The timeseries API also supports fetching all types if we pass a wildcard or known prefix
    # Let's fetch with many possible names and see what comes back
    possible_cf_types = [
        "annualOperatingCashFlow",
        "annualCashFlowFromContinuingOperatingActivities",
        "annualNetCashProvidedByOperatingActivities",
        "annualNetCashProvidedByFinancingActivities",
        "annualNetCashProvidedByInvestingActivities",
        "annualDividendsPaid",
        "annualDividendPaid",
        "annualCommonStockDividendPaid",
        "annualRepurchaseOfCapitalStock",
        "annualCommonStockRepurchase",
        "annualDepreciationAndAmortization",
        "annualDeferredIncomeTax",
        "annualChangeInWorkingCapital",
        "annualChangesInWorkingCapital",
        "annualPurchaseOfBusiness",
        "annualCapitalExpenditureReported",
        "annualNetInvestingCashFlow",
        "annualNetFinancingCashFlow",
        "annualNetOperatingCashFlow",
        "annualBeginningCashPosition",
        "annualEndCashPosition",
        "annualChangesInCash",
        "annualCashDividendsPaid",
    ]
    type_str = ",".join(possible_cf_types)
    url = (
        f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}"
        f"?merge=false&padTimeSeries=true&period1=493590046&period2=2750557599"
        f"&type={type_str}&lang=en-NZ&region=NZ"
    )

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results_arr = data.get("timeseries", {}).get("result", [])

        print(f"\n  Returned {len(results_arr)} objects for {len(possible_cf_types)} requested types\n")
        for result in results_arr:
            meta = result.get("meta", {})
            feat_type = meta.get("type", ["?"])[0]
            if feat_type in result and result[feat_type]:
                data_points = [p for p in result[feat_type] if p is not None]
                if data_points:
                    dates = [p["asOfDate"] for p in data_points]
                    vals = [p.get("reportedValue", {}).get("raw") for p in data_points]
                    print(f"  [FOUND] {feat_type}: {len(data_points)} obs, {min(dates)}–{max(dates)}")
                    for d, v in zip(dates, vals):
                        print(f"         {d}: {v/1e9:.3f}B" if v else f"         {d}: null")
    except Exception as e:
        print(f"  ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBE B: History depth — how far back does YF data go for NZX tickers?
# ─────────────────────────────────────────────────────────────────────────────

def probe_history_depth():
    section("PROBE B: History depth — yfinance annual & quarterly")

    tickers = ["FPH.NZ", "SPK.NZ", "RYM.NZ", "MEL.NZ", "AIR.NZ", "ATM.NZ", "FNZ.NZ", "SKT.NZ"]

    for t_str in tickers:
        t = yf.Ticker(t_str)
        try:
            bs = t.balance_sheet
            qbs = t.quarterly_balance_sheet

            if bs is not None and not bs.empty:
                dates = sorted(bs.columns)
                oldest = str(dates[0].date())
                newest = str(dates[-1].date())
                n_years = len(dates)
            else:
                oldest = newest = "N/A"
                n_years = 0

            if qbs is not None and not qbs.empty:
                qdates = sorted(qbs.columns)
                q_oldest = str(qdates[0].date())
                q_newest = str(qdates[-1].date())
                n_q = len(qdates)
            else:
                q_oldest = q_newest = "N/A"
                n_q = 0

            print(f"  {t_str}: annual {n_years}yr [{oldest}→{newest}] | quarterly {n_q}q [{q_oldest}→{q_newest}]")
        except Exception as e:
            print(f"  {t_str}: ERROR - {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBE C: Full NZX universe coverage via yfinance
# ─────────────────────────────────────────────────────────────────────────────

def probe_nzx_universe_coverage():
    section("PROBE C: NZX universe coverage via yfinance balance_sheet + sector")

    # Sample of 30 NZX tickers across different caps
    nzx_sample = [
        "FPH.NZ", "SPK.NZ", "RYM.NZ", "MEL.NZ", "AIR.NZ",
        "ATM.NZ", "FNZ.NZ", "SKT.NZ", "IFT.NZ", "MFT.NZ",
        "ARG.NZ", "PCT.NZ", "CEN.NZ", "MCY.NZ", "HGH.NZ",
        "NPX.NZ", "VCT.NZ", "IPL.NZ", "NZR.NZ", "THL.NZ",
        "SML.NZ", "OCA.NZ", "KMD.NZ", "EBO.NZ", "GTK.NZ",
        "PFI.NZ", "CMO.NZ", "VGL.NZ", "AFT.NZ", "NWF.NZ",
    ]

    has_bs = 0
    has_cf = 0
    has_sector = 0
    no_data = []
    sector_counts = {}

    for t_str in nzx_sample:
        t = yf.Ticker(t_str)
        try:
            info = t.info
            bs = t.balance_sheet
            cf = t.cashflow
            sector = info.get("sector", "N/A")

            got_bs = bs is not None and not bs.empty
            got_cf = cf is not None and not cf.empty
            got_sector = sector not in ("N/A", None, "")

            if got_bs: has_bs += 1
            if got_cf: has_cf += 1
            if got_sector:
                has_sector += 1
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            status = "BS+" if got_bs else "BS-"
            status += " CF+" if got_cf else " CF-"
            print(f"  {t_str}: {status} | sector={sector}")

            if not got_bs and not got_cf:
                no_data.append(t_str)
        except Exception as e:
            print(f"  {t_str}: ERROR - {e}")
            no_data.append(t_str)

    n = len(nzx_sample)
    print(f"\n  Summary ({n} tickers sampled):")
    print(f"    Balance sheet:  {has_bs}/{n} ({100*has_bs/n:.0f}%)")
    print(f"    Cash flow:      {has_cf}/{n} ({100*has_cf/n:.0f}%)")
    print(f"    Sector:         {has_sector}/{n} ({100*has_sector/n:.0f}%)")
    print(f"    No data at all: {no_data}")
    print(f"    Sector distribution: {sector_counts}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBE D: SimFin — NZX coverage test
# ─────────────────────────────────────────────────────────────────────────────

def probe_simfin():
    section("PROBE D: SimFin API — NZX coverage")

    # SimFin free API: https://simfin.com/api/v2/
    # No key needed for basic queries; rate limited
    SIMFIN_BASE = "https://simfin.com/api/v2"

    # Search for NZX companies
    print("\nSearching SimFin for NZX companies (market=NZ)...")
    try:
        resp = requests.get(f"{SIMFIN_BASE}/companies/list?market=NZ", timeout=15)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Response type: {type(data)}")
            print(f"  Preview: {str(data)[:500]}")
        else:
            print(f"  Response: {resp.text[:300]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Try with API key header if needed
    print("\nSearching SimFin for FPH specifically...")
    try:
        resp = requests.get(f"{SIMFIN_BASE}/companies/search?query=Fisher+Paykel", timeout=15)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Preview: {str(data)[:500]}")
        else:
            print(f"  Response: {resp.text[:300]}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBE E: Alpha Vantage — NZX fundamental coverage
# ─────────────────────────────────────────────────────────────────────────────

def probe_alpha_vantage():
    section("PROBE E: Alpha Vantage — NZX fundamental coverage")

    # Alpha Vantage free: 25 req/day without key, limited to demo
    # With free key: 25 req/day
    # Let's test if NZX tickers work at all
    AV_BASE = "https://www.alphavantage.co/query"
    KEY = "demo"  # demo key, works only for certain tickers

    # Test balance sheet for NZX ticker (AV uses different format: NZE:FPH?)
    # Alpha Vantage typically uses EXCHANGE:TICKER format for non-US
    test_cases = ["NZE:FPH", "FPH.NZ", "FPH"]

    for ticker_fmt in test_cases:
        print(f"\nTesting Alpha Vantage BALANCE_SHEET for '{ticker_fmt}'...")
        try:
            resp = requests.get(
                AV_BASE,
                params={"function": "BALANCE_SHEET", "symbol": ticker_fmt, "apikey": KEY},
                timeout=15
            )
            data = resp.json()
            if "annualReports" in data:
                reports = data["annualReports"]
                print(f"  Got {len(reports)} annual reports")
                if reports:
                    print(f"  Dates: {[r['fiscalDateEnding'] for r in reports]}")
                    print(f"  Keys: {list(reports[0].keys())[:10]}")
            elif "Information" in data:
                print(f"  Info: {data['Information'][:200]}")
            elif "Note" in data:
                print(f"  Note: {data['Note'][:200]}")
            else:
                print(f"  Response: {str(data)[:300]}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Also test OVERVIEW for sector
    for ticker_fmt in test_cases[:2]:
        print(f"\nTesting Alpha Vantage OVERVIEW for '{ticker_fmt}'...")
        try:
            resp = requests.get(
                AV_BASE,
                params={"function": "OVERVIEW", "symbol": ticker_fmt, "apikey": KEY},
                timeout=15
            )
            data = resp.json()
            if "Sector" in data:
                print(f"  Sector: {data.get('Sector')}")
                print(f"  Industry: {data.get('Industry')}")
                print(f"  Exchange: {data.get('Exchange')}")
            else:
                print(f"  Response: {str(data)[:300]}")
        except Exception as e:
            print(f"  ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBE F: YF timeseries — how far back does annual data go?
# Try with extended period and check actual history
# ─────────────────────────────────────────────────────────────────────────────

def probe_yf_history_depth():
    section("PROBE F: YF timeseries API — actual historical depth for key types")

    # Test a few tickers that have been listed for a long time
    tickers = ["SPK.NZ", "AIR.NZ", "FPH.NZ"]  # SPK and AIR have been listed since 1990s
    headers = {"User-Agent": "Mozilla/5.0"}

    for ticker in tickers:
        print(f"\n  {ticker}:")
        url = (
            f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}"
            f"?merge=false&padTimeSeries=true&period1=493590046&period2=2750557599"
            f"&type=annualTotalAssets,annualStockholdersEquity,annualTotalRevenue&lang=en-NZ&region=NZ"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("timeseries", {}).get("result", [])
            for result in results:
                meta = result.get("meta", {})
                feat = meta.get("type", ["?"])[0]
                if feat in result and result[feat]:
                    pts = [p for p in result[feat] if p is not None]
                    if pts:
                        dates = [p["asOfDate"] for p in pts]
                        vals = [p.get("reportedValue", {}).get("raw") for p in pts]
                        print(f"    {feat}: {len(pts)} obs [{min(dates)}–{max(dates)}]")
                        # Show all available dates
                        for d, v in zip(dates, vals):
                            print(f"      {d}: {v/1e9:.3f}B" if v else f"      {d}: null")
        except Exception as e:
            print(f"    ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROBE G: Check yfinance quarterly data depth
# ─────────────────────────────────────────────────────────────────────────────

def probe_quarterly_depth():
    section("PROBE G: yfinance quarterly balance sheet depth")

    for t_str in ["FPH.NZ", "SPK.NZ", "AIR.NZ"]:
        t = yf.Ticker(t_str)
        try:
            qbs = t.quarterly_balance_sheet
            qcf = t.quarterly_cashflow

            if qbs is not None and not qbs.empty:
                qdates = sorted(qbs.columns, reverse=True)
                print(f"\n  {t_str} quarterly BS: {len(qdates)} quarters")
                print(f"    Dates: {[str(d.date()) for d in qdates]}")
                if "Total Assets" in qbs.index:
                    vals = {str(d.date()): f"{qbs.loc['Total Assets', d]/1e9:.3f}B" for d in qdates if pd.notna(qbs.loc['Total Assets', d])}
                    print(f"    Total Assets: {vals}")
            else:
                print(f"\n  {t_str} quarterly BS: EMPTY")

            if qcf is not None and not qcf.empty:
                qcf_dates = sorted(qcf.columns, reverse=True)
                print(f"  {t_str} quarterly CF: {len(qcf_dates)} quarters")
            else:
                print(f"  {t_str} quarterly CF: EMPTY")
        except Exception as e:
            print(f"  {t_str}: ERROR - {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"NZX Deep Data Probe — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    probe_ocf_type_names()
    probe_yf_history_depth()
    probe_quarterly_depth()
    probe_history_depth()
    probe_nzx_universe_coverage()
    probe_simfin()
    probe_alpha_vantage()

    print(f"\n{SEP}\nDeep probe complete.\n{SEP}\n")
